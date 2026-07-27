#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <doca_comch.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>
#include <mpi.h>

#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "../common/comch_mpi_common.h"
#include "../common/dma_common.h"
#include "../common/doca_rdma_utils.h"

#include "uthash.h"
#include <pthread.h>   // ★ 追加, スレッドセーフになるようにmutex制御
#include <errno.h>
/* stdatomic.h は doca_rdma_utils.h 経由で C/C++ 対応済み */

#include "../common/timing_utils.h"

DOCA_LOG_REGISTER(COMCH_CLIENT);

/* =====================================================
 * Phase 5 Step 4: RDMA command slot structure
 * Must match DPU-side definition in comch_server.c
 * ===================================================== */
#define CMD_SLOT_SIZE 4096

struct rdma_compact_cmd {
    /* ---- Header (8 bytes) ---- */
    volatile uint64_t seq;               /* sequence number (0 = empty, written LAST by Host) */
    /* ---- Payload ---- */
    uint32_t op_type;                    /* COLLECTIVE_ALL_GATHER, etc. */
    uint32_t _pad0;
    uint64_t src_addr;                   /* GPU src virtual address */
    uint64_t src_len;                    /* src buffer length */
    uint64_t dst_addr;                   /* GPU dst virtual address */
    uint64_t dst_len;                    /* dst buffer length */
    /* Export descriptors (variable length, packed inline) */
    uint32_t src_export_len;
    uint32_t dst_export_len;
    uint32_t src_export_len_rail1;
    uint32_t dst_export_len_rail1;
    /* local_cpu fields */
    uint64_t local_cpu_addr;
    uint64_t local_cpu_len;
    uint32_t local_cpu_export_len;
    uint32_t _pad2;                      /* align local_cpu_flag_addr to 8 bytes */
    uint64_t local_cpu_flag_addr;
    uint64_t local_cpu_flag_len;
    uint32_t local_cpu_flag_export_len;
    uint32_t _pad1;
    /* Variable-length export data follows */
    uint8_t export_data[];
};

struct comch_ctrl_path_objects {
	struct doca_dev *hw_dev;	  /* Device used in the sample */
	struct doca_pe *pe;		  /* PE object used in the sample */
	struct doca_comch_client *client; /* Client object used in the sample */
	const char *text;		  /* Message to send to the server */
	uint32_t text_len;		  /* Length of message to send to the server */
	doca_error_t result;		  /* Holds result will be updated in callbacks */
	bool finish;			  /* Controls whether progress loop should be run */
	uint64_t rank;

    struct finish_flags_entry *finish_flags_map;
	/* DOCA RDMA: Host-DPU 接続用コンテキスト */
	struct doca_rdma_ctx_t host_rdma_ctx;

	/* Multi-Rail: 2 番目のポートの RDMA コンテキスト */
	struct doca_dev *hw_dev_rail1;         /* 2 番目の NIC デバイス */
	struct doca_rdma_ctx_t host_rdma_ctx_rail1;
	bool dual_rail;

	/* Phase 5 Step 3: RDMA Doorbell (DPU→Host 完了通知) */
	volatile uint64_t doorbell __attribute__((aligned(64)));
	struct doca_mmap *doorbell_mmap;
	const void *doorbell_export_desc;
	size_t doorbell_export_desc_len;

	/* Phase 5 Step 4: RDMA Command Slot (Host→DPU) */
	void *cmd_local_buf;                           /* local staging buffer for RDMA Write */
	struct doca_mmap *cmd_local_mmap;              /* local mmap for cmd_local_buf */
	struct doca_remote_mem_t dpu_cmd_rmem;         /* DPU command slot remote mmap */
	bool cmd_ring_enabled;
	uint64_t cmd_seq;                              /* monotonically increasing sequence number */
};

/* id ごとに持つ finish フラグ */
struct finish_flags_entry {
 	uint64_t id;  /* ハッシュキー */

    atomic_bool doca_send_task_finish;
	atomic_bool ucp_connect_host_dpu_finish; //UCPワーカ接続確立したかどうか
    atomic_bool ucp_create_ring_finish;
    atomic_bool ucp_collective_finish;

    UT_hash_handle hh; /* uthash 用ハンドル */
};

static struct comch_ctrl_path_objects global_sample_objects;
/* ★ finish_flags_mutex: ハッシュテーブル操作 (HASH_FIND/ADD/DEL) 専用 */
static pthread_mutex_t finish_flags_mutex = PTHREAD_MUTEX_INITIALIZER;

/* id に対応するエントリを取得。存在しなければ必要に応じて作成 */
static struct finish_flags_entry *get_finish_flags_entry(uint64_t id, bool create_if_missing)
{
    struct finish_flags_entry *entry = NULL;
    /* ★ ここで map 全体をロック */
    pthread_mutex_lock(&finish_flags_mutex);
    HASH_FIND(hh, global_sample_objects.finish_flags_map, &id, sizeof(id), entry);
    if (entry == NULL && create_if_missing) {
        entry = (struct finish_flags_entry *)calloc(1, sizeof(*entry));
        if (entry == NULL) {
            pthread_mutex_unlock(&finish_flags_mutex);
            return NULL;
        }
        entry->id = id;
        /* bool のフィールドは calloc で全部 false */
        HASH_ADD(hh, global_sample_objects.finish_flags_map, id,
                 sizeof(entry->id), entry);
    }
    pthread_mutex_unlock(&finish_flags_mutex);
    return entry;
}

static void free_finish_flags_map(void)
{
    struct finish_flags_entry *cur, *tmp;
    pthread_mutex_lock(&finish_flags_mutex);      // ★ 追加
    HASH_ITER(hh, global_sample_objects.finish_flags_map, cur, tmp) {
        HASH_DEL(global_sample_objects.finish_flags_map, cur);
        free(cur);
    }
    global_sample_objects.finish_flags_map = NULL;
    pthread_mutex_unlock(&finish_flags_mutex);    // ★ 追加
}

/* (addr,len) → rkey_buf のキャッシュエントリ */
struct rkey_cache_entry {
    uint64_t key;          /* make_addr_len_key(addr,len) の結果 */
    void    *rkey_buf;     /* DOCA: doca_mmap_export_rdma の export_desc */
    size_t   rkey_size;    /* DOCA: export_desc_len */
    struct doca_mmap *mmap; /* DOCA: エクスポート元の mmap (破棄用) */
    /* Phase 8: PCI export for Cross-GVMI */
    void    *pci_export_buf;
    size_t   pci_export_size;
    UT_hash_handle hh;
};

static inline uint64_t
make_addr_len_key(uint64_t addr, uint64_t len)
{
    uint64_t x = addr;
    x ^= len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x;
}

/* SRC/DST で別々にしておくとデバッグしやすい */
static struct rkey_cache_entry *g_src_rkey_cache = NULL;
static struct rkey_cache_entry *g_dst_rkey_cache = NULL;

/* Multi-Rail: rail1 用キャッシュ */
static struct rkey_cache_entry *g_src_rkey_cache_rail1 = NULL;
static struct rkey_cache_entry *g_dst_rkey_cache_rail1 = NULL;

static struct rkey_cache_entry *g_local_cpu_rkey_cache = NULL;
static struct rkey_cache_entry *g_local_cpu_flag_rkey_cache = NULL;

/* Phase 14: GPU flag pool globals.
 *   Python (run_zero) で torch.zeros(N, int32, cuda) を確保した後、
 *   register_flag_pool_py(addr, len) を呼ぶと:
 *     1. このアドレスに対して doca_mmap_export_rdma を行い rkey_buf を取得
 *     2. rkey_buf を CONTROL_CMD_UCP_INIT_FLAG_POOL で DPU に送信
 *     3. グローバル変数に保存し、各 AG cmd に flag_gpu_addr/flag_value を入れる
 *   DPU 側は import 後にキャッシュし、AG 完了時に inline RDMA Write でフラグを書く。 */
static uint64_t g_flag_pool_base_addr = 0;
static uint64_t g_flag_pool_total_len = 0;
static struct doca_mmap *g_flag_pool_mmap = NULL;
static struct doca_mmap *g_flag_pool_mmap_rail1 = NULL;
static const void *g_flag_pool_rkey_buf = NULL;
static size_t      g_flag_pool_rkey_buf_len = 0;
static const void *g_flag_pool_rkey_buf_rail1 = NULL;
static size_t      g_flag_pool_rkey_buf_len_rail1 = 0;
/* Phase 15: Cross-GVMI PCI export for flag pool (allows DPU to RDMA Write to GPU memory) */
static const void *g_flag_pool_pci_export_buf = NULL;
static size_t      g_flag_pool_pci_export_len = 0;
static const void *g_flag_pool_pci_export_buf_rail1 = NULL;
static size_t      g_flag_pool_pci_export_len_rail1 = 0;
static bool        g_flag_pool_enabled = false;

static struct rkey_cache_entry *
rkey_cache_find(struct rkey_cache_entry *cache, uint64_t key)
{
    struct rkey_cache_entry *e = NULL;
    HASH_FIND(hh, cache, &key, sizeof(key), e);
    return e;
}

static struct rkey_cache_entry *
rkey_cache_insert(struct rkey_cache_entry **cache,
                  uint64_t key,
                  void *rkey_buf, size_t rkey_size)
{
    struct rkey_cache_entry *e = (struct rkey_cache_entry *)malloc(sizeof(*e));
    if (!e)
        return NULL;

    e->key       = key;
    e->rkey_buf  = rkey_buf;
    e->rkey_size = rkey_size;

    HASH_ADD(hh, *cache, key, sizeof(e->key), e);
    return e;
}

void clean_comch_sample_objects();
void comch_client_shutdown(void);
static doca_error_t init_comch_ctrl_path_objects(const char *server_name, const char *dev_pci_addr);

//コマンドの長さは#define CONTROL_CMD_MAX_SIZE  256で固定
doca_error_t comch_send_control_cmd(struct control_cmd *send_cmd, uint64_t id)
{
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};
	struct doca_comch_task_send *task;
	struct doca_task *task_obj;
	struct doca_comch_connection *connection;
	doca_error_t result;
	union doca_data user_data;
	
	/* この id 用のエントリを取得（なければ作成） */
    struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to allocate finish_flags_entry");
        return DOCA_ERROR_NO_MEMORY; /* 適当なエラーコードに変更してOK */
    }

    atomic_store_explicit(&entry->doca_send_task_finish, false, memory_order_release);

	//送信用コマンドをそのままバッファであるsend_bufに詰め替える。長さはCONTROL_CMD_MAX_SIZEで固定する
	uint8_t send_buf[CONTROL_CMD_MAX_SIZE];
	size_t  send_cap = CONTROL_CMD_MAX_SIZE;
	doca_error_t st = control_cmd_pack(send_cmd, &send_cap, send_buf);
	if (st != DOCA_SUCCESS) {
		DOCA_LOG_ERR("control_cmd_pack failed");
		return st;
	}

	DOCA_CHECK(doca_comch_client_get_connection(global_sample_objects.client, &connection));
	//接続ハンドルにユーザデータを紐づけて、後で受信コールバックなどから逆引きできるようにする
	user_data.ptr = (void *)(&global_sample_objects);
	DOCA_CHECK(doca_comch_connection_set_user_data(connection, user_data));
	DOCA_CHECK(doca_comch_client_task_send_alloc_init(global_sample_objects.client, connection, send_buf, CONTROL_CMD_MAX_SIZE, &task));

	task_obj = doca_comch_task_send_as_task(task);

	/* タスクの user_data に id を詰める（void* 経由） */
    union doca_data task_user_data;
    task_user_data.ptr = (void *)(uintptr_t)id;  /* uint64_t → ポインタ変換 */
    doca_task_set_user_data(task_obj, task_user_data);

    DOCA_CHECK(doca_task_submit(task_obj));
	
	while (!atomic_load_explicit(&entry->doca_send_task_finish, memory_order_acquire)) {
		doca_pe_progress(global_sample_objects.pe);
	}

	return DOCA_SUCCESS;
}

/* =============================================================
 * Phase 14: GPU flag pool register
 *   Python から register_flag_pool_py(addr, len) で呼ばれる。
 *   - 同 GPU メモリ領域に対して doca_mmap_export_rdma を 1 度だけ実行
 *     (rail1 がある場合は rail1 dev でも export)
 *   - rkey_buf をグローバルに保存
 *   - INIT_FLAG_POOL コマンドを ComCh で DPU に送信
 *   - DPU は doca_remote_mem_create でインポートしてキャッシュ
 *
 *   想定タイミング: torch.cuda.init() → torch.zeros(.., int32, cuda='cuda') の後、
 *   かつ ucp_create_ring_request() の後 (ComCh が確立済みの状態)。
 * ============================================================= */
doca_error_t comch_register_flag_pool(uint64_t base_addr, uint64_t total_len)
{
    if (base_addr == 0 || total_len == 0) {
        DOCA_LOG_ERR("comch_register_flag_pool: invalid args (addr=0x%lx len=%lu)",
                     base_addr, total_len);
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (g_flag_pool_enabled) {
        DOCA_LOG_WARN("comch_register_flag_pool: already enabled, ignoring");
        return DOCA_SUCCESS;
    }

    struct doca_dev *rdma_dev = global_sample_objects.host_rdma_ctx.dev;
    if (!rdma_dev) {
        DOCA_LOG_ERR("comch_register_flag_pool: host rdma_dev not initialized");
        return DOCA_ERROR_BAD_STATE;
    }

    /* Phase 15: permissions に PCI_READ_WRITE を追加 (AG dst と同じパターン)。
     * これにより flag pool が GPU memory 上でも DPU 側が Cross-GVMI 経由で
     * アクセスできるようになる。 */
    const uint32_t flag_pool_access_flags =
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
        DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE |
        DOCA_ACCESS_FLAG_PCI_READ_WRITE;

    /* rail0 export */
    doca_error_t dret;
    dret = doca_mmap_create(&g_flag_pool_mmap);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool mmap create failed"); return dret; }
    dret = doca_mmap_set_permissions(g_flag_pool_mmap, flag_pool_access_flags);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool set_permissions failed"); doca_mmap_destroy(g_flag_pool_mmap); g_flag_pool_mmap = NULL; return dret; }
    dret = doca_mmap_set_memrange(g_flag_pool_mmap, (void *)(uintptr_t)base_addr, total_len);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool set_memrange failed"); doca_mmap_destroy(g_flag_pool_mmap); g_flag_pool_mmap = NULL; return dret; }
    dret = doca_mmap_add_dev(g_flag_pool_mmap, rdma_dev);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool add_dev rail0 failed"); doca_mmap_destroy(g_flag_pool_mmap); g_flag_pool_mmap = NULL; return dret; }
    dret = doca_mmap_start(g_flag_pool_mmap);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool start failed"); doca_mmap_destroy(g_flag_pool_mmap); g_flag_pool_mmap = NULL; return dret; }
    dret = doca_mmap_export_rdma(g_flag_pool_mmap, rdma_dev,
                                 &g_flag_pool_rkey_buf, &g_flag_pool_rkey_buf_len);
    if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("flag pool export_rdma rail0 failed"); doca_mmap_destroy(g_flag_pool_mmap); g_flag_pool_mmap = NULL; return dret; }

    /* Phase 15: PCI export (Cross-GVMI) — best effort。失敗しても継続 (legacy RDMA path で動作)。 */
    {
        doca_error_t pret = doca_mmap_export_pci(g_flag_pool_mmap, rdma_dev,
                                                 &g_flag_pool_pci_export_buf,
                                                 &g_flag_pool_pci_export_len);
        if (pret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("flag pool export_pci rail0 failed: %s (fallback to rdma rkey path)",
                          doca_error_get_descr(pret));
            g_flag_pool_pci_export_buf = NULL;
            g_flag_pool_pci_export_len = 0;
        }
    }

    /* rail1 export (best effort) */
    if (global_sample_objects.dual_rail) {
        struct doca_dev *rdma_dev_rail1 = global_sample_objects.host_rdma_ctx_rail1.dev;
        if (rdma_dev_rail1) {
            doca_error_t r1ret = doca_mmap_create(&g_flag_pool_mmap_rail1);
            if (r1ret == DOCA_SUCCESS) r1ret = doca_mmap_set_permissions(g_flag_pool_mmap_rail1,
                flag_pool_access_flags);
            if (r1ret == DOCA_SUCCESS) r1ret = doca_mmap_set_memrange(g_flag_pool_mmap_rail1, (void *)(uintptr_t)base_addr, total_len);
            if (r1ret == DOCA_SUCCESS) r1ret = doca_mmap_add_dev(g_flag_pool_mmap_rail1, rdma_dev_rail1);
            if (r1ret == DOCA_SUCCESS) r1ret = doca_mmap_start(g_flag_pool_mmap_rail1);
            if (r1ret == DOCA_SUCCESS) r1ret = doca_mmap_export_rdma(g_flag_pool_mmap_rail1, rdma_dev_rail1,
                                                                      &g_flag_pool_rkey_buf_rail1,
                                                                      &g_flag_pool_rkey_buf_len_rail1);
            if (r1ret != DOCA_SUCCESS) {
                DOCA_LOG_WARN("flag pool rail1 export failed, falling back to rail0 only");
                if (g_flag_pool_mmap_rail1) { doca_mmap_destroy(g_flag_pool_mmap_rail1); g_flag_pool_mmap_rail1 = NULL; }
                g_flag_pool_rkey_buf_rail1 = NULL;
                g_flag_pool_rkey_buf_len_rail1 = 0;
            } else {
                /* Phase 15: rail1 の PCI export も best effort */
                doca_error_t pret = doca_mmap_export_pci(g_flag_pool_mmap_rail1, rdma_dev_rail1,
                                                         &g_flag_pool_pci_export_buf_rail1,
                                                         &g_flag_pool_pci_export_len_rail1);
                if (pret != DOCA_SUCCESS) {
                    DOCA_LOG_WARN("flag pool rail1 export_pci failed: %s",
                                  doca_error_get_descr(pret));
                    g_flag_pool_pci_export_buf_rail1 = NULL;
                    g_flag_pool_pci_export_len_rail1 = 0;
                }
            }
        }
    }

    g_flag_pool_base_addr = base_addr;
    g_flag_pool_total_len = total_len;

    /* INIT_FLAG_POOL コマンドを DPU に送信 */
    struct control_cmd init_cmd;
    memset(&init_cmd, 0, sizeof(init_cmd));
    init_cmd.type = CONTROL_CMD_UCP_INIT_FLAG_POOL;
    init_cmd.ucp_init_flag_pool.base_addr = base_addr;
    init_cmd.ucp_init_flag_pool.length = total_len;
    init_cmd.ucp_init_flag_pool.rkey_buf = (void *)g_flag_pool_rkey_buf;
    init_cmd.ucp_init_flag_pool.rkey_buf_len = g_flag_pool_rkey_buf_len;
    init_cmd.ucp_init_flag_pool.rkey_buf_rail1 = (void *)g_flag_pool_rkey_buf_rail1;
    init_cmd.ucp_init_flag_pool.rkey_buf_len_rail1 = g_flag_pool_rkey_buf_len_rail1;
    /* Phase 15: Cross-GVMI PCI export desc (optional; empty if export_pci failed) */
    init_cmd.ucp_init_flag_pool.pci_export_buf = (void *)g_flag_pool_pci_export_buf;
    init_cmd.ucp_init_flag_pool.pci_export_buf_len = g_flag_pool_pci_export_len;
    init_cmd.ucp_init_flag_pool.pci_export_buf_rail1 = (void *)g_flag_pool_pci_export_buf_rail1;
    init_cmd.ucp_init_flag_pool.pci_export_buf_len_rail1 = g_flag_pool_pci_export_len_rail1;

    /* INIT 送信用に専用の id を使う (既存の collective id と衝突しない値) */
    uint64_t init_id = (uint64_t)0x14F1A6 /* 'FLAG' */;
    doca_error_t send_ret = comch_send_control_cmd(&init_cmd, init_id);
    if (send_ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send INIT_FLAG_POOL: %s", doca_error_get_descr(send_ret));
        doca_mmap_destroy(g_flag_pool_mmap);
        g_flag_pool_mmap = NULL;
        if (g_flag_pool_mmap_rail1) { doca_mmap_destroy(g_flag_pool_mmap_rail1); g_flag_pool_mmap_rail1 = NULL; }
        return send_ret;
    }

    g_flag_pool_enabled = true;
    if (global_sample_objects.rank == 0) {
        printf("[Phase15] flag pool registered: addr=0x%lx len=%lu rkey_len=%zu rail1=%s pci_len=%zu pci_rail1=%s\n",
               base_addr, total_len, g_flag_pool_rkey_buf_len,
               g_flag_pool_rkey_buf_rail1 ? "yes" : "no",
               g_flag_pool_pci_export_len,
               (g_flag_pool_pci_export_buf_rail1 && g_flag_pool_pci_export_len_rail1 > 0) ? "yes" : "no");
        fflush(stdout);
    }
    return DOCA_SUCCESS;
}

/* ---- DOCA RDMA ダミーコールバック (Host は RDMA タスクを発行しないが ctx 作成に必要) ---- */
static void host_rdma_read_comp_cb(struct doca_rdma_task_read *task,
                                    union doca_data task_user_data,
                                    union doca_data ctx_user_data)
{
	(void)task; (void)task_user_data; (void)ctx_user_data;
}
static void host_rdma_read_err_cb(struct doca_rdma_task_read *task,
                                   union doca_data task_user_data,
                                   union doca_data ctx_user_data)
{
	(void)task; (void)task_user_data; (void)ctx_user_data;
	DOCA_LOG_ERR("host_rdma_read_err_cb called");
}
/* Phase 5 Step 4: callback data for Host RDMA Write */
struct host_write_cb_data {
	struct doca_buf *src_buf;
	struct doca_buf *dst_buf;
	atomic_bool *done;
};

static void host_rdma_write_comp_cb(struct doca_rdma_task_write *task,
                                     union doca_data task_user_data,
                                     union doca_data ctx_user_data)
{
	(void)ctx_user_data;
	struct host_write_cb_data *cb = (struct host_write_cb_data *)task_user_data.ptr;
	if (cb) {
		if (cb->src_buf) doca_buf_dec_refcount(cb->src_buf, NULL);
		if (cb->dst_buf) doca_buf_dec_refcount(cb->dst_buf, NULL);
		doca_task_free(doca_rdma_task_write_as_task(task));
		if (cb->done) atomic_store_explicit(cb->done, true, memory_order_release);
		free(cb);
	} else {
		doca_task_free(doca_rdma_task_write_as_task(task));
	}
}
static void host_rdma_write_err_cb(struct doca_rdma_task_write *task,
                                    union doca_data task_user_data,
                                    union doca_data ctx_user_data)
{
	(void)ctx_user_data;
	DOCA_LOG_ERR("host_rdma_write_err_cb called");
	struct host_write_cb_data *cb = (struct host_write_cb_data *)task_user_data.ptr;
	if (cb) {
		if (cb->src_buf) doca_buf_dec_refcount(cb->src_buf, NULL);
		if (cb->dst_buf) doca_buf_dec_refcount(cb->dst_buf, NULL);
		doca_task_free(doca_rdma_task_write_as_task(task));
		if (cb->done) atomic_store_explicit(cb->done, true, memory_order_release);
		free(cb);
	} else {
		doca_task_free(doca_rdma_task_write_as_task(task));
	}
}

/* Phase 5 Step 4: Submit a single RDMA Write and wait for completion.
 * src_addr is within rdma_ctx->local_mmap, dst is within rmem->remote_mmap. */
static doca_error_t host_rdma_write_sync(
	struct doca_rdma_ctx_t *rdma_ctx,
	void *src_addr, size_t src_len,
	struct doca_remote_mem_t *rmem, void *dst_addr, size_t dst_len)
{
	doca_error_t ret;
	struct doca_buf *src_buf = NULL, *dst_buf = NULL;

	ret = doca_buf_inventory_buf_get_by_addr(rdma_ctx->buf_inv, rdma_ctx->local_mmap,
	                                          src_addr, src_len, &src_buf);
	if (ret != DOCA_SUCCESS) return ret;
	ret = doca_buf_set_data(src_buf, src_addr, src_len);
	if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(src_buf, NULL); return ret; }

	ret = doca_buf_inventory_buf_get_by_addr(rdma_ctx->buf_inv, rmem->remote_mmap,
	                                          dst_addr, dst_len, &dst_buf);
	if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(src_buf, NULL); return ret; }

	struct host_write_cb_data *cb =
		(struct host_write_cb_data *)calloc(1, sizeof(*cb));
	if (!cb) { doca_buf_dec_refcount(src_buf, NULL); doca_buf_dec_refcount(dst_buf, NULL); return DOCA_ERROR_NO_MEMORY; }
	atomic_bool write_done;
	atomic_store_explicit(&write_done, false, memory_order_release);
	cb->src_buf = src_buf;
	cb->dst_buf = dst_buf;
	cb->done = &write_done;

	union doca_data ud;
	ud.ptr = cb;

	struct doca_rdma_task_write *write_task;
	ret = doca_rdma_task_write_allocate_init(rdma_ctx->rdma,
	    rdma_ctx->connections[0], src_buf, dst_buf, ud, &write_task);
	if (ret != DOCA_SUCCESS) {
		doca_buf_dec_refcount(src_buf, NULL);
		doca_buf_dec_refcount(dst_buf, NULL);
		free(cb);
		return ret;
	}
	ret = doca_task_submit(doca_rdma_task_write_as_task(write_task));
	if (ret != DOCA_SUCCESS) {
		doca_task_free(doca_rdma_task_write_as_task(write_task));
		doca_buf_dec_refcount(src_buf, NULL);
		doca_buf_dec_refcount(dst_buf, NULL);
		free(cb);
		return ret;
	}

	while (!atomic_load_explicit(&write_done, memory_order_acquire))
		doca_pe_progress(rdma_ctx->pe);

	return DOCA_SUCCESS;
}

void ucp_connect_host_dpu_request(uint64_t id)
{
	struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to allocate finish_flags_entry");
        return;
    }
    atomic_store_explicit(&entry->ucp_connect_host_dpu_finish, false, memory_order_release);

	/* Phase 5 Step 4: Allocate local buffer for RDMA Write commands */
	global_sample_objects.cmd_local_buf = aligned_alloc(64, CMD_SLOT_SIZE);
	if (global_sample_objects.cmd_local_buf)
		memset(global_sample_objects.cmd_local_buf, 0, CMD_SLOT_SIZE);
	global_sample_objects.cmd_ring_enabled = false;
	global_sample_objects.cmd_seq = 1;

	/* DOCA RDMA コンテキスト作成 (Host-DPU 接続用、Phase 5 Step 4: local mmap for cmd staging) */
	doca_error_t ret;
	struct doca_rdma_ctx_t *rdma_host = &global_sample_objects.host_rdma_ctx;

	ret = doca_rdma_ctx_init(rdma_host, global_sample_objects.hw_dev, NULL,
	                         global_sample_objects.cmd_local_buf, CMD_SLOT_SIZE,
	                         DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
	                         32, 0,    /* recv_q_size=0: Host は Receive しない */
	                         3);       /* gid_index=3: RoCE v2 (Host 側) */
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("host rdma_ctx init failed: %s", doca_error_get_name(ret));
		return;
	}

	ret = doca_rdma_ctx_configure_and_start(rdma_host, 1, 4, 0, 0,
	                                        host_rdma_read_comp_cb, host_rdma_read_err_cb,
	                                        host_rdma_write_comp_cb, host_rdma_write_err_cb,
	                                        NULL, NULL, NULL, NULL,
	                                        NULL);
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("host rdma_ctx configure_and_start failed: %s", doca_error_get_name(ret));
		doca_rdma_ctx_destroy(rdma_host);
		return;
	}

	ret = doca_rdma_ctx_export(rdma_host, 0);
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("host rdma_ctx export failed: %s", doca_error_get_name(ret));
		doca_rdma_ctx_destroy(rdma_host);
		return;
	}

	/* ---- Multi-Rail: 2 番目のポートの RDMA コンテキスト作成 ---- */
	struct doca_rdma_ctx_t *rdma_rail1 = &global_sample_objects.host_rdma_ctx_rail1;
	global_sample_objects.dual_rail = false;

	/* Host PCI アドレスから他方のポートを推定 (e.g., 0000:b3:00.0 → 0000:b3:00.1) */
	{
		char pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE] = {0};
		doca_error_t dret = doca_devinfo_get_pci_addr_str(
		    doca_dev_as_devinfo(global_sample_objects.hw_dev), pci_addr);
		if (dret == DOCA_SUCCESS) {
			/* 末尾の function 番号を反転 (0→1, 1→0) */
			char other_pci[DOCA_DEVINFO_PCI_ADDR_SIZE];
			strncpy(other_pci, pci_addr, sizeof(other_pci) - 1);
			other_pci[sizeof(other_pci) - 1] = '\0';
			size_t len = strlen(other_pci);
			if (len > 0) {
				char last = other_pci[len - 1];
				other_pci[len - 1] = (last == '0') ? '1' : '0';
			}

			dret = open_doca_device_with_pci(other_pci, NULL, &global_sample_objects.hw_dev_rail1);
			if (dret == DOCA_SUCCESS) {
				DOCA_LOG_INFO("Multi-Rail: opened rail1 device: %s", other_pci);

				dret = doca_rdma_ctx_init(rdma_rail1, global_sample_objects.hw_dev_rail1, NULL,
				                          NULL, 0,
				                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
				                          32, 0, 3);
				if (dret == DOCA_SUCCESS) {
					dret = doca_rdma_ctx_configure_and_start(rdma_rail1, 1, 1, 0, 0,
					    host_rdma_read_comp_cb, host_rdma_read_err_cb,
					    host_rdma_write_comp_cb, host_rdma_write_err_cb,
					    NULL, NULL, NULL, NULL, NULL);
					if (dret == DOCA_SUCCESS) {
						dret = doca_rdma_ctx_export(rdma_rail1, 0);
						if (dret == DOCA_SUCCESS) {
							global_sample_objects.dual_rail = true;
							DOCA_LOG_INFO("Multi-Rail: rail1 RDMA context ready");
						} else {
							DOCA_LOG_WARN("Multi-Rail: rail1 export failed");
							doca_rdma_ctx_destroy(rdma_rail1);
						}
					} else {
						DOCA_LOG_WARN("Multi-Rail: rail1 configure failed");
						doca_rdma_ctx_destroy(rdma_rail1);
					}
				} else {
					DOCA_LOG_WARN("Multi-Rail: rail1 init failed");
				}
			} else {
				DOCA_LOG_WARN("Multi-Rail: failed to open rail1 device '%s'", other_pci);
			}
		}
	}

	/* ---- Phase 5 Step 3: Doorbell mmap 作成・エクスポート ---- */
	global_sample_objects.doorbell = 0;
	global_sample_objects.doorbell_mmap = NULL;
	global_sample_objects.doorbell_export_desc = NULL;
	global_sample_objects.doorbell_export_desc_len = 0;
	{
		struct doca_mmap *db_mmap = NULL;
		doca_error_t dret = doca_mmap_create(&db_mmap);
		if (dret == DOCA_SUCCESS) dret = doca_mmap_add_dev(db_mmap, rdma_host->dev);
		if (dret == DOCA_SUCCESS) dret = doca_mmap_set_memrange(db_mmap,
		    (void *)&global_sample_objects.doorbell, sizeof(uint64_t));
		if (dret == DOCA_SUCCESS) dret = doca_mmap_set_permissions(db_mmap,
		    DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE);
		if (dret == DOCA_SUCCESS) dret = doca_mmap_start(db_mmap);
		if (dret == DOCA_SUCCESS) {
			const void *exp_desc = NULL;
			size_t exp_len = 0;
			dret = doca_mmap_export_rdma(db_mmap, rdma_host->dev, &exp_desc, &exp_len);
			if (dret == DOCA_SUCCESS) {
				global_sample_objects.doorbell_mmap = db_mmap;
				global_sample_objects.doorbell_export_desc = exp_desc;
				global_sample_objects.doorbell_export_desc_len = exp_len;
				DOCA_LOG_INFO("Doorbell mmap exported (len=%zu)", exp_len);
			} else {
				DOCA_LOG_WARN("Doorbell mmap export failed: %s", doca_error_get_name(dret));
				doca_mmap_destroy(db_mmap);
			}
		} else {
			DOCA_LOG_WARN("Doorbell mmap setup failed: %s", doca_error_get_name(dret));
			if (db_mmap) doca_mmap_destroy(db_mmap);
		}
	}

	/* コマンドの作成・送信: DOCA RDMA 接続記述子を DPU に送る */
	struct control_cmd send_cmd;
	send_cmd.type = CONTROL_CMD_UCP_CONNECT_HOST_DPU;
	send_cmd.ucp_connect_host_dpu.id = id;
	send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address = (void *)rdma_host->local_conn_desc;
	send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address_len = rdma_host->local_conn_desc_len;

	/* Multi-Rail: rail1 の接続記述子を含める */
	if (global_sample_objects.dual_rail) {
		send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = (void *)rdma_rail1->local_conn_desc;
		send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 = rdma_rail1->local_conn_desc_len;
	} else {
		send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = NULL;
		send_cmd.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 = 0;
	}

	/* Doorbell export desc を含める */
	send_cmd.ucp_connect_host_dpu.doorbell_export_desc = (void *)global_sample_objects.doorbell_export_desc;
	send_cmd.ucp_connect_host_dpu.doorbell_export_desc_len = global_sample_objects.doorbell_export_desc_len;

	doca_error_t result = comch_send_control_cmd(&send_cmd, id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to comch send control cmd");
		return;
	}
	/* 接続確立は DPU からの通知受信時に execute_ucp_connect_host_dpu_notify で行う */
}

void ucp_create_ring_request(uint64_t id)
{
	doca_error_t result;
	uint32_t max_msg_size;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

	struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to get finish_flags_entry");
        return;
    }
	while (!atomic_load_explicit(&entry->ucp_connect_host_dpu_finish, memory_order_acquire)) {
		doca_pe_progress(global_sample_objects.pe);
	}

	atomic_store_explicit(&entry->ucp_create_ring_finish, false, memory_order_release);

	//ここでリング作成コマンドを作成する
	struct control_cmd send_cmd;
	send_cmd.type = CONTROL_CMD_UCP_CREATE_RING;
	send_cmd.ucp_create_ring.id = id;
	
	result = comch_send_control_cmd(&send_cmd, id);
	if(result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to comch send control cmd");
		return ;
	}

	while (!atomic_load_explicit(&entry->ucp_create_ring_finish, memory_order_acquire)) {
		doca_pe_progress(global_sample_objects.pe);
	}
	return ;
}

/* Doorbell monotonic sequence (sync パスでも使用するため前方定義) */
static volatile uint64_t g_doorbell_submit_seq = 0;

/* Submit mutex: mmap 作成 + ComCh 送信を直列化 (DOCA API はスレッドセーフでない) */
static pthread_mutex_t g_impl_submit_mtx = PTHREAD_MUTEX_INITIALIZER;

/* ---- Host-side mmap cache (submit mutex 内でアクセスするためロック不要) ---- */
#define HOST_MMAP_CACHE_SIZE 4096
struct host_mmap_cache_entry {
    uint64_t addr;
    uint64_t len;
    struct doca_dev *dev;
    struct doca_mmap *mmap;
    const void *export_desc;
    size_t export_desc_len;
    int valid;
};
static struct host_mmap_cache_entry g_host_mmap_cache[HOST_MMAP_CACHE_SIZE];
static uint64_t g_mmap_cache_hits = 0, g_mmap_cache_misses = 0;

static struct doca_mmap *host_mmap_cache_get(uint64_t addr, uint64_t len, struct doca_dev *dev,
                                              const void **out_desc, size_t *out_len)
{
    uint64_t h = (addr >> 12) ^ (len >> 4);
    uint32_t idx = (uint32_t)(h % HOST_MMAP_CACHE_SIZE);
    struct host_mmap_cache_entry *e = &g_host_mmap_cache[idx];
    if (e->valid && e->addr == addr && e->len == len && e->dev == dev) {
        *out_desc = e->export_desc;
        *out_len = e->export_desc_len;
        g_mmap_cache_hits++;
        return e->mmap;
    }
    return NULL;
}

static void host_mmap_cache_put(uint64_t addr, uint64_t len, struct doca_dev *dev,
                                struct doca_mmap *mmap, const void *desc, size_t desc_len)
{
    uint64_t h = (addr >> 12) ^ (len >> 4);
    uint32_t idx = (uint32_t)(h % HOST_MMAP_CACHE_SIZE);
    struct host_mmap_cache_entry *e = &g_host_mmap_cache[idx];
    if (e->valid && e->mmap && e->mmap != mmap) {
        doca_mmap_destroy(e->mmap);  /* 古いエントリを破棄 */
    }
    e->addr = addr; e->len = len; e->dev = dev;
    e->mmap = mmap; e->export_desc = desc; e->export_desc_len = desc_len;
    e->valid = 1;
    g_mmap_cache_misses++;
}

/* キャッシュ付き mmap 作成ヘルパー */
static doca_error_t host_get_or_create_mmap(uint64_t addr, uint64_t len, struct doca_dev *dev,
                                             const void **out_desc, size_t *out_desc_len,
                                             struct doca_mmap **out_mmap)
{
    struct doca_mmap *cached = host_mmap_cache_get(addr, len, dev, out_desc, out_desc_len);
    if (cached) {
        *out_mmap = cached;
        return DOCA_SUCCESS;
    }
    /* cache miss: 新規作成 */
    struct doca_mmap *mmap = NULL;
    doca_error_t dret;
    dret = doca_mmap_create(&mmap);
    if (dret != DOCA_SUCCESS) return dret;
    dret = doca_mmap_set_permissions(mmap,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ |
        DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
    if (dret != DOCA_SUCCESS) { doca_mmap_destroy(mmap); return dret; }
    dret = doca_mmap_set_memrange(mmap, (void *)addr, len);
    if (dret != DOCA_SUCCESS) { doca_mmap_destroy(mmap); return dret; }
    dret = doca_mmap_add_dev(mmap, dev);
    if (dret != DOCA_SUCCESS) { doca_mmap_destroy(mmap); return dret; }
    dret = doca_mmap_start(mmap);
    if (dret != DOCA_SUCCESS) { doca_mmap_destroy(mmap); return dret; }
    dret = doca_mmap_export_rdma(mmap, dev, out_desc, out_desc_len);
    if (dret != DOCA_SUCCESS) { doca_mmap_destroy(mmap); return dret; }
    host_mmap_cache_put(addr, len, dev, mmap, *out_desc, *out_desc_len);
    *out_mmap = mmap;
    return DOCA_SUCCESS;
}

/* local_cpu_buffer_len == 0 のときは通常の集合通信、!=0 のときはローカル CPU バッファリング併用。 */
doca_error_t ucp_collective_request_impl(uint64_t id, uint64_t src_buffer_address, uint64_t src_buffer_len,
	uint64_t dst_buffer_address, uint64_t dst_buffer_len, struct CollectiveRequest collective_request,
    uint64_t local_cpu_buffer_address, uint64_t local_cpu_buffer_len, uint64_t local_cpu_flag_buffer_address, uint64_t local_cpu_flag_buffer_len,
    bool wait_doorbell, uint64_t *out_submit_seq,
    uint64_t flag_gpu_addr, uint32_t flag_value)
{
	/* Submit mutex: mmap 作成〜ComCh 送信を直列化 (doorbell wait 前に解放) */
	pthread_mutex_lock(&g_impl_submit_mtx);

	struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to get finish_flags_entry");
        pthread_mutex_unlock(&g_impl_submit_mtx);
        return DOCA_ERROR_BAD_STATE;
    }
    atomic_store_explicit(&entry->ucp_collective_finish, false, memory_order_release);

	//メモリ周りの登録 (DOCA mmap + export_rdma)
	struct doca_dev *rdma_dev = global_sample_objects.host_rdma_ctx.dev;

	/* ======== SRC: (addr,len) → mmap export desc をキャッシュ ======== */
    uint64_t src_key = make_addr_len_key(src_buffer_address, src_buffer_len);

    struct rkey_cache_entry *src_entry =
        rkey_cache_find(g_src_rkey_cache, src_key);

    void *src_rkey_buf   = NULL;
    size_t src_rkey_size = 0;

    if (src_entry) {
        /* 既にエクスポート済み */
        src_rkey_buf   = src_entry->rkey_buf;
        src_rkey_size  = src_entry->rkey_size;
    } else {
        /* 初回: doca_mmap 作成 + export_rdma */
        struct doca_mmap *src_mmap = NULL;
        const void *src_export_desc = NULL;
        size_t src_export_desc_len = 0;
        doca_error_t dret;

        dret = doca_mmap_create(&src_mmap);
        if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("Failed to create src mmap"); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_set_permissions(src_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(src_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_set_memrange(src_mmap, (void *)src_buffer_address, src_buffer_len);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(src_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_add_dev(src_mmap, rdma_dev);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(src_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_start(src_mmap);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(src_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_export_rdma(src_mmap, rdma_dev, &src_export_desc, &src_export_desc_len);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(src_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }

        src_rkey_buf  = (void *)src_export_desc;
        src_rkey_size = src_export_desc_len;

        /* Phase 9: PCI export of src (for GPU Direct Ring step 0 Send) */
        const void *src_pci_desc = NULL;
        size_t src_pci_desc_len = 0;
        dret = doca_mmap_export_pci(src_mmap, rdma_dev, &src_pci_desc, &src_pci_desc_len);
        if (dret != DOCA_SUCCESS) src_pci_desc_len = 0;

        /* キャッシュに挿入 */
        src_entry = rkey_cache_insert(&g_src_rkey_cache,
                                      src_key, src_rkey_buf, src_rkey_size);
        if (src_entry) {
            src_entry->mmap = src_mmap;
            src_entry->pci_export_buf = (void *)src_pci_desc;
            src_entry->pci_export_size = src_pci_desc_len;
        } else {
            DOCA_LOG_ERR("Failed to insert src rkey cache");
        }
    }

    /* ======== DST 側もほぼ同じ処理 (DOCA mmap) ======== */
    uint64_t dst_key = make_addr_len_key(dst_buffer_address, dst_buffer_len);

    struct rkey_cache_entry *dst_entry =
        rkey_cache_find(g_dst_rkey_cache, dst_key);

    void *dst_rkey_buf   = NULL;
    size_t dst_rkey_size = 0;

    if (dst_entry) {
        dst_rkey_buf   = dst_entry->rkey_buf;
        dst_rkey_size  = dst_entry->rkey_size;
    } else {
        struct doca_mmap *dst_mmap = NULL;
        const void *dst_export_desc = NULL;
        size_t dst_export_desc_len = 0;
        doca_error_t dret;

        dret = doca_mmap_create(&dst_mmap);
        if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("Failed to create dst mmap"); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_set_permissions(dst_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(dst_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_set_memrange(dst_mmap, (void *)dst_buffer_address, dst_buffer_len);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(dst_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_add_dev(dst_mmap, rdma_dev);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(dst_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_start(dst_mmap);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(dst_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dret = doca_mmap_export_rdma(dst_mmap, rdma_dev, &dst_export_desc, &dst_export_desc_len);
        if (dret != DOCA_SUCCESS) { doca_mmap_destroy(dst_mmap); pthread_mutex_unlock(&g_impl_submit_mtx); return dret; }
        dst_rkey_buf  = (void *)dst_export_desc;
        dst_rkey_size = dst_export_desc_len;

        /* PCI export (Cross-GVMI: DPU が GPU メモリをローカル mmap としてインポート) */
        const void *dst_pci_desc = NULL;
        size_t dst_pci_desc_len = 0;
        dret = doca_mmap_export_pci(dst_mmap, rdma_dev, &dst_pci_desc, &dst_pci_desc_len);
        if (dret != DOCA_SUCCESS) dst_pci_desc_len = 0;

        dst_entry = rkey_cache_insert(&g_dst_rkey_cache, dst_key, dst_rkey_buf, dst_rkey_size);
        if (dst_entry) {
            dst_entry->mmap = dst_mmap;
            dst_entry->pci_export_buf = (void *)dst_pci_desc;
            dst_entry->pci_export_size = dst_pci_desc_len;
        } else {
            DOCA_LOG_ERR("Failed to insert dst rkey cache");
        }
    }

    /* ローカル CPU バッファ用の rkey 登録 (SRC/DST と同形) */
    void *local_cpu_rkey_buf = NULL;
    size_t local_cpu_rkey_size = 0;
    void *local_cpu_flag_rkey_buf = NULL;
    size_t local_cpu_flag_rkey_size = 0;
    if(local_cpu_buffer_address != 0) {
        //ローカルCPUバッファリング (DOCA mmap)
        uint64_t local_cpu_key = make_addr_len_key(local_cpu_buffer_address, local_cpu_buffer_len);

        struct rkey_cache_entry *local_cpu_entry =
            rkey_cache_find(g_local_cpu_rkey_cache, local_cpu_key);

        if (local_cpu_entry) {
            local_cpu_rkey_buf   = local_cpu_entry->rkey_buf;
            local_cpu_rkey_size  = local_cpu_entry->rkey_size;
        } else {
            struct doca_mmap *lc_mmap = NULL;
            const void *lc_export_desc = NULL;
            size_t lc_export_desc_len = 0;
            doca_error_t dret;

            dret = doca_mmap_create(&lc_mmap);
            if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("Failed to create local_cpu mmap"); return dret; }
            dret = doca_mmap_add_dev(lc_mmap, rdma_dev);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lc_mmap); return dret; }
            dret = doca_mmap_set_memrange(lc_mmap, (void *)local_cpu_buffer_address, local_cpu_buffer_len);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lc_mmap); return dret; }
            dret = doca_mmap_set_permissions(lc_mmap,
                DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lc_mmap); return dret; }
            dret = doca_mmap_start(lc_mmap);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lc_mmap); return dret; }
            dret = doca_mmap_export_rdma(lc_mmap, rdma_dev, &lc_export_desc, &lc_export_desc_len);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lc_mmap); return dret; }

            local_cpu_rkey_buf  = (void *)lc_export_desc;
            local_cpu_rkey_size = lc_export_desc_len;

            local_cpu_entry = rkey_cache_insert(&g_local_cpu_rkey_cache,
                                        local_cpu_key, local_cpu_rkey_buf, local_cpu_rkey_size);
            if (local_cpu_entry) local_cpu_entry->mmap = lc_mmap;
        }

        //ローカルCPU用フラグ (DOCA mmap)
        uint64_t local_cpu_flag_key = make_addr_len_key(local_cpu_flag_buffer_address, local_cpu_flag_buffer_len);

        struct rkey_cache_entry *local_cpu_flag_entry =
            rkey_cache_find(g_local_cpu_flag_rkey_cache, local_cpu_flag_key);

        if (local_cpu_flag_entry) {
            local_cpu_flag_rkey_buf   = local_cpu_flag_entry->rkey_buf;
            local_cpu_flag_rkey_size  = local_cpu_flag_entry->rkey_size;
        } else {
            struct doca_mmap *lcf_mmap = NULL;
            const void *lcf_export_desc = NULL;
            size_t lcf_export_desc_len = 0;
            doca_error_t dret;

            dret = doca_mmap_create(&lcf_mmap);
            if (dret != DOCA_SUCCESS) { DOCA_LOG_ERR("Failed to create local_cpu_flag mmap"); return dret; }
            dret = doca_mmap_add_dev(lcf_mmap, rdma_dev);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lcf_mmap); return dret; }
            dret = doca_mmap_set_memrange(lcf_mmap, (void *)local_cpu_flag_buffer_address, local_cpu_flag_buffer_len);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lcf_mmap); return dret; }
            dret = doca_mmap_set_permissions(lcf_mmap,
                DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lcf_mmap); return dret; }
            dret = doca_mmap_start(lcf_mmap);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lcf_mmap); return dret; }
            dret = doca_mmap_export_rdma(lcf_mmap, rdma_dev, &lcf_export_desc, &lcf_export_desc_len);
            if (dret != DOCA_SUCCESS) { doca_mmap_destroy(lcf_mmap); return dret; }

            local_cpu_flag_rkey_buf  = (void *)lcf_export_desc;
            local_cpu_flag_rkey_size = lcf_export_desc_len;

            local_cpu_flag_entry = rkey_cache_insert(&g_local_cpu_flag_rkey_cache,
                                        local_cpu_flag_key, local_cpu_flag_rkey_buf, local_cpu_flag_rkey_size);
            if (local_cpu_flag_entry) local_cpu_flag_entry->mmap = lcf_mmap;
        }
    }


	/* ---- Multi-Rail: rail1 デバイスでも SRC/DST mmap をエクスポート ---- */
	void *src_rkey_buf_rail1 = NULL;
	size_t src_rkey_size_rail1 = 0;
	void *dst_rkey_buf_rail1 = NULL;
	size_t dst_rkey_size_rail1 = 0;

	if (global_sample_objects.dual_rail) {
		struct doca_dev *rdma_dev_rail1 = global_sample_objects.host_rdma_ctx_rail1.dev;

		/* SRC rail1 export */
		struct rkey_cache_entry *src_entry_r1 = rkey_cache_find(g_src_rkey_cache_rail1, src_key);
		if (src_entry_r1) {
			src_rkey_buf_rail1 = src_entry_r1->rkey_buf;
			src_rkey_size_rail1 = src_entry_r1->rkey_size;
		} else {
			struct doca_mmap *src_mmap_r1 = NULL;
			const void *src_exp_r1 = NULL;
			size_t src_exp_len_r1 = 0;
			doca_error_t dret;

			dret = doca_mmap_create(&src_mmap_r1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_add_dev(src_mmap_r1, rdma_dev_rail1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_set_memrange(src_mmap_r1, (void *)src_buffer_address, src_buffer_len);
			/* Phase 12.2: rail1 でも PCI export が必要なので PCI_READ_WRITE 権限を追加 */
			if (dret == DOCA_SUCCESS) dret = doca_mmap_set_permissions(src_mmap_r1,
			    DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_start(src_mmap_r1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_export_rdma(src_mmap_r1, rdma_dev_rail1, &src_exp_r1, &src_exp_len_r1);
			if (dret == DOCA_SUCCESS) {
				src_rkey_buf_rail1 = (void *)src_exp_r1;
				src_rkey_size_rail1 = src_exp_len_r1;
				src_entry_r1 = rkey_cache_insert(&g_src_rkey_cache_rail1, src_key, src_rkey_buf_rail1, src_rkey_size_rail1);
				if (src_entry_r1) {
					src_entry_r1->mmap = src_mmap_r1;
					/* Phase 12.2: rail1 PCI export (GPU Direct dual-rail Ring 用) */
					const void *src_pci_r1 = NULL;
					size_t src_pci_len_r1 = 0;
					doca_error_t pret = doca_mmap_export_pci(src_mmap_r1, rdma_dev_rail1, &src_pci_r1, &src_pci_len_r1);
					if (pret == DOCA_SUCCESS) {
						src_entry_r1->pci_export_buf = (void *)src_pci_r1;
						src_entry_r1->pci_export_size = src_pci_len_r1;
					}
				}
			} else {
				DOCA_LOG_WARN("Multi-Rail: src mmap rail1 export failed");
				if (src_mmap_r1) doca_mmap_destroy(src_mmap_r1);
			}
		}

		/* DST rail1 export */
		struct rkey_cache_entry *dst_entry_r1 = rkey_cache_find(g_dst_rkey_cache_rail1, dst_key);
		if (dst_entry_r1) {
			dst_rkey_buf_rail1 = dst_entry_r1->rkey_buf;
			dst_rkey_size_rail1 = dst_entry_r1->rkey_size;
		} else {
			struct doca_mmap *dst_mmap_r1 = NULL;
			const void *dst_exp_r1 = NULL;
			size_t dst_exp_len_r1 = 0;
			doca_error_t dret;

			dret = doca_mmap_create(&dst_mmap_r1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_add_dev(dst_mmap_r1, rdma_dev_rail1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_set_memrange(dst_mmap_r1, (void *)dst_buffer_address, dst_buffer_len);
			/* Phase 12.2: rail1 でも PCI export が必要なので PCI_READ_WRITE 権限を追加 */
			if (dret == DOCA_SUCCESS) dret = doca_mmap_set_permissions(dst_mmap_r1,
			    DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_PCI_READ_WRITE);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_start(dst_mmap_r1);
			if (dret == DOCA_SUCCESS) dret = doca_mmap_export_rdma(dst_mmap_r1, rdma_dev_rail1, &dst_exp_r1, &dst_exp_len_r1);
			if (dret == DOCA_SUCCESS) {
				dst_rkey_buf_rail1 = (void *)dst_exp_r1;
				dst_rkey_size_rail1 = dst_exp_len_r1;
				dst_entry_r1 = rkey_cache_insert(&g_dst_rkey_cache_rail1, dst_key, dst_rkey_buf_rail1, dst_rkey_size_rail1);
				if (dst_entry_r1) {
					dst_entry_r1->mmap = dst_mmap_r1;
					/* Phase 12.2: rail1 PCI export (GPU Direct dual-rail Ring 用) */
					const void *dst_pci_r1 = NULL;
					size_t dst_pci_len_r1 = 0;
					doca_error_t pret = doca_mmap_export_pci(dst_mmap_r1, rdma_dev_rail1, &dst_pci_r1, &dst_pci_len_r1);
					if (pret == DOCA_SUCCESS) {
						dst_entry_r1->pci_export_buf = (void *)dst_pci_r1;
						dst_entry_r1->pci_export_size = dst_pci_len_r1;
					}
				}
			} else {
				DOCA_LOG_WARN("Multi-Rail: dst mmap rail1 export failed");
				if (dst_mmap_r1) doca_mmap_destroy(dst_mmap_r1);
			}
		}
	}

	/* ---- Host timing ---- */
	struct timespec _hts;
	clock_gettime(CLOCK_MONOTONIC, &_hts);
	uint64_t t_host_pre_send_ns = (uint64_t)_hts.tv_sec * 1000000000ULL + (uint64_t)_hts.tv_nsec;

	/* Doorbell: monotonic counter — リセット不要 */

	/* Phase 5 Step 4: RDMA Write command (disabled — regresses latency due to per-write overhead) */
	bool cmd_sent_via_rdma = false;
	if (false /* global_sample_objects.cmd_ring_enabled */) {
		struct rdma_compact_cmd *cmd =
			(struct rdma_compact_cmd *)global_sample_objects.cmd_local_buf;
		memset(cmd, 0, CMD_SLOT_SIZE);
		cmd->op_type = collective_request.collective_op;
		cmd->src_addr = src_buffer_address;
		cmd->src_len = src_buffer_len;
		cmd->dst_addr = dst_buffer_address;
		cmd->dst_len = dst_buffer_len;
		/* Pack export descriptors inline */
		uint8_t *ep = cmd->export_data;
		if (src_rkey_size > 0) {
			memcpy(ep, src_rkey_buf, src_rkey_size);
			cmd->src_export_len = (uint32_t)src_rkey_size;
			ep += src_rkey_size;
		}
		if (dst_rkey_size > 0) {
			memcpy(ep, dst_rkey_buf, dst_rkey_size);
			cmd->dst_export_len = (uint32_t)dst_rkey_size;
			ep += dst_rkey_size;
		}
		if (src_rkey_size_rail1 > 0) {
			memcpy(ep, src_rkey_buf_rail1, src_rkey_size_rail1);
			cmd->src_export_len_rail1 = (uint32_t)src_rkey_size_rail1;
			ep += src_rkey_size_rail1;
		}
		if (dst_rkey_size_rail1 > 0) {
			memcpy(ep, dst_rkey_buf_rail1, dst_rkey_size_rail1);
			cmd->dst_export_len_rail1 = (uint32_t)dst_rkey_size_rail1;
			ep += dst_rkey_size_rail1;
		}
		/* local_cpu fields */
		cmd->local_cpu_addr = local_cpu_buffer_address;
		cmd->local_cpu_len = local_cpu_buffer_len;
		if (local_cpu_rkey_size > 0) {
			memcpy(ep, local_cpu_rkey_buf, local_cpu_rkey_size);
			cmd->local_cpu_export_len = (uint32_t)local_cpu_rkey_size;
			ep += local_cpu_rkey_size;
		}
		cmd->local_cpu_flag_addr = local_cpu_flag_buffer_address;
		cmd->local_cpu_flag_len = local_cpu_flag_buffer_len;
		if (local_cpu_flag_rkey_size > 0) {
			memcpy(ep, local_cpu_flag_rkey_buf, local_cpu_flag_rkey_size);
			cmd->local_cpu_flag_export_len = (uint32_t)local_cpu_flag_rkey_size;
			ep += local_cpu_flag_rkey_size;
		}

		uint64_t seq = global_sample_objects.cmd_seq++;

		/* RDMA Write 1: payload (skip seq field at offset 0) */
		size_t payload_offset = offsetof(struct rdma_compact_cmd, op_type);
		size_t total_data_len = (size_t)(ep - (uint8_t *)cmd);
		size_t payload_size = total_data_len - payload_offset;
		/* Ensure minimum payload covers fixed fields */
		size_t min_payload = sizeof(struct rdma_compact_cmd) - payload_offset;
		if (payload_size < min_payload) payload_size = min_payload;

		struct doca_rdma_ctx_t *rdma_host = &global_sample_objects.host_rdma_ctx;
		struct doca_remote_mem_t *rmem = &global_sample_objects.dpu_cmd_rmem;

		doca_error_t wret = host_rdma_write_sync(rdma_host,
			(uint8_t *)cmd + payload_offset, payload_size,
			rmem,
			(void *)((uint8_t *)(uintptr_t)rmem->remote_addr + payload_offset), payload_size);
		if (wret == DOCA_SUCCESS) {
			/* RDMA Write 2: seq (doorbell) — 8 bytes at offset 0 */
			cmd->seq = seq;
			wret = host_rdma_write_sync(rdma_host,
				(void *)&cmd->seq, sizeof(uint64_t),
				rmem,
				(void *)(uintptr_t)rmem->remote_addr, sizeof(uint64_t));
			if (wret == DOCA_SUCCESS) {
				cmd_sent_via_rdma = true;
			} else {
				DOCA_LOG_ERR("RDMA Write seq failed: %s", doca_error_get_name(wret));
				pthread_mutex_unlock(&g_impl_submit_mtx);
				return DOCA_ERROR_BAD_STATE;
			}
		} else {
			DOCA_LOG_ERR("RDMA Write payload failed: %s, falling back to ComCh", doca_error_get_name(wret));
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &_hts);
	uint64_t t_host_pre_comch_ns = (uint64_t)_hts.tv_sec * 1000000000ULL + (uint64_t)_hts.tv_nsec;

	if (!cmd_sent_via_rdma) {
		/* ComCh fallback */
		struct control_cmd send_cmd;
		send_cmd.type = CONTROL_CMD_UCP_COLLECTIVE;
		send_cmd.ucp_collective.id = id;

		send_cmd.ucp_collective.remote_src_buffer_address = src_buffer_address;
		send_cmd.ucp_collective.remote_src_buffer_len = src_buffer_len;
		send_cmd.ucp_collective.src_rkey_buf = src_rkey_buf;
		send_cmd.ucp_collective.src_rkey_buf_len = src_rkey_size;

		send_cmd.ucp_collective.remote_dst_buffer_address = dst_buffer_address;
		send_cmd.ucp_collective.remote_dst_buffer_len = dst_buffer_len;
		send_cmd.ucp_collective.dst_rkey_buf = dst_rkey_buf;
		send_cmd.ucp_collective.dst_rkey_buf_len = dst_rkey_size;

		send_cmd.ucp_collective.local_cpu_buffer_address = local_cpu_buffer_address;
		send_cmd.ucp_collective.local_cpu_buffer_len = local_cpu_buffer_len;
		send_cmd.ucp_collective.local_cpu_rkey_buf = local_cpu_rkey_buf;
		send_cmd.ucp_collective.local_cpu_rkey_buf_len = local_cpu_rkey_size;

		send_cmd.ucp_collective.local_cpu_flag_address = local_cpu_flag_buffer_address;
		send_cmd.ucp_collective.local_cpu_flag_len = local_cpu_flag_buffer_len;
		send_cmd.ucp_collective.local_cpu_flag_rkey_buf = local_cpu_flag_rkey_buf;
		send_cmd.ucp_collective.local_cpu_flag_rkey_buf_len = local_cpu_flag_rkey_size;

		/* Multi-Rail: rail1 export desc */
		send_cmd.ucp_collective.src_rkey_buf_rail1 = src_rkey_buf_rail1;
		send_cmd.ucp_collective.src_rkey_buf_len_rail1 = src_rkey_size_rail1;
		send_cmd.ucp_collective.dst_rkey_buf_rail1 = dst_rkey_buf_rail1;
		send_cmd.ucp_collective.dst_rkey_buf_len_rail1 = dst_rkey_size_rail1;

		/* Phase 8: Cross-GVMI PCI export of dst */
		send_cmd.ucp_collective.dst_pci_export_buf = dst_entry ? dst_entry->pci_export_buf : NULL;
		send_cmd.ucp_collective.dst_pci_export_buf_len = dst_entry ? dst_entry->pci_export_size : 0;

		/* Phase 9: Cross-GVMI PCI export of src */
		send_cmd.ucp_collective.src_pci_export_buf = src_entry ? src_entry->pci_export_buf : NULL;
		send_cmd.ucp_collective.src_pci_export_buf_len = src_entry ? src_entry->pci_export_size : 0;

		/* Phase 12.2: Cross-GVMI PCI export rail1 (dual-rail GPU Direct) */
		{
			struct rkey_cache_entry *src_e_r1 = global_sample_objects.dual_rail
				? rkey_cache_find(g_src_rkey_cache_rail1, src_key) : NULL;
			struct rkey_cache_entry *dst_e_r1 = global_sample_objects.dual_rail
				? rkey_cache_find(g_dst_rkey_cache_rail1, dst_key) : NULL;
			send_cmd.ucp_collective.src_pci_export_buf_rail1 = src_e_r1 ? src_e_r1->pci_export_buf : NULL;
			send_cmd.ucp_collective.src_pci_export_buf_len_rail1 = src_e_r1 ? src_e_r1->pci_export_size : 0;
			send_cmd.ucp_collective.dst_pci_export_buf_rail1 = dst_e_r1 ? dst_e_r1->pci_export_buf : NULL;
			send_cmd.ucp_collective.dst_pci_export_buf_len_rail1 = dst_e_r1 ? dst_e_r1->pci_export_size : 0;
		}

		send_cmd.ucp_collective.collective_request = collective_request;

		/* Phase 14: GPU flag completion fields */
		send_cmd.ucp_collective.flag_gpu_addr = flag_gpu_addr;
		send_cmd.ucp_collective.flag_value    = flag_value;

		doca_error_t result = comch_send_control_cmd(&send_cmd, id);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to comch send control cmd");
			pthread_mutex_unlock(&g_impl_submit_mtx);
			return DOCA_ERROR_BAD_STATE;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &_hts);
	uint64_t t_host_post_send_ns = (uint64_t)_hts.tv_sec * 1000000000ULL + (uint64_t)_hts.tv_nsec;

	/* ComCh 送信完了 → submit mutex 解放 (別スレッドが次の AG を送信可能になる) */
	uint64_t my_seq = __sync_add_and_fetch(&g_doorbell_submit_seq, 1);
	if (out_submit_seq) *out_submit_seq = my_seq;
	pthread_mutex_unlock(&g_impl_submit_mtx);

	//終了まで待機: RDMA Doorbell (monotonic counter) で DPU からの完了通知を受け取る
	if (wait_doorbell) {
		if (global_sample_objects.doorbell_export_desc_len > 0) {
			while (global_sample_objects.doorbell < my_seq) {
				/* spin */
			}
		} else {
			while (!atomic_load_explicit(&entry->ucp_collective_finish, memory_order_acquire)) {
				doca_pe_progress(global_sample_objects.pe);
			}
		}
	}

	return DOCA_SUCCESS;
}

int comch_client_init(const char *base_server_name, const char *dev_pci_addr)
{
	uint32_t max_msg_size;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

    doca_error_t result;
    struct doca_log_backend *sdk_log;
    result = doca_log_backend_create_standard();
    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_INITIALIZATION;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_INITIALIZATION;

	int tmp_rank, tmp_size;
    //MPI_Init(NULL, NULL); 初期化はPython側で行うことができるからいいらしい
	MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &tmp_size);
    uint64_t rank = (uint64_t)tmp_rank, size = (uint64_t)tmp_size;
	global_sample_objects.rank = rank;
	char server_name_buf[256];
    snprintf(server_name_buf, sizeof(server_name_buf), "%s_%d", base_server_name, tmp_rank);
    // 以降はこの server_name を使う
    const char *server_name = server_name_buf;

	global_sample_objects.finish_flags_map = NULL; //フラッグのハッシュテーブルのマップ

    DOCA_CHECK(init_comch_ctrl_path_objects(server_name, dev_pci_addr));
    DOCA_CHECK(doca_comch_cap_get_max_msg_size(doca_dev_as_devinfo(global_sample_objects.hw_dev), &max_msg_size));

	enum doca_ctx_states state;
    do {
        doca_pe_progress(global_sample_objects.pe);
        result = doca_ctx_get_state(doca_comch_client_as_ctx(global_sample_objects.client), &state);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_ctx_get_state() failed: %s", doca_error_get_descr(result));
            return result;
        }
    } while (state != DOCA_CTX_STATE_RUNNING);

	global_sample_objects.finish = false;

	return 0;
}


//send完了コールバック
static void send_task_completion_callback(struct doca_comch_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data)
{
	struct comch_ctrl_path_objects *sample_objects = (struct comch_ctrl_path_objects *)ctx_user_data.ptr;

	uint64_t id = (uint64_t)(uintptr_t)task_user_data.ptr;

    struct finish_flags_entry *entry = get_finish_flags_entry(id, false);
	if (entry != NULL) {
		atomic_store_explicit(&entry->doca_send_task_finish, true, memory_order_release);
	} else {
		DOCA_LOG_ERR("send_task_completion_callback: finish_flags_entry not found for id=%lu",
					(unsigned long)id);
	}

    doca_task_free(doca_comch_task_send_as_task(task));
}

static void send_task_completion_err_callback(struct doca_comch_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data)
{
	struct comch_ctrl_path_objects *sample_objects = (struct comch_ctrl_path_objects *)ctx_user_data.ptr;
    uint64_t id = (uint64_t)(uintptr_t)task_user_data.ptr;
    struct finish_flags_entry *entry = get_finish_flags_entry(id, false);
    if (entry != NULL) {
        atomic_store_explicit(&entry->doca_send_task_finish, true, memory_order_release);
    } else {
        DOCA_LOG_ERR("send_task_completion_err_callback: finish_flags_entry not found for id=%lu",
                     (unsigned long)id);
    }

    sample_objects->result = doca_task_get_status(doca_comch_task_send_as_task(task));
    DOCA_LOG_ERR("Message (id=%lu) failed to send with error = %s",
                 (unsigned long)id,
                 doca_error_get_name(sample_objects->result));
    doca_task_free(doca_comch_task_send_as_task(task));
}

void execute_ucp_connect_host_dpu_notify(struct control_notify *recv_notify)
{
	if(global_sample_objects.rank == 0)
		DOCA_LOG_INFO("execute ucp connect host dpu notify, id = %lu", recv_notify->ucp_connect_host_dpu.id);

	uint64_t id = recv_notify->ucp_connect_host_dpu.id;

	/* DPU の DOCA RDMA 接続記述子を受け取り、接続を確立する */
	const void *dpu_conn_desc = recv_notify->ucp_connect_host_dpu.remote_ucp_worker_address;
	size_t dpu_conn_desc_len = recv_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len;

	struct doca_rdma_ctx_t *rdma_host = &global_sample_objects.host_rdma_ctx;
	doca_error_t ret = doca_rdma_ctx_connect(rdma_host, 0, dpu_conn_desc, dpu_conn_desc_len);
	if (ret != DOCA_SUCCESS) {
		DOCA_LOG_ERR("host rdma_ctx connect failed: %s", doca_error_get_name(ret));
		return;
	}

	/* Multi-Rail: rail1 の接続も確立 */
	if (global_sample_objects.dual_rail &&
	    recv_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 > 0) {
		struct doca_rdma_ctx_t *rdma_rail1 = &global_sample_objects.host_rdma_ctx_rail1;
		ret = doca_rdma_ctx_connect(rdma_rail1, 0,
		    recv_notify->ucp_connect_host_dpu.remote_ucp_worker_address_rail1,
		    recv_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1);
		if (ret != DOCA_SUCCESS) {
			DOCA_LOG_WARN("Multi-Rail: host rdma_ctx_rail1 connect failed: %s", doca_error_get_name(ret));
			global_sample_objects.dual_rail = false;
		} else {
			DOCA_LOG_INFO("Multi-Rail: rail1 connection established");
		}
	}

	/* Phase 5 Step 4: Create remote mmap for DPU command slot */
	if (recv_notify->ucp_connect_host_dpu.cmd_slot_export_desc_len > 0 &&
	    global_sample_objects.cmd_local_buf) {
		struct doca_rdma_ctx_t *rh = &global_sample_objects.host_rdma_ctx;
		doca_error_t mret = doca_remote_mem_create(
			&global_sample_objects.dpu_cmd_rmem,
			rh->dev,
			recv_notify->ucp_connect_host_dpu.cmd_slot_export_desc,
			recv_notify->ucp_connect_host_dpu.cmd_slot_export_desc_len);
		if (mret == DOCA_SUCCESS) {
			global_sample_objects.cmd_ring_enabled = true;
			printf("[RANK%lu] RDMA command ring enabled (remote_addr=0x%lx, remote_len=%zu)\n",
			       global_sample_objects.rank,
			       (unsigned long)global_sample_objects.dpu_cmd_rmem.remote_addr,
			       global_sample_objects.dpu_cmd_rmem.remote_len);
		} else {
			DOCA_LOG_WARN("Failed to create DPU cmd remote mmap: %s, using ComCh fallback",
			              doca_error_get_name(mret));
			global_sample_objects.cmd_ring_enabled = false;
		}
	}

	printf("[RANK%lu] comch client: DOCA RDMA connection to DPU established (dual_rail=%d, cmd_ring=%d)\n",
	       global_sample_objects.rank, global_sample_objects.dual_rail,
	       global_sample_objects.cmd_ring_enabled);

	/* finish フラグを立てる */
	struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to get finish_flags_entry");
        return;
    }
    atomic_store_explicit(&entry->ucp_connect_host_dpu_finish, true, memory_order_release);
}

void execute_ucp_create_ring_notify(struct control_notify *recv_notify)
{
	if(global_sample_objects.rank == 0)
		DOCA_LOG_INFO("execute ucp create ring notify, id = %lu", recv_notify->ucp_create_ring.id);
	
	uint64_t id = recv_notify->ucp_create_ring.id;
    struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to get finish_flags_entry");
        return;
    }
    atomic_store_explicit(&entry->ucp_create_ring_finish, true, memory_order_release);
	return ;
}

void execute_ucp_collective_notify(struct control_notify *recv_notify)
{
	uint64_t id = recv_notify->ucp_collective.id;
    struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to get finish_flags_entry");
        return;
    }
    atomic_store_explicit(&entry->ucp_collective_finish, true, memory_order_release);

}

static void message_recv_callback(struct doca_comch_event_msg_recv *event, uint8_t *recv_buffer, uint32_t msg_len, struct doca_comch_connection *comch_connection)
{
	//コネクションからユーザデータを取得
	union doca_data user_data = doca_comch_connection_get_user_data(comch_connection);
	struct comch_ctrl_path_objects *sample_objects = (struct comch_ctrl_path_objects *)user_data.ptr;
	(void)event;
	//もらった通知に応じて処理を変えるよ
	struct control_notify *recv_notify = NULL;
	doca_error_t st = control_notify_unpack(recv_buffer, CONTROL_NOTIFY_MAX_SIZE, &recv_notify);
	if(recv_notify != NULL) {
		switch(recv_notify->type) {
		case CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU:
			execute_ucp_connect_host_dpu_notify(recv_notify);
			break;
		case CONTROL_NOTIFY_UCP_CREATE_RING:
			execute_ucp_create_ring_notify(recv_notify);
			break;
		case CONTROL_NOTIFY_UCP_COLLECTIVE:
			execute_ucp_collective_notify(recv_notify);
			break;
		}
	}
	return ;
}

//クライアント側で確保したリソースの後始末を行う
void clean_comch_sample_objects()
{
    /*
     * This function may be called from multiple exit paths. Make it idempotent.
     * (Double-destroy of DOCA/UCX objects can easily lead to SIGSEGV in libucs.)
     */
    static int cleaned = 0;
    if (cleaned)
        return;
    cleaned = 1;

    doca_error_t result;

    /* Stop auxiliary threads that may still touch UCX/DOCA objects. */
    comch_client_shutdown();

    /* finish フラグ用ハッシュテーブルの解放 */
    free_finish_flags_map();

    /*
     * Stop the comch context before destroying the client/PE/device.
     * DOCA_ERROR_IN_PROGRESS means "stop already in progress" and is expected.
     */
    if (global_sample_objects.client != NULL && global_sample_objects.pe != NULL) {
        struct doca_ctx *ctx = doca_comch_client_as_ctx(global_sample_objects.client);
        enum doca_ctx_states state;

        (void)doca_ctx_get_state(ctx, &state);
        if (state != DOCA_CTX_STATE_IDLE && state != DOCA_CTX_STATE_STOPPING) {
            result = doca_ctx_stop(ctx);
            if (result != DOCA_SUCCESS && result != DOCA_ERROR_IN_PROGRESS && result != DOCA_ERROR_BAD_STATE) {
                DOCA_LOG_ERR("doca_ctx_stop() failed: %s", doca_error_get_name(result));
            }
        }

        /* Progress PE until the context reaches IDLE. */
        do {
            doca_pe_progress(global_sample_objects.pe);

            result = doca_ctx_get_state(ctx, &state);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("doca_ctx_get_state() failed during shutdown: %s", doca_error_get_name(result));
                break;
            }
            if (state != DOCA_CTX_STATE_IDLE)
                __asm__ __volatile__("pause");
        } while (state != DOCA_CTX_STATE_IDLE);
    }

    /* Phase 5 Step 4: cmd ring cleanup */
    if (global_sample_objects.dpu_cmd_rmem.valid) {
        doca_remote_mem_destroy(&global_sample_objects.dpu_cmd_rmem);
    }
    if (global_sample_objects.cmd_local_buf) {
        free(global_sample_objects.cmd_local_buf);
        global_sample_objects.cmd_local_buf = NULL;
    }

    /* DOCA RDMA コンテキスト破棄 */
    doca_rdma_ctx_destroy(&global_sample_objects.host_rdma_ctx);
    if (global_sample_objects.dual_rail)
        doca_rdma_ctx_destroy(&global_sample_objects.host_rdma_ctx_rail1);

    /* Now it is safe to destroy ctrl-path objects and device. */
    clean_comch_ctrl_path_client(global_sample_objects.client, global_sample_objects.pe);
    global_sample_objects.client = NULL;
    global_sample_objects.pe = NULL;

    if (global_sample_objects.hw_dev != NULL) {
        result = doca_dev_close(global_sample_objects.hw_dev);
        if (result != DOCA_SUCCESS)
            DOCA_LOG_ERR("Failed to close hw device properly with error = %s", doca_error_get_name(result));
        global_sample_objects.hw_dev = NULL;
    }

    /* Multi-Rail: rail1 デバイスを閉じる */
    if (global_sample_objects.hw_dev_rail1 != NULL) {
        result = doca_dev_close(global_sample_objects.hw_dev_rail1);
        if (result != DOCA_SUCCESS)
            DOCA_LOG_ERR("Failed to close rail1 device: %s", doca_error_get_name(result));
        global_sample_objects.hw_dev_rail1 = NULL;
    }

}

//クライアントコンテキストの状態が変わったときに呼ばれるコールバック。prev_state→next_stateに遷移済み
static void comch_client_state_changed_callback(const union doca_data user_data,
						struct doca_ctx *ctx,
						enum doca_ctx_states prev_state,
						enum doca_ctx_states next_state)
{
	(void)ctx;
	(void)prev_state;
	//struct comch_ctrl_path_objects *sample_objects = (struct comch_ctrl_path_objects *)user_data.ptr;
	switch (next_state) {
	case DOCA_CTX_STATE_IDLE:
		if(global_sample_objects.rank == 0)
			DOCA_LOG_INFO("CC client context has been stopped");
		global_sample_objects.finish = true;
		break;
	case DOCA_CTX_STATE_STARTING:
		if(global_sample_objects.rank == 0)
			DOCA_LOG_INFO("CC client context entered into starting state. Waiting for connection establishment");
		break;
	case DOCA_CTX_STATE_RUNNING:
		if(global_sample_objects.rank == 0)
			DOCA_LOG_INFO("CC client context is running. Sending message");
		break;
	case DOCA_CTX_STATE_STOPPING:
		if(global_sample_objects.rank == 0)
			DOCA_LOG_INFO("CC client context entered into stopping state. Waiting for connection termination");
		break;
	default:
		break;
	}
}

//初期化関数宣言。サーバ名・デバイスPCI・状態構造体を受け取り、デバイス/クライアント/PE を作成する。
static doca_error_t init_comch_ctrl_path_objects(const char *server_name, const char *dev_pci_addr)
{
	doca_error_t result;
	struct comch_ctrl_path_client_cb_config cfg = {.send_task_comp_cb = send_task_completion_callback,
						       .send_task_comp_err_cb = send_task_completion_err_callback,
						       .msg_recv_cb = message_recv_callback,
						       .data_path_mode = false,
						       .new_consumer_cb = NULL,
						       .expired_consumer_cb = NULL,
						       .ctx_user_data = &global_sample_objects,
						       .ctx_state_changed_cb = comch_client_state_changed_callback};

    DOCA_CHECK(open_doca_device_with_pci(dev_pci_addr, NULL, &(global_sample_objects.hw_dev)));
    DOCA_CHECK(init_comch_ctrl_path_client(server_name, global_sample_objects.hw_dev, &cfg,
					     &(global_sample_objects.client), &(global_sample_objects.pe)));
	return DOCA_SUCCESS;
}


/* --------------今はもう利用していない--------------------- */
int main(int argc, char **argv)
{
    if(argc < 3) {
        fprintf(stderr, "usage: %s <server name> <host pci address>\n", argv[0]);
        fprintf(stderr, "  e.g. %s collective_server b3:00.0\n", argv[0]);
    }
    const char *base_server_name = argv[1];
    const char *dev_pci_addr = argv[2];
    const char *text = "HELLO WORLD";
    uint32_t text_len = 11;

	uint32_t max_msg_size;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

    doca_error_t result;
    struct doca_log_backend *sdk_log;
    result = doca_log_backend_create_standard();
    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_INITIALIZATION;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_INITIALIZATION;

	int tmp_rank, tmp_size;
    MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &tmp_size);
    uint64_t rank = (uint64_t)tmp_rank, size = (uint64_t)tmp_size;
    printf("MPI rank: %lu", rank);
	char server_name_buf[256];
    snprintf(server_name_buf, sizeof(server_name_buf), "%s_%d", base_server_name, tmp_rank);
    // 以降はこの server_name を使う
    const char *server_name = server_name_buf;
    printf("server_name = %s, dev_pci_addr = %s\n", server_name, dev_pci_addr);

    DOCA_CHECK(init_comch_ctrl_path_objects(server_name, dev_pci_addr));
    DOCA_CHECK(doca_comch_cap_get_max_msg_size(doca_dev_as_devinfo(global_sample_objects.hw_dev), &max_msg_size));

	enum doca_ctx_states state;
    do {
		doca_pe_progress(global_sample_objects.pe);
        result = doca_ctx_get_state(doca_comch_client_as_ctx(global_sample_objects.client), &state);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_ctx_get_state() failed: %s", doca_error_get_descr(result));
            return result;
        }
    } while (state != DOCA_CTX_STATE_RUNNING);

	if (text_len > max_msg_size) {
		DOCA_LOG_ERR("Failed to run sample, text size is larger than supported message size. text_len = %u, supported message size = %u", text_len, max_msg_size);
		return DOCA_ERROR_INVALID_VALUE;
	}
	global_sample_objects.text = text;
	global_sample_objects.text_len = text_len;
	global_sample_objects.finish = false;

	//result = comch_create_send_control_cmd(CONTROL_CMD_CREATE_RING_AND_COLLECTIVE);
	if(result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to comch create send control cmd");
		goto mpi_abort;
	}

	while (!global_sample_objects.finish) {
		int progressed = doca_pe_progress(global_sample_objects.pe);
		if (progressed == 0) {
            __asm__ __volatile__("pause");
		}
	}
	clean_comch_sample_objects();

mpi_abort:
    if(result != DOCA_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}

/* =========================
 *  Collective single-flight queue (FIFO) + handle API
 *
 *  enqueue 時に local_cpu_*_size == 0 なら通常集合通信、!=0 ならローカル CPU
 *  バッファリングを併用する。
 * ========================= */
typedef struct collective_job_u64 {
    uint64_t id;
    uint64_t src_addr, src_size;
    uint64_t dst_addr, dst_size;
    struct CollectiveRequest req;

    uint64_t local_addr, local_size;
    uint64_t local_flag_addr, local_flag_size;

    /* Phase 14: GPU flag (host-side request fills these; DPU writes
     *           flag_value to (flag_gpu_addr) when AG done). 0 = legacy. */
    uint64_t flag_gpu_addr;
    uint32_t flag_value;

    uint64_t submit_seq;   /* monotonic doorbell sequence for this job */
    int completed;
    uint64_t result_u64;
    int refcnt;

    pthread_mutex_t mtx;
    pthread_cond_t  cv;

    struct collective_job_u64 *next;
} collective_job_u64_t;

/* Phase 11: submit (1本) + completion (1本) 構成 */
static pthread_t g_submit_thread;
static pthread_t g_completion_thread;
static int g_collective_worker_running = 0;
static volatile int g_collective_stop = 0;

/* Inflight ring (SPSC: submit produces, completion consumes)
 * 順序保証: submit が単一スレッドなので push 順 = FIFO 順 = doorbell seq 順 */
#define INFLIGHT_CAP 256  /* 2 の冪 */
static collective_job_u64_t * volatile g_inflight_ring[INFLIGHT_CAP];
static volatile uint64_t g_inflight_head = 0;  /* completion が読み書き */
static volatile uint64_t g_inflight_tail = 0;  /* submit が読み書き */

/* Completion thread を空 ring 時にブロックさせる cond var
 * (busy spin で CPU を消費すると NCCL ベンチや他スレッドを妨害するため) */
static pthread_mutex_t g_inflight_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_inflight_cv  = PTHREAD_COND_INITIALIZER;

static pthread_mutex_t g_collective_q_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_collective_q_cv  = PTHREAD_COND_INITIALIZER;

static collective_job_u64_t *g_collective_q_head = NULL;
static collective_job_u64_t *g_collective_q_tail = NULL;

static inline uint64_t _worker_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void collective_q_push(collective_job_u64_t *j)
{
    pthread_mutex_lock(&g_collective_q_mtx);
    j->next = NULL;
    if (g_collective_q_tail) g_collective_q_tail->next = j;
    else g_collective_q_head = j;
    g_collective_q_tail = j;
    pthread_cond_signal(&g_collective_q_cv);
    pthread_mutex_unlock(&g_collective_q_mtx);
}

static collective_job_u64_t *collective_q_pop(void)
{
    collective_job_u64_t *j = g_collective_q_head;
    if (!j) return NULL;
    g_collective_q_head = j->next;
    if (!g_collective_q_head) g_collective_q_tail = NULL;
    j->next = NULL;
    return j;
}

static void collective_job_free(collective_job_u64_t *j)
{
    pthread_cond_destroy(&j->cv);
    pthread_mutex_destroy(&j->mtx);
    free(j);
}

static void collective_job_release(collective_job_u64_t *j)
{
    int do_free = 0;
    pthread_mutex_lock(&j->mtx);
    j->refcnt--;
    if (j->refcnt == 0) do_free = 1;
    pthread_mutex_unlock(&j->mtx);
    if (do_free) collective_job_free(j);
}

/* ---- Submit thread (1本): FIFO 順で dequeue → mmap+ComCh 送信 → inflight push ---- */
static void *collective_submit_main(void *arg)
{
    (void)arg;
    uint64_t prof_ops = 0;
    uint64_t prof_submit_ns = 0;

    pthread_mutex_lock(&g_collective_q_mtx);
    while (!g_collective_stop) {
        while (!g_collective_q_head && !g_collective_stop) {
            pthread_cond_wait(&g_collective_q_cv, &g_collective_q_mtx);
        }
        if (g_collective_stop) break;

        collective_job_u64_t *j = collective_q_pop();
        pthread_mutex_unlock(&g_collective_q_mtx);

        uint64_t t0 = _worker_now_ns();

        /* mmap 作成 + ComCh 送信のみ (doorbell 待ちなし)
         * j->submit_seq に doorbell シーケンス番号を確実に保存 */
        uint64_t my_seq = 0;
        doca_error_t st = ucp_collective_request_impl(
            j->id,
            j->src_addr, j->src_size,
            j->dst_addr, j->dst_size,
            j->req,
            j->local_addr, j->local_size,
            j->local_flag_addr,
            j->local_flag_size,
            false,  /* wait_doorbell = false: doorbell 待ちは completion thread */
            &my_seq,
            j->flag_gpu_addr, j->flag_value  /* Phase 14: GPU flag fields */
        );
        j->submit_seq = my_seq;
        j->result_u64 = (uint64_t)st;

        uint64_t t1 = _worker_now_ns();

        /* inflight ring に push (single producer なので FIFO 順序保証) */
        while ((g_inflight_tail - g_inflight_head) >= INFLIGHT_CAP) {
            __builtin_ia32_pause();  /* ring full: 通常到達しない */
        }
        g_inflight_ring[g_inflight_tail & (INFLIGHT_CAP - 1)] = j;
        __sync_synchronize();  /* completion thread が j を読む前に submit_seq が見えるよう保証 */
        g_inflight_tail++;

        /* completion thread を起こす (空 ring からブロック解除) */
        pthread_mutex_lock(&g_inflight_mtx);
        pthread_cond_signal(&g_inflight_cv);
        pthread_mutex_unlock(&g_inflight_mtx);

        /* profiling */
        prof_ops++;
        prof_submit_ns += (t1 - t0);
#if 0 /* 計測ログ [HOST SUBMIT PROF] を無効化 (2026-07-27) */
        if (prof_ops % 5000 == 0) {
            double avg_us = (double)prof_submit_ns / prof_ops / 1000.0;
            printf("[HOST SUBMIT PROF] ops=%lu | avg_submit=%.1fus\n", prof_ops, avg_us);
            fflush(stdout);
        }
#endif

        pthread_mutex_lock(&g_collective_q_mtx);
    }
    pthread_mutex_unlock(&g_collective_q_mtx);
    return NULL;
}

/* ---- Completion thread (1本): doorbell ポーリング → 完了シグナル ----
 * 重要: sched_yield/usleep は使わない (μs オーダーの latency が必要) */
static void *collective_completion_main(void *arg)
{
    (void)arg;
    uint64_t prof_ops = 0;
    uint64_t prof_wait_ns = 0;

    while (!g_collective_stop) {
        /* inflight ring が空: cond var でブロック (CPU 消費ゼロ)
         * 学習中はほぼ常に job がある & ベンチマーク中は他スレッドを邪魔しない */
        if (g_inflight_head >= g_inflight_tail) {
            pthread_mutex_lock(&g_inflight_mtx);
            while (g_inflight_head >= g_inflight_tail && !g_collective_stop) {
                pthread_cond_wait(&g_inflight_cv, &g_inflight_mtx);
            }
            pthread_mutex_unlock(&g_inflight_mtx);
            if (g_collective_stop) break;
        }

        collective_job_u64_t *j = g_inflight_ring[g_inflight_head & (INFLIGHT_CAP - 1)];
        __sync_synchronize();  /* j->submit_seq の可視性を保証 */

        /* doorbell が job の seq に到達するまで busy spin */
        uint64_t t0 = _worker_now_ns();
        if (global_sample_objects.doorbell_export_desc_len > 0) {
            while (global_sample_objects.doorbell < j->submit_seq) {
                __builtin_ia32_pause();
            }
        } else {
            /* doorbell 非対応時の fallback */
            struct finish_flags_entry *entry = get_finish_flags_entry(j->id, false);
            if (entry) {
                while (!atomic_load_explicit(&entry->ucp_collective_finish, memory_order_acquire)) {
                    __builtin_ia32_pause();
                }
            }
        }
        uint64_t t1 = _worker_now_ns();

        /* 完了シグナル (Python 側 wait() を解放) */
        pthread_mutex_lock(&j->mtx);
        j->completed = 1;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mtx);

        /* inflight ring から pop + 参照解放 */
        g_inflight_head++;
        collective_job_release(j);

        /* profiling */
        prof_ops++;
        prof_wait_ns += (t1 - t0);
#if 0 /* 計測ログ [HOST COMPLETION PROF] を無効化 (2026-07-27) */
        if (prof_ops % 5000 == 0) {
            double avg_us = (double)prof_wait_ns / prof_ops / 1000.0;
            printf("[HOST COMPLETION PROF] ops=%lu | avg_doorbell_wait=%.1fus\n",
                   prof_ops, avg_us);
            fflush(stdout);
        }
#endif
    }
    return NULL;
}

static void _pin_thread(pthread_t tid, int core, const char *name)
{
    if (core < 0) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (pthread_setaffinity_np(tid, sizeof(cpuset), &cpuset) == 0) {
        DOCA_LOG_INFO("Pinned %s to core %d", name, core);
    } else {
        DOCA_LOG_WARN("Failed to pin %s to core %d", name, core);
    }
}

static void ensure_collective_worker_started(void)
{
    pthread_mutex_lock(&g_collective_q_mtx);
    if (!g_collective_worker_running) {
        g_collective_stop = 0;

        if (global_sample_objects.rank == 0) {
            printf("collective submit + completion threads create! (Phase 11 inflight pipelining)\n");
        }

        /* Submit thread (FIFO 順序保証) */
        if (pthread_create(&g_submit_thread, NULL, collective_submit_main, NULL) != 0) {
            DOCA_LOG_ERR("Failed to create submit thread");
            abort();
        }

        /* Completion thread (busy spin で doorbell ポーリング) */
        if (pthread_create(&g_completion_thread, NULL, collective_completion_main, NULL) != 0) {
            DOCA_LOG_ERR("Failed to create completion thread");
            abort();
        }

        /* コアピンニング (同一 NUMA ノード内の専用コア) */
        const char *wenv = getenv("COMCH_WORKER_CORE");
        const char *penv = getenv("COMCH_POLLER_CORE");
        if (wenv) _pin_thread(g_submit_thread, atoi(wenv), "submit");
        if (penv) _pin_thread(g_completion_thread, atoi(penv), "completion");
        /* SCHED_FIFO は使わない: 他スレッドの飢餓を引き起こすため */

        g_collective_worker_running = 1;
    }
    pthread_mutex_unlock(&g_collective_q_mtx);
}

/* ===== 公開: enqueue (uint64_t handle) ===== */
int ucp_collective_enqueue_u64_with_flag(uint64_t id,
                                         uint64_t src_addr, uint64_t src_size,
                                         uint64_t dst_addr, uint64_t dst_size,
                                         struct CollectiveRequest collective_request,
                                         uint64_t *out_handle,
                                         uint64_t local_addr, uint64_t local_size,
                                         uint64_t local_flag_addr, uint64_t local_flag_size,
                                         uint64_t flag_gpu_addr, uint32_t flag_value);

int ucp_collective_enqueue_u64(uint64_t id,
                               uint64_t src_addr, uint64_t src_size,
                               uint64_t dst_addr, uint64_t dst_size,
                               struct CollectiveRequest collective_request,
                               uint64_t *out_handle,
                               uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size)
{
    return ucp_collective_enqueue_u64_with_flag(id, src_addr, src_size, dst_addr, dst_size,
                                                collective_request, out_handle,
                                                local_addr, local_size, local_flag_addr, local_flag_size,
                                                0, 0);
}

int ucp_collective_enqueue_u64_with_flag(uint64_t id,
                                         uint64_t src_addr, uint64_t src_size,
                                         uint64_t dst_addr, uint64_t dst_size,
                                         struct CollectiveRequest collective_request,
                                         uint64_t *out_handle,
                                         uint64_t local_addr, uint64_t local_size,
                                         uint64_t local_flag_addr, uint64_t local_flag_size,
                                         uint64_t flag_gpu_addr, uint32_t flag_value)
{
    if (!out_handle) return -EINVAL;
    ensure_collective_worker_started();

    collective_job_u64_t *j = (collective_job_u64_t*)calloc(1, sizeof(*j));
    if (!j) return -ENOMEM;

    j->id = id;
    j->src_addr = src_addr;
    j->src_size = src_size;
    j->dst_addr = dst_addr;
    j->dst_size = dst_size;
    j->req = collective_request;
    j->local_addr = local_addr;
    j->local_size = local_size;
    j->local_flag_addr = local_flag_addr;
    j->local_flag_size = local_flag_size;
    j->flag_gpu_addr = flag_gpu_addr;
    j->flag_value    = flag_value;

    pthread_mutex_init(&j->mtx, NULL);
    pthread_cond_init(&j->cv, NULL);

    j->completed = 0;
    j->result_u64 = 0;
    j->refcnt = 2; /* caller + queue */

    collective_q_push(j);

    *out_handle = (uint64_t)(uintptr_t)j;
    return 0;
}

int comch_req_test_u64(uint64_t h, uint64_t *out_result_u64)
{
    collective_job_u64_t *j = (collective_job_u64_t*)(uintptr_t)h;
    if (!j) return -EINVAL;

    pthread_mutex_lock(&j->mtx);
    int done = j->completed;
    uint64_t res = j->result_u64;
    pthread_mutex_unlock(&j->mtx);

    if (done && out_result_u64) *out_result_u64 = res;
    return done ? 1 : 0;
}

int comch_req_wait_u64(uint64_t h, uint64_t *out_result_u64)
{
    collective_job_u64_t *j = (collective_job_u64_t*)(uintptr_t)h;
    if (!j) return -EINVAL;

    pthread_mutex_lock(&j->mtx);
    while (!j->completed) {
        pthread_cond_wait(&j->cv, &j->mtx);
    }
    uint64_t res = j->result_u64;
    pthread_mutex_unlock(&j->mtx);

    if (out_result_u64) *out_result_u64 = res;
    return 0;
}

void comch_req_release_u64(uint64_t h)
{
    collective_job_u64_t *j = (collective_job_u64_t*)(uintptr_t)h;
    if (!j) return;
    collective_job_release(j);
}

/* ===== 公開 API ===== */
/*
 * 変更点：
 * - 直接実行しない
 * - job をキューに積み、完了を待つ
 */
void ucp_collective_request(uint64_t id,
                            uint64_t src_addr, uint64_t src_size,
                            uint64_t dst_addr, uint64_t dst_size,
                            struct CollectiveRequest collective_request,
                            uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size)
{
    uint64_t handle = 0;
    int rc = ucp_collective_enqueue_u64_with_flag(id, src_addr, src_size, dst_addr, dst_size,
                                       collective_request, &handle, local_addr, local_size, local_flag_addr, local_flag_size,
                                       0, 0);
    if (rc < 0) {
        DOCA_LOG_ERR("ucp_collective_enqueue_u64 failed rc=%d", rc);
        return;
    }

    uint64_t result = 0;
    rc = comch_req_wait_u64(handle, &result);
    if (rc < 0) {
        DOCA_LOG_ERR("comch_req_wait_u64 failed rc=%d", rc);
        comch_req_release_u64(handle);
        return;
    }

    comch_req_release_u64(handle);

    /* result は doca_error_t 相当（必要ならログ） */
    if ((doca_error_t)result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("collective failed: %s", doca_error_get_descr((doca_error_t)result));
    }
}

void comch_client_shutdown(void)
{
    /* 既存の shutdown 処理があるなら先に実行 */

    /* 追加：collective worker を停止 */
    pthread_mutex_lock(&g_collective_q_mtx);
    if (g_collective_worker_running) {
        g_collective_stop = 1;
        pthread_cond_broadcast(&g_collective_q_cv);
        pthread_mutex_unlock(&g_collective_q_mtx);

        /* completion thread も cond_wait から解放 */
        pthread_mutex_lock(&g_inflight_mtx);
        pthread_cond_broadcast(&g_inflight_cv);
        pthread_mutex_unlock(&g_inflight_mtx);

        pthread_join(g_submit_thread, NULL);
        pthread_join(g_completion_thread, NULL);

        pthread_mutex_lock(&g_collective_q_mtx);
        g_collective_worker_running = 0;
    }
    pthread_mutex_unlock(&g_collective_q_mtx);

    /* キューに残った job をキャンセル扱いで完了させるならここで drain/cancel も可能 */
}
/*====================キューによる単一スレッド化ここまで====================*/