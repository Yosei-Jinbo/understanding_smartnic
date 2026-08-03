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

#include "comch_client.h"
#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "../common/comch_mpi_common.h"
#include "../common/doca_rdma_utils.h"

#include "uthash.h"
#include <pthread.h>   //スレッドセーフになるようにmutex制御
#include <errno.h>
/* stdatomic.h は doca_rdma_utils.h 経由で C/C++ 対応済み */

#include "../common/timing_utils.h"

DOCA_LOG_REGISTER(COLLECTIVE_QUEUE);
#include "comch_client_internal.h"

/* (addr,len) → rkey_buf のキャッシュエントリ */
struct rkey_cache_entry {
    uint64_t key;          /* make_addr_len_key(addr,len) の結果 */
    void    *rkey_buf;     /* DOCA: doca_mmap_export_rdma の export_desc */
    size_t   rkey_size;    /* DOCA: export_desc_len */
    struct doca_mmap *mmap; /* DOCA: エクスポート元の mmap (破棄用) */
    /* PCI export for Cross-GVMI */
    void    *pci_export_buf;
    size_t   pci_export_size;
    UT_hash_handle hh;
};

static inline uint64_t
make_addr_len_key(uint64_t addr, uint64_t len)
{
    uint64_t x = addr;
    x ^= len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2); //ハッシュキー
    return x;
}

/* SRC/DST で別々にしておくとデバッグしやすい */
static struct rkey_cache_entry *g_src_rkey_cache = NULL;
static struct rkey_cache_entry *g_dst_rkey_cache = NULL;

/* Multi-Rail: rail1 用キャッシュ */
static struct rkey_cache_entry *g_src_rkey_cache_rail1 = NULL;
static struct rkey_cache_entry *g_dst_rkey_cache_rail1 = NULL;

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

/* Doorbell monotonic sequence */
static volatile uint64_t g_doorbell_submit_seq = 0;

/* Submit mutex: mmap 作成 + ComCh 送信を直列化 (DOCA API はスレッドセーフでない) */
static pthread_mutex_t g_impl_submit_mtx = PTHREAD_MUTEX_INITIALIZER;

static doca_error_t ucp_collective_request_impl(uint64_t id, uint64_t src_buffer_address, uint64_t src_buffer_len,
	uint64_t dst_buffer_address, uint64_t dst_buffer_len, struct CollectiveRequest collective_request,
	uint64_t flag_gpu_addr, uint32_t flag_value,
	uint64_t *out_submit_seq)
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

        /* PCI export of src (for GPU Direct Ring step 0 Send) */
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
			/* rail1 でも PCI export が必要なので PCI_READ_WRITE 権限を追加 */
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
					/* rail1 PCI export (GPU Direct dual-rail Ring 用) */
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
			/* rail1 でも PCI export が必要なので PCI_READ_WRITE 権限を追加 */
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
					/* rail1 PCI export (GPU Direct dual-rail Ring 用) */
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

	/* Doorbell: monotonic counter — リセット不要 */

	{
		/* ComCh でコマンド送信 */
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

		/* local CPU バッファリング経路は廃止済み — wire 互換のため 0/NULL を送る */
		send_cmd.ucp_collective.local_cpu_buffer_address = 0;
		send_cmd.ucp_collective.local_cpu_buffer_len = 0;
		send_cmd.ucp_collective.local_cpu_rkey_buf = NULL;
		send_cmd.ucp_collective.local_cpu_rkey_buf_len = 0;

		send_cmd.ucp_collective.local_cpu_flag_address = 0;
		send_cmd.ucp_collective.local_cpu_flag_len = 0;
		send_cmd.ucp_collective.local_cpu_flag_rkey_buf = NULL;
		send_cmd.ucp_collective.local_cpu_flag_rkey_buf_len = 0;

		/* Multi-Rail: rail1 export desc */
		send_cmd.ucp_collective.src_rkey_buf_rail1 = src_rkey_buf_rail1;
		send_cmd.ucp_collective.src_rkey_buf_len_rail1 = src_rkey_size_rail1;
		send_cmd.ucp_collective.dst_rkey_buf_rail1 = dst_rkey_buf_rail1;
		send_cmd.ucp_collective.dst_rkey_buf_len_rail1 = dst_rkey_size_rail1;

		/* Cross-GVMI PCI export of dst */
		send_cmd.ucp_collective.dst_pci_export_buf = dst_entry ? dst_entry->pci_export_buf : NULL;
		send_cmd.ucp_collective.dst_pci_export_buf_len = dst_entry ? dst_entry->pci_export_size : 0;

		/* Cross-GVMI PCI export of src */
		send_cmd.ucp_collective.src_pci_export_buf = src_entry ? src_entry->pci_export_buf : NULL;
		send_cmd.ucp_collective.src_pci_export_buf_len = src_entry ? src_entry->pci_export_size : 0;

		/* Cross-GVMI PCI export rail1 (dual-rail GPU Direct) */
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

		/* GPU flag completion fields (0 なら DPU は flag write しない = legacy) */
		send_cmd.ucp_collective.flag_gpu_addr = flag_gpu_addr;
		send_cmd.ucp_collective.flag_value    = flag_value;

		doca_error_t result = comch_send_control_cmd(&send_cmd, id);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to comch send control cmd");
			pthread_mutex_unlock(&g_impl_submit_mtx);
			return DOCA_ERROR_BAD_STATE;
		}
	}

	/* ComCh 送信完了 → submit mutex 解放 (別スレッドが次の AG を送信可能になる)
	 * 完了待ちは completion thread が doorbell (monotonic counter) で行う。 */
	uint64_t my_seq = __sync_add_and_fetch(&g_doorbell_submit_seq, 1);
	if (out_submit_seq) *out_submit_seq = my_seq;
	pthread_mutex_unlock(&g_impl_submit_mtx);

	return DOCA_SUCCESS;
}

/* =========================
 *  Collective single-flight queue (FIFO) + handle API
 * ========================= */
typedef struct collective_job_u64 {
    uint64_t id;
    uint64_t src_addr, src_size;
    uint64_t dst_addr, dst_size;
    struct CollectiveRequest req;

    /* GPU flag (host-side request fills these; DPU writes flag_value to
     * (flag_gpu_addr) when AG done). 0 = legacy. */
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

/* submit (1本) + completion (1本) 構成 */
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

    pthread_mutex_lock(&g_collective_q_mtx);
    while (!g_collective_stop) {
        while (!g_collective_q_head && !g_collective_stop) {
            pthread_cond_wait(&g_collective_q_cv, &g_collective_q_mtx);
        }
        if (g_collective_stop) break;

        collective_job_u64_t *j = collective_q_pop();
        pthread_mutex_unlock(&g_collective_q_mtx);

        /* mmap 作成 + ComCh 送信のみ (doorbell 待ちは completion thread)
         * j->submit_seq に doorbell シーケンス番号を確実に保存 */
        uint64_t my_seq = 0;
        doca_error_t st = ucp_collective_request_impl(
            j->id,
            j->src_addr, j->src_size,
            j->dst_addr, j->dst_size,
            j->req,
            j->flag_gpu_addr, j->flag_value,
            &my_seq
        );
        j->submit_seq = my_seq;
        j->result_u64 = (uint64_t)st;

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

        /* 完了シグナル (Python 側 wait() を解放) */
        pthread_mutex_lock(&j->mtx);
        j->completed = 1;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mtx);

        /* inflight ring から pop + 参照解放 */
        g_inflight_head++;
        collective_job_release(j);
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
            printf("collective submit + completion threads create! (inflight pipelining)\n");
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

/* 公開: enqueue (uint64_t handle)。flag_gpu_addr != 0 なら DPU が完了時に
 * flag_value を RDMA Write する (GPU flag completion sync)。 */
int ucp_collective_enqueue_with_flag_u64(uint64_t id,
                               uint64_t src_addr, uint64_t src_size,
                               uint64_t dst_addr, uint64_t dst_size,
                               struct CollectiveRequest collective_request,
                               uint64_t flag_gpu_addr, uint32_t flag_value,
                               uint64_t *out_handle)
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
    j->flag_gpu_addr = flag_gpu_addr;
    j->flag_value = flag_value;

    pthread_mutex_init(&j->mtx, NULL);
    pthread_cond_init(&j->cv, NULL);

    j->completed = 0;
    j->result_u64 = 0;
    j->refcnt = 2; /* caller + queue */

    collective_q_push(j);

    *out_handle = (uint64_t)(uintptr_t)j;
    return 0;
}

int ucp_collective_enqueue_u64(uint64_t id,
                               uint64_t src_addr, uint64_t src_size,
                               uint64_t dst_addr, uint64_t dst_size,
                               struct CollectiveRequest collective_request,
                               uint64_t *out_handle)
{
    return ucp_collective_enqueue_with_flag_u64(id, src_addr, src_size,
                                                dst_addr, dst_size,
                                                collective_request,
                                                0 /* flag なし */, 0,
                                                out_handle);
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

void comch_client_shutdown(void)
{
    /* collective worker を停止 */
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
