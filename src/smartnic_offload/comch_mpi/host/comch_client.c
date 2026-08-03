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

DOCA_LOG_REGISTER(COMCH_CLIENT);
#include "comch_client_internal.h"
#include "host_config.h"

/* RDMA command slot (must match CMD_SLOT_SIZE on the DPU side in comch_server.c) */
#define CMD_SLOT_SIZE 4096

struct comch_ctrl_path_objects global_sample_objects;
/* finish_flags_mutex: ハッシュテーブル操作 (HASH_FIND/ADD/DEL) 専用 */
static pthread_mutex_t finish_flags_mutex = PTHREAD_MUTEX_INITIALIZER;

static doca_error_t init_comch_ctrl_path_objects(const char *server_name, const char *dev_pci_addr);

/* id に対応するエントリを取得。存在しなければ必要に応じて作成 */
struct finish_flags_entry *get_finish_flags_entry(uint64_t id, bool create_if_missing)
{
    struct finish_flags_entry *entry = NULL;
    /* ここで map 全体をロック */
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
    pthread_mutex_lock(&finish_flags_mutex);
    HASH_ITER(hh, global_sample_objects.finish_flags_map, cur, tmp) {
        HASH_DEL(global_sample_objects.finish_flags_map, cur);
        free(cur);
    }
    global_sample_objects.finish_flags_map = NULL;
    pthread_mutex_unlock(&finish_flags_mutex);
}

//コマンドの長さは#define CONTROL_CMD_MAX_SIZEで固定
doca_error_t comch_send_control_cmd(struct control_cmd *send_cmd, uint64_t id)
{
	struct doca_comch_task_send *task;
	struct doca_task *task_obj;
	struct doca_comch_connection *connection;
	union doca_data user_data;

	/* この id 用のエントリを取得（なければ作成） */
    struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to allocate finish_flags_entry");
        return DOCA_ERROR_NO_MEMORY;
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

/* ---- GPU flag pool 登録 (cuStreamWaitValue32 completion sync) ----
 *   Python から register_flag_pool_py(addr, len) で呼ばれる。
 *   - 同 GPU メモリ領域に対して doca_mmap_export_rdma を 1 度だけ実行
 *     (rail1 がある場合は rail1 dev でも export)
 *   - rkey_buf をグローバルに保存
 *   - INIT_FLAG_POOL コマンドを ComCh で DPU に送信
 *   - DPU は doca_remote_mem_create でインポートしてキャッシュし、
 *     AG 完了時に inline RDMA Write でフラグを書く。
 *
 *   想定タイミング: torch.cuda.init() → torch.zeros(.., int32, cuda='cuda') の後、
 *   かつ ucp_create_ring_request() の後 (ComCh が確立済みの状態)。 */
static struct doca_mmap *g_flag_pool_mmap = NULL;
static struct doca_mmap *g_flag_pool_mmap_rail1 = NULL;
static const void *g_flag_pool_rkey_buf = NULL;
static size_t      g_flag_pool_rkey_buf_len = 0;
static const void *g_flag_pool_rkey_buf_rail1 = NULL;
static size_t      g_flag_pool_rkey_buf_len_rail1 = 0;
/* Cross-GVMI PCI export for flag pool (allows DPU to RDMA Write to GPU memory) */
static const void *g_flag_pool_pci_export_buf = NULL;
static size_t      g_flag_pool_pci_export_len = 0;
static const void *g_flag_pool_pci_export_buf_rail1 = NULL;
static size_t      g_flag_pool_pci_export_len_rail1 = 0;
static bool        g_flag_pool_enabled = false;

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

    /* permissions に PCI_READ_WRITE を含める (AG dst と同じパターン)。
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

    /* PCI export (Cross-GVMI) — best effort。失敗しても継続 (legacy RDMA path で動作)。 */
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
                /* rail1 の PCI export も best effort */
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
    /* Cross-GVMI PCI export desc (optional; empty if export_pci failed) */
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
        printf("[FlagPool] registered: addr=0x%lx len=%lu rkey_len=%zu rail1=%s pci_len=%zu pci_rail1=%s\n",
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
/* Host RMA context 登録用の write コールバック (host は write を発行しないため no-op)。
 * DOCA ctx 設定に渡すために存在する。 */
static void host_rdma_write_comp_cb(struct doca_rdma_task_write *task,
                                     union doca_data task_user_data,
                                     union doca_data ctx_user_data)
{
	(void)task; (void)task_user_data; (void)ctx_user_data;
}
static void host_rdma_write_err_cb(struct doca_rdma_task_write *task,
                                    union doca_data task_user_data,
                                    union doca_data ctx_user_data)
{
	(void)task; (void)task_user_data; (void)ctx_user_data;
	DOCA_LOG_ERR("host_rdma_write_err_cb called");
}

void ucp_connect_host_dpu_request(uint64_t id)
{
	struct finish_flags_entry *entry = get_finish_flags_entry(id, true);
    if (entry == NULL) {
        DOCA_LOG_ERR("Failed to allocate finish_flags_entry");
        return;
    }
    atomic_store_explicit(&entry->ucp_connect_host_dpu_finish, false, memory_order_release);

	/* host RMA context 用ローカルバッファを確保 (DOCA ctx_init が要求) */
	global_sample_objects.cmd_local_buf = aligned_alloc(64, CMD_SLOT_SIZE);
	if (global_sample_objects.cmd_local_buf)
		memset(global_sample_objects.cmd_local_buf, 0, CMD_SLOT_SIZE);

	/* DOCA RDMA コンテキスト作成 (Host-DPU 接続用、local mmap は cmd staging 用) */
	doca_error_t ret;
	struct doca_rdma_ctx_t *rdma_host = &global_sample_objects.host_rdma_ctx;

	ret = doca_rdma_ctx_init(rdma_host, global_sample_objects.hw_dev, NULL,
	                         global_sample_objects.cmd_local_buf, CMD_SLOT_SIZE,
	                         DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
	                         32, 0,    /* recv_q_size=0: Host は Receive しない */
	                         HOST_GID_INDEX);
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
				                          32, 0, HOST_GID_INDEX);
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

	/* ---- Doorbell mmap 作成・エクスポート ---- */
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

int comch_client_init(const char *base_server_name, const char *dev_pci_addr)
{
	uint32_t max_msg_size;

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
    uint64_t rank = (uint64_t)tmp_rank;
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

	return 0;
}


//send完了コールバック
static void send_task_completion_callback(struct doca_comch_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data)
{
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

	printf("[RANK%lu] comch client: DOCA RDMA connection to DPU established (dual_rail=%d)\n",
	       global_sample_objects.rank, global_sample_objects.dual_rail);

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
	(void)user_data;
	(void)event;
	//もらった通知に応じて処理を変えるよ
	struct control_notify *recv_notify = NULL;
	control_notify_unpack(recv_buffer, CONTROL_NOTIFY_MAX_SIZE, &recv_notify);
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
     * (Double-destroy of DOCA objects can easily lead to SIGSEGV.)
     */
    static int cleaned = 0;
    if (cleaned)
        return;
    cleaned = 1;

    doca_error_t result;

    /* Stop auxiliary threads that may still touch DOCA objects. */
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
	(void)user_data;
	(void)ctx;
	(void)prev_state;
	switch (next_state) {
	case DOCA_CTX_STATE_IDLE:
		if(global_sample_objects.rank == 0)
			DOCA_LOG_INFO("CC client context has been stopped");
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
