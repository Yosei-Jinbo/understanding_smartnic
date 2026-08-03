#define _GNU_SOURCE
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
#include <doca_rdma.h>
#include <mpi.h>
#include "uthash.h"
#include <pthread.h>

#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "comch_server.h"
#include "../common/comch_mpi_common.h"
#include "../common/timing_utils.h"
#include "../common/doca_rdma_utils.h"
#include <sched.h>
#include <math.h>
#include "spsc_queue.h"

#include "rdma_task.h"
#include "rs_pool.h"
#include "union_tracker.h"
#include "dpu_config.h"
#include "collective.h"

DOCA_LOG_REGISTER(COMCH_SERVER);

/* =====================================================
 * コア割り当て設定（1 DPU = 2 ランク, 計 16 コア想定）
 * レイアウトと制約は dpu_config.h のコメントを参照。
 * ===================================================== */
int g_comm_cores    = COMM_CORES_DEFAULT;
int g_compute_cores = COMPUTE_CORES_DEFAULT;
/* CrossGVMI アブレーション用: 1 なら AG の GPU 直接アクセス(PCI cross-GVMI import)を無効化し、
 * host_dst_rmem/host_src_rmem 経由の RDMA staging 経路に落とす（gpu_direct を強制 off）。
 * RS には影響させない（RS は reduction のため常に DPU に取り込むので staging の概念が無い）。 */
int g_force_staging = 0;
/* AG の rail 選択。既定は single-rail（実測で dual≈single、律速は DPU↔GPU の cross-GVMI DMA
 * 経路で通信並列度は効かないため single を採用）。dual-rail コードは残しており、
 * FORCE_SINGLE_RAIL=0 で dual を有効化できる。 */
int g_force_single_rail = 1;
/* RS の Recv 先行 post（pre-post）。1 なら全 step の Recv をループ前に一括 post する。
 *
 * これは **チャンク分割とは独立の最適化** である。両者を 1 つのフラグで束ねていた時期があり、
 * その結果 chunk off（AG_PIECE_MAX=1、および chunk_size < AG_PIECE_MIN_SIZE の小メッセージ）で
 * 先行 post まで失われ、受信側が recv を post する前に送信が到達して RoCE の RNR リトライが
 * 発生し、サイズ非依存の固定遅延（実測 ~3ms）が乗っていた。
 * 既定 1（= main 相当の挙動）。アブレーションで効果を測るときだけ RS_PREPOST=0 にする。 */
int g_rs_prepost = 1;

static void core_alloc_init_from_env(void)
{
    const char *c = getenv("COMM_CORES");
    const char *k = getenv("COMPUTE_CORES");
    const char *fs = getenv("FORCE_STAGING");
    const char *sr = getenv("FORCE_SINGLE_RAIL");
    const char *pp = getenv("RS_PREPOST");
    if (c) { int v = atoi(c); if (v >= 1 && v <= (int)DOCA_WORKER_TYPE_COUNT) g_comm_cores = v; }
    if (k) { int v = atoi(k); if (v >= 1) g_compute_cores = v; }
    g_force_staging = (fs && atoi(fs) != 0) ? 1 : 0;
    /* FORCE_SINGLE_RAIL: 未設定なら既定(1=single)を維持。設定時のみ上書き(0=dual,1=single)。 */
    if (sr) g_force_single_rail = (atoi(sr) != 0) ? 1 : 0;
    /* RS_PREPOST: 未設定なら既定(1=有効)を維持。設定時のみ上書き。 */
    if (pp) g_rs_prepost = (atoi(pp) != 0) ? 1 : 0;
    int maxk = 16 - 4 - 2 * g_comm_cores;      /* 2*C + K + 4 <= 16 */
    if (g_compute_cores > maxk) g_compute_cores = maxk;
    if (g_compute_cores < 1)   g_compute_cores = 1;
    DOCA_LOG_INFO("core alloc: COMM_CORES=%d COMPUTE_CORES=%d (2*C+K+4=%d) FORCE_STAGING=%d FORCE_SINGLE_RAIL=%d RS_PREPOST=%d",
                  g_comm_cores, g_compute_cores, 2 * g_comm_cores + g_compute_cores + 4,
                  g_force_staging, g_force_single_rail, g_rs_prepost);
}

/* shutdown 時の ctx 停止待ちポーリング間隔 (旧 dma_common.h 由来) */
#define SLEEP_IN_NANOS (1000)

/* * バッファプール (pthread_spinlock, 連続確保) */

void cw_buffer_pool_init(struct cw_buffer_pool *pool, void *base, size_t slot_size)
{
    pthread_spin_init(&pool->lock, PTHREAD_PROCESS_PRIVATE);
    pool->base = base;
    pool->total_size = slot_size * CW_BUFFER_POOL_SLOTS;
    for (int i = 0; i < CW_BUFFER_POOL_SLOTS; i++) {
        pool->slots[i].buf = (char *)base + i * slot_size;
        pool->slots[i].capacity = slot_size;
        pool->slots[i].in_use = 0;
    }
}

int cw_buffer_pool_acquire(struct cw_buffer_pool *pool, size_t size, void **buf_out, size_t *capacity_out)
{
    pthread_spin_lock(&pool->lock);
    int free_idx = -1;
    for (int i = 0; i < CW_BUFFER_POOL_SLOTS; i++)
        if (!pool->slots[i].in_use) { free_idx = i; break; }
    if (free_idx < 0) { pthread_spin_unlock(&pool->lock); return -1; }

    struct cw_buffer_slot *slot = &pool->slots[free_idx];
    if (slot->capacity < size) {
        pthread_spin_unlock(&pool->lock);
        DOCA_LOG_ERR("Buffer slot %d capacity %zu < requested %zu (no realloc)", free_idx, slot->capacity, size);
        return -2;
    }
    slot->in_use = 1;
    *buf_out = slot->buf;
    if (capacity_out) *capacity_out = slot->capacity;
    pthread_spin_unlock(&pool->lock);
    return free_idx;
}

void cw_buffer_pool_release(struct cw_buffer_pool *pool, int slot_idx)
{
    if (slot_idx < 0 || slot_idx >= CW_BUFFER_POOL_SLOTS) return;
    pthread_spin_lock(&pool->lock);
    pool->slots[slot_idx].in_use = 0;
    pthread_spin_unlock(&pool->lock);
}

void cw_buffer_pool_destroy(struct cw_buffer_pool *pool)
{
    pthread_spin_destroy(&pool->lock);
}

static pthread_mutex_t send_notify_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_flag pe_progress_spin = ATOMIC_FLAG_INIT;

/* =====================================================
 * RDMA command slot (Host → DPU の RDMA-Write コマンド受信)
 * Host は cmd_slot_buf にこの構造体を RDMA Write する。
 * seq は offset 0; Host は payload を書いてから seq を書く (doorbell)。
 * ===================================================== */

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
    /* Variable-length export data follows (concatenated in order:
       src_export, dst_export, src_export_rail1, dst_export_rail1,
       local_cpu_export, local_cpu_flag_export) */
    uint8_t export_data[];
};

/* * メッセージワーカスレッドプール */

#define MSG_POOL_THREADS  1
#define MSG_QUEUE_CAP     4096  /* 8 bytes/entry = 32KB */

static void *message_worker_thread(void *arg);

struct msg_task { struct message_worker_arg *w; };

struct msg_thread_pool {
    pthread_t threads[MSG_POOL_THREADS];
    /* Lock-free SPSC ring buffer */
    struct msg_task q[MSG_QUEUE_CAP];
    _Alignas(64) atomic_uint head;
    _Alignas(64) atomic_uint tail;
    atomic_int stop;
    /* back-pointer for RDMA cmd_slot polling */
    struct comch_ctrl_path_server_objects *sample_objects;
};

static void *msg_pool_worker_main(void *arg)
{
    struct msg_thread_pool *p = (struct msg_thread_pool *)arg;
    while (!atomic_load_explicit(&p->stop, memory_order_acquire)) {
        /* ---- Fast path — poll RDMA command slot ---- */
        if (p->sample_objects) {
            struct collective_worker_t *cw = &p->sample_objects->collective_worker;
            if (cw->cmd_slot_enabled && cw->cmd_slot_buf) {
                volatile struct rdma_compact_cmd *cmd =
                    (volatile struct rdma_compact_cmd *)cw->cmd_slot_buf;
                uint64_t seq = cmd->seq;
                if (seq == cw->cmd_expected_seq) {
                    /* New command arrived via RDMA! */
                    cw->cmd_expected_seq++;
                    __sync_synchronize();  /* ensure payload reads happen after seq read */

                    /* Build control_cmd from compact format */
                    struct control_cmd recv_cmd;
                    memset(&recv_cmd, 0, sizeof(recv_cmd));
                    recv_cmd.type = CONTROL_CMD_UCP_COLLECTIVE;
                    recv_cmd.ucp_collective.id = seq;
                    recv_cmd.ucp_collective.remote_src_buffer_address = cmd->src_addr;
                    recv_cmd.ucp_collective.remote_src_buffer_len = cmd->src_len;
                    recv_cmd.ucp_collective.remote_dst_buffer_address = cmd->dst_addr;
                    recv_cmd.ucp_collective.remote_dst_buffer_len = cmd->dst_len;
                    recv_cmd.ucp_collective.collective_request.collective_op = cmd->op_type;

                    /* Extract export descriptors from inline data */
                    const uint8_t *ep = (const uint8_t *)cmd->export_data;
                    if (cmd->src_export_len > 0) {
                        recv_cmd.ucp_collective.src_rkey_buf = (void *)ep;
                        recv_cmd.ucp_collective.src_rkey_buf_len = cmd->src_export_len;
                        ep += cmd->src_export_len;
                    }
                    if (cmd->dst_export_len > 0) {
                        recv_cmd.ucp_collective.dst_rkey_buf = (void *)ep;
                        recv_cmd.ucp_collective.dst_rkey_buf_len = cmd->dst_export_len;
                        ep += cmd->dst_export_len;
                    }
                    if (cmd->src_export_len_rail1 > 0) {
                        recv_cmd.ucp_collective.src_rkey_buf_rail1 = (void *)ep;
                        recv_cmd.ucp_collective.src_rkey_buf_len_rail1 = cmd->src_export_len_rail1;
                        ep += cmd->src_export_len_rail1;
                    }
                    if (cmd->dst_export_len_rail1 > 0) {
                        recv_cmd.ucp_collective.dst_rkey_buf_rail1 = (void *)ep;
                        recv_cmd.ucp_collective.dst_rkey_buf_len_rail1 = cmd->dst_export_len_rail1;
                        ep += cmd->dst_export_len_rail1;
                    }
                    /* local_cpu fields */
                    recv_cmd.ucp_collective.local_cpu_buffer_address = cmd->local_cpu_addr;
                    recv_cmd.ucp_collective.local_cpu_buffer_len = cmd->local_cpu_len;
                    if (cmd->local_cpu_export_len > 0) {
                        recv_cmd.ucp_collective.local_cpu_rkey_buf = (void *)ep;
                        recv_cmd.ucp_collective.local_cpu_rkey_buf_len = cmd->local_cpu_export_len;
                        ep += cmd->local_cpu_export_len;
                    }
                    recv_cmd.ucp_collective.local_cpu_flag_address = cmd->local_cpu_flag_addr;
                    recv_cmd.ucp_collective.local_cpu_flag_len = cmd->local_cpu_flag_len;
                    if (cmd->local_cpu_flag_export_len > 0) {
                        recv_cmd.ucp_collective.local_cpu_flag_rkey_buf = (void *)ep;
                        recv_cmd.ucp_collective.local_cpu_flag_rkey_buf_len = cmd->local_cpu_flag_export_len;
                        ep += cmd->local_cpu_flag_export_len;
                    }

                    /* Execute directly (no malloc, no unpack, no SPSC queue)
                     * Note: this is the RDMA-Write command path (disabled in current
                     * production), defaults to Ring 0. Production uses ComCh + ring
                     * dispatcher which correctly assigns ring_id. */
                    execute_doca_collective_cmd(&recv_cmd, p->sample_objects, 0);

                    /* Clear slot for next use */
                    __sync_synchronize();
                    cmd->seq = 0;
                    __sync_synchronize();

                    continue;  /* Back to poll loop */
                }
            }
        }

        /* ---- Slow path: ComCh message queue ---- */
        uint32_t h = atomic_load_explicit(&p->head, memory_order_acquire);
        uint32_t t = atomic_load_explicit(&p->tail, memory_order_relaxed);
        if (h == t) {
            /* empty — spin */
            continue;
        }
        struct message_worker_arg *w = p->q[h].w;
        atomic_store_explicit(&p->head, (h + 1) % MSG_QUEUE_CAP, memory_order_release);
        (void)message_worker_thread(w);
    }
    return NULL;
}

static int msg_pool_init(struct msg_thread_pool *p, int mpi_rank,
                         struct comch_ctrl_path_server_objects *sample_objects)
{
    memset(p, 0, sizeof(*p));
    atomic_store(&p->head, 0);
    atomic_store(&p->tail, 0);
    atomic_store(&p->stop, 0);
    p->sample_objects = sample_objects;  /* for cmd_slot polling */
    for (int i = 0; i < MSG_POOL_THREADS; i++) {
        if (pthread_create(&p->threads[i], NULL, msg_pool_worker_main, p) != 0) {
            atomic_store(&p->stop, 1);
            for (int j = 0; j < i; j++) pthread_join(p->threads[j], NULL);
            return -1;
        }
        cpu_set_t cpuset; CPU_ZERO(&cpuset);
        CPU_SET(msg_pool_core(mpi_rank), &cpuset);
        pthread_setaffinity_np(p->threads[i], sizeof(cpu_set_t), &cpuset);
    }
    return 0;
}

static void msg_pool_destroy(struct msg_thread_pool *p)
{
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    for (int i = 0; i < MSG_POOL_THREADS; i++) if (p->threads[i]) pthread_join(p->threads[i], NULL);
    /* drain remaining */
    uint32_t h = atomic_load(&p->head);
    uint32_t t = atomic_load(&p->tail);
    while (h != t) {
        struct msg_task task = p->q[h]; h = (h + 1) % MSG_QUEUE_CAP;
        if (task.w) { if (task.w->recv_cmd) free(task.w->recv_cmd); free(task.w); }
    }
}

static int msg_pool_try_submit(struct msg_thread_pool *p, struct message_worker_arg *w)
{
    if (atomic_load_explicit(&p->stop, memory_order_acquire)) return -1;
    uint32_t t = atomic_load_explicit(&p->tail, memory_order_relaxed);
    uint32_t next_t = (t + 1) % MSG_QUEUE_CAP;
    if (next_t == atomic_load_explicit(&p->head, memory_order_acquire)) return -1; /* full */
    p->q[t].w = w;
    atomic_store_explicit(&p->tail, next_t, memory_order_release);
    return 0;
}

/* * ComCh 通知ユーティリティ */

static struct send_notify_entry *get_send_notify_entry(struct comch_ctrl_path_server_objects *objects, uint64_t id, bool create_if_missing)
{
    struct send_notify_entry *entry = NULL;
    pthread_mutex_lock(&send_notify_mutex);
    HASH_FIND(hh, objects->send_notify_table, &id, sizeof(uint64_t), entry);
    if (entry == NULL && create_if_missing) {
        entry = (struct send_notify_entry *)calloc(1, sizeof(*entry));
        if (!entry) { pthread_mutex_unlock(&send_notify_mutex); return NULL; }
        entry->id = id;
        atomic_init(&entry->finished, false);
        HASH_ADD(hh, objects->send_notify_table, id, sizeof(uint64_t), entry);
    }
    pthread_mutex_unlock(&send_notify_mutex);
    return entry;
}

doca_error_t comch_send_control_notify(struct control_notify *send_notify, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct doca_comch_task_send *task;
    struct doca_task *task_obj;
    doca_error_t result;
    union doca_data user_data;

    uint64_t notify_id;
    switch (send_notify->type) {
    case CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU: notify_id = send_notify->ucp_connect_host_dpu.id; break;
    case CONTROL_NOTIFY_UCP_CREATE_RING:      notify_id = send_notify->ucp_create_ring.id; break;
    case CONTROL_NOTIFY_UCP_COLLECTIVE:       notify_id = send_notify->ucp_collective.id; break;
    default: return DOCA_ERROR_INVALID_VALUE;
    }

    uint8_t send_buf[CONTROL_NOTIFY_MAX_SIZE];
    size_t send_cap = CONTROL_NOTIFY_MAX_SIZE;
    result = control_notify_pack(send_notify, &send_cap, send_buf);
    if (result != DOCA_SUCCESS) return result;

    struct send_notify_entry *entry = get_send_notify_entry(sample_objects, notify_id, true);
    if (!entry) return DOCA_ERROR_NO_MEMORY;
    atomic_store_explicit(&entry->finished, false, memory_order_release);

    user_data.u64 = notify_id;
    DOCA_CHECK(doca_comch_server_task_send_alloc_init(sample_objects->server, sample_objects->connection,
                                                      send_buf, CONTROL_NOTIFY_MAX_SIZE, &task));
    task_obj = doca_comch_task_send_as_task(task);
    doca_task_set_user_data(task_obj, user_data);
    DOCA_CHECK(doca_task_submit(task_obj));

    while (!atomic_load_explicit(&entry->finished, memory_order_acquire)) {
        if (!atomic_flag_test_and_set_explicit(&pe_progress_spin, memory_order_acquire)) {
            doca_pe_progress(sample_objects->pe);
            atomic_flag_clear_explicit(&pe_progress_spin, memory_order_release);
        }
    }
    return DOCA_SUCCESS;
}

/* * ComCh コールバック */

void send_task_completion_callback(struct doca_comch_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    struct comch_ctrl_path_server_objects *sample_objects = (struct comch_ctrl_path_server_objects *)ctx_user_data.ptr;
    uint64_t id = task_user_data.u64;
    struct send_notify_entry *entry = get_send_notify_entry(sample_objects, id, false);
    if (entry) { atomic_store_explicit(&entry->finished, true, memory_order_release); }
    sample_objects->result = DOCA_SUCCESS;
    doca_task_free(doca_comch_task_send_as_task(task));
}

void send_task_completion_err_callback(struct doca_comch_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    struct comch_ctrl_path_server_objects *sample_objects = (struct comch_ctrl_path_server_objects *)ctx_user_data.ptr;
    uint64_t id = task_user_data.u64;
    struct send_notify_entry *entry = get_send_notify_entry(sample_objects, id, false);
    if (entry) { atomic_store_explicit(&entry->finished, true, memory_order_release); }
    sample_objects->result = doca_task_get_status(doca_comch_task_send_as_task(task));
    doca_task_free(doca_comch_task_send_as_task(task));
}

void server_connection_event_callback(struct doca_comch_event_connection_status_changed *event, struct doca_comch_connection *comch_conn, uint8_t change_success)
{
    union doca_data user_data;
    struct doca_comch_server *comch_server;
    struct comch_ctrl_path_server_objects *sample_objects;
    (void)event;
    comch_server = doca_comch_server_get_server_ctx(comch_conn);
    if (doca_ctx_get_user_data(doca_comch_server_as_ctx(comch_server), &user_data) != DOCA_SUCCESS) return;
    sample_objects = (struct comch_ctrl_path_server_objects *)user_data.ptr;
    if (!change_success) { DOCA_LOG_ERR("Failed connection received"); return; }
    sample_objects->num_connected_clients++;
    if (sample_objects->rank == 0) DOCA_LOG_INFO("New client connected to server");
}

void server_disconnection_event_callback(struct doca_comch_event_connection_status_changed *event, struct doca_comch_connection *comch_conn, uint8_t change_success)
{
    union doca_data user_data;
    struct doca_comch_server *comch_server;
    struct comch_ctrl_path_server_objects *sample_objects;
    (void)event; (void)change_success;
    comch_server = doca_comch_server_get_server_ctx(comch_conn);
    if (doca_ctx_get_user_data(doca_comch_server_as_ctx(comch_server), &user_data) != DOCA_SUCCESS) return;
    sample_objects = (struct comch_ctrl_path_server_objects *)user_data.ptr;
    sample_objects->num_connected_clients--;
    if (sample_objects->rank == 0) DOCA_LOG_INFO("A client was disconnected from server");
    /* DPU Union Tracker: 切断時 (= 学習終了時) に必ず出力。
     * 短い run (ViT ag_smartnic 等) でも aggregate を確実に得られるようにする。 */
    dpu_union_tracker_print(sample_objects->rank);
    sample_objects->finish = true;
}

/* =====================================================
 * Per-ring job queue and processing thread
 *
 * Ring 0 / Ring 1 それぞれが自分の job queue を持ち、独立したスレッドで処理する。
 * これにより Ring 0 と Ring 1 が DPU 上で並列実行される。
 *
 * 順序保証: 各 queue は FIFO。Ring assignment は ag_seq_counter で deterministic。
 * 全 rank で同じ AG は同じ ring に割り当てられるため、Ring algorithm は正しく動作する。
 * ===================================================== */

/* Per-ring queue 初期化 */
int ring_queues_init(struct collective_worker_t *cw)
{
    for (int i = 0; i < N_RINGS; i++) {
        cw->ring_queue_head[i] = NULL;
        cw->ring_queue_tail[i] = NULL;
        if (pthread_mutex_init(&cw->ring_queue_mtx[i], NULL) != 0) return -1;
        if (pthread_cond_init(&cw->ring_queue_cv[i], NULL) != 0) return -1;
        atomic_store(&cw->ring_queue_stop[i], false);
        atomic_store(&cw->ring_proc_running[i], false);
    }
    atomic_store(&cw->ag_seq_counter, 0);
    return 0;
}

/* Ring queue に job を追加 (FIFO, cond_signal で worker を wake)
 *
 * 3-tier adaptive wait の worker を確実に起こすため、push 時に必ず
 * cond_signal を送る。hot path では worker は Tier 1/2 で capture してくれるため
 * cond_signal は無害 (no sleeper)。idle 時は signal で Tier 3 から即 wake。
 *
 * signal はロック内部で発行 (pthread 標準パターン)。解放順序を守ることで
 * missed wake-up を避ける。 */
static void ring_queue_push(struct collective_worker_t *cw, int ring_id, struct ring_job *job)
{
    pthread_mutex_lock(&cw->ring_queue_mtx[ring_id]);
    job->next = NULL;
    if (cw->ring_queue_tail[ring_id]) {
        cw->ring_queue_tail[ring_id]->next = job;
    } else {
        cw->ring_queue_head[ring_id] = job;
    }
    cw->ring_queue_tail[ring_id] = job;
    pthread_cond_signal(&cw->ring_queue_cv[ring_id]);
    pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
}

/* Ring queue から 1 job を取り出す (3-tier adaptive wait)
 *
 * 過去の設計失敗:
 *   - 純 cond_wait → Linux scheduler の wake-up が 10ms 級で rank skew が拡大
 *   - 純 busy-poll → unpinned thread × 4 が critical pinned thread を阻害
 *
 * 解決: 3-tier adaptive hybrid
 *   Tier 1 (fast path):    head != NULL なら即 pop。hot path (queue が常に非空) で 0 overhead
 *   Tier 2 (short spin):   ~10000 × cpu_relax (~10-50μs) で短 gap を吸収。
 *                          cond_wait のコストを回避しつつ、rank skew を 10μs 級に抑制
 *   Tier 3 (cond_wait):    長 idle 時のみ sleep。CPU を消費しないので pinned thread と競合しない
 *
 * Training 時の挙動分析:
 *   - AG arrival 間隔 ≈ 5ms、処理時間 ≈ 5ms → steady state で queue は常に 0-few 項目
 *   - Tier 1 で 80-90% のケースを捕獲
 *   - Tier 2 で burst 間の短 gap を吸収
 *   - Tier 3 は step 境界や idle 区間のみ (wake-up latency 許容)
 *
 * 注意: spin 中も stop flag を check して shutdown race を避ける */
static struct ring_job *ring_queue_pop_blocking(struct collective_worker_t *cw, int ring_id)
{
    /* ---- Tier 1: fast path (no spin, no cond_wait) ---- */
    if (cw->ring_queue_head[ring_id] != NULL) {
        pthread_mutex_lock(&cw->ring_queue_mtx[ring_id]);
        if (cw->ring_queue_head[ring_id] != NULL) {
            struct ring_job *job = cw->ring_queue_head[ring_id];
            cw->ring_queue_head[ring_id] = job->next;
            if (!cw->ring_queue_head[ring_id]) cw->ring_queue_tail[ring_id] = NULL;
            job->next = NULL;
            pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
            return job;
        }
        pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
    }

    /* ---- Tier 2: short spin (10000 × cpu_relax ≈ 10-50μs) ---- */
    for (int i = 0; i < 10000; i++) {
        if (atomic_load_explicit(&cw->ring_queue_stop[ring_id], memory_order_acquire))
            return NULL;
        if (cw->ring_queue_head[ring_id] != NULL) {
            pthread_mutex_lock(&cw->ring_queue_mtx[ring_id]);
            if (cw->ring_queue_head[ring_id] != NULL) {
                struct ring_job *job = cw->ring_queue_head[ring_id];
                cw->ring_queue_head[ring_id] = job->next;
                if (!cw->ring_queue_head[ring_id]) cw->ring_queue_tail[ring_id] = NULL;
                job->next = NULL;
                pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
                return job;
            }
            pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
        }
        cpu_relax();
    }

    /* ---- Tier 3: cond_wait (long idle, low-priority) ----
     * Lock を保持した状態で predicate を check し、空なら cond_wait で sleep。
     * signaled または stop フラグで wake up する。
     * spurious wake-up に対して while ループで再チェック。 */
    pthread_mutex_lock(&cw->ring_queue_mtx[ring_id]);
    while (!atomic_load_explicit(&cw->ring_queue_stop[ring_id], memory_order_acquire) &&
           cw->ring_queue_head[ring_id] == NULL) {
        pthread_cond_wait(&cw->ring_queue_cv[ring_id], &cw->ring_queue_mtx[ring_id]);
    }
    if (atomic_load_explicit(&cw->ring_queue_stop[ring_id], memory_order_acquire)) {
        pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
        return NULL;
    }
    struct ring_job *job = cw->ring_queue_head[ring_id];
    if (job) {
        cw->ring_queue_head[ring_id] = job->next;
        if (!cw->ring_queue_head[ring_id]) cw->ring_queue_tail[ring_id] = NULL;
        job->next = NULL;
    }
    pthread_mutex_unlock(&cw->ring_queue_mtx[ring_id]);
    return job;
}

/* Per-ring 処理スレッドの起動パラメータ */
struct ring_proc_init_arg {
    struct comch_ctrl_path_server_objects *sample_objects;
    int ring_id;
};

/* Per-ring 処理スレッドの main loop */
static void *ring_proc_main(void *arg)
{
    struct ring_proc_init_arg *init_arg = (struct ring_proc_init_arg *)arg;
    struct comch_ctrl_path_server_objects *sample_objects = init_arg->sample_objects;
    int ring_id = init_arg->ring_id;
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    free(init_arg);

    atomic_store(&cw->ring_proc_running[ring_id], true);

    /* CPU affinity: 既存の worker thread が comm_core() を使っている。
     * ring proc thread は別コアに配置したいが、DPU は 16 コアで余裕が少ない。
     * 明示的には pinning せず OS に任せる (kernel が適切なコアに配置する)。 */

    while (!atomic_load(&cw->ring_queue_stop[ring_id])) {
        struct ring_job *job = ring_queue_pop_blocking(cw, ring_id);
        if (!job) break;  /* stopped */

        /* Execute collective on this ring */
        execute_doca_collective_cmd(job->cmd, job->sample_objects, ring_id);

        /* job->cmd and job->buf_copy point to the SAME memory (control_cmd_unpack
         * returns packed_cmd as-is). Free once via buf_copy. */
        if (job->buf_copy) free(job->buf_copy);
        free(job);
    }

    atomic_store(&cw->ring_proc_running[ring_id], false);
    return NULL;
}

/* Per-ring 処理スレッド起動 (create_ring の最後で呼ぶ) */
static int ring_proc_threads_start(struct comch_ctrl_path_server_objects *sample_objects)
{
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    /* Ring 1 が無効な場合は 1 thread だけ起動 (Ring 0 のみ) */
    int n = cw->ring_r1_enabled ? N_RINGS : 1;
    for (int i = 0; i < n; i++) {
        struct ring_proc_init_arg *init_arg = (struct ring_proc_init_arg *)calloc(1, sizeof(*init_arg));
        if (!init_arg) { DOCA_LOG_ERR("ring_proc init_arg alloc failed"); return -1; }
        init_arg->sample_objects = sample_objects;
        init_arg->ring_id = i;
        if (pthread_create(&cw->ring_proc_thread[i], NULL, ring_proc_main, init_arg) != 0) {
            DOCA_LOG_ERR("ring_proc thread %d create failed", i);
            free(init_arg);
            return -1;
        }
    }
    if (sample_objects->rank == 0) {
        printf("[DPU] %d ring processing thread(s) started\n", n);
        fflush(stdout);
    }
    return 0;
}

/* Per-ring 処理スレッド停止 (cleanup で呼ぶ)
 * stop flag セットに加えて cond_broadcast で sleeping worker を起こす。
 * これがないと Tier 3 cond_wait 中の worker が shutdown を検出できず hang する。 */
static void ring_proc_threads_stop(struct collective_worker_t *cw)
{
    int n = cw->ring_r1_enabled ? N_RINGS : 1;
    for (int i = 0; i < n; i++) {
        atomic_store(&cw->ring_queue_stop[i], true);
        /* sleeping worker を起こす (Tier 3 cond_wait から抜けさせる) */
        pthread_mutex_lock(&cw->ring_queue_mtx[i]);
        pthread_cond_broadcast(&cw->ring_queue_cv[i]);
        pthread_mutex_unlock(&cw->ring_queue_mtx[i]);
    }
    for (int i = 0; i < n; i++) {
        if (atomic_load(&cw->ring_proc_running[i])) {
            pthread_join(cw->ring_proc_thread[i], NULL);
        }
    }
}

/* * メッセージハンドラ */

static void *message_worker_thread(void *arg)
{
    struct message_worker_arg *w = (struct message_worker_arg *)arg;
    struct comch_ctrl_path_server_objects *sample_objects = w->sample_objects;
    struct control_cmd *recv_cmd = w->recv_cmd;

    /* 重要: control_cmd_unpack は packed_cmd (= buf_copy) を return するため
     * recv_cmd == w->buf_copy。1 回の free で両方解放される。
     * ここでは free(recv_cmd) が buf_copy も解放することに注意。 */

    if (recv_cmd) {
        switch (recv_cmd->type) {
        case CONTROL_CMD_UCP_CREATE_RING:
            execute_doca_create_ring_cmd(recv_cmd, sample_objects);
            /* Parallel ring dispatch (env var PHASE15_ENABLE_RING_QUEUES=1):
             *   Ring 0 / Ring 1 の 2 worker thread を起動し、AG を round-robin で並列実行する。
             *
             *   busy-poll 版は unpinned thread × 4 が critical pinned thread を阻害して
             *   全体悪化 → 失敗。現行は adaptive hybrid: Tier 1 (fast path) +
             *   Tier 2 (short spin 50us) + Tier 3 (cond_wait)。idle 時は CPU 消費ゼロで
             *   pinned thread と競合しない。
             *
             *   デフォルトは無効 (安全側)。PHASE15_ENABLE_RING_QUEUES=1 で opt-in 有効化。 */
            {
                static int use_ring_queues = -1;
                if (use_ring_queues < 0) {
                    const char *e = getenv("PHASE15_ENABLE_RING_QUEUES");
                    use_ring_queues = (e && atoi(e) != 0) ? 1 : 0;
                }
                if (use_ring_queues) {
                    (void)ring_proc_threads_start(sample_objects);
                } else {
                    if (sample_objects->rank == 0) {
                        printf("[DPU] ring_proc threads NOT started (PHASE15_ENABLE_RING_QUEUES unset), using inline path\n");
                        fflush(stdout);
                    }
                }
            }
            free(recv_cmd);  /* = free(buf_copy) */
            free(w);
            return NULL;
        case CONTROL_CMD_UCP_COLLECTIVE:
            /* Dispatch (env PHASE15_ENABLE_RING_QUEUES=1):
             *   - AG: ag_seq % N_RINGS で Ring 0/1 に round-robin 配分 → 2-way 並列
             *   - RS: 常に Ring 0 (既存 semantics 維持)
             *   - デフォルト (env unset): inline 実行パス (serial)
             *
             *   per-op 計測で DPU inflight=1 が判明 (msg_pool = 1 thread の結果)。
             *   adaptive hybrid wait により parallel dispatch を安全に enable できる。 */
            {
                static int use_ring_queues = -1;
                if (use_ring_queues < 0) {
                    const char *e = getenv("PHASE15_ENABLE_RING_QUEUES");
                    use_ring_queues = (e && atoi(e) != 0) ? 1 : 0;
                    if (sample_objects->rank == 0) {
                        printf("[DPU] PHASE15_ENABLE_RING_QUEUES=%d — %s\n",
                               use_ring_queues,
                               use_ring_queues ? "parallel ring queue (2 workers)" : "inline serial (default)");
                        fflush(stdout);
                    }
                }

                if (!use_ring_queues) {
                    /* デフォルト: inline 実行。ring_id は常に 0。RS も AG もこのパスを通る。 */
                    execute_doca_collective_cmd(recv_cmd, sample_objects, 0);
                    break;
                }

                /* Ring queue path (opt-in only) */
                struct collective_worker_t *cw = &sample_objects->collective_worker;
                bool is_ag = (recv_cmd->ucp_collective.collective_request.collective_op == COLLECTIVE_ALL_GATHER);
                int ring_id;
                if (is_ag) {
                    if (cw->ring_r1_enabled) {
                        ring_id = (int)(atomic_fetch_add(&cw->ag_seq_counter, 1) % N_RINGS);
                    } else {
                        ring_id = 0;
                        atomic_fetch_add(&cw->ag_seq_counter, 1);
                    }
                } else {
                    /* RS は常に Ring 0 (Ring 0 ctx を共有するため) */
                    ring_id = 0;
                }

                struct ring_job *job = (struct ring_job *)calloc(1, sizeof(*job));
                if (!job) {
                    DOCA_LOG_ERR("ring_job alloc failed, falling back to inline execution");
                    execute_doca_collective_cmd(recv_cmd, sample_objects, 0);
                    free(recv_cmd);  /* = free(buf_copy) */
                    free(w);
                    return NULL;
                }
                job->cmd = recv_cmd;
                job->buf_copy = w->buf_copy;  /* same as recv_cmd */
                job->ring_id = ring_id;
                job->sample_objects = sample_objects;

                ring_queue_push(cw, ring_id, job);
                /* buf_copy / recv_cmd は job が所有する。ここでは free しない */
                free(w);  /* w だけ free */
                return NULL;
            }
            break;
        case CONTROL_CMD_UCP_CONNECT_HOST_DPU:
            execute_doca_connect_host_dpu_cmd(recv_cmd, sample_objects);
            break;
        case CONTROL_CMD_UCP_INIT_FLAG_POOL:
            execute_doca_init_flag_pool_cmd(recv_cmd, sample_objects);
            break;
        default:
            break;
        }
    }
    free(recv_cmd);  /* = free(buf_copy) */
    free(w);
    return NULL;
}

static void message_recv_callback(struct doca_comch_event_msg_recv *event,
                                  uint8_t *recv_buffer, uint32_t msg_len,
                                  struct doca_comch_connection *comch_connection)
{
    union doca_data user_data;
    struct doca_comch_server *comch_server;
    struct comch_ctrl_path_server_objects *sample_objects;
    (void)event;

    comch_server = doca_comch_server_get_server_ctx(comch_connection);
    if (doca_ctx_get_user_data(doca_comch_server_as_ctx(comch_server), &user_data) != DOCA_SUCCESS) return;
    sample_objects = (struct comch_ctrl_path_server_objects *)user_data.ptr;
    sample_objects->connection = comch_connection;

    void *buf_copy = malloc(msg_len);
    if (!buf_copy) { DOCA_LOG_ERR("malloc failed"); return; }
    memcpy(buf_copy, recv_buffer, msg_len);

    struct control_cmd *recv_cmd = NULL;
    doca_error_t st = control_cmd_unpack(buf_copy, msg_len, &recv_cmd);
    if (st != DOCA_SUCCESS || !recv_cmd) { free(buf_copy); return; }

    struct message_worker_arg *w = malloc(sizeof(*w));
    if (!w) { free(recv_cmd); free(buf_copy); return; }
    w->sample_objects = sample_objects;
    w->recv_cmd = recv_cmd;
    w->buf_copy = buf_copy;

    if (!sample_objects->msg_pool || msg_pool_try_submit(sample_objects->msg_pool, w) != 0) {
        /* msg_pool 満杯または無効: drop して解放
         * 注意: recv_cmd は buf_copy 内部を指すポインタなので free(recv_cmd) してはいけない */
        if (sample_objects->msg_pool)
            DOCA_LOG_WARN("msg_pool queue full, dropping collective command");
        free(w); free(buf_copy);
        return;
    }
}

/* * クリーンアップ */

static void clean_comch_sample_objects(struct comch_ctrl_path_server_objects *sample_objects)
{
    static int cleaned = 0;
    if (cleaned) return;
    cleaned = 1;
    doca_error_t result;

    /* Per-ring 処理スレッドを先に停止 (RDMA ctx 破棄前) */
    ring_proc_threads_stop(&sample_objects->collective_worker);

    /* Per-ring MPI_Comm を解放 (MPI_Finalize より先に)
     * MPI_Comm_free は collective 操作なので全 rank で同じ順序で呼ぶ */
    for (int ri = 0; ri < N_RINGS; ri++) {
        if (sample_objects->collective_worker.ring_comm_valid[ri] &&
            sample_objects->collective_worker.ring_comm[ri] != MPI_COMM_NULL) {
            MPI_Comm_free(&sample_objects->collective_worker.ring_comm[ri]);
            sample_objects->collective_worker.ring_comm_valid[ri] = false;
        }
    }

    if (sample_objects->collective_worker.worker_pool) {
        doca_worker_thread_pool_destroy(sample_objects->collective_worker.worker_pool);
        free(sample_objects->collective_worker.worker_pool);
        sample_objects->collective_worker.worker_pool = NULL;
    }
    if (sample_objects->collective_worker.rs_pool) {
        rs_thread_pool_destroy(sample_objects->collective_worker.rs_pool);
        sample_objects->collective_worker.rs_pool = NULL;
    }

    /* RDB の remote mmap を破棄 (RDMA ctx より先に) */
    if (sample_objects->collective_worker.rdb_enabled) {
        doca_remote_mem_destroy(&sample_objects->collective_worker.rdb_next_rmem);
        sample_objects->collective_worker.rdb_enabled = false;
    }

    /* DOCA RDMA コンテキスト破棄 (PCI mmap より先に破棄し、buf を解放する) */
    doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_rma);
    if (sample_objects->collective_worker.dual_rail)
        doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_rma_rail1);
    doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_send);
    doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_recv);
    /* Multi-Rail Ring: rail1 コンテキスト破棄 */
    if (sample_objects->collective_worker.conn_to_next_rail1)
        doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_send_rail1);
    if (sample_objects->collective_worker.conn_from_prev_rail1)
        doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_recv_rail1);
    /* Ring 1 コンテキスト破棄 */
    if (sample_objects->collective_worker.ring_r1_enabled) {
        doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_send_r1);
        doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_recv_r1);
        if (sample_objects->collective_worker.ring_r1_rail1_enabled) {
            doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_send_r1_rail1);
            doca_rdma_ctx_destroy(&sample_objects->collective_worker.rdma_ring_recv_r1_rail1);
        }
    }

    /* Remote mmap キャッシュ破棄 */
    rmem_cache_destroy(sample_objects->collective_worker.src_rmem_cache);
    rmem_cache_destroy(sample_objects->collective_worker.dst_rmem_cache);
    /* Multi-Rail: rail1 キャッシュ破棄 */
    rmem_cache_destroy(sample_objects->collective_worker.src_rmem_cache_rail1);
    rmem_cache_destroy(sample_objects->collective_worker.dst_rmem_cache_rail1);
    /* PCI mmap キャッシュ破棄 (RDMA ctx 破棄後なので buf は全て解放済み) */
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_dst_pci_cache);
    sample_objects->collective_worker.gpu_dst_pci_cache = NULL;
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_src_pci_cache);
    sample_objects->collective_worker.gpu_src_pci_cache = NULL;
    /* rail1 PCI mmap キャッシュ破棄 */
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_dst_pci_cache_rail1);
    sample_objects->collective_worker.gpu_dst_pci_cache_rail1 = NULL;
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_src_pci_cache_rail1);
    sample_objects->collective_worker.gpu_src_pci_cache_rail1 = NULL;
    if (sample_objects->msg_pool) { msg_pool_destroy(sample_objects->msg_pool); free(sample_objects->msg_pool); sample_objects->msg_pool = NULL; }

    /* cmd_slot 解放 */
    if (sample_objects->collective_worker.cmd_slot_mmap) {
        doca_mmap_stop(sample_objects->collective_worker.cmd_slot_mmap);
        doca_mmap_destroy(sample_objects->collective_worker.cmd_slot_mmap);
        sample_objects->collective_worker.cmd_slot_mmap = NULL;
    }
    if (sample_objects->collective_worker.cmd_slot_buf) {
        free(sample_objects->collective_worker.cmd_slot_buf);
        sample_objects->collective_worker.cmd_slot_buf = NULL;
    }

    /* Working buffer 解放 */
    free(sample_objects->collective_worker.working_buf);
    sample_objects->collective_worker.working_buf = NULL;

    if (sample_objects->server && sample_objects->pe) {
        struct doca_ctx *ctx = doca_comch_server_as_ctx(sample_objects->server);
        enum doca_ctx_states state;
        (void)doca_ctx_get_state(ctx, &state);
        if (state != DOCA_CTX_STATE_IDLE && state != DOCA_CTX_STATE_STOPPING) {
            result = doca_ctx_stop(ctx);
            (void)result;
        }
        struct timespec ts = {.tv_sec = 0, .tv_nsec = SLEEP_IN_NANOS};
        do {
            doca_pe_progress(sample_objects->pe);
            result = doca_ctx_get_state(ctx, &state);
            if (result != DOCA_SUCCESS) break;
            if (state != DOCA_CTX_STATE_IDLE) nanosleep(&ts, NULL);
        } while (state != DOCA_CTX_STATE_IDLE);
    }
    if (sample_objects->server) { doca_comch_server_destroy(sample_objects->server); sample_objects->server = NULL; }
    if (sample_objects->pe) { doca_pe_destroy(sample_objects->pe); sample_objects->pe = NULL; }
    if (sample_objects->rep_dev) { doca_dev_rep_close(sample_objects->rep_dev); sample_objects->rep_dev = NULL; }
    if (sample_objects->hw_dev) { doca_dev_close(sample_objects->hw_dev); sample_objects->hw_dev = NULL; }
}

/* * ComCh サーバ状態変更コールバック */

static void comch_server_state_changed_callback(const union doca_data user_data, struct doca_ctx *ctx, enum doca_ctx_states prev_state, enum doca_ctx_states next_state)
{
    (void)ctx; (void)prev_state;
    struct comch_ctrl_path_server_objects *sample_objects = (struct comch_ctrl_path_server_objects *)user_data.ptr;
    switch (next_state) {
    case DOCA_CTX_STATE_IDLE: sample_objects->finish = true; break;
    case DOCA_CTX_STATE_RUNNING: break;
    default: break;
    }
}

/* * ComCh サーバ初期化 */

static doca_error_t init_comch_ctrl_path_server_objects(const char *server_name, const char *dev_pci_addr, const char *dev_rep_pci_addr, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct comch_ctrl_path_server_cb_config cfg = {
        .send_task_comp_cb = send_task_completion_callback,
        .send_task_comp_err_cb = send_task_completion_err_callback,
        .msg_recv_cb = message_recv_callback,
        .server_connection_event_cb = server_connection_event_callback,
        .server_disconnection_event_cb = server_disconnection_event_callback,
        .data_path_mode = false,
        .new_consumer_cb = NULL,
        .expired_consumer_cb = NULL,
        .ctx_user_data = sample_objects,
        .ctx_state_changed_cb = comch_server_state_changed_callback
    };
    DOCA_CHECK(open_doca_device_with_pci(dev_pci_addr, NULL, &(sample_objects->hw_dev)));
    DOCA_CHECK(open_doca_device_rep_with_pci(sample_objects->hw_dev, DOCA_DEVINFO_REP_FILTER_NET, dev_rep_pci_addr, &(sample_objects->rep_dev)));
    DOCA_CHECK(init_comch_ctrl_path_server(server_name, sample_objects->hw_dev, sample_objects->rep_dev, &cfg, &(sample_objects->server), &(sample_objects->pe)));

    sample_objects->msg_pool = malloc(sizeof(*sample_objects->msg_pool));
    if (!sample_objects->msg_pool) return DOCA_ERROR_NO_MEMORY;
    if (msg_pool_init(sample_objects->msg_pool, (int)sample_objects->rank, sample_objects) != 0) {
        free(sample_objects->msg_pool); sample_objects->msg_pool = NULL;
        return DOCA_ERROR_NO_MEMORY;
    }
    return DOCA_SUCCESS;
}

/* * main */

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s <server name> <dpu pci address> <representor pci addr> <device_name>\n", argv[0]);
        return 1;
    }
    const char *base_server_name = argv[1];
    const char *dev_pci_addr     = argv[2];
    const char *dev_rep_pci_addr = argv[3];
    const char *device_name      = argv[4];
    doca_error_t status = DOCA_SUCCESS;
    struct comch_ctrl_path_server_objects sample_objects = {0};
    uint32_t max_msg_size;

    struct doca_log_backend *sdk_log;
    status = doca_log_backend_create_standard();
    if (status != DOCA_SUCCESS) goto mpi_fail;
    status = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (status != DOCA_SUCCESS) goto mpi_fail;
    status = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
    if (status != DOCA_SUCCESS) goto mpi_fail;

    /* MPI_THREAD_MULTIPLE 必須 (msg_pool_worker スレッドから MPI_Barrier を呼ぶため) */
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    if (mpi_provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "[WARN] MPI_THREAD_MULTIPLE not supported (provided=%d), MPI_Barrier in collective may fail\n", mpi_provided);
    }
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
    uint64_t rank = (uint64_t)tmp_rank, world_size = (uint64_t)tmp_world_size;

    core_alloc_init_from_env();   /* COMM_CORES / COMPUTE_CORES を env から反映（全ピン留めより前） */

    {
        cpu_set_t cpuset; CPU_ZERO(&cpuset);
        int main_cpu_id = main_core(tmp_rank);
        CPU_SET(main_cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    char server_name_buf[256];
    snprintf(server_name_buf, sizeof(server_name_buf), "%s_%d", base_server_name, tmp_rank);
    const char *server_name = server_name_buf;

    memset(&(sample_objects.collective_worker), 0, sizeof(sample_objects.collective_worker));
    sample_objects.rank = rank;
    sample_objects.world_size = world_size;
    sample_objects.dev_pci_addr = dev_pci_addr;
    sample_objects.device_name = device_name;

    status = init_comch_ctrl_path_server_objects(server_name, dev_pci_addr, dev_rep_pci_addr, &sample_objects);
    if (status != DOCA_SUCCESS) goto cleanup;
    status = doca_comch_cap_get_max_msg_size(doca_dev_as_devinfo(sample_objects.hw_dev), &max_msg_size);
    if (status != DOCA_SUCCESS) goto cleanup;

    while (!sample_objects.finish) {
        if (!atomic_flag_test_and_set_explicit(&pe_progress_spin, memory_order_acquire)) {
            doca_pe_progress(sample_objects.pe);
            atomic_flag_clear_explicit(&pe_progress_spin, memory_order_release);
        }
    }

    cw_buffer_pool_destroy(&sample_objects.collective_worker.send_pool);
    cw_buffer_pool_destroy(&sample_objects.collective_worker.recv_pool);

cleanup:
    clean_comch_sample_objects(&sample_objects);
    if (status != DOCA_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
mpi_fail:
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}
