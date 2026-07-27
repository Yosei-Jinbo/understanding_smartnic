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
#include "../common/rdma_common.h"
#include "../common/comch_mpi_common.h"
#include "../common/timing_utils.h"
#include "../common/doca_rdma_utils.h"
#include <arm_neon.h>
#include <sched.h>
#include <math.h>
#include "spsc_queue.h"

DOCA_LOG_REGISTER(COMCH_SERVER);

/* =====================================================
 * Phase 7 Step 0: 詳細計測インフラ
 * データは配列に蓄積し、全イテレーション後にまとめて出力。
 * ホットパスでの fprintf/printf は一切行わない。
 * ===================================================== */

/* 計測を無効化するには -DAG_TIMING_DISABLED を定義 */
#ifdef AG_TIMING_DISABLED
#define AG_TIMING_ENABLED 0
#else
#define AG_TIMING_ENABLED 1
#endif

#define AG_TIMING_MAX_OPS    20
#define AG_TIMING_MAX_ITERS  100

/* =====================================================
 * コア割り当て設定（1 DPU = 2 ランク, 計 16 コア想定）
 *   ワークロードの性質で配置方針を変える:
 *     - 通信 progress : busy-poll のため rank 分離（3 worker を C コアに packing）
 *     - 計算 RS 集約  : バースト的・メモリ律速のため rank 共有（動的バースト＋MLP）
 *   env で実行時変更可（再ビルド不要でスイープ）:
 *     COMM_CORES    : 通信 progress の rank あたりコア数 C (1..DOCA_WORKER_TYPE_COUNT, default 3)
 *     COMPUTE_CORES : 計算 RS 集約スレッド数 K（rank 共有, default 6）
 *   制約: 2*C + K + 4 <= 16（超過分は K を自動クランプ）
 *   レイアウト（自動計算）:
 *     comm[0..2C-1](rank別) | msg[2C,2C+1](rank別) | compute[2C+2 .. +K-1](共有) | main(rank別)
 * ===================================================== */
#define COMM_CORES_DEFAULT     3
#define COMPUTE_CORES_DEFAULT  6

static int g_comm_cores    = COMM_CORES_DEFAULT;
static int g_compute_cores = COMPUTE_CORES_DEFAULT;
/* CrossGVMI アブレーション用: 1 なら AG の GPU 直接アクセス(PCI cross-GVMI import)を無効化し、
 * host_dst_rmem/host_src_rmem 経由の RDMA staging 経路に落とす（gpu_direct を強制 off）。
 * RS には影響させない（RS は reduction のため常に DPU に取り込むので staging の概念が無い）。 */
static int g_force_staging = 0;

static void core_alloc_init_from_env(void)
{
    const char *c = getenv("COMM_CORES");
    const char *k = getenv("COMPUTE_CORES");
    const char *fs = getenv("FORCE_STAGING");
    if (c) { int v = atoi(c); if (v >= 1 && v <= (int)DOCA_WORKER_TYPE_COUNT) g_comm_cores = v; }
    if (k) { int v = atoi(k); if (v >= 1) g_compute_cores = v; }
    g_force_staging = (fs && atoi(fs) != 0) ? 1 : 0;
    int maxk = 16 - 4 - 2 * g_comm_cores;      /* 2*C + K + 4 <= 16 */
    if (g_compute_cores > maxk) g_compute_cores = maxk;
    if (g_compute_cores < 1)   g_compute_cores = 1;
    DOCA_LOG_INFO("core alloc: COMM_CORES=%d COMPUTE_CORES=%d (2*C+K+4=%d) FORCE_STAGING=%d",
                  g_comm_cores, g_compute_cores, 2 * g_comm_cores + g_compute_cores + 4, g_force_staging);
}

/* 3 progress worker を C コアに packing（rank 分離） */
static inline int comm_core(int rank, int i)  { return (rank % 2) * g_comm_cores + (i % g_comm_cores); }
static inline int msg_pool_core(int rank)     { return 2 * g_comm_cores + (rank % 2); }
static inline int compute_core_base(void)     { return 2 * g_comm_cores + 2; }
static inline int compute_core(int i)         { return compute_core_base() + i; }               /* 共有 */
static inline int main_core(int rank)         { return compute_core_base() + g_compute_cores + (rank % 2); }

enum ag_op_name {
    AG_OP_READ = 0,
    AG_OP_WRITE,
    AG_OP_RING_SENDRECV,
    AG_OP_RING_PUT,
    AG_OP_DOORBELL,
    AG_OP_BARRIER,
};

static const char *ag_op_name_str[] = {
    "Read", "Write", "Ring S+R", "Ring Put", "Doorbell", "Barrier"
};

struct ag_op_timing {
    uint64_t submit_ns;   /* submit API calls (buf alloc, task alloc, doca_task_submit) */
    uint64_t wait_ns;     /* PE progress loop until completion */
    int      name_idx;
};

struct ag_timing_record {
    struct ag_op_timing ops[AG_TIMING_MAX_OPS];
    int      num_ops;
    uint64_t total_ns;
    uint64_t buffer_len;
};

static struct {
    struct ag_timing_record records[AG_TIMING_MAX_ITERS];
    int      count;
    bool     active;
} g_ag_timing = {0};

static void print_ag_timing_summary(uint64_t rank, uint64_t buffer_len, uint64_t world_size)
{
    int n = g_ag_timing.count;
    if (n == 0) return;

    int max_ops = 0;
    for (int i = 0; i < n; i++)
        if (g_ag_timing.records[i].num_ops > max_ops)
            max_ops = g_ag_timing.records[i].num_ops;

    uint64_t N_per_rank = buffer_len / (2 * world_size);  /* fp16 → element count */

    fprintf(stderr, "\n[AG TIMING rank%lu N=%lu buflen=%lu] %d iterations:\n",
            rank, N_per_rank, buffer_len, n);

    for (int op = 0; op < max_ops; op++) {
        uint64_t sum_submit = 0, sum_wait = 0;
        uint64_t min_total = UINT64_MAX, max_total = 0;
        int count = 0;
        int name_idx = -1;
        double sum_sq = 0;
        uint64_t totals[AG_TIMING_MAX_ITERS];

        for (int i = 0; i < n; i++) {
            if (op < g_ag_timing.records[i].num_ops) {
                struct ag_op_timing *t = &g_ag_timing.records[i].ops[op];
                uint64_t total = t->submit_ns + t->wait_ns;
                sum_submit += t->submit_ns;
                sum_wait += t->wait_ns;
                if (total < min_total) min_total = total;
                if (total > max_total) max_total = total;
                name_idx = t->name_idx;
                totals[count] = total;
                count++;
            }
        }
        if (count == 0) continue;

        double avg_total = (double)(sum_submit + sum_wait) / count;
        for (int i = 0; i < count; i++) {
            double d = (double)totals[i] - avg_total;
            sum_sq += d * d;
        }
        double stddev = (count > 1) ? sqrt(sum_sq / (count - 1)) : 0;

        const char *name = (name_idx >= 0 && name_idx < 6) ? ag_op_name_str[name_idx] : "???";
        fprintf(stderr, "  Op%2d %-10s submit=%7.1fus wait=%7.1fus total=%7.1fus "
                "(min=%7.1f max=%7.1f sd=%6.1f)\n",
                op, name,
                (double)sum_submit / count / 1e3,
                (double)sum_wait / count / 1e3,
                avg_total / 1e3,
                (double)min_total / 1e3,
                (double)max_total / 1e3,
                stddev / 1e3);
    }

    /* Total summary */
    uint64_t sum_total = 0, min_total = UINT64_MAX, max_total = 0;
    double sum_sq = 0;
    for (int i = 0; i < n; i++) {
        uint64_t t = g_ag_timing.records[i].total_ns;
        sum_total += t;
        if (t < min_total) min_total = t;
        if (t > max_total) max_total = t;
    }
    double avg = (double)sum_total / n;
    for (int i = 0; i < n; i++) {
        double d = (double)g_ag_timing.records[i].total_ns - avg;
        sum_sq += d * d;
    }
    double stddev = (n > 1) ? sqrt(sum_sq / (n - 1)) : 0;

    fprintf(stderr, "  TOTAL: avg=%.1fus min=%.1fus max=%.1fus stddev=%.1fus (%.1f%%)\n\n",
            avg / 1e3, (double)min_total / 1e3, (double)max_total / 1e3,
            stddev / 1e3, (avg > 0) ? stddev / avg * 100.0 : 0);
}

/* =====================================================
 * Phase 5 Step 4: RDMA command slot structure
 * Host RDMA Writes this to DPU cmd_slot_buf.
 * seq is at offset 0; Host writes payload first, then seq (doorbell).
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
    /* Variable-length export data follows (concatenated in order:
       src_export, dst_export, src_export_rail1, dst_export_rail1,
       local_cpu_export, local_cpu_flag_export) */
    uint8_t export_data[];
};

/* =====================================================
 * DOCA RDMA コールバックデータ + コールバック関数
 * ===================================================== */

struct doca_task_cb_data {
    int stride_id;
    struct doca_buf *buf1;
    struct doca_buf *buf2;
    struct doca_worker_thread_ctx *worker;  /* NULL in inline mode */
    atomic_int *inline_pending;              /* inline mode: decrement on completion */
    /* Phase 12.2 H-3: 診断用情報 (RDMA op エラー時に出力) */
    uint32_t  op_type;        /* doca_rdma_task_type_t (READ/WRITE/SEND/RECV) */
    uint32_t  length;
    uint64_t  local_addr;
    uint64_t  remote_addr;
    void     *rdma_ctx_ptr;   /* どの RDMA context か (rail0/rail1 識別用) */
};

/* Phase 12.2 H-3: エラーログ throttling — 最初の N 件は full detail、以降はカウントのみ */
#define RDMA_ERR_LOG_DETAIL_MAX  10
static atomic_int g_rdma_err_recv_count;
static atomic_int g_rdma_err_send_count;
static atomic_int g_rdma_err_read_count;
static atomic_int g_rdma_err_write_count;

static const char *_op_type_name(uint32_t op_type)
{
    switch (op_type) {
    case DOCA_RDMA_TASK_READ:      return "READ";
    case DOCA_RDMA_TASK_WRITE:     return "WRITE";
    case DOCA_RDMA_TASK_RING_SEND: return "RING_SEND";
    case DOCA_RDMA_TASK_RING_RECV: return "RING_RECV";
    default:                       return "UNKNOWN";
    }
}

static inline void task_cb_finish(struct doca_task_cb_data *cb, completion_status_t status)
{
    int sid = cb->stride_id;
    if (cb->inline_pending) {
        atomic_fetch_sub(cb->inline_pending, 1);
    } else if (cb->worker) {
        spsc_completion_queue_t *cq = (spsc_completion_queue_t *)cb->worker->completion_queue;
        completion_event_t ev = { .status = status, .stride_id = sid };
        while (!spsc_completion_queue_push(cq, &ev)) cpu_relax();
    }
    free(cb);
}

/* Phase 12.2 H-3: 共通エラーログヘルパー */
static inline void _log_rdma_err(const char *op_kind,
                                  doca_error_t status,
                                  struct doca_task_cb_data *cb,
                                  atomic_int *counter)
{
    int n = atomic_fetch_add(counter, 1) + 1;
    if (n <= RDMA_ERR_LOG_DETAIL_MAX) {
        DOCA_LOG_ERR("RDMA %s task #%d failed: %s "
                     "[op=%s ctx=%p len=%u local=0x%lx remote=0x%lx stride=%d]",
                     op_kind, n, doca_error_get_descr(status),
                     _op_type_name(cb->op_type),
                     cb->rdma_ctx_ptr, cb->length,
                     cb->local_addr, cb->remote_addr, cb->stride_id);
    } else if (n == RDMA_ERR_LOG_DETAIL_MAX + 1) {
        DOCA_LOG_ERR("RDMA %s errors: throttling further messages, will print every 1000",
                     op_kind);
    } else if (n % 1000 == 0) {
        DOCA_LOG_ERR("RDMA %s errors: total=%d", op_kind, n);
    }
}

/* ---- Read 完了/エラー ---- */
static void rdma_read_comp_cb(struct doca_rdma_task_read *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_read_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

static void rdma_read_err_cb(struct doca_rdma_task_read *task,
                              union doca_data task_user_data,
                              union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    doca_error_t status = doca_task_get_status(doca_rdma_task_read_as_task(task));
    _log_rdma_err("read", status, cb, &g_rdma_err_read_count);
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_read_as_task(task));
    task_cb_finish(cb, COMP_ERROR);
}

/* ---- Write 完了/エラー ---- */
static void rdma_write_comp_cb(struct doca_rdma_task_write *task,
                                union doca_data task_user_data,
                                union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_write_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

static void rdma_write_err_cb(struct doca_rdma_task_write *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    doca_error_t status = doca_task_get_status(doca_rdma_task_write_as_task(task));
    _log_rdma_err("write", status, cb, &g_rdma_err_write_count);
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_write_as_task(task));
    task_cb_finish(cb, COMP_ERROR);
}

/* ---- Send 完了/エラー ---- */
static void rdma_send_comp_cb(struct doca_rdma_task_send *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_send_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

static void rdma_send_err_cb(struct doca_rdma_task_send *task,
                              union doca_data task_user_data,
                              union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    doca_error_t status = doca_task_get_status(doca_rdma_task_send_as_task(task));
    _log_rdma_err("send", status, cb, &g_rdma_err_send_count);
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_send_as_task(task));
    task_cb_finish(cb, COMP_ERROR);
}

/* ---- Receive 完了/エラー ---- */
static void rdma_recv_comp_cb(struct doca_rdma_task_receive *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_receive_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

static void rdma_recv_err_cb(struct doca_rdma_task_receive *task,
                              union doca_data task_user_data,
                              union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    doca_error_t status = doca_task_get_status(doca_rdma_task_receive_as_task(task));
    _log_rdma_err("recv", status, cb, &g_rdma_err_recv_count);
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_receive_as_task(task));
    task_cb_finish(cb, COMP_ERROR);
}

/* =====================================================
 * DOCA タスク提出 (desc → DOCA API 呼び出し)
 * ===================================================== */

static void submit_doca_task_from_desc(struct doca_rdma_ctx_t *rdma_ctx,
                                        const struct doca_task_desc *desc,
                                        struct doca_worker_thread_ctx *worker)
{
    doca_error_t ret;
    struct doca_buf *local_buf = NULL, *remote_buf = NULL;
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)calloc(1, sizeof(*cb));
    if (!cb) {
        if (desc->inline_pending) {
            atomic_fetch_sub(desc->inline_pending, 1);
        } else {
            completion_event_t ev = { .status = COMP_ERROR, .stride_id = desc->stride_id };
            spsc_completion_queue_push((spsc_completion_queue_t *)worker->completion_queue, &ev);
        }
        return;
    }
    cb->stride_id = desc->stride_id;
    if (desc->inline_pending) {
        cb->worker = NULL;
        cb->inline_pending = desc->inline_pending;
    } else {
        cb->worker = worker;
        cb->inline_pending = NULL;
    }
    /* Phase 12.2 H-3: 診断用情報をコピー */
    cb->op_type      = (uint32_t)desc->type;
    cb->length       = (uint32_t)desc->length;
    cb->local_addr   = (uint64_t)(uintptr_t)desc->local_addr;
    cb->remote_addr  = (uint64_t)(uintptr_t)desc->remote_addr;
    cb->rdma_ctx_ptr = (void *)rdma_ctx;

    union doca_data ud;
    ud.ptr = cb;

    /* Phase 6 Step F: PE spinlock protects buf/task alloc + submit + PE progress */
    struct pe_spin_t *spin = desc->pe_spin;
    if (spin) pe_spin_lock(spin);

    switch (desc->type) {
    case DOCA_RDMA_TASK_READ: {
        ret = doca_rdma_get_remote_buf(rdma_ctx, desc->remote_mem, desc->remote_addr, desc->length, &remote_buf);
        if (ret != DOCA_SUCCESS) goto fail;
        ret = doca_buf_set_data(remote_buf, desc->remote_addr, desc->length);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(remote_buf, NULL); goto fail; }

        /* Phase 9: local_mmap_override → Read の dst を GPU dst (PCI import mmap) に */
        if (desc->local_mmap_override) {
            ret = doca_buf_inventory_buf_get_by_addr(rdma_ctx->buf_inv, desc->local_mmap_override,
                                                      desc->local_addr, desc->length, &local_buf);
        } else {
            ret = doca_rdma_get_local_buf(rdma_ctx, desc->local_addr, desc->length, &local_buf);
        }
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(remote_buf, NULL); goto fail; }

        cb->buf1 = remote_buf;
        cb->buf2 = local_buf;

        struct doca_rdma_task_read *read_task;
        ret = doca_rdma_task_read_allocate_init(rdma_ctx->rdma, desc->connection, remote_buf, local_buf, ud, &read_task);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(remote_buf, NULL); doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        ret = doca_task_submit(doca_rdma_task_read_as_task(read_task));
        if (ret != DOCA_SUCCESS) { doca_task_free(doca_rdma_task_read_as_task(read_task)); doca_buf_dec_refcount(remote_buf, NULL); doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        break;
    }
    case DOCA_RDMA_TASK_WRITE: {
        ret = doca_rdma_get_local_buf(rdma_ctx, desc->local_addr, desc->length, &local_buf);
        if (ret != DOCA_SUCCESS) goto fail;
        ret = doca_buf_set_data(local_buf, desc->local_addr, desc->length);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); goto fail; }

        ret = doca_rdma_get_remote_buf(rdma_ctx, desc->remote_mem, desc->remote_addr, desc->length, &remote_buf);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); goto fail; }

        cb->buf1 = local_buf;
        cb->buf2 = remote_buf;

        struct doca_rdma_task_write *write_task;
        ret = doca_rdma_task_write_allocate_init(rdma_ctx->rdma, desc->connection, local_buf, remote_buf, ud, &write_task);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); doca_buf_dec_refcount(remote_buf, NULL); goto fail; }
        ret = doca_task_submit(doca_rdma_task_write_as_task(write_task));
        if (ret != DOCA_SUCCESS) { doca_task_free(doca_rdma_task_write_as_task(write_task)); doca_buf_dec_refcount(local_buf, NULL); doca_buf_dec_refcount(remote_buf, NULL); goto fail; }
        break;
    }
    case DOCA_RDMA_TASK_RING_SEND: {
        /* Phase 8: local_mmap_override → Cross-GVMI PCI import した GPU メモリを使用 */
        if (desc->local_mmap_override) {
            ret = doca_buf_inventory_buf_get_by_addr(rdma_ctx->buf_inv, desc->local_mmap_override,
                                                      desc->local_addr, desc->length, &local_buf);
        } else {
            ret = doca_rdma_get_local_buf(rdma_ctx, desc->local_addr, desc->length, &local_buf);
        }
        if (ret != DOCA_SUCCESS) goto fail;
        ret = doca_buf_set_data(local_buf, desc->local_addr, desc->length);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); goto fail; }

        cb->buf1 = local_buf;
        cb->buf2 = NULL;

        struct doca_rdma_task_send *send_task;
        ret = doca_rdma_task_send_allocate_init(rdma_ctx->rdma, desc->connection, local_buf, ud, &send_task);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        ret = doca_task_submit(doca_rdma_task_send_as_task(send_task));
        if (ret != DOCA_SUCCESS) { doca_task_free(doca_rdma_task_send_as_task(send_task)); doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        break;
    }
    case DOCA_RDMA_TASK_RING_RECV: {
        /* Phase 8: local_mmap_override → Cross-GVMI PCI import した GPU メモリを使用 */
        if (desc->local_mmap_override) {
            ret = doca_buf_inventory_buf_get_by_addr(rdma_ctx->buf_inv, desc->local_mmap_override,
                                                      desc->local_addr, desc->length, &local_buf);
        } else {
            ret = doca_rdma_get_local_buf(rdma_ctx, desc->local_addr, desc->length, &local_buf);
        }
        if (ret != DOCA_SUCCESS) goto fail;

        cb->buf1 = local_buf;
        cb->buf2 = NULL;

        struct doca_rdma_task_receive *recv_task;
        ret = doca_rdma_task_receive_allocate_init(rdma_ctx->rdma, local_buf, ud, &recv_task);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        ret = doca_task_submit(doca_rdma_task_receive_as_task(recv_task));
        if (ret != DOCA_SUCCESS) { doca_task_free(doca_rdma_task_receive_as_task(recv_task)); doca_buf_dec_refcount(local_buf, NULL); goto fail; }
        break;
    }
    default:
        goto fail;
    }
    if (spin) pe_spin_unlock(spin);
    return;

fail:
    if (spin) pe_spin_unlock(spin);
    /* Phase 12.2 H-3: 提出失敗の詳細ログ (throttled) */
    {
        static atomic_int g_submit_fail_count;
        int n = atomic_fetch_add(&g_submit_fail_count, 1) + 1;
        if (n <= RDMA_ERR_LOG_DETAIL_MAX) {
            DOCA_LOG_ERR("submit_doca_task_from_desc fail #%d: ret=%s "
                         "[op=%s ctx=%p len=%zu local=%p remote=%p stride=%d local_mmap_override=%p]",
                         n, doca_error_get_descr(ret),
                         _op_type_name((uint32_t)desc->type),
                         (void *)rdma_ctx, desc->length,
                         desc->local_addr, desc->remote_addr, desc->stride_id,
                         (void *)desc->local_mmap_override);
        } else if (n == RDMA_ERR_LOG_DETAIL_MAX + 1) {
            DOCA_LOG_ERR("submit_doca_task errors: throttling further");
        }
    }
    free(cb);
    if (desc->inline_pending) {
        atomic_fetch_sub(desc->inline_pending, 1);
    } else if (worker) {
        completion_event_t ev = { .status = COMP_ERROR, .stride_id = desc->stride_id };
        while (!spsc_completion_queue_push((spsc_completion_queue_t *)worker->completion_queue, &ev))
            cpu_relax();
    }
}

/* =====================================================
 * DOCA ワーカスレッド メインループ
 * ===================================================== */

static void *doca_worker_thread_main(void *arg)
{
    struct doca_worker_thread_ctx *ctx = (struct doca_worker_thread_ctx *)arg;
    spsc_doca_task_queue_t *tq = (spsc_doca_task_queue_t *)ctx->task_queue;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ctx->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (atomic_load(&ctx->running)) {
        bool did_work = false;
        struct doca_task_desc task;

        /* task_queue は EXIT 通知のみ（hot path のタスクは inline 提出） */
        while (spsc_doca_task_queue_pop(tq, &task)) {
            if (task.type == DOCA_RDMA_TASK_EXIT) {
                atomic_store(&ctx->running, false);
                break;
            }
        }

        /* 担当 PE を round-robin で progress（pe_spin で inline 実行と排他） */
        for (int j = 0; j < ctx->n_pe; j++) {
            struct doca_rdma_ctx_t *rc = ctx->pe_ctx[j];
            struct pe_spin_t *spin = ctx->pe_spins[j];
            if (!rc || !rc->pe) continue;
            if (spin) {
                if (try_pe_progress(rc->pe, spin)) did_work = true;
            } else {
                if (doca_pe_progress(rc->pe)) did_work = true;
            }
        }

        if (!did_work) {
            cpu_relax();
        }
    }
    return NULL;
}

/* =====================================================
 * DOCA ワーカスレッドプール
 * ===================================================== */

static int doca_worker_thread_pool_init(
    struct doca_worker_thread_pool_t *pool,
    struct collective_worker_t *cw,
    int mpi_rank)
{
    if (!pool || !cw) return -1;
    memset(pool, 0, sizeof(*pool));
    pool->mpi_rank = mpi_rank;
    atomic_init(&pool->initialized, false);

    struct doca_rdma_ctx_t *ctx_map[DOCA_WORKER_TYPE_COUNT] = {
        &cw->rdma_rma,
        &cw->rdma_ring_recv,
        &cw->rdma_ring_send,
    };

    /* Phase 6 Step F: PE spinlock 初期化 */
    pe_spin_init(&cw->rma_pe_spin);
    pe_spin_init(&cw->rma_pe_spin_rail1);
    pe_spin_init(&cw->ring_send_pe_spin);
    pe_spin_init(&cw->ring_send_pe_spin_rail1);
    pe_spin_init(&cw->ring_recv_pe_spin);
    pe_spin_init(&cw->ring_recv_pe_spin_rail1);

    /* Phase 6 Step F: PE spinlock マッピングテーブル */
    struct pe_spin_t *spin_map[DOCA_WORKER_TYPE_COUNT] = {
        &cw->rma_pe_spin,
        &cw->ring_recv_pe_spin,
        &cw->ring_send_pe_spin,
    };
    struct pe_spin_t *spin_rail1_map[DOCA_WORKER_TYPE_COUNT] = {
        cw->dual_rail ? &cw->rma_pe_spin_rail1 : NULL,
        cw->conn_from_prev_rail1 ? &cw->ring_recv_pe_spin_rail1 : NULL,
        cw->conn_to_next_rail1 ? &cw->ring_send_pe_spin_rail1 : NULL,
    };

    /* rail1 の context マップ（rail1 spinlock が非 NULL のときのみ有効） */
    struct doca_rdma_ctx_t *rail1_ctx_map[DOCA_WORKER_TYPE_COUNT] = {
        &cw->rdma_rma_rail1,
        &cw->rdma_ring_recv_rail1,
        &cw->rdma_ring_send_rail1,
    };

    /* Option 2: N = g_comm_cores 本のワーカ。役割 → ワーカを role % N で配分し、
     * 各ワーカが担当役割の rail0/rail1 PE を round-robin progress する。
     * role 順 [RMA(0), RECV(1), SEND(2)] なので N>=2 で RECV/SEND は別ワーカになる。 */
    int N = g_comm_cores;
    if (N < 1) N = 1;
    if (N > DOCA_WORKER_TYPE_COUNT) N = DOCA_WORKER_TYPE_COUNT;
    pool->n_workers = N;

    for (int w = 0; w < N; w++) {
        struct doca_worker_thread_ctx *wk = &pool->workers[w];
        wk->worker_type = (doca_worker_type_t)w;
        wk->core_id = comm_core(mpi_rank, w);
        wk->n_pe = 0;
    }
    for (int r = 0; r < DOCA_WORKER_TYPE_COUNT; r++) {
        struct doca_worker_thread_ctx *wk = &pool->workers[r % N];
        wk->pe_ctx[wk->n_pe]   = ctx_map[r];       /* rail0 */
        wk->pe_spins[wk->n_pe] = spin_map[r];
        wk->n_pe++;
        if (spin_rail1_map[r]) {                    /* rail1（条件成立時のみ） */
            wk->pe_ctx[wk->n_pe]   = rail1_ctx_map[r];
            wk->pe_spins[wk->n_pe] = spin_rail1_map[r];
            wk->n_pe++;
        }
    }

    for (int w = 0; w < N; w++) {
        struct doca_worker_thread_ctx *wk = &pool->workers[w];
        spsc_doca_task_queue_t *tq = (spsc_doca_task_queue_t *)calloc(1, sizeof(*tq));
        spsc_completion_queue_t *cq = (spsc_completion_queue_t *)calloc(1, sizeof(*cq));
        if (!tq || !cq) { free(tq); free(cq); return -1; }
        spsc_doca_task_queue_init(tq);
        spsc_completion_queue_init(cq);
        wk->task_queue = tq;
        wk->completion_queue = cq;
        atomic_init(&wk->running, true);
        atomic_init(&wk->paused, false);
        atomic_init(&wk->paused_ack, false);

        DOCA_LOG_INFO("comm worker %d: core=%d n_pe=%d", w, wk->core_id, wk->n_pe);
        if (pthread_create(&wk->thread, NULL, doca_worker_thread_main, wk) != 0) {
            atomic_store(&wk->running, false);
            free(tq); free(cq);
            return -1;
        }
    }
    atomic_store(&pool->initialized, true);
    return 0;
}

static void doca_worker_thread_pool_destroy(struct doca_worker_thread_pool_t *pool)
{
    if (!pool || !atomic_load(&pool->initialized)) return;

    for (int i = 0; i < pool->n_workers; i++) {
        struct doca_worker_thread_ctx *w = &pool->workers[i];
        spsc_doca_task_queue_t *tq = (spsc_doca_task_queue_t *)w->task_queue;
        if (tq) {
            struct doca_task_desc exit_task;
            memset(&exit_task, 0, sizeof(exit_task));
            exit_task.type = DOCA_RDMA_TASK_EXIT;
            spsc_doca_task_queue_push(tq, &exit_task);
        }
    }
    for (int i = 0; i < pool->n_workers; i++) {
        struct doca_worker_thread_ctx *w = &pool->workers[i];
        if (w->thread) pthread_join(w->thread, NULL);
        free(w->task_queue);
        free(w->completion_queue);
        w->task_queue = NULL;
        w->completion_queue = NULL;
    }
    atomic_store(&pool->initialized, false);
}

/* =====================================================
 * タスク提出・完了待機ヘルパー
 * ===================================================== */

static doca_worker_type_t doca_task_type_to_worker_type(doca_rdma_task_type_t t)
{
    switch (t) {
    case DOCA_RDMA_TASK_READ:
    case DOCA_RDMA_TASK_WRITE:
        return DOCA_WORKER_TYPE_RMA;
    case DOCA_RDMA_TASK_RING_SEND:
        return DOCA_WORKER_TYPE_RING_SEND;
    case DOCA_RDMA_TASK_RING_RECV:
        return DOCA_WORKER_TYPE_RING_RECV;
    default:
        return DOCA_WORKER_TYPE_RMA;
    }
}

static bool submit_doca_task_to_queue(
    struct doca_worker_thread_pool_t *pool,
    struct doca_task_desc *task)
{
    if (!pool || !atomic_load(&pool->initialized)) return false;
    doca_worker_type_t wt = doca_task_type_to_worker_type(task->type);
    struct doca_worker_thread_ctx *w = &pool->workers[wt];
    spsc_doca_task_queue_t *tq = (spsc_doca_task_queue_t *)w->task_queue;
    if (!tq) return false;
    return spsc_doca_task_queue_push(tq, task);
}

static void poll_wait_for_completion(
    struct doca_worker_thread_pool_t *pool,
    doca_worker_type_t worker_type,
    int expected_stride_id)
{
    if (!pool || !atomic_load(&pool->initialized)) return;
    struct doca_worker_thread_ctx *w = &pool->workers[worker_type];
    spsc_completion_queue_t *cq = (spsc_completion_queue_t *)w->completion_queue;
    if (!cq) return;

    completion_event_t event;
    uint64_t start_ns = now_monotonic_ns();
    while (1) {
        if (spsc_completion_queue_pop(cq, &event)) {
            if (event.stride_id == expected_stride_id) return;
        }
        uint64_t elapsed = now_monotonic_ns() - start_ns;
        if (elapsed < 200000) cpu_relax();
        else sched_yield();
    }
}

static void submit_and_wait_doca(
    struct doca_worker_thread_pool_t *pool,
    struct doca_task_desc *task)
{
    /* Phase 6 Step F: inline submit + PE progress from main thread */
    atomic_int pending;
    atomic_init(&pending, 1);
    task->inline_pending = &pending;

    struct doca_rdma_ctx_t *ctx = task->rdma_ctx;
    struct pe_spin_t *spin = task->pe_spin;

    /* Direct submit (bypass SPSC queue) */
    submit_doca_task_from_desc(ctx, task, NULL);

    /* Main thread PE progress to drive completion */
    while (atomic_load_explicit(&pending, memory_order_acquire) > 0) {
        if (spin && ctx && ctx->pe)
            try_pe_progress(ctx->pe, spin);
        else if (ctx && ctx->pe)
            doca_pe_progress(ctx->pe);
    }
}

/* =====================================================
 * 集約用スレッドプール (rs_thread_pool) — UNCHANGED
 * ===================================================== */

/* RS_NUM_THREADS: フォールバック用デフォルト。実際の計算スレッド数は
 * 実行時 g_compute_cores（env COMPUTE_CORES）を使用。ピン留めは compute_core(i)（rank 共有）。 */
#define RS_NUM_THREADS           COMPUTE_CORES_DEFAULT
#define RS_AGGREGATION_THRESHOLD (256 * 1024)

#define RS_JOB_OP_ADD   0

struct rs_add_job_t {
    fp16_t   *dst;
    fp16_t   *src;
    uint64_t start;
    uint64_t end;
    int      op;
};

struct rs_thread_pool_t {
    int                      num_threads;
    pthread_t               *threads;
    struct rs_worker_arg_t  *worker_args;
    atomic_bool              running;
    spsc_rs_task_queue_t    *task_queue;
    spsc_rs_completion_queue_t *completion_queue;
    pthread_mutex_t          barrier_mutex;
    pthread_cond_t           barrier_cond;
    struct rs_task           current_task;
    atomic_int               done_count;
    atomic_int               ready_count;
    volatile int             task_generation;  /* Phase 9: busy-poll 用に volatile */
};

struct rs_worker_arg_t {
    struct rs_thread_pool_t *pool;
    int                      tid;
};

static inline void
rs_add_range_neon(fp16_t *dst, fp16_t *src, uint64_t start, uint64_t end)
{
    (void)(end - start);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint64_t vec_len  = len & ~7ULL;
    uint64_t i        = start;
    uint64_t loop_end = start + vec_len;
    for (; i < loop_end; i += 8) {
        float16x8_t vec_a = vld1q_f16((const __fp16 *)&dst[i]);
        float16x8_t vec_b = vld1q_f16((const __fp16 *)&src[i]);
        float16x8_t vec_r = vaddq_f16(vec_a, vec_b);
        vst1q_f16((__fp16 *)&dst[i], vec_r);
    }
    for (; i < end; i++)
        dst[i] = (fp16_t)(dst[i] + src[i]);
#else
    for (uint64_t i = start; i < end; i++)
        dst[i] = (fp16_t)(dst[i] + src[i]);
#endif
}

static inline void __attribute__((unused))
rs_copy_range_neon(fp16_t *dst, fp16_t *src, uint64_t start, uint64_t end)
{
    uint64_t len = end - start;
    memcpy(&dst[start], &src[start], len * sizeof(fp16_t));
}

static void *
rs_worker_main(void *arg)
{
    struct rs_worker_arg_t  *wa   = (struct rs_worker_arg_t *)arg;
    struct rs_thread_pool_t *pool = wa->pool;
    const int tid = wa->tid;
    const int num_threads = pool->num_threads;
    int my_generation = 0;

    while (atomic_load(&pool->running)) {
        if (tid == 0) {
            struct rs_task task;
            pthread_mutex_lock(&pool->barrier_mutex);
            while (!spsc_rs_task_queue_pop(pool->task_queue, &task) && atomic_load(&pool->running))
                pthread_cond_wait(&pool->barrier_cond, &pool->barrier_mutex);
            if (!atomic_load(&pool->running)) { pthread_mutex_unlock(&pool->barrier_mutex); break; }
            pool->current_task = task;
            pool->task_generation++;
            atomic_store(&pool->done_count, 0);
            atomic_store(&pool->ready_count, 0);
            pthread_cond_broadcast(&pool->barrier_cond);
            pthread_mutex_unlock(&pool->barrier_mutex);
            my_generation = pool->task_generation;
        } else {
            pthread_mutex_lock(&pool->barrier_mutex);
            while (pool->task_generation == my_generation && atomic_load(&pool->running))
                pthread_cond_wait(&pool->barrier_cond, &pool->barrier_mutex);
            my_generation = pool->task_generation;
            pthread_mutex_unlock(&pool->barrier_mutex);
            if (!atomic_load(&pool->running)) break;
        }

        struct rs_task *task = &pool->current_task;
        uint64_t total = task->end - task->start;
        uint64_t base  = total / (uint64_t)num_threads;
        uint64_t rem   = total % (uint64_t)num_threads;
        uint64_t my_start = task->start + (uint64_t)tid * base + (uint64_t)((tid < (int)rem) ? tid : (int)rem);
        uint64_t my_len   = base + ((tid < (int)rem) ? 1ULL : 0ULL);
        uint64_t my_end   = my_start + my_len;

        if (my_len != 0)
            rs_add_range_neon(task->dst, task->src, my_start, my_end);

        int count = atomic_fetch_add(&pool->done_count, 1) + 1;
        if (count == num_threads) {
            completion_event_t event = { .status = COMP_SUCCESS, .stride_id = task->stride_id };
            uint64_t push_start_ns = now_monotonic_ns();
            while (!spsc_rs_completion_queue_push(pool->completion_queue, &event)) {
                uint64_t elapsed = now_monotonic_ns() - push_start_ns;
                if (elapsed < 200000) cpu_relax(); else sched_yield();
            }
        }
        atomic_fetch_add(&pool->ready_count, 1);
        if (tid == 0) {
            while (atomic_load(&pool->ready_count) < num_threads) cpu_relax();
        }
    }
    return NULL;
}

static doca_error_t
rs_thread_pool_create(struct rs_thread_pool_t **out_pool, int num_threads)
{
    if (num_threads <= 0) num_threads = RS_NUM_THREADS;
    struct rs_thread_pool_t *pool = (struct rs_thread_pool_t *)calloc(1, sizeof(*pool));
    if (!pool) return DOCA_ERROR_NO_MEMORY;

    pool->num_threads = num_threads;
    pool->threads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    pool->worker_args = (struct rs_worker_arg_t *)calloc(num_threads, sizeof(struct rs_worker_arg_t));
    pool->task_queue = (spsc_rs_task_queue_t *)calloc(1, sizeof(spsc_rs_task_queue_t));
    pool->completion_queue = (spsc_rs_completion_queue_t *)calloc(1, sizeof(spsc_rs_completion_queue_t));
    if (!pool->threads || !pool->worker_args || !pool->task_queue || !pool->completion_queue) {
        free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
        return DOCA_ERROR_NO_MEMORY;
    }
    spsc_rs_task_queue_init(pool->task_queue);
    spsc_rs_completion_queue_init(pool->completion_queue);
    pthread_mutex_init(&pool->barrier_mutex, NULL);
    pthread_cond_init(&pool->barrier_cond, NULL);
    pool->task_generation = 0;
    atomic_init(&pool->done_count, 0);
    atomic_init(&pool->ready_count, 0);
    atomic_init(&pool->running, true);

    for (int i = 0; i < num_threads; i++) {
        pool->worker_args[i].pool = pool;
        pool->worker_args[i].tid  = i;
        int rc = pthread_create(&pool->threads[i], NULL, rs_worker_main, &pool->worker_args[i]);
        if (rc != 0) {
            atomic_store(&pool->running, false);
            pthread_cond_broadcast(&pool->barrier_cond);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->barrier_mutex);
            pthread_cond_destroy(&pool->barrier_cond);
            free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
            return DOCA_ERROR_NO_MEMORY;
        }
        cpu_set_t cpuset; CPU_ZERO(&cpuset);
        CPU_SET(compute_core(i), &cpuset);
        pthread_setaffinity_np(pool->threads[i], sizeof(cpu_set_t), &cpuset);
    }
    *out_pool = pool;
    return DOCA_SUCCESS;
}

static void rs_thread_pool_destroy(struct rs_thread_pool_t *pool)
{
    if (!pool) return;
    pthread_mutex_lock(&pool->barrier_mutex);
    atomic_store(&pool->running, false);
    pthread_cond_broadcast(&pool->barrier_cond);
    pthread_mutex_unlock(&pool->barrier_mutex);
    for (int i = 0; i < pool->num_threads; i++) pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->barrier_mutex);
    pthread_cond_destroy(&pool->barrier_cond);
    free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
}

static inline void
submit_rs_task(struct rs_thread_pool_t *pool, fp16_t *dst, fp16_t *src, size_t start, size_t end, int stride_id)
{
    if (!pool || start >= end) return;
    struct rs_task task = { .dst = dst, .src = src, .start = start, .end = end, .stride_id = stride_id };
    while (!spsc_rs_task_queue_push(pool->task_queue, &task)) cpu_relax();
    pthread_mutex_lock(&pool->barrier_mutex);
    pthread_cond_broadcast(&pool->barrier_cond);
    pthread_mutex_unlock(&pool->barrier_mutex);
}

static inline void
poll_wait_rs_completion(struct rs_thread_pool_t *pool, int stride_id)
{
    if (!pool) return;
    completion_event_t event;
    uint64_t start_ns = now_monotonic_ns();
    while (1) {
        if (spsc_rs_completion_queue_pop(pool->completion_queue, &event))
            if (event.stride_id == stride_id) return;
        uint64_t elapsed = now_monotonic_ns() - start_ns;
        if (elapsed < 200000) cpu_relax(); else sched_yield();
    }
}

static void __attribute__((unused))
rs_thread_pool_add_and_wait(struct rs_thread_pool_t *pool, fp16_t *dst, fp16_t *src, uint64_t start, uint64_t end)
{
    static atomic_int rs_stride_counter = 0;
    int stride_id = atomic_fetch_add(&rs_stride_counter, 1);
    submit_rs_task(pool, dst, src, start, end, stride_id);
    poll_wait_rs_completion(pool, stride_id);
}

/* =====================================================
 * ComCh 関連の構造体・ユーティリティ
 * ===================================================== */

static pthread_mutex_t send_notify_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_flag pe_progress_spin = ATOMIC_FLAG_INIT;

struct msg_thread_pool;

static inline uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }
static doca_error_t init_comch_ctrl_path_server_objects(const char *server_name, const char *dev_pci_addr, const char *dev_rep_pci_addr, struct comch_ctrl_path_server_objects *sample_objects);
/* Phase 5 Step 4: forward declaration for RDMA cmd_slot fast path */
void execute_doca_collective_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects, uint64_t t_recv_ns, int ring_id);

/* Phase 15: Forward decls for per-ring queue helpers (defined later near message handlers) */
static int ring_queues_init(struct collective_worker_t *cw);
static int ring_proc_threads_start(struct comch_ctrl_path_server_objects *sample_objects);
static void ring_proc_threads_stop(struct collective_worker_t *cw);

struct comch_ctrl_path_server_objects {
    struct doca_dev *hw_dev;
    struct doca_dev_rep *rep_dev;
    struct doca_pe *pe;
    struct doca_comch_server *server;
    struct doca_comch_connection *connection;
    struct msg_thread_pool *msg_pool;
    uint32_t num_connected_clients;
    doca_error_t result;
    bool finish;
    const char *dev_pci_addr;
    const char *device_name;
    uint64_t rank;
    uint64_t world_size;
    struct send_notify_entry *send_notify_table;
    struct collective_worker_t collective_worker;
};

struct send_notify_entry {
    uint64_t id;
    atomic_bool finished;
    UT_hash_handle hh;
};

/* =====================================================
 * バッファプール (pthread_spinlock, 連続確保)
 * ===================================================== */

static void cw_buffer_pool_init(struct cw_buffer_pool *pool, void *base, size_t slot_size)
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

static int cw_buffer_pool_acquire(struct cw_buffer_pool *pool, size_t size, void **buf_out, size_t *capacity_out)
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

static void cw_buffer_pool_release(struct cw_buffer_pool *pool, int slot_idx)
{
    if (slot_idx < 0 || slot_idx >= CW_BUFFER_POOL_SLOTS) return;
    pthread_spin_lock(&pool->lock);
    pool->slots[slot_idx].in_use = 0;
    pthread_spin_unlock(&pool->lock);
}

static void cw_buffer_pool_destroy(struct cw_buffer_pool *pool)
{
    pthread_spin_destroy(&pool->lock);
}

/* =====================================================
 * メッセージワーカスレッドプール — UNCHANGED
 * ===================================================== */

#define MSG_POOL_THREADS  1
#define MSG_QUEUE_CAP     4096  /* 8 bytes/entry = 32KB. Phase 11 で inflight 数増加に対応 */

static void *message_worker_thread(void *arg);

struct msg_task { struct message_worker_arg *w; };

struct msg_thread_pool {
    pthread_t threads[MSG_POOL_THREADS];
    /* Lock-free SPSC ring buffer */
    struct msg_task q[MSG_QUEUE_CAP];
    _Alignas(64) atomic_uint head;
    _Alignas(64) atomic_uint tail;
    atomic_int stop;
    /* Phase 5 Step 4: back-pointer for RDMA cmd_slot polling */
    struct comch_ctrl_path_server_objects *sample_objects;
};

/* ============================================================
 * DPU Union Tracker (interval union で AG/RS/overlap を wall で測る)
 *
 * 目的: AG と RS が並列実行されても double-count せず、
 *       "pure AG wall busy" / "pure RS wall busy" / "overlap" を正しく取る。
 *
 * 方式: active counter 方式 (online interval union)。
 *   - ag_active/rs_active: 現在走っている AG/RS の個数 (並列 inflight)
 *   - ag_busy_start: ag_active が 0→1 になった時刻
 *   - both_start: (ag_active>0 AND rs_active>0) になった時刻
 *   - 各 total_*_ns: 累積 busy 時間 (interval union)
 *
 * Sum invariant: AG + RS − overlap = ANY (AG or RS active wall time)
 * ============================================================ */
static pthread_mutex_t g_union_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_union_ag_active = 0;
static int g_union_rs_active = 0;
static uint64_t g_union_ag_busy_start_ns = 0;
static uint64_t g_union_rs_busy_start_ns = 0;
static uint64_t g_union_both_start_ns = 0;
static uint64_t g_union_ag_busy_total_ns = 0;
static uint64_t g_union_rs_busy_total_ns = 0;
static uint64_t g_union_both_total_ns = 0;
static uint64_t g_union_ag_count = 0;
static uint64_t g_union_rs_count = 0;
static uint64_t g_union_max_ag_inflight = 0;
static uint64_t g_union_max_rs_inflight = 0;
static int g_union_printed = 0;  /* 重複出力防止 */

static inline uint64_t union_now_ns(void) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
}

static void dpu_union_tracker_enter(bool is_ag) {
    uint64_t t = union_now_ns();
    pthread_mutex_lock(&g_union_lock);

    bool was_both = (g_union_ag_active > 0) && (g_union_rs_active > 0);

    if (is_ag) {
        if (g_union_ag_active == 0) g_union_ag_busy_start_ns = t;
        g_union_ag_active++;
        g_union_ag_count++;
        if ((uint64_t)g_union_ag_active > g_union_max_ag_inflight)
            g_union_max_ag_inflight = (uint64_t)g_union_ag_active;
    } else {
        if (g_union_rs_active == 0) g_union_rs_busy_start_ns = t;
        g_union_rs_active++;
        g_union_rs_count++;
        if ((uint64_t)g_union_rs_active > g_union_max_rs_inflight)
            g_union_max_rs_inflight = (uint64_t)g_union_rs_active;
    }

    bool is_both_now = (g_union_ag_active > 0) && (g_union_rs_active > 0);
    if (!was_both && is_both_now) g_union_both_start_ns = t;

    pthread_mutex_unlock(&g_union_lock);
}

static void dpu_union_tracker_exit(bool is_ag) {
    uint64_t t = union_now_ns();
    pthread_mutex_lock(&g_union_lock);

    bool was_both = (g_union_ag_active > 0) && (g_union_rs_active > 0);

    if (is_ag) {
        g_union_ag_active--;
        if (g_union_ag_active == 0 && g_union_ag_busy_start_ns > 0) {
            g_union_ag_busy_total_ns += t - g_union_ag_busy_start_ns;
        }
    } else {
        g_union_rs_active--;
        if (g_union_rs_active == 0 && g_union_rs_busy_start_ns > 0) {
            g_union_rs_busy_total_ns += t - g_union_rs_busy_start_ns;
        }
    }

    bool is_both_now = (g_union_ag_active > 0) && (g_union_rs_active > 0);
    if (was_both && !is_both_now && g_union_both_start_ns > 0) {
        g_union_both_total_ns += t - g_union_both_start_ns;
    }

    pthread_mutex_unlock(&g_union_lock);
}

static void dpu_union_tracker_print(int rank) {
    pthread_mutex_lock(&g_union_lock);
    if (g_union_printed || rank != 0) {
        pthread_mutex_unlock(&g_union_lock);
        return;
    }
    g_union_printed = 1;

    uint64_t ag_ns  = g_union_ag_busy_total_ns;
    uint64_t rs_ns  = g_union_rs_busy_total_ns;
    uint64_t both_ns = g_union_both_total_ns;
    uint64_t ag_cnt = g_union_ag_count;
    uint64_t rs_cnt = g_union_rs_count;
    uint64_t max_ag = g_union_max_ag_inflight;
    uint64_t max_rs = g_union_max_rs_inflight;
    pthread_mutex_unlock(&g_union_lock);

    double ag_s   = ag_ns / 1e9;
    double rs_s   = rs_ns / 1e9;
    double both_s = both_ns / 1e9;
    double any_s  = ag_s + rs_s - both_s;
    double ag_only_s = ag_s - both_s;
    double rs_only_s = rs_s - both_s;

    printf("\n========== [DPU UNION TRACKER rank=%d] (interval union / parallelism-aware) ==========\n"
           "  AG  busy (union): %8.3f s | ops=%lu | max inflight=%lu\n"
           "  RS  busy (union): %8.3f s | ops=%lu | max inflight=%lu\n"
           "  Both  (overlap) : %8.3f s  (AG and RS simultaneously active)\n"
           "  AG only         : %8.3f s  (AG active, RS idle)\n"
           "  RS only         : %8.3f s  (RS active, AG idle)\n"
           "  Any comm (union): %8.3f s  (= AG + RS − overlap)\n"
           "==========================================================================\n",
           rank,
           ag_s, ag_cnt, max_ag,
           rs_s, rs_cnt, max_rs,
           both_s, ag_only_s, rs_only_s, any_s);
    fflush(stdout);
}

/* ============================================================
 * Phase 12 Calc C: per-op DPU timing record (for outlier analysis)
 *
 * 各 collective 操作の (size, setup_us, barrier_us, ring_us, notify_us)
 * を全件記録し、定期的に top-N を出力する。
 * 32MB AG が 200ms+ かかる原因が setup / barrier / ring / notify のどこに
 * あるかを切り分けるための診断用計測。
 *
 * - 1 process あたり 1 つの msg_pool_worker からのみ書き込まれるため、
 *   ロック不要 (atomic_int は念のため)。
 * - リングではなく溢れたら捨てる (容量 12000 = 1 epoch 程度を想定)。
 * - 出力は count が 10000 のときに 1 回 + 以降 10000 op ごと、rank 0 のみ。
 * ============================================================ */
#define DPU_PER_OP_TIMING_MAX  12000

struct dpu_per_op_record {
    uint64_t id;
    uint32_t size_kb;
    uint32_t setup_us;     /* t_pre_pause_ns - t_entry_ns (PCI import + rmem + bufpool) */
    uint32_t barrier_us;   /* MPI_Barrier inside collective_*() */
    uint32_t ring_us;      /* (t_post_collective_ns - t_post_pause_ns) - barrier_us */
    uint32_t notify_us;    /* t_done_ns - t_post_collective_ns (doorbell + cleanup) */
    /* Phase 16: ring 内部 breakdown (AG only; RS は 0) */
    uint32_t recv_post_us;        /* 初期 Recv bulk post 時間 */
    uint32_t step_submit_us;      /* 全 Ring step の Send submit 合計 */
    uint32_t step_wait_us;        /* 全 Ring step の Wait 合計 */
    uint16_t num_pieces;          /* piece 数 */
    uint16_t num_steps;           /* total_steps */
    /* Phase 17: training-specific overhead 原因切り分け用 */
    uint32_t queue_wait_us;       /* t_entry_ns - t_recv_ns (msg_pool dispatch latency) */
    uint16_t inflight_at_entry;   /* execute_doca_collective_cmd 入口での同時実行 AG/RS 数 */
    uint8_t  is_ag;               /* 1=AG, 0=RS */
    uint8_t  pad[1];
};

static struct dpu_per_op_record g_dpu_per_op[DPU_PER_OP_TIMING_MAX];
static atomic_int g_dpu_per_op_count;

/* Phase 17: DPU 上で実行中の collective (AG/RS) 数。
 *   execute_doca_collective_cmd の入口で inc、出口で dec。
 *   仮説: per-AG 時間が inflight 数と正相関すれば directly
 *        "concurrent AG が直列化されて互いに待っている" ことの証拠になる。 */
static atomic_int g_dpu_ag_inflight;

/* collective_*() が MPI_Barrier 経過時間を書き込むためのスレッドローカル */
static __thread uint64_t g_per_op_barrier_ns;
/* Phase 16: collective_all_gather 内部の breakdown を per-op で伝えるための TLS */
static __thread uint64_t g_per_op_recv_post_ns;
static __thread uint64_t g_per_op_step_submit_ns;
static __thread uint64_t g_per_op_step_wait_ns;
static __thread uint16_t g_per_op_num_pieces;
static __thread uint16_t g_per_op_num_steps;

/* ============================================================
 * Phase 13-A: per-rank arrival time → rank skew measurement
 *
 * 同じ AG (同じ id) について、4 ranks の barrier_ns の MAX = その AG の rank skew。
 * 「一番早く着いた rank が最も長く待つ」原理を利用。
 *
 * 各 rank は (id, barrier_us) を全 AG 分ローカル記録し、10000 op 達成時に
 * MPI_Gather で rank 0 に集約。rank 0 が id ごとの max barrier を計算して
 * 統計と top-N を出力する。
 *
 * 既存の per-op 計測 (g_dpu_per_op) とは独立に動く。
 * - g_dpu_per_op: rank 0 のみ、setup/barrier/ring/notify の breakdown
 * - g_phase13a:   全 rank、(id, barrier_us) のみ、MPI gather 対応
 * ============================================================ */
#define PHASE13A_MAX_OPS 12000

struct phase13a_record {
    uint32_t id;
    uint32_t barrier_us;
};

static struct phase13a_record g_phase13a_records[PHASE13A_MAX_OPS];
static atomic_int g_phase13a_count;
static int g_phase13a_gathered = 0;  /* 0 = まだ、1 = 集計済み (1 回だけ実行) */

/* ============================================================
 * Phase 12.2 G: Runtime-tunable AG piece configuration
 *
 * 環境変数:
 *   AG_PIECE_MAX     - chunk あたりの最大 piece 数 (default: 8, hard max: 32)
 *   AG_PIECE_TARGET  - piece size の目標バイト数 (default: 8 MB)
 *
 * num_pieces 計算:
 *   target = AG_PIECE_TARGET
 *   max_n  = AG_PIECE_MAX
 *   num_pieces = clamp(ceil(chunk_size / target), 1, max_n)
 *
 * 例 (max=16, target=8MB):
 *   chunk=12.25MB → 2 pieces of 6.1MB
 *   chunk=49MB    → 7 pieces of 7MB → cap 8 で 8 pieces of 6.1MB
 *   chunk=147MB   → 19 pieces → cap 16 で 16 pieces of 9.2MB
 * ============================================================ */
static int  g_ag_piece_max    = AG_PIECE_MAX_COUNT_DEFAULT;
static uint64_t g_ag_piece_target_bytes = AG_PIECE_TARGET_BYTES;
static int  g_ag_piece_inited = 0;

static void ag_piece_config_init_once(void)
{
    if (g_ag_piece_inited) return;
    g_ag_piece_inited = 1;

    const char *env_max = getenv("AG_PIECE_MAX");
    if (env_max) {
        int v = atoi(env_max);
        if (v >= 1 && v <= AG_PIECE_HARD_MAX) {
            g_ag_piece_max = v;
        } else {
            DOCA_LOG_WARN("AG_PIECE_MAX=%s out of range [1,%d], using default %d",
                          env_max, AG_PIECE_HARD_MAX, g_ag_piece_max);
        }
    }

    const char *env_target = getenv("AG_PIECE_TARGET");
    if (env_target) {
        long long v = atoll(env_target);
        if (v >= (long long)AG_PIECE_MIN_SIZE && v <= (long long)(1024 * 1024 * 1024)) {
            g_ag_piece_target_bytes = (uint64_t)v;
        } else {
            DOCA_LOG_WARN("AG_PIECE_TARGET=%s out of range, using default %lu",
                          env_target, g_ag_piece_target_bytes);
        }
    }

    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        printf("[AG PIECE CONFIG rank0] max=%d target=%lu bytes (%.1f MB)\n",
               g_ag_piece_max, g_ag_piece_target_bytes,
               (double)g_ag_piece_target_bytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

/* chunk_size から num_pieces を計算 (size-adaptive) */
static inline int ag_compute_num_pieces(uint64_t chunk_size)
{
    /* target に近い piece 数を求める */
    uint64_t n = (chunk_size + g_ag_piece_target_bytes - 1) / g_ag_piece_target_bytes;
    if (n < 1) n = 1;
    if (n > (uint64_t)g_ag_piece_max) n = (uint64_t)g_ag_piece_max;
    /* MIN_SIZE 制約: piece が小さすぎる場合は piece 数を減らす */
    while (n > 1 && (chunk_size / n) < AG_PIECE_MIN_SIZE) {
        n--;
    }
    return (int)n;
}

/* ============================================================
 * Phase 13-A: rank skew gather + summary
 * ============================================================ */

struct phase13a_skew_entry {
    uint32_t id;
    uint32_t max_barrier_us;       /* max across ranks for this id */
    uint32_t min_barrier_us;       /* min across ranks */
    uint32_t skew_us;              /* max - min */
    int n_ranks;                   /* how many ranks reported this id */
};

static int _cmp_skew_by_skew_desc(const void *a, const void *b)
{
    const struct phase13a_skew_entry *ea = (const struct phase13a_skew_entry *)a;
    const struct phase13a_skew_entry *eb = (const struct phase13a_skew_entry *)b;
    if (ea->skew_us > eb->skew_us) return -1;
    if (ea->skew_us < eb->skew_us) return  1;
    return 0;
}

/* MPI Gather + 集計 + 出力 (各 rank で 1 回だけ呼ぶ collective 関数) */
static void phase13a_gather_and_print(void)
{
    if (g_phase13a_gathered) return;
    g_phase13a_gathered = 1;

    int my_rank = -1, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_n = atomic_load(&g_phase13a_count);
    if (local_n > PHASE13A_MAX_OPS) local_n = PHASE13A_MAX_OPS;

    /* 全 rank で同じサイズに揃える: 各 rank の local_n の min を取る */
    int common_n = 0;
    MPI_Allreduce(&local_n, &common_n, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (common_n <= 0) return;

    /* 各 rank が common_n 個のレコードを送信。
     * MPI_Gather で rank 0 に world_size × common_n × sizeof(record) 集約 */
    struct phase13a_record *all_records = NULL;
    if (my_rank == 0) {
        all_records = (struct phase13a_record *)malloc(
            sizeof(struct phase13a_record) * (size_t)common_n * (size_t)world_size);
        if (!all_records) {
            DOCA_LOG_WARN("phase13a: malloc failed for gather buffer");
            return;
        }
    }

    /* 各 rank の最初の common_n 個を送る */
    MPI_Gather(g_phase13a_records,
               common_n * (int)sizeof(struct phase13a_record), MPI_BYTE,
               all_records,
               common_n * (int)sizeof(struct phase13a_record), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (my_rank != 0) {
        /* 非 rank 0 はここで終了 */
        return;
    }

    /* rank 0 集計: 同じスロット index は同じ id (host が同じ順序で送るため)
     * 各スロット index について 4 ranks の barrier_us から max/min を取る。
     * id ミスマッチがあれば skip (順序不一致は無視)。 */
    struct phase13a_skew_entry *entries = (struct phase13a_skew_entry *)
        calloc((size_t)common_n, sizeof(struct phase13a_skew_entry));
    if (!entries) {
        free(all_records);
        DOCA_LOG_WARN("phase13a: malloc failed for entries");
        return;
    }

    /* 注意: 各 rank が同じ順序で AG を処理しているはず (host が同じ id 列を送るため)
     * → all_records[r * common_n + i] と all_records[r' * common_n + i] は同じ id のはず
     * 念のため id を確認しつつマージする */
    int n_entries = 0;
    for (int i = 0; i < common_n; i++) {
        uint32_t id_ref = all_records[0 * common_n + i].id;
        uint32_t max_b = 0;
        uint32_t min_b = UINT32_MAX;
        bool consistent = true;
        for (int r = 0; r < world_size; r++) {
            struct phase13a_record *rec = &all_records[r * common_n + i];
            if (rec->id != id_ref) {
                consistent = false;
            }
            if (rec->barrier_us > max_b) max_b = rec->barrier_us;
            if (rec->barrier_us < min_b) min_b = rec->barrier_us;
        }
        if (!consistent) {
            /* id ミスマッチ — 順序が崩れている、スキップ */
            continue;
        }
        entries[n_entries].id = id_ref;
        entries[n_entries].max_barrier_us = max_b;
        entries[n_entries].min_barrier_us = min_b;
        entries[n_entries].skew_us = max_b - min_b;
        entries[n_entries].n_ranks = world_size;
        n_entries++;
    }

    /* 統計計算 */
    uint64_t total_skew_us = 0;
    uint64_t total_max_barrier_us = 0;
    uint32_t max_skew_us = 0;
    uint32_t max_id = 0;
    int n_skew_gt_100us = 0;
    int n_skew_gt_1ms   = 0;
    int n_skew_gt_10ms  = 0;
    int n_skew_gt_100ms = 0;
    for (int i = 0; i < n_entries; i++) {
        uint32_t s = entries[i].skew_us;
        total_skew_us += s;
        total_max_barrier_us += entries[i].max_barrier_us;
        if (s > max_skew_us) { max_skew_us = s; max_id = entries[i].id; }
        if (s > 100)    n_skew_gt_100us++;
        if (s > 1000)   n_skew_gt_1ms++;
        if (s > 10000)  n_skew_gt_10ms++;
        if (s > 100000) n_skew_gt_100ms++;
    }

    printf("\n========== [Phase 13-A: Rank Skew Summary] ==========\n");
    printf("World size:        %d\n", world_size);
    printf("AGs gathered:      %d (consistent ids across ranks)\n", n_entries);
    printf("Total rank skew:   %.3f s (= sum of (max-min) barrier across ranks per AG)\n",
           (double)total_skew_us / 1e6);
    printf("Total max barrier: %.3f s (= sum of max-barrier per AG, upper bound on skew impact)\n",
           (double)total_max_barrier_us / 1e6);
    if (n_entries > 0) {
        printf("Avg skew per AG:   %.1f us\n", (double)total_skew_us / n_entries);
        printf("Max skew (1 AG):   %u us = %.2f ms (id=%u)\n",
               max_skew_us, max_skew_us / 1e3, max_id);
    }
    printf("Skew distribution:\n");
    printf("  > 100us:   %d (%.1f%%)\n", n_skew_gt_100us, n_entries > 0 ? 100.0 * n_skew_gt_100us / n_entries : 0.0);
    printf("  > 1ms:     %d (%.1f%%)\n", n_skew_gt_1ms,   n_entries > 0 ? 100.0 * n_skew_gt_1ms   / n_entries : 0.0);
    printf("  > 10ms:    %d (%.1f%%)\n", n_skew_gt_10ms,  n_entries > 0 ? 100.0 * n_skew_gt_10ms  / n_entries : 0.0);
    printf("  > 100ms:   %d (%.1f%%)\n", n_skew_gt_100ms, n_entries > 0 ? 100.0 * n_skew_gt_100ms / n_entries : 0.0);

    /* Top 20 skew */
    qsort(entries, (size_t)n_entries, sizeof(struct phase13a_skew_entry), _cmp_skew_by_skew_desc);
    int top = n_entries < 20 ? n_entries : 20;
    printf("Top %d AGs by rank skew:\n", top);
    printf("  %-8s  %-12s  %-12s  %-12s\n", "id", "skew(ms)", "max_bar(ms)", "min_bar(ms)");
    for (int i = 0; i < top; i++) {
        printf("  %-8u  %-12.3f  %-12.3f  %-12.3f\n",
               entries[i].id,
               entries[i].skew_us / 1e3,
               entries[i].max_barrier_us / 1e3,
               entries[i].min_barrier_us / 1e3);
    }
    printf("=====================================================\n");
    fflush(stdout);

    free(entries);
    free(all_records);
}

static int _cmp_per_op_by_ring_us(const void *a, const void *b)
{
    const struct dpu_per_op_record *ra = (const struct dpu_per_op_record *)a;
    const struct dpu_per_op_record *rb = (const struct dpu_per_op_record *)b;
    if (ra->ring_us > rb->ring_us) return -1;
    if (ra->ring_us < rb->ring_us) return  1;
    return 0;
}

static int _cmp_per_op_by_barrier_us(const void *a, const void *b)
{
    const struct dpu_per_op_record *ra = (const struct dpu_per_op_record *)a;
    const struct dpu_per_op_record *rb = (const struct dpu_per_op_record *)b;
    if (ra->barrier_us > rb->barrier_us) return -1;
    if (ra->barrier_us < rb->barrier_us) return  1;
    return 0;
}

static int _cmp_per_op_by_setup_us(const void *a, const void *b)
{
    const struct dpu_per_op_record *ra = (const struct dpu_per_op_record *)a;
    const struct dpu_per_op_record *rb = (const struct dpu_per_op_record *)b;
    if (ra->setup_us > rb->setup_us) return -1;
    if (ra->setup_us < rb->setup_us) return  1;
    return 0;
}

static void dpu_per_op_print_summary(int rank, int count)
{
    if (rank != 0) return;
    int n = count;
    if (n > DPU_PER_OP_TIMING_MAX) n = DPU_PER_OP_TIMING_MAX;
    if (n <= 0) return;

    /* スナップショットを取って sort (元配列は msg_pool_worker が書き続ける) */
    struct dpu_per_op_record *snap = (struct dpu_per_op_record *)
        malloc(sizeof(struct dpu_per_op_record) * (size_t)n);
    if (!snap) return;
    memcpy(snap, g_dpu_per_op, sizeof(struct dpu_per_op_record) * (size_t)n);

    /* 基本統計 */
    uint64_t sum_setup = 0, sum_bar = 0, sum_ring = 0, sum_notify = 0;
    uint32_t max_setup = 0, max_bar = 0, max_ring = 0, max_notify = 0;
    int ag_count = 0, rs_count = 0;
    for (int i = 0; i < n; i++) {
        sum_setup += snap[i].setup_us;
        sum_bar   += snap[i].barrier_us;
        sum_ring  += snap[i].ring_us;
        sum_notify+= snap[i].notify_us;
        if (snap[i].setup_us  > max_setup)  max_setup  = snap[i].setup_us;
        if (snap[i].barrier_us> max_bar)    max_bar    = snap[i].barrier_us;
        if (snap[i].ring_us   > max_ring)   max_ring   = snap[i].ring_us;
        if (snap[i].notify_us > max_notify) max_notify = snap[i].notify_us;
        if (snap[i].is_ag) ag_count++; else rs_count++;
    }

    printf("\n========== [DPU PER-OP rank=0] ops=%d (AG=%d RS=%d) ==========\n",
           n, ag_count, rs_count);
    printf("  avg(us): setup=%.1f barrier=%.1f ring=%.1f notify=%.1f\n",
           (double)sum_setup / n, (double)sum_bar / n,
           (double)sum_ring / n, (double)sum_notify / n);
    printf("  max(us): setup=%u barrier=%u ring=%u notify=%u\n",
           max_setup, max_bar, max_ring, max_notify);

    /* ================================================================
     * Phase 16: Ring 内部 breakdown + per-size bucket 集計 (AG only)
     *
     * ring_us の内訳が recv_post + step_submit + step_wait のどれに偏って
     * いるかを見て、次の攻撃対象を決める。
     *
     * bucket: 1-16KB (small, 主にバイアス), 1-16MB (medium), >16MB (large)
     * ================================================================ */
    {
        /* per-bucket aggregator */
        struct bucket_stats {
            const char *name;
            uint64_t lo_kb, hi_kb;  /* 包括範囲 [lo, hi] */
            int      count;
            uint64_t sum_setup, sum_barrier, sum_ring, sum_notify;
            uint64_t sum_recv_post, sum_step_submit, sum_step_wait;
            uint64_t sum_pieces, sum_steps;
            uint32_t max_step_wait;
        } buckets[4] = {
            { "  0-1KB", 0,        1,        0, 0,0,0,0, 0,0,0, 0,0, 0 },
            { "1-16KB",  1,        16,       0, 0,0,0,0, 0,0,0, 0,0, 0 },
            { "1-16MB",  1024,     16*1024,  0, 0,0,0,0, 0,0,0, 0,0, 0 },
            { ">16MB",   16*1024+1, UINT64_MAX, 0, 0,0,0,0, 0,0,0, 0,0, 0 },
        };
        for (int i = 0; i < n; i++) {
            if (!snap[i].is_ag) continue;  /* AG のみ集計 */
            uint32_t kb = snap[i].size_kb;
            int bidx = -1;
            if (kb <= 1) bidx = 0;
            else if (kb <= 16) bidx = 1;
            else if (kb <= 16*1024) bidx = 2;
            else bidx = 3;
            struct bucket_stats *b = &buckets[bidx];
            b->count++;
            b->sum_setup       += snap[i].setup_us;
            b->sum_barrier     += snap[i].barrier_us;
            b->sum_ring        += snap[i].ring_us;
            b->sum_notify      += snap[i].notify_us;
            b->sum_recv_post   += snap[i].recv_post_us;
            b->sum_step_submit += snap[i].step_submit_us;
            b->sum_step_wait   += snap[i].step_wait_us;
            b->sum_pieces      += snap[i].num_pieces;
            b->sum_steps       += snap[i].num_steps;
            if (snap[i].step_wait_us > b->max_step_wait) b->max_step_wait = snap[i].step_wait_us;
        }

        printf("\n  ---- Phase 16: AG Ring breakdown per size bucket ----\n");
        printf("  bucket    count    setup  barrier recv_post step_sub step_wait  notify    ring  pcs  stp  max_wait\n");
        for (int bi = 0; bi < 4; bi++) {
            struct bucket_stats *b = &buckets[bi];
            if (b->count == 0) continue;
            double c = (double)b->count;
            printf("  %-7s %6d  %7.1f  %7.1f   %7.1f  %7.1f   %7.1f  %6.1f  %6.1f %4.1f %4.1f  %7u\n",
                   b->name, b->count,
                   b->sum_setup / c,
                   b->sum_barrier / c,
                   b->sum_recv_post / c,
                   b->sum_step_submit / c,
                   b->sum_step_wait / c,
                   b->sum_notify / c,
                   b->sum_ring / c,
                   (double)b->sum_pieces / c,
                   (double)b->sum_steps / c,
                   b->max_step_wait);
        }
        printf("  (all columns in us; pcs=pieces/op, stp=ring steps/op, max_wait=worst step_wait)\n");

        /* Ring 内部の占有率 (全 AG 総和) */
        uint64_t tot_ring = 0, tot_rp = 0, tot_ss = 0, tot_sw = 0;
        for (int bi = 0; bi < 4; bi++) {
            tot_ring += buckets[bi].sum_ring;
            tot_rp   += buckets[bi].sum_recv_post;
            tot_ss   += buckets[bi].sum_step_submit;
            tot_sw   += buckets[bi].sum_step_wait;
        }
        if (tot_ring > 0) {
            printf("  Ring 内訳 (AG total): recv_post=%.1f%%  step_submit=%.1f%%  step_wait=%.1f%%  (rest=%.1f%%)\n",
                   100.0 * tot_rp / tot_ring,
                   100.0 * tot_ss / tot_ring,
                   100.0 * tot_sw / tot_ring,
                   100.0 * (double)((int64_t)tot_ring - (int64_t)(tot_rp + tot_ss + tot_sw)) / tot_ring);
        }
    }

    /* ================================================================
     * Phase 17: queue_wait + inflight 分析
     *   仮説 H1 (queue_wait): t_recv → t_entry が ms 級なら DPU msg_pool
     *                          dispatcher 遅延が bottleneck
     *   仮説 H2 (inflight):   inflight=1 の per-AG 時間が micro-bench と一致し、
     *                          inflight≥2 で遅くなれば concurrent AG の serialization
     * ================================================================ */
    {
        /* queue_wait 分布 (AG/RS 両方、全 op) */
        uint64_t qw_sum = 0;
        uint32_t qw_max = 0;
        int qw_gt_1ms = 0, qw_gt_5ms = 0, qw_gt_10ms = 0;
        for (int i = 0; i < n; i++) {
            qw_sum += snap[i].queue_wait_us;
            if (snap[i].queue_wait_us > qw_max) qw_max = snap[i].queue_wait_us;
            if (snap[i].queue_wait_us > 1000)  qw_gt_1ms++;
            if (snap[i].queue_wait_us > 5000)  qw_gt_5ms++;
            if (snap[i].queue_wait_us > 10000) qw_gt_10ms++;
        }
        printf("\n  ---- Phase 17: queue_wait (t_recv → t_entry) distribution ----\n");
        printf("  avg=%.1fus  max=%uus  >1ms:%d  >5ms:%d  >10ms:%d\n",
               (double)qw_sum / n, qw_max, qw_gt_1ms, qw_gt_5ms, qw_gt_10ms);
        printf("  → if avg > 1000us: H1 confirmed (DPU msg_pool dispatch slow)\n");

        /* inflight グループ別の per-AG 統計 (AG のみ) */
        printf("\n  ---- Phase 17: per-AG time grouped by inflight_at_entry ----\n");
        struct infl_bucket {
            const char *name;
            int lo, hi;
            int count;
            uint64_t sum_ring, sum_barrier, sum_step_wait, sum_setup;
            uint32_t max_ring;
        } infl[5] = {
            { "infl=1",   1, 1, 0, 0,0,0,0, 0 },
            { "infl=2",   2, 2, 0, 0,0,0,0, 0 },
            { "infl=3",   3, 3, 0, 0,0,0,0, 0 },
            { "infl=4-8", 4, 8, 0, 0,0,0,0, 0 },
            { "infl>=9",  9, 65535, 0, 0,0,0,0, 0 },
        };
        for (int i = 0; i < n; i++) {
            if (!snap[i].is_ag) continue;
            int ii = snap[i].inflight_at_entry;
            int idx = -1;
            for (int bi = 0; bi < 5; bi++) {
                if (ii >= infl[bi].lo && ii <= infl[bi].hi) { idx = bi; break; }
            }
            if (idx < 0) continue;
            struct infl_bucket *b = &infl[idx];
            b->count++;
            b->sum_ring      += snap[i].ring_us;
            b->sum_barrier   += snap[i].barrier_us;
            b->sum_step_wait += snap[i].step_wait_us;
            b->sum_setup     += snap[i].setup_us;
            if (snap[i].ring_us > b->max_ring) b->max_ring = snap[i].ring_us;
        }
        printf("  bucket   count    ring(avg)  barrier(avg)  step_wait(avg)  setup(avg)   ring(max)\n");
        for (int bi = 0; bi < 5; bi++) {
            struct infl_bucket *b = &infl[bi];
            if (b->count == 0) continue;
            double c = (double)b->count;
            printf("  %-8s %5d   %7.1fus  %7.1fus   %7.1fus     %5.1fus   %7uus\n",
                   b->name, b->count,
                   b->sum_ring / c,
                   b->sum_barrier / c,
                   b->sum_step_wait / c,
                   b->sum_setup / c,
                   b->max_ring);
        }
        printf("  → if infl=1 ring is much smaller than infl>=2: H2 confirmed (concurrent AG serialization)\n");
        printf("  → if infl=1 ring ≈ micro-bench expected: training overhead is inflight-driven\n");
        printf("  → if infl=1 ring still large: overhead is NOT inflight-related (check H1, H3)\n");
    }

    /* Top 20 by ring_us */
    qsort(snap, (size_t)n, sizeof(struct dpu_per_op_record), _cmp_per_op_by_ring_us);
    printf("  Top 20 by ring_us (Phase 16 breakdown: recv_post | step_sub | step_wait):\n");
    int top = n < 20 ? n : 20;
    for (int i = 0; i < top; i++) {
        printf("    [%c] id=%-6lu size=%7uKB pcs=%u stp=%u | setup=%5.2f bar=%6.2f ring=%7.2f (rp=%5.2f ss=%5.2f sw=%7.2f) not=%5.2f (ms)\n",
               snap[i].is_ag ? 'A' : 'R', snap[i].id, snap[i].size_kb,
               snap[i].num_pieces, snap[i].num_steps,
               snap[i].setup_us       / 1000.0,
               snap[i].barrier_us     / 1000.0,
               snap[i].ring_us        / 1000.0,
               snap[i].recv_post_us   / 1000.0,
               snap[i].step_submit_us / 1000.0,
               snap[i].step_wait_us   / 1000.0,
               snap[i].notify_us      / 1000.0);
    }

    /* Top 20 by barrier_us */
    qsort(snap, (size_t)n, sizeof(struct dpu_per_op_record), _cmp_per_op_by_barrier_us);
    printf("  Top 20 by barrier_us:\n");
    for (int i = 0; i < top; i++) {
        printf("    [%c] id=%-6lu size=%7uKB | setup=%6.2fms barrier=%6.2fms ring=%7.2fms notify=%6.2fms\n",
               snap[i].is_ag ? 'A' : 'R', snap[i].id, snap[i].size_kb,
               snap[i].setup_us  / 1000.0,
               snap[i].barrier_us/ 1000.0,
               snap[i].ring_us   / 1000.0,
               snap[i].notify_us / 1000.0);
    }

    /* Top 20 by setup_us */
    qsort(snap, (size_t)n, sizeof(struct dpu_per_op_record), _cmp_per_op_by_setup_us);
    printf("  Top 20 by setup_us:\n");
    for (int i = 0; i < top; i++) {
        printf("    [%c] id=%-6lu size=%7uKB | setup=%6.2fms barrier=%6.2fms ring=%7.2fms notify=%6.2fms\n",
               snap[i].is_ag ? 'A' : 'R', snap[i].id, snap[i].size_kb,
               snap[i].setup_us  / 1000.0,
               snap[i].barrier_us/ 1000.0,
               snap[i].ring_us   / 1000.0,
               snap[i].notify_us / 1000.0);
    }
    printf("================================================================\n");
    fflush(stdout);
    free(snap);
}

static void *msg_pool_worker_main(void *arg)
{
    struct msg_thread_pool *p = (struct msg_thread_pool *)arg;
    while (!atomic_load_explicit(&p->stop, memory_order_acquire)) {
        /* ---- Phase 5 Step 4: Fast path — poll RDMA command slot ---- */
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
                     * production), defaults to Ring 0. Production uses ComCh + Phase 15
                     * dispatcher which correctly assigns ring_id. */
                    execute_doca_collective_cmd(&recv_cmd, p->sample_objects, 0, 0);

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
    p->sample_objects = sample_objects;  /* Phase 5 Step 4: for cmd_slot polling */
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

/* =====================================================
 * DOCA タスク初期化ヘルパー
 * ===================================================== */

static inline void init_read_task(struct doca_task_desc *t, void *local_addr,
                                   void *remote_addr, size_t length,
                                   struct doca_remote_mem_t *remote_mem,
                                   struct doca_rdma_ctx_t *rdma_ctx,
                                   struct doca_rdma_connection *connection,
                                   int stride_id)
{
    memset(t, 0, sizeof(*t));
    t->type = DOCA_RDMA_TASK_READ;
    t->local_addr = local_addr;
    t->remote_addr = remote_addr;
    t->length = length;
    t->remote_mem = remote_mem;
    t->rdma_ctx = rdma_ctx;
    t->connection = connection;
    t->stride_id = stride_id;
}

static inline void init_write_task(struct doca_task_desc *t, void *local_addr,
                                    void *remote_addr, size_t length,
                                    struct doca_remote_mem_t *remote_mem,
                                    struct doca_rdma_ctx_t *rdma_ctx,
                                    struct doca_rdma_connection *connection,
                                    int stride_id)
{
    memset(t, 0, sizeof(*t));
    t->type = DOCA_RDMA_TASK_WRITE;
    t->local_addr = local_addr;
    t->remote_addr = remote_addr;
    t->length = length;
    t->remote_mem = remote_mem;
    t->rdma_ctx = rdma_ctx;
    t->connection = connection;
    t->stride_id = stride_id;
}

static inline void init_ring_send_task(struct doca_task_desc *t, void *local_addr,
                                        size_t length,
                                        struct doca_rdma_ctx_t *rdma_ctx,
                                        struct doca_rdma_connection *connection,
                                        int stride_id)
{
    memset(t, 0, sizeof(*t));
    t->type = DOCA_RDMA_TASK_RING_SEND;
    t->local_addr = local_addr;
    t->length = length;
    t->rdma_ctx = rdma_ctx;
    t->connection = connection;
    t->stride_id = stride_id;
}

static inline void init_ring_recv_task(struct doca_task_desc *t, void *local_addr,
                                        size_t length,
                                        struct doca_rdma_ctx_t *rdma_ctx,
                                        int stride_id)
{
    memset(t, 0, sizeof(*t));
    t->type = DOCA_RDMA_TASK_RING_RECV;
    t->local_addr = local_addr;
    t->length = length;
    t->rdma_ctx = rdma_ctx;
    t->stride_id = stride_id;
}

/* =====================================================
 * Phase 12.1 Plan A: RDMA Doorbell Barrier (RDB)
 *
 * MPI_Barrier (avg 325μs, max 281ms) を 2-pass Ring barrier に置換する。
 * 隣接 Ring 接続のみを使用 (フルメッシュ禁止)。
 *
 * 設計原則:
 *   - OS バイパス: RDMA Write inline のみ、カーネル経由なし
 *   - 既存 Ring 接続再利用: rdma_ring_send + conn_to_next
 *   - working_buf 末尾の 192B を slot 領域として使用 (追加メモリ登録不要)
 *   - 隣接 rank の rdb_local_slot のリモートアドレスを起動時に MPI 1 回交換
 *   - monotonic seq でリセット不要、複数 barrier を区別
 *
 * 2-pass プロトコル (barrier #k, k = 1, 2, ...):
 *   if rank == 0:
 *     Write(2k-1) → next.slot      // gather wave start
 *     wait my.slot >= 2k-1          // gather wave returned (= 全 rank が到達)
 *     Write(2k)   → next.slot      // release wave start
 *     wait my.slot >= 2k            // release wave returned
 *   else:
 *     wait my.slot >= 2k-1          // gather wave from prev
 *     Write(2k-1) → next.slot      // forward gather wave
 *     wait my.slot >= 2k            // release wave from prev
 *     Write(2k)   → next.slot      // forward release wave
 *
 * 重要な性質:
 *   - 2-pass にすることで、release wave が始まる前に必ず gather wave が
 *     完走する。よって「ある rank が exit した時、全 rank は少なくとも
 *     barrier に到達済み」を保証 (1-pass では破られる)。
 *   - 想定 latency: N=4 なら 2(N-1)+2 = 8 hops × ~5μs = ~40μs。
 *     現状 MPI_Barrier の avg 325μs / max 281ms に比べ大幅改善。
 *
 * Slot レイアウト (working_buf 末尾 192 bytes):
 *   [working_buf - 192] rdb_send_data_a (8B + padding to 64B)  // gather Write 用
 *   [working_buf - 128] rdb_send_data_b (8B + padding to 64B)  // release Write 用
 *   [working_buf -  64] rdb_local_slot  (8B + padding to 64B)  // 前 rank が書く
 *
 * 2 つの送信バッファを用意することで、gather Write の NIC 読み込みが
 * 完了する前に release Write のバッファを書き換える race を回避。
 * ===================================================== */

#define RDB_SLOT_REGION_SIZE  192
#define RDB_OFFSET_SEND_A     192
#define RDB_OFFSET_SEND_B     128
#define RDB_OFFSET_LOCAL_SLOT 64

/* CPU pause/yield (アーキテクチャ別) */
#if defined(__aarch64__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define cpu_relax() __builtin_ia32_pause()
#endif

static inline uint64_t _rdb_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* MPI で交換するハンドシェイク構造 (固定サイズ、~120 byte の export desc を 256 byte で余裕を持って格納) */
struct rdb_handshake {
    uint64_t slot_va;          /* 自分の rdb_local_slot のローカル VA */
    uint32_t export_len;       /* working_buf mmap export の長さ */
    uint32_t reserved;
    uint8_t  export_desc[256]; /* working_buf mmap export descriptor */
};

/* Plan A 用 send buffer ポインタ (gather と release で別々のバッファ) */
static inline uint64_t *rdb_get_send_buf_a(struct collective_worker_t *cw) {
    return (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_A);
}
static inline uint64_t *rdb_get_send_buf_b(struct collective_worker_t *cw) {
    return (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_B);
}

/* 1 hop の RDMA Write を発行 (submit_and_wait — inline submit + PE progress 待ち) */
static inline void rdb_write_next(struct collective_worker_t *cw,
                                  uint64_t *local_buf, uint64_t value)
{
    *local_buf = value;

    struct doca_task_desc write_task;
    init_write_task(&write_task,
                    local_buf,
                    (void *)cw->rdb_next_remote_slot_addr,
                    sizeof(uint64_t),
                    &cw->rdb_next_rmem,
                    &cw->rdma_ring_send, cw->conn_to_next, 0);
    write_task.pe_spin = &cw->ring_send_pe_spin;
    submit_and_wait_doca(cw->worker_pool, &write_task);
}

/* 自分の slot を spin で待つ (前 rank が target 以上の値を書き込むまで) */
static inline void rdb_wait_local_slot(struct collective_worker_t *cw, uint64_t target)
{
    while (*cw->rdb_local_slot < target) {
        cpu_relax();
    }
    __sync_synchronize();  /* slot 読み出しが後続のメモリ操作より先に行われることを保証 */
}

/* MPI_COMM_WORLD の rank を取得 (profiling 用、頻繁には呼ばない) */
static int rdb_get_rank(void) {
    int r = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

/* RDB 初期化 (Ring 接続確立後に 1 回だけ呼ぶ) */
static doca_error_t rdb_init(struct collective_worker_t *cw, int rank, int world_size)
{
    /* RDB_DISABLE=1 で無効化 */
    const char *dis = getenv("RDB_DISABLE");
    if (dis && atoi(dis) != 0) {
        DOCA_LOG_INFO("RDB_DISABLE=1, falling back to MPI_Barrier");
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    if (cw->working_buf_size < RDB_SLOT_REGION_SIZE) {
        DOCA_LOG_WARN("RDB: working_buf too small (%zu), falling back to MPI_Barrier",
                      cw->working_buf_size);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    if (!cw->conn_to_next) {
        DOCA_LOG_WARN("RDB: conn_to_next not established, falling back to MPI_Barrier");
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* slot 領域の初期化 (working_buf 末尾) */
    uint64_t *send_a = (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_A);
    uint64_t *send_b = (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_B);
    volatile uint64_t *local_slot = (volatile uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_LOCAL_SLOT);
    *send_a = 0;
    *send_b = 0;
    *local_slot = 0;
    cw->rdb_local_slot = local_slot;
    cw->rdb_send_data  = send_a;  /* 互換性のため (legacy フィールド) */

    /* working_buf を rdma_ring_send 経由で export */
    const void *my_export = NULL;
    size_t      my_export_len = 0;
    doca_error_t ret = doca_rdma_ctx_export_mmap(&cw->rdma_ring_send,
                                                  &my_export, &my_export_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_WARN("RDB: doca_rdma_ctx_export_mmap failed: %s, falling back to MPI_Barrier",
                      doca_error_get_name(ret));
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }
    if (my_export_len > sizeof(((struct rdb_handshake *)0)->export_desc)) {
        DOCA_LOG_WARN("RDB: export_desc too large (%zu > 256), falling back to MPI_Barrier",
                      my_export_len);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* 隣接 rank と handshake を MPI_Sendrecv で交換
     *   prev_rank に送信 (PREV にとって自分は NEXT なので、PREV は自分を書きたい)
     *   next_rank から受信 (自分は NEXT を書きたいので、NEXT の情報を得る)
     */
    int prev_rank = (rank - 1 + world_size) % world_size;
    int next_rank = (rank + 1) % world_size;

    struct rdb_handshake my_hs, next_hs;
    memset(&my_hs,   0, sizeof(my_hs));
    memset(&next_hs, 0, sizeof(next_hs));
    my_hs.slot_va    = (uint64_t)(uintptr_t)cw->rdb_local_slot;
    my_hs.export_len = (uint32_t)my_export_len;
    memcpy(my_hs.export_desc, my_export, my_export_len);

    int mret = MPI_Sendrecv(&my_hs,   sizeof(my_hs),   MPI_BYTE, prev_rank, 0x12345,
                            &next_hs, sizeof(next_hs), MPI_BYTE, next_rank, 0x12345,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (mret != MPI_SUCCESS) {
        DOCA_LOG_WARN("RDB: MPI_Sendrecv failed (%d), falling back to MPI_Barrier", mret);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* 受け取った export descriptor を import (next_rank の working_buf を remote_mem 化) */
    ret = doca_remote_mem_create(&cw->rdb_next_rmem, cw->ring_dev,
                                  next_hs.export_desc, next_hs.export_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_WARN("RDB: doca_remote_mem_create failed: %s, falling back to MPI_Barrier",
                      doca_error_get_name(ret));
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    cw->rdb_next_remote_slot_addr = next_hs.slot_va;
    cw->rdb_my_seq        = 0;
    cw->rdb_total_count   = 0;
    cw->rdb_total_ns      = 0;
    cw->rdb_max_ns        = 0;
    cw->rdb_enabled       = true;

    DOCA_LOG_INFO("[rank%d] RDB initialized: my_slot_va=0x%lx next_slot_va=0x%lx (prev=%d, next=%d)",
                  rank, (uint64_t)(uintptr_t)cw->rdb_local_slot,
                  cw->rdb_next_remote_slot_addr, prev_rank, next_rank);
    return DOCA_SUCCESS;
}

/* 2-pass Ring barrier (rank と world_size を引数に取る) */
static doca_error_t rdb_barrier(struct collective_worker_t *cw, int rank, int world_size, int ring_id)
{
    (void)world_size;  /* 現在は不使用 (将来 N>16 等で使用予定) */

    if (!cw->rdb_enabled) {
        /* RDB 未初期化 → MPI_Barrier フォールバック。
         * Phase 15: Ring 0 と Ring 1 が同時に barrier を呼ぶ可能性があるため、
         * ring 専用の MPI_Comm を使う (MPI_COMM_WORLD を共有すると barrier が
         * 混線して Ring algorithm が破綻する)。 */
        MPI_Comm comm = (ring_id >= 0 && ring_id < N_RINGS && cw->ring_comm_valid[ring_id])
                        ? cw->ring_comm[ring_id]
                        : MPI_COMM_WORLD;
        MPI_Barrier(comm);
        return DOCA_SUCCESS;
    }

    uint64_t t0 = _rdb_now_ns();
    uint64_t my_seq         = ++cw->rdb_my_seq;
    uint64_t gather_target  = 2 * my_seq - 1;
    uint64_t release_target = 2 * my_seq;

    uint64_t *buf_a = rdb_get_send_buf_a(cw);
    uint64_t *buf_b = rdb_get_send_buf_b(cw);

    if (rank == 0) {
        /* Phase 1: gather wave 開始 */
        rdb_write_next(cw, buf_a, gather_target);
        rdb_wait_local_slot(cw, gather_target);
        /* Phase 2: release wave 開始 */
        rdb_write_next(cw, buf_b, release_target);
        rdb_wait_local_slot(cw, release_target);
    } else {
        /* Phase 1: gather wave 受信 → 転送 */
        rdb_wait_local_slot(cw, gather_target);
        rdb_write_next(cw, buf_a, gather_target);
        /* Phase 2: release wave 受信 → 転送 */
        rdb_wait_local_slot(cw, release_target);
        rdb_write_next(cw, buf_b, release_target);
    }

    uint64_t t1 = _rdb_now_ns();
    uint64_t elapsed = t1 - t0;
    cw->rdb_total_count++;
    cw->rdb_total_ns += elapsed;
    if (elapsed > cw->rdb_max_ns) cw->rdb_max_ns = elapsed;

    /* per-op TLS profiling (collective 単位の barrier 計測用) */
    g_per_op_barrier_ns = elapsed;

    /* 5000 回ごとに rank 0 から profiling 出力 */
    if (cw->rdb_total_count % 5000 == 0) {
        int my_rank = rdb_get_rank();
        if (my_rank == 0) {
            double avg_us = (double)cw->rdb_total_ns / cw->rdb_total_count / 1000.0;
            double max_us = (double)cw->rdb_max_ns / 1000.0;
            printf("[RDB rank0] barriers=%lu avg=%.1fus max=%.1fus\n",
                   cw->rdb_total_count, avg_us, max_us);
            fflush(stdout);
        }
    }

    return DOCA_SUCCESS;
}

/* =====================================================
 * Remote mmap キャッシュ (Phase 3)
 * ===================================================== */

static inline uint64_t rmem_cache_hash(uint64_t addr, size_t len)
{
    uint64_t x = addr;
    x ^= (uint64_t)len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x & RMEM_CACHE_MASK;
}

static struct rmem_cache_entry *rmem_cache_find(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len)
{
    int idx = (int)rmem_cache_hash(addr, len);
    if (cache[idx].valid && cache[idx].addr == addr && cache[idx].len == len)
        return &cache[idx];
    return NULL;
}

static void rmem_cache_store(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len,
    const struct doca_remote_mem_t *rmem)
{
    int idx = (int)rmem_cache_hash(addr, len);
    if (cache[idx].valid)
        doca_remote_mem_destroy(&cache[idx].rmem);
    cache[idx].addr  = addr;
    cache[idx].len   = len;
    cache[idx].rmem  = *rmem;
    cache[idx].valid = true;
}

static void rmem_cache_destroy(struct rmem_cache_entry *cache)
{
    for (int i = 0; i < RMEM_CACHE_SIZE; i++) {
        if (cache[i].valid) {
            doca_remote_mem_destroy(&cache[i].rmem);
            cache[i].valid = false;
        }
    }
}

/* ---- PCI import mmap キャッシュ (16384 エントリ、直接マッピング) ---- */

static inline uint64_t pci_mmap_cache_hash(uint64_t addr, size_t len)
{
    uint64_t x = addr;
    x ^= (uint64_t)len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x & PCI_MMAP_CACHE_MASK;
}

static struct pci_mmap_cache_entry *pci_mmap_cache_alloc(void)
{
    struct pci_mmap_cache_entry *cache = (struct pci_mmap_cache_entry *)calloc(
        PCI_MMAP_CACHE_SIZE, sizeof(struct pci_mmap_cache_entry));
    return cache;
}

/* 線形探索の最大ステップ数 (無限ループ防止) */
#define PCI_MMAP_CACHE_MAX_PROBE  64

static struct doca_mmap *pci_mmap_cache_find(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len)
{
    if (!cache) return NULL;
    int idx = (int)pci_mmap_cache_hash(addr, len);
    for (int probe = 0; probe < PCI_MMAP_CACHE_MAX_PROBE; probe++) {
        int i = (idx + probe) & PCI_MMAP_CACHE_MASK;
        if (!cache[i].valid)
            return NULL;  /* 空きスロットに到達 → 存在しない */
        if (cache[i].addr == addr && cache[i].len == len)
            return cache[i].mmap;
    }
    return NULL;
}

static void pci_mmap_cache_store(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len,
    struct doca_mmap *mmap)
{
    if (!cache) return;
    int idx = (int)pci_mmap_cache_hash(addr, len);
    for (int probe = 0; probe < PCI_MMAP_CACHE_MAX_PROBE; probe++) {
        int i = (idx + probe) & PCI_MMAP_CACHE_MASK;
        if (!cache[i].valid) {
            /* 空きスロット発見 → 格納 */
            cache[i].addr  = addr;
            cache[i].len   = len;
            cache[i].mmap  = mmap;
            cache[i].valid = true;
            return;
        }
        if (cache[i].addr == addr && cache[i].len == len) {
            /* 同じキーが既に存在 → 更新不要 */
            return;
        }
    }
    /* キャッシュ満杯 — 呼び出し側は新しい mmap で動作するが次回 re-import が発生する */
}

static void pci_mmap_cache_destroy(struct pci_mmap_cache_entry *cache)
{
    if (!cache) return;
    for (int i = 0; i < PCI_MMAP_CACHE_SIZE; i++) {
        if (cache[i].valid && cache[i].mmap) {
            doca_mmap_destroy(cache[i].mmap);
            cache[i].valid = false;
        }
    }
    free(cache);
}

/* =====================================================
 * Phase 7 Step 0-D: DOCA API micro-benchmark
 * 各 DOCA API 呼び出しの個別コストを 1000 回計測し出力する。
 * ワーカスレッド起動前に 1 回だけ呼ぶこと。
 * ===================================================== */

static void run_doca_api_microbenchmark(struct collective_worker_t *cw)
{
    const int N = 1000;
    uint64_t t0, t1;
    struct doca_rdma_ctx_t *ctx = &cw->rdma_rma;
    void *addr = cw->working_buf;

    /* 1. calloc / free */
    uint64_t sum_calloc = 0, sum_free = 0;
    uint64_t min_calloc = UINT64_MAX, max_calloc = 0;
    uint64_t min_free = UINT64_MAX, max_free = 0;
    for (int i = 0; i < N; i++) {
        t0 = now_monotonic_ns();
        void *p = calloc(1, sizeof(struct doca_task_cb_data));
        t1 = now_monotonic_ns();
        uint64_t d = t1 - t0;
        sum_calloc += d;
        if (d < min_calloc) min_calloc = d;
        if (d > max_calloc) max_calloc = d;

        t0 = now_monotonic_ns();
        free(p);
        t1 = now_monotonic_ns();
        d = t1 - t0;
        sum_free += d;
        if (d < min_free) min_free = d;
        if (d > max_free) max_free = d;
    }

    /* 2. doca_buf_inventory_buf_get_by_addr + dec_refcount */
    uint64_t sum_buf_get = 0, sum_buf_dec = 0;
    uint64_t min_buf_get = UINT64_MAX, max_buf_get = 0;
    for (int i = 0; i < N; i++) {
        struct doca_buf *buf = NULL;
        t0 = now_monotonic_ns();
        doca_rdma_get_local_buf(ctx, addr, 64, &buf);
        t1 = now_monotonic_ns();
        uint64_t d = t1 - t0;
        sum_buf_get += d;
        if (d < min_buf_get) min_buf_get = d;
        if (d > max_buf_get) max_buf_get = d;
        if (buf) {
            t0 = now_monotonic_ns();
            doca_buf_dec_refcount(buf, NULL);
            t1 = now_monotonic_ns();
            sum_buf_dec += t1 - t0;
        }
    }

    /* 3. doca_pe_progress (empty — no pending tasks) */
    uint64_t sum_pe = 0;
    uint64_t min_pe = UINT64_MAX, max_pe = 0;
    for (int i = 0; i < N; i++) {
        t0 = now_monotonic_ns();
        doca_pe_progress(ctx->pe);
        t1 = now_monotonic_ns();
        uint64_t d = t1 - t0;
        sum_pe += d;
        if (d < min_pe) min_pe = d;
        if (d > max_pe) max_pe = d;
    }

    /* 4. atomic_flag test_and_set + clear (spinlock overhead) */
    atomic_flag test_lock = ATOMIC_FLAG_INIT;
    uint64_t sum_lock = 0;
    for (int i = 0; i < N; i++) {
        t0 = now_monotonic_ns();
        atomic_flag_test_and_set(&test_lock);
        atomic_flag_clear(&test_lock);
        t1 = now_monotonic_ns();
        sum_lock += t1 - t0;
    }

    fprintf(stderr, "\n[DOCA API MICROBENCH rank%d] %d iterations:\n"
            "  calloc(cb_data):    avg=%5.0fns  min=%5luns  max=%5luns\n"
            "  free(cb_data):      avg=%5.0fns  min=%5luns  max=%5luns\n"
            "  buf_get_by_addr:    avg=%5.0fns  min=%5luns  max=%5luns\n"
            "  buf_dec_refcount:   avg=%5.0fns\n"
            "  pe_progress(empty): avg=%5.0fns  min=%5luns  max=%5luns\n"
            "  atomic_flag lock:   avg=%5.0fns\n\n",
            cw->mpi_rank, N,
            (double)sum_calloc / N, min_calloc, max_calloc,
            (double)sum_free / N, min_free, max_free,
            (double)sum_buf_get / N, min_buf_get, max_buf_get,
            (double)sum_buf_dec / N,
            (double)sum_pe / N, min_pe, max_pe,
            (double)sum_lock / N);
}

/* =====================================================
 * 集合通信アルゴリズム (DOCA RDMA版, NUM_STRIDES=1固定)
 * ===================================================== */

/* 前方宣言 (collective_reduce_scatter から参照) */
static inline void progress_rma_pes(struct collective_worker_t *cw, bool use_dual);
static inline void progress_ring_pes(struct collective_worker_t *cw, bool use_dual_ring);

static doca_error_t collective_reduce_scatter(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf, void *recv_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    uint64_t id)
{
    (void)id;
    fp16_t *local_send_buffer = (fp16_t *)send_buf;
    fp16_t *local_receive_buffer = (fp16_t *)recv_buf;
    uint64_t chunk_size = buffer_len / world_size;
    uint64_t elements_per_chunk = chunk_size / sizeof(fp16_t);
    int steps = (int)world_size - 1;
    struct doca_task_desc task;

    /* Phase 7 Step 2 + Phase 12.1 Plan A: RDB (RDMA Doorbell Barrier) で全 rank 同期。
     * MPI_Barrier (avg 325μs / max 281ms) を 2-pass Ring barrier (~40μs) に置換。
     * RDB 未初期化時は内部で MPI_Barrier フォールバック。
     * Phase 15: RS は常に Ring 0 を使用 (RS は並列化していない) */
    {
        uint64_t _bar_t0 = now_monotonic_ns();
        rdb_barrier(cw, (int)rank, (int)world_size, 0);
        g_per_op_barrier_ns = now_monotonic_ns() - _bar_t0;
    }

    /* DPU Union Tracker (ring-only): barrier 後〜ring 完了直前までを wrap。
     * 純 NIC 転送時間のみを wall union で測る。 */
    dpu_union_tracker_enter(false /* is_ag=false for RS */);

    /* 全 Ring step の Recv を一括先行 post (AllGather と同じ手法) */
    atomic_int recv_pendings[steps > 0 ? steps : 1];
    for (int step = 0; step < steps; step++) {
        uint64_t r_idx = (rank + 2*world_size - 2 - step) % world_size;
        atomic_init(&recv_pendings[step], 1);
        init_ring_recv_task(&task, (void *)((uintptr_t)recv_buf + r_idx * chunk_size),
                            chunk_size, &cw->rdma_ring_recv, step);
        task.inline_pending = &recv_pendings[step];
        task.pe_spin = &cw->ring_recv_pe_spin;
        submit_doca_task_from_desc(&cw->rdma_ring_recv, &task, NULL);
    }

    /* Read[0]: chunk[(rank-1)%ws] from GPU src → send_buf */
    #define RS_INLINE_READ(idx) do { \
        uint64_t _off = (idx) * chunk_size; \
        atomic_int _p; atomic_init(&_p, 1); \
        init_read_task(&task, (void *)((uintptr_t)send_buf + _off), \
                       (void *)(remote_src_addr + _off), \
                       chunk_size, &cw->host_src_rmem, &cw->rdma_rma, cw->host_conn, 0); \
        task.inline_pending = &_p; task.pe_spin = &cw->rma_pe_spin; \
        submit_doca_task_from_desc(&cw->rdma_rma, &task, NULL); \
        while (atomic_load_explicit(&_p, memory_order_acquire) > 0) \
            progress_rma_pes(cw, false); \
    } while(0)

    uint64_t read_index = (rank - 1 + world_size) % world_size;
    RS_INLINE_READ(read_index);

    for (int step = 0; step < steps; step++) {
        uint64_t s_idx = (rank + world_size - 1 - step) % world_size;

        /* Send (inline) — Recv は既に pre-post 済み */
        atomic_int send_p;
        atomic_init(&send_p, 1);
        init_ring_send_task(&task, (void *)((uintptr_t)send_buf + s_idx * chunk_size),
                            chunk_size, &cw->rdma_ring_send, cw->conn_to_next, step);
        task.inline_pending = &send_p; task.pe_spin = &cw->ring_send_pe_spin;
        submit_doca_task_from_desc(&cw->rdma_ring_send, &task, NULL);

        /* Read next chunk — Send+Recv と並行 */
        read_index = (rank - step - 2 + 2*world_size) % world_size;
        atomic_int read_p;
        atomic_init(&read_p, 1);
        {
            uint64_t _off = read_index * chunk_size;
            init_read_task(&task, (void *)((uintptr_t)send_buf + _off),
                           (void *)(remote_src_addr + _off),
                           chunk_size, &cw->host_src_rmem, &cw->rdma_rma, cw->host_conn, 0);
            task.inline_pending = &read_p; task.pe_spin = &cw->rma_pe_spin;
            submit_doca_task_from_desc(&cw->rdma_rma, &task, NULL);
        }

        /* Send + Recv + Read 全完了待ち */
        while (atomic_load_explicit(&recv_pendings[step], memory_order_acquire) > 0 ||
               atomic_load_explicit(&send_p, memory_order_acquire) > 0 ||
               atomic_load_explicit(&read_p, memory_order_acquire) > 0) {
            progress_ring_pes(cw, false);
            progress_rma_pes(cw, false);
        }

        /* Aggregate: send_buf += recv_buf */
        uint64_t agg_index = (rank + world_size - 1 - (step + 1)) % world_size;
        uint64_t start = elements_per_chunk * agg_index;
        uint64_t end = start + elements_per_chunk;
        if (elements_per_chunk <= RS_AGGREGATION_THRESHOLD / sizeof(fp16_t)) {
            rs_add_range_neon(local_send_buffer, local_receive_buffer, start, end);
        } else {
            submit_rs_task(cw->rs_pool, local_send_buffer, local_receive_buffer, start, end, step);
            poll_wait_rs_completion(cw->rs_pool, step);
        }
    }

    /* Put: send_buf[rank] → GPU dst */
    {
        atomic_int _p; atomic_init(&_p, 1);
        init_write_task(&task, (void *)((uintptr_t)send_buf + rank * chunk_size),
                        (void *)(remote_dst_addr),
                        chunk_size, &cw->host_dst_rmem, &cw->rdma_rma, cw->host_conn, 0);
        task.inline_pending = &_p; task.pe_spin = &cw->rma_pe_spin;
        submit_doca_task_from_desc(&cw->rdma_rma, &task, NULL);
        while (atomic_load_explicit(&_p, memory_order_acquire) > 0)
            progress_rma_pes(cw, false);
    }

    #undef RS_INLINE_READ
    /* DPU Union Tracker (ring-only): RS 終了 */
    dpu_union_tracker_exit(false);
    return DOCA_SUCCESS;
}

/* Phase 6 Step F: main thread PE progress helpers for AllGather */
static inline void progress_rma_pes(struct collective_worker_t *cw, bool use_dual)
{
    try_pe_progress(cw->rdma_rma.pe, &cw->rma_pe_spin);
    if (use_dual && cw->rdma_rma_rail1.pe)
        try_pe_progress(cw->rdma_rma_rail1.pe, &cw->rma_pe_spin_rail1);
}

/* Phase 15: Ring-aware context bundle. 各 AG は ring_id (0 or 1) を受け、
 * 対応する RDMA context / connection / pe_spin をこの構造体経由でアクセスする。
 * collective_all_gather の開始時に get_ring_ctx() で初期化する。 */
struct ring_ctx_ptrs {
    struct doca_rdma_ctx_t *send;
    struct doca_rdma_ctx_t *recv;
    struct doca_rdma_ctx_t *send_rail1;
    struct doca_rdma_ctx_t *recv_rail1;
    struct doca_rdma_connection *conn_to_next;
    struct doca_rdma_connection *conn_from_prev;
    struct doca_rdma_connection *conn_to_next_rail1;
    struct doca_rdma_connection *conn_from_prev_rail1;
    struct pe_spin_t *send_pe_spin;
    struct pe_spin_t *recv_pe_spin;
    struct pe_spin_t *send_pe_spin_rail1;
    struct pe_spin_t *recv_pe_spin_rail1;
    int ring_id;  /* 0 or 1 (for debug logs) */
};

static inline void get_ring_ctx(struct collective_worker_t *cw, int ring_id,
                                 struct ring_ctx_ptrs *rc)
{
    rc->ring_id = ring_id;
    if (ring_id == 0) {
        rc->send = &cw->rdma_ring_send;
        rc->recv = &cw->rdma_ring_recv;
        rc->send_rail1 = &cw->rdma_ring_send_rail1;
        rc->recv_rail1 = &cw->rdma_ring_recv_rail1;
        rc->conn_to_next = cw->conn_to_next;
        rc->conn_from_prev = cw->conn_from_prev;
        rc->conn_to_next_rail1 = cw->conn_to_next_rail1;
        rc->conn_from_prev_rail1 = cw->conn_from_prev_rail1;
        rc->send_pe_spin = &cw->ring_send_pe_spin;
        rc->recv_pe_spin = &cw->ring_recv_pe_spin;
        rc->send_pe_spin_rail1 = &cw->ring_send_pe_spin_rail1;
        rc->recv_pe_spin_rail1 = &cw->ring_recv_pe_spin_rail1;
    } else {
        /* Ring 1 (Phase 15) */
        rc->send = &cw->rdma_ring_send_r1;
        rc->recv = &cw->rdma_ring_recv_r1;
        rc->send_rail1 = &cw->rdma_ring_send_r1_rail1;
        rc->recv_rail1 = &cw->rdma_ring_recv_r1_rail1;
        rc->conn_to_next = cw->conn_to_next_r1;
        rc->conn_from_prev = cw->conn_from_prev_r1;
        rc->conn_to_next_rail1 = cw->conn_to_next_r1_rail1;
        rc->conn_from_prev_rail1 = cw->conn_from_prev_r1_rail1;
        rc->send_pe_spin = &cw->ring_send_pe_spin_r1;
        rc->recv_pe_spin = &cw->ring_recv_pe_spin_r1;
        rc->send_pe_spin_rail1 = &cw->ring_send_pe_spin_r1_rail1;
        rc->recv_pe_spin_rail1 = &cw->ring_recv_pe_spin_r1_rail1;
    }
}

/* Ring 0 / Ring 1 共通の PE progress. ring_ctx_ptrs で対象 ring の PE を直接指定。 */
static inline void progress_ring_pes_rc(struct ring_ctx_ptrs *rc, bool use_dual_ring)
{
    try_pe_progress(rc->send->pe, rc->send_pe_spin);
    try_pe_progress(rc->recv->pe, rc->recv_pe_spin);
    if (use_dual_ring) {
        if (rc->send_rail1->pe)
            try_pe_progress(rc->send_rail1->pe, rc->send_pe_spin_rail1);
        if (rc->recv_rail1->pe)
            try_pe_progress(rc->recv_rail1->pe, rc->recv_pe_spin_rail1);
    }
}

/* 既存コード (collective_reduce_scatter 等) 互換: Ring 0 のみ progress */
static inline void progress_ring_pes(struct collective_worker_t *cw, bool use_dual_ring)
{
    try_pe_progress(cw->rdma_ring_send.pe, &cw->ring_send_pe_spin);
    try_pe_progress(cw->rdma_ring_recv.pe, &cw->ring_recv_pe_spin);
    if (use_dual_ring) {
        if (cw->rdma_ring_send_rail1.pe)
            try_pe_progress(cw->rdma_ring_send_rail1.pe, &cw->ring_send_pe_spin_rail1);
        if (cw->rdma_ring_recv_rail1.pe)
            try_pe_progress(cw->rdma_ring_recv_rail1.pe, &cw->ring_recv_pe_spin_rail1);
    }
}

static doca_error_t collective_all_gather(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    struct doca_mmap *gpu_src_local_mmap,
    uint64_t id,
    struct doca_mmap *gpu_dst_local_mmap,
    /* Phase 12.2: dual-rail GPU Direct (rail1 mmaps; NULL ならシングルレール) */
    struct doca_mmap *gpu_src_local_mmap_rail1,
    struct doca_mmap *gpu_dst_local_mmap_rail1,
    /* Phase 15: ring_id (0 or 1) で Ring instance を選ぶ */
    int ring_id)
{
    (void)id;
    uint64_t chunk_size = buffer_len / world_size;

    /* ---- Phase 15: ring_id に対応する context/conn/pe_spin を取得 ---- */
    struct ring_ctx_ptrs rc;
    get_ring_ctx(cw, ring_id, &rc);

    /* ---- ピース分割パラメータ (Phase 12.2 G: size-adaptive + runtime tunable) ----
     * AG_PIECE_MAX (env) と AG_PIECE_TARGET (env) で実行時調整可能。
     * デフォルトは max=8, target=8MB (Phase 12.2 段階の挙動と同じ)。 */
    int num_pieces = ag_compute_num_pieces(chunk_size);
    uint64_t piece_size = chunk_size / (uint64_t)num_pieces;
    uint64_t last_piece_size = chunk_size - piece_size * (uint64_t)(num_pieces - 1);

    struct doca_task_desc task;

    /* Multi-Rail helpers */
    bool use_dual = cw->dual_rail && cw->host_src_rmem_rail1.valid && cw->host_dst_rmem_rail1.valid;
    bool use_dual_ring = (rc.conn_to_next_rail1 != NULL) && (rc.conn_from_prev_rail1 != NULL);

    /* Phase 12.2: GPU Direct でも dual-rail を使う条件
     *   - rail0 / rail1 の両方に Ring 接続がある (use_dual_ring == true)
     *   - GPU Direct で必要な mmap が両 rail で揃っている
     * full_gpu_direct (Phase 9) では src/dst 両方に mmap が必要、
     * gpu_direct のみ (Phase 8) では dst だけで足りる。 */
    bool gpu_direct_dual_ring_dst_ok = (gpu_dst_local_mmap != NULL) && (gpu_dst_local_mmap_rail1 != NULL);
    bool gpu_direct_dual_ring_src_ok = (gpu_src_local_mmap != NULL) && (gpu_src_local_mmap_rail1 != NULL);
    if (gpu_dst_local_mmap) {
        /* GPU Direct 時: rail1 mmap が揃っていなければシングルレールにフォールバック */
        if (!gpu_direct_dual_ring_dst_ok)
            use_dual_ring = false;
        /* full_gpu_direct (src も使う) なら src 側 rail1 も必須 */
        if (gpu_src_local_mmap && !gpu_direct_dual_ring_src_ok)
            use_dual_ring = false;
    }

    /* ---- Phase 7 Step 0: timing (no print in hot path) ---- */
    struct ag_timing_record *trec = NULL;
    if (g_ag_timing.active && g_ag_timing.count < AG_TIMING_MAX_ITERS) {
        trec = &g_ag_timing.records[g_ag_timing.count];
        trec->num_ops = 0;
        trec->buffer_len = buffer_len;
    }
    int _oi = 0;             /* operation index */
    uint64_t _t0 = 0, _t1 = 0;  /* submit start, submit end / wait start */

    /* Phase 8/9: GPU Direct — PCI import した mmap が有効なら使用 */
    bool gpu_direct = (gpu_dst_local_mmap != NULL);
    /* Phase 9: src GVMI も有効なら world_size ステップ (Read+Write+Barrier 全廃) */
    bool full_gpu_direct = gpu_direct && (gpu_src_local_mmap != NULL);

    /* Phase 9: full_gpu_direct なら world_size ステップ (自分のチャンクも Ring 経由で受信) */
    int total_steps = full_gpu_direct ? (int)world_size : (int)world_size - 1;

    /* ========================================
     * Phase 7 Step 2 + Phase 12.1 Plan A: RDB barrier で全 rank 同期。
     * MPI_Barrier (avg 325μs / max 281ms) を 2-pass Ring barrier (~40μs) に置換。
     * RDB 未初期化時は内部で MPI_Barrier フォールバック。
     * Phase 15: ring_id 専用の MPI_Comm を使って Ring 0/1 の barrier が混線しないようにする
     * ======================================== */
    if (trec) _t0 = now_monotonic_ns();
    {
        uint64_t _bar_t0 = now_monotonic_ns();
        rdb_barrier(cw, (int)rank, (int)world_size, ring_id);
        g_per_op_barrier_ns = now_monotonic_ns() - _bar_t0;
    }
    if (trec && _oi < AG_TIMING_MAX_OPS) {
        uint64_t _t2 = now_monotonic_ns();
        trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = 0, .wait_ns = _t2 - _t0, .name_idx = AG_OP_BARRIER };
        _oi++;
    }

    /* DPU Union Tracker (ring-only): barrier 後〜ring 完了直前までを wrap。
     * 純 NIC 転送時間のみを wall union で測る。 */
    dpu_union_tracker_enter(true /* is_ag=true */);

    /* ========================================
     * Phase 7 Step 1: 全 Ring step の Recv を一括先行 post
     * Phase 8: GPU Direct → gpu_dst_local_mmap に直接 Recv (Put 不要)
     * ======================================== */
    /* Phase 16: per-op 初期化 */
    g_per_op_recv_post_ns = 0;
    g_per_op_step_submit_ns = 0;
    g_per_op_step_wait_ns = 0;
    g_per_op_num_pieces = (uint16_t)num_pieces;
    g_per_op_num_steps = (uint16_t)total_steps;

    uint64_t _recv_post_t0 = now_monotonic_ns();
    if (trec) _t0 = _recv_post_t0;
    atomic_int recv_pendings[total_steps > 0 ? total_steps : 1];
    for (int step = 0; step < total_steps; step++) {
        uint64_t receive_index = (rank - 1 - (uint64_t)step + world_size * 2) % world_size;
        uint64_t receive_offset = receive_index * chunk_size;
        atomic_init(&recv_pendings[step], num_pieces);
        for (int p = 0; p < num_pieces; p++) {
            uint64_t p_off = (uint64_t)p * piece_size;
            uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
            bool piece_rail1 = use_dual_ring && (p & 1);
            /* Phase 15: ring_id に応じた ctx (rc.recv / rc.recv_rail1) を選択 */
            struct doca_rdma_ctx_t *recv_ctx = piece_rail1 ? rc.recv_rail1 : rc.recv;
            if (gpu_direct) {
                init_ring_recv_task(&task,
                    (void *)(remote_dst_addr + receive_offset + p_off),
                    p_len, recv_ctx, step * num_pieces + p);
                /* Phase 12.2: rail1 piece は ring_dev_rail1 で import した mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_dst_local_mmap_rail1 : gpu_dst_local_mmap;
            } else {
                init_ring_recv_task(&task,
                    (void *)((uintptr_t)send_buf + receive_offset + p_off),
                    p_len, recv_ctx, step * num_pieces + p);
            }
            task.inline_pending = &recv_pendings[step];
            task.pe_spin = piece_rail1 ? rc.recv_pe_spin_rail1 : rc.recv_pe_spin;
            submit_doca_task_from_desc(recv_ctx, &task, NULL);
        }
    }
    uint64_t _recv_post_t1 = now_monotonic_ns();
    g_per_op_recv_post_ns = _recv_post_t1 - _recv_post_t0;
    if (trec) _t1 = _recv_post_t1;
    if (trec && _oi < AG_TIMING_MAX_OPS) {
        trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = _t1 - _t0, .wait_ns = 0, .name_idx = AG_OP_RING_SENDRECV };
        _oi++;
    }

    /* ========================================
     * INITIAL フェーズ: Read + Write (full_gpu_direct なら全スキップ)
     * Phase 9: full_gpu_direct では world_size ステップ Ring で全データ転送
     * ======================================== */
    if (!full_gpu_direct) {
        uint64_t my_offset = rank * chunk_size;

        {
            /* Legacy: Read → DPU DDR → Write → GPU dst
             * (Phase 9 の Read→GPU dst 直書きは PCIe U ターン不可のため使えない) */
            if (trec) _t0 = now_monotonic_ns();
            atomic_int read_pending;
            atomic_init(&read_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *src_rmem = (use_dual && (p & 1)) ? &cw->host_src_rmem_rail1 : &cw->host_src_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_read_task(&task,
                               (void *)((uintptr_t)send_buf + my_offset + p_off),
                               (void *)(remote_src_addr + p_off),
                               p_len, src_rmem, rma_ctx, conn, p);
                task.inline_pending = &read_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task, NULL);
            }
            if (trec) _t1 = now_monotonic_ns();
            while (atomic_load_explicit(&read_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);
            if (trec && _oi < AG_TIMING_MAX_OPS) {
                uint64_t _t2 = now_monotonic_ns();
                trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = _t1 - _t0, .wait_ns = _t2 - _t1, .name_idx = AG_OP_READ };
                _oi++;
            }

            /* Write: DPU send_buf → Host GPU dst[my_rank] */
            if (trec) _t0 = now_monotonic_ns();
            atomic_int write_pending;
            atomic_init(&write_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *dst_rmem = (use_dual && (p & 1)) ? &cw->host_dst_rmem_rail1 : &cw->host_dst_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_write_task(&task,
                                (void *)((uintptr_t)send_buf + my_offset + p_off),
                                (void *)(remote_dst_addr + my_offset + p_off),
                                p_len, dst_rmem, rma_ctx, conn, p);
                task.inline_pending = &write_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task, NULL);
            }
            if (trec) _t1 = now_monotonic_ns();
            while (atomic_load_explicit(&write_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);
            if (trec && _oi < AG_TIMING_MAX_OPS) {
                uint64_t _t2 = now_monotonic_ns();
                trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = _t1 - _t0, .wait_ns = _t2 - _t1, .name_idx = AG_OP_WRITE };
                _oi++;
            }
        }

        /* MPI_Barrier は Recv bulk post 前に実行済み */
    }

    /* ========================================
     * Ring フェーズ
     * ======================================== */
    /* Phase 12.2 G-3: per-step timing for big AGs (size >= 1MB)
     *
     * 各 unique size について最初の 3 件を全 rank で印字する。
     * これにより:
     *   - 150MB embedding を確実に捕捉できる (旧版は 500 件に 1 件で運次第だった)
     *   - 全 rank を印字することで rank skew を可視化できる
     *   - 中サイズ (1MB-50MB) も捕捉できるので piece 数調整の検証に使える
     *
     * size_seen テーブル: 32 エントリの直接マッピング (size_kb → count)
     * 各 size について count < 3 まで印字、3 を超えたらスキップ。 */
    #define _AG_DBG_MAX_SIZES 32
    #define _AG_DBG_PRINT_PER_SIZE 3
    static struct {
        uint32_t size_kb;
        uint8_t  printed_count;
    } _ag_dbg_seen[_AG_DBG_MAX_SIZES];
    static int _ag_dbg_seen_n = 0;

    bool _ag_step_dbg_print = false;
    if (buffer_len >= (1UL << 20) && total_steps <= 16) {
        uint32_t sz_kb = (uint32_t)(buffer_len / 1024);
        int idx = -1;
        for (int i = 0; i < _ag_dbg_seen_n; i++) {
            if (_ag_dbg_seen[i].size_kb == sz_kb) { idx = i; break; }
        }
        if (idx < 0 && _ag_dbg_seen_n < _AG_DBG_MAX_SIZES) {
            idx = _ag_dbg_seen_n++;
            _ag_dbg_seen[idx].size_kb = sz_kb;
            _ag_dbg_seen[idx].printed_count = 0;
        }
        if (idx >= 0 && _ag_dbg_seen[idx].printed_count < _AG_DBG_PRINT_PER_SIZE) {
            _ag_dbg_seen[idx].printed_count++;
            /* 計測ログ [AG STEP ...] を無効化 (2026-07-27) */
            /* _ag_step_dbg_print = true; */
        }
    }
    uint64_t _ag_step_submit_ns[16] = {0};
    uint64_t _ag_step_wait_ns[16]   = {0};

    for (int step = 0; step < total_steps; step++) {
        uint64_t send_index = (rank - (uint64_t)step + world_size) % world_size;
        uint64_t receive_index = (rank - 1 - (uint64_t)step + world_size * 2) % world_size;
        uint64_t send_offset = send_index * chunk_size;
        uint64_t receive_offset = receive_index * chunk_size;

        /* Phase 16: per-step timing は全 AG で集計 (_ag_step_dbg_print は subset 用) */
        uint64_t _p16_step_t0 = now_monotonic_ns();
        uint64_t _step_t0 = _ag_step_dbg_print ? _p16_step_t0 : 0;

        /* Send 投入 */
        if (trec) _t0 = now_monotonic_ns();
        atomic_int send_pending;
        atomic_init(&send_pending, num_pieces);
        for (int p = 0; p < num_pieces; p++) {
            uint64_t p_off = (uint64_t)p * piece_size;
            uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
            bool piece_rail1 = use_dual_ring && (p & 1);
            /* Phase 15: ring_id に応じた ctx (rc.send / rc.send_rail1) と conn を選択 */
            struct doca_rdma_ctx_t *send_ctx = piece_rail1 ? rc.send_rail1 : rc.send;
            struct doca_rdma_connection *send_conn = piece_rail1 ? rc.conn_to_next_rail1 : rc.conn_to_next;

            if (full_gpu_direct && step == 0) {
                /* Phase 9: step 0 は GPU src から Send */
                init_ring_send_task(&task,
                    (void *)(remote_src_addr + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
                /* Phase 12.2: rail1 piece は src rail1 mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_src_local_mmap_rail1 : gpu_src_local_mmap;
            } else if (gpu_direct) {
                /* Phase 8/9: step 1+ は GPU dst から Send */
                init_ring_send_task(&task,
                    (void *)(remote_dst_addr + send_offset + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
                /* Phase 12.2: rail1 piece は dst rail1 mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_dst_local_mmap_rail1 : gpu_dst_local_mmap;
            } else {
                /* Legacy: DPU send_buf から Send */
                init_ring_send_task(&task,
                    (void *)((uintptr_t)send_buf + send_offset + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
            }
            task.inline_pending = &send_pending;
            task.pe_spin = piece_rail1 ? rc.send_pe_spin_rail1 : rc.send_pe_spin;
            submit_doca_task_from_desc(send_ctx, &task, NULL);
        }

        /* Phase 16: submit 完了時刻を常に記録 */
        uint64_t _p16_submit_done = now_monotonic_ns();
        if (trec) _t1 = _p16_submit_done;
        uint64_t _step_t1 = _ag_step_dbg_print ? _p16_submit_done : 0;
        while (atomic_load_explicit(&recv_pendings[step], memory_order_acquire) > 0 ||
               atomic_load_explicit(&send_pending, memory_order_acquire) > 0)
            progress_ring_pes_rc(&rc, use_dual_ring);
        uint64_t _p16_wait_done = now_monotonic_ns();
        /* Phase 16: 全 AG で step submit/wait を合計 */
        g_per_op_step_submit_ns += _p16_submit_done - _p16_step_t0;
        g_per_op_step_wait_ns   += _p16_wait_done - _p16_submit_done;
        if (trec && _oi < AG_TIMING_MAX_OPS) {
            uint64_t _t2 = _p16_wait_done;
            trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = _t1 - _t0, .wait_ns = _t2 - _t1, .name_idx = AG_OP_RING_SENDRECV };
            _oi++;
        }
        if (_ag_step_dbg_print && step < 16) {
            _ag_step_submit_ns[step] = _p16_submit_done - _step_t0;
            _ag_step_wait_ns[step]   = _p16_wait_done - _p16_submit_done;
        }

        /* Put: gpu_direct なら不要 (Recv が GPU dst に直接着地済み) */
        if (!gpu_direct) {
            if (trec) _t0 = now_monotonic_ns();
            atomic_int put_pending;
            atomic_init(&put_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *dst_rmem = (use_dual && (p & 1)) ? &cw->host_dst_rmem_rail1 : &cw->host_dst_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_write_task(&task,
                                (void *)((uintptr_t)send_buf + receive_offset + p_off),
                                (void *)(remote_dst_addr + receive_offset + p_off),
                                p_len, dst_rmem, rma_ctx, conn, p);
                task.inline_pending = &put_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task, NULL);
            }
            if (trec) _t1 = now_monotonic_ns();
            while (atomic_load_explicit(&put_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);
            if (trec && _oi < AG_TIMING_MAX_OPS) {
                uint64_t _t2 = now_monotonic_ns();
                trec->ops[_oi] = (struct ag_op_timing){ .submit_ns = _t1 - _t0, .wait_ns = _t2 - _t1, .name_idx = AG_OP_RING_PUT };
                _oi++;
            }
        }
    }

    /* Phase 12.2 G-3 / Phase 15: per-step breakdown 出力 (ring_id も併記) */
    if (_ag_step_dbg_print) {
        uint64_t total = 0;
        for (int s = 0; s < total_steps && s < 16; s++) total += _ag_step_submit_ns[s] + _ag_step_wait_ns[s];
        printf("[AG STEP rank=%lu ring=%d] size=%lukB pieces=%d psize=%.1fMB steps=%d gpu_direct=%d full=%d dualring=%d total=%.2fms\n",
               rank, ring_id, buffer_len / 1024, num_pieces,
               (double)piece_size / (1024.0 * 1024.0),
               total_steps, gpu_direct, full_gpu_direct, use_dual_ring, total / 1e6);
        for (int s = 0; s < total_steps && s < 16; s++) {
            printf("  [r%lu ring=%d sz=%lukB] step%d: submit=%6.2fus  wait=%8.2fus  total=%8.2fus\n",
                   rank, ring_id, buffer_len / 1024, s,
                   _ag_step_submit_ns[s] / 1e3,
                   _ag_step_wait_ns[s]   / 1e3,
                   (_ag_step_submit_ns[s] + _ag_step_wait_ns[s]) / 1e3);
        }
        fflush(stdout);
    }

    /* Record total */
    if (trec) {
        uint64_t total = 0;
        for (int i = 0; i < _oi; i++)
            total += trec->ops[i].submit_ns + trec->ops[i].wait_ns;
        trec->total_ns = total;
        trec->num_ops = _oi;
        g_ag_timing.count++;
    }

    /* DPU Union Tracker (ring-only): AG 終了 */
    dpu_union_tracker_exit(true);
    return DOCA_SUCCESS;
}

/* =====================================================
 * ComCh 通知ユーティリティ — UNCHANGED
 * ===================================================== */

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

/* =====================================================
 * 接続セットアップ関数
 * ===================================================== */

void execute_doca_connect_host_dpu_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    doca_error_t ret;

    cw->dev = sample_objects->hw_dev;
    cw->mpi_rank = (int)sample_objects->rank;

    /* RDMA 用ネットワークデバイスを開く (インターフェース名で指定、GID テーブルを持つデバイスを確実に取得) */
    ret = open_doca_device_with_iface_name(
        (const uint8_t *)sample_objects->device_name,
        strlen(sample_objects->device_name),
        NULL, &cw->rdma_dev);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open RDMA network device (iface='%s'): %s",
                     sample_objects->device_name, doca_error_get_name(ret));
        return;
    }
    DOCA_LOG_INFO("Opened RDMA network device (RMA): iface=%s", sample_objects->device_name);

    /* Ring 用デバイスを開く (常に enp3s0f0s0 = port 0 を使用) */
    const char *ring_iface = "enp3s0f0s0";
    if (strcmp(sample_objects->device_name, ring_iface) == 0) {
        /* RMA と同じポートなら共有 */
        cw->ring_dev = cw->rdma_dev;
        DOCA_LOG_INFO("Ring device shares RMA device: iface=%s", ring_iface);
    } else {
        ret = open_doca_device_with_iface_name(
            (const uint8_t *)ring_iface,
            strlen(ring_iface),
            NULL, &cw->ring_dev);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to open Ring network device (iface='%s'): %s",
                         ring_iface, doca_error_get_name(ret));
            return;
        }
        DOCA_LOG_INFO("Opened Ring network device: iface=%s", ring_iface);
    }

    /* Multi-Rail Ring: 2 番目のポート (enp3s0f1s0) を開く */
    const char *ring_iface_rail1 = "enp3s0f1s0";
    /* rdma_dev or rdma_dev_rail1 で既に開いている場合は共有 */
    if (strcmp(sample_objects->device_name, ring_iface_rail1) == 0) {
        cw->ring_dev_rail1 = cw->rdma_dev;
        DOCA_LOG_INFO("Ring rail1 device shares RMA device: iface=%s", ring_iface_rail1);
    } else {
        /* rdma_dev_rail1 が後で開かれるので、ここでは別途開く */
        ret = open_doca_device_with_iface_name(
            (const uint8_t *)ring_iface_rail1,
            strlen(ring_iface_rail1),
            NULL, &cw->ring_dev_rail1);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Multi-Rail Ring: failed to open ring_dev_rail1 (iface='%s'): %s",
                          ring_iface_rail1, doca_error_get_name(ret));
            cw->ring_dev_rail1 = NULL;
        } else {
            DOCA_LOG_INFO("Opened Ring rail1 network device: iface=%s", ring_iface_rail1);
        }
    }

    /* Host からの DOCA RDMA 接続記述子をコピーして保存 */
    size_t desc_len = recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;
    void *desc_copy = malloc(desc_len);
    if (!desc_copy) { DOCA_LOG_ERR("Failed to alloc host conn_desc copy"); return; }
    memcpy(desc_copy, recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address, desc_len);
    cw->host_rma_conn_desc = desc_copy;
    cw->host_rma_conn_desc_len = desc_len;

    /* Multi-Rail: Host から rail1 接続記述子が来ているか確認 */
    size_t desc_len_rail1 = recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
    if (desc_len_rail1 > 0) {
        void *desc_copy_rail1 = malloc(desc_len_rail1);
        if (desc_copy_rail1) {
            memcpy(desc_copy_rail1, recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_rail1, desc_len_rail1);
            cw->host_rma_conn_desc_rail1 = desc_copy_rail1;
            cw->host_rma_conn_desc_len_rail1 = desc_len_rail1;
        }
    }

    /* ---- Working buffer 確保 (create_ring で使うバッファプールの元) ---- */
    /* doca_mmap は 64B アライメントを要求するため aligned_alloc を使用 */
    size_t slot_size = CW_SLOT_DEFAULT_BYTES;
    size_t pool_total = slot_size * CW_BUFFER_POOL_SLOTS * 2;  /* send + recv */
    /* Phase 12.1 Plan A: working_buf 末尾に RDB barrier の slot 領域を 4KB 確保。
     * これにより buffer pool と RDB slot が衝突しないことを保証する。 */
    size_t total_size = pool_total + 4096;
    void *working_buf = aligned_alloc(64, total_size);
    if (!working_buf) { DOCA_LOG_ERR("Failed to allocate working buffer"); free(desc_copy); return; }
    memset(working_buf, 0, total_size);
    cw->working_buf = working_buf;
    cw->working_buf_size = total_size;

    /* ---- PCI mmap キャッシュ初期化 (rail0) ---- */
    if (!cw->gpu_dst_pci_cache) cw->gpu_dst_pci_cache = pci_mmap_cache_alloc();
    if (!cw->gpu_src_pci_cache) cw->gpu_src_pci_cache = pci_mmap_cache_alloc();
    /* Phase 12.2: rail1 用の PCI mmap キャッシュ */
    if (!cw->gpu_dst_pci_cache_rail1) cw->gpu_dst_pci_cache_rail1 = pci_mmap_cache_alloc();
    if (!cw->gpu_src_pci_cache_rail1) cw->gpu_src_pci_cache_rail1 = pci_mmap_cache_alloc();

    /* ---- RMA (Host-DPU Read/Write) RDMA コンテキスト作成 (永続) ---- */
    ret = doca_rdma_ctx_init(&cw->rdma_rma, cw->rdma_dev, NULL,
                              working_buf, total_size,
                              DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                              64, 0,   /* recv_q_size=0: RMA は Read/Write のみ */
                              1);      /* gid_index=1: RoCE v2 (DPU 側) */
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma init failed: %s", doca_error_get_name(ret)); return; }

    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_rma, 32, 32, 0, 0,
                                             rdma_read_comp_cb, rdma_read_err_cb,
                                             rdma_write_comp_cb, rdma_write_err_cb,
                                             NULL, NULL, NULL, NULL,
                                             NULL);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma configure failed"); return; }

    /* Export + Connect */
    ret = doca_rdma_ctx_export(&cw->rdma_rma, 0);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma export failed"); return; }

    ret = doca_rdma_ctx_connect(&cw->rdma_rma, 0,
                                 cw->host_rma_conn_desc, cw->host_rma_conn_desc_len);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma connect to Host failed: %s", doca_error_get_name(ret)); return; }
    cw->host_conn = cw->rdma_rma.connections[0];

    /* ---- Multi-Rail: 2 番目のポートの RMA RDMA コンテキスト作成 ---- */
    struct control_notify send_notify;
    send_notify.type = CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU;
    send_notify.ucp_connect_host_dpu.id = recv_cmd->ucp_connect_host_dpu.id;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len = cw->rdma_rma.local_conn_desc_len;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address = (void *)cw->rdma_rma.local_conn_desc;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = NULL;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 = 0;

    if (desc_len_rail1 > 0) {
        /* 他方のポートを特定して開く */
        const char *other_iface = NULL;
        if (strcmp(sample_objects->device_name, "enp3s0f0s0") == 0)
            other_iface = "enp3s0f1s0";
        else
            other_iface = "enp3s0f0s0";

        ret = open_doca_device_with_iface_name(
            (const uint8_t *)other_iface, strlen(other_iface),
            NULL, &cw->rdma_dev_rail1);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Multi-Rail: failed to open rail1 device (iface='%s'): %s, continuing single-rail",
                          other_iface, doca_error_get_name(ret));
        } else {
            DOCA_LOG_INFO("Multi-Rail: opened rail1 device: iface=%s", other_iface);

            /* rail1 RMA RDMA コンテキスト作成 (working_buf を共有) */
            ret = doca_rdma_ctx_init(&cw->rdma_rma_rail1, cw->rdma_dev_rail1, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, 1);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 init failed: %s", doca_error_get_name(ret));
            } else {
                ret = doca_rdma_ctx_configure_and_start(&cw->rdma_rma_rail1, 32, 32, 0, 0,
                                                         rdma_read_comp_cb, rdma_read_err_cb,
                                                         rdma_write_comp_cb, rdma_write_err_cb,
                                                         NULL, NULL, NULL, NULL, NULL);
                if (ret != DOCA_SUCCESS) {
                    DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 configure failed");
                } else {
                    ret = doca_rdma_ctx_export(&cw->rdma_rma_rail1, 0);
                    if (ret != DOCA_SUCCESS) {
                        DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 export failed");
                    } else {
                        ret = doca_rdma_ctx_connect(&cw->rdma_rma_rail1, 0,
                                                     cw->host_rma_conn_desc_rail1,
                                                     cw->host_rma_conn_desc_len_rail1);
                        if (ret != DOCA_SUCCESS) {
                            DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 connect failed: %s", doca_error_get_name(ret));
                        } else {
                            cw->host_conn_rail1 = cw->rdma_rma_rail1.connections[0];
                            cw->dual_rail = true;
                            DOCA_LOG_INFO("Multi-Rail: rail1 RMA connection established");

                            /* rail1 の接続記述子を通知に含める */
                            send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 =
                                (void *)cw->rdma_rma_rail1.local_conn_desc;
                            send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 =
                                cw->rdma_rma_rail1.local_conn_desc_len;
                        }
                    }
                }
            }
        }
    }

    /* ---- Phase 5 Step 3: Host doorbell remote mmap のインポート ---- */
    cw->doorbell_enabled = false;
    cw->doorbell_seq = 0;
    if (recv_cmd->ucp_connect_host_dpu.doorbell_export_desc_len > 0) {
        doca_error_t dbret = doca_remote_mem_create(&cw->host_doorbell_rmem, cw->rdma_dev,
            recv_cmd->ucp_connect_host_dpu.doorbell_export_desc,
            recv_cmd->ucp_connect_host_dpu.doorbell_export_desc_len);
        if (dbret == DOCA_SUCCESS) {
            cw->doorbell_enabled = true;
            DOCA_LOG_INFO("Doorbell remote mmap imported (addr=0x%lx, len=%zu)",
                          cw->host_doorbell_rmem.remote_addr, cw->host_doorbell_rmem.remote_len);
        } else {
            DOCA_LOG_WARN("Doorbell remote_mem create failed: %s", doca_error_get_name(dbret));
        }
    }

    /* ---- Phase 5 Step 4: RDMA command slot 作成 (Host RDMA Write 受信用) ---- */
    cw->cmd_slot_enabled = false;
    cw->cmd_slot_buf = aligned_alloc(64, CMD_SLOT_SIZE);
    if (cw->cmd_slot_buf) {
        memset(cw->cmd_slot_buf, 0, CMD_SLOT_SIZE);
        cw->cmd_expected_seq = 1;  /* first expected seq */

        doca_error_t mret = doca_mmap_create(&cw->cmd_slot_mmap);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_add_dev(cw->cmd_slot_mmap, cw->rdma_dev);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_set_memrange(cw->cmd_slot_mmap, cw->cmd_slot_buf, CMD_SLOT_SIZE);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_set_permissions(cw->cmd_slot_mmap,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_start(cw->cmd_slot_mmap);
        if (mret == DOCA_SUCCESS) {
            mret = doca_mmap_export_rdma(cw->cmd_slot_mmap, cw->rdma_dev,
                                          &cw->cmd_slot_export_desc, &cw->cmd_slot_export_desc_len);
        }
        if (mret == DOCA_SUCCESS) {
            cw->cmd_slot_enabled = true;
            DOCA_LOG_INFO("RDMA command slot ready (%zu bytes export)", cw->cmd_slot_export_desc_len);
        } else {
            DOCA_LOG_WARN("RDMA command slot setup failed, using ComCh fallback");
        }
    }

    /* Phase 5 Step 4: Include cmd_slot export desc in notify */
    if (cw->cmd_slot_enabled) {
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc = (void *)cw->cmd_slot_export_desc;
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc_len = cw->cmd_slot_export_desc_len;
    } else {
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc = NULL;
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc_len = 0;
    }

    /* 通知: DPU の接続記述子を Host に返送 */
    doca_error_t result = comch_send_control_notify(&send_notify, sample_objects);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send control notify");
    }

    DOCA_LOG_INFO("Host-DPU RDMA connection established (rank %d, dual_rail=%d, doorbell=%d, cmd_slot=%d)",
                  cw->mpi_rank, cw->dual_rail, cw->doorbell_enabled, cw->cmd_slot_enabled);
}

void execute_doca_create_ring_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    doca_error_t ret;
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t rank = sample_objects->rank;
    uint64_t world_size = sample_objects->world_size;

    /* Phase 15: per-ring job queues を初期化 (mutex/cv/stop flag) */
    if (ring_queues_init(cw) != 0) {
        DOCA_LOG_ERR("[Phase15] ring_queues_init failed");
        return;
    }

    /* ---- 1. バッファプール初期化 (working buffer は connect_host_dpu で確保済み) ---- */
    size_t slot_size = CW_SLOT_DEFAULT_BYTES;
    void *working_buf = cw->working_buf;
    size_t total_size = cw->working_buf_size;
    if (!working_buf) { DOCA_LOG_ERR("Working buffer not allocated (connect_host_dpu not called?)"); return; }

    cw_buffer_pool_init(&cw->send_pool, working_buf, slot_size);
    cw_buffer_pool_init(&cw->recv_pool, (char *)working_buf + slot_size * CW_BUFFER_POOL_SLOTS, slot_size);

    /* ---- 2. rdma_rma は connect_host_dpu で作成・接続済み ---- */

    /* ---- 3. Ring (DPU-DPU Send/Recv) RDMA コンテキスト作成 + 接続 ---- */
    for (size_t i = 0; i < world_size; i++) {
        if (rank != i && rank != ((i - 1 + world_size) % world_size))
            goto ring_barrier;

        if (rank == i) {
            /* rank i は recv 側: rdma_ring_recv を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_recv, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, 1);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv init failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Phase 12.2-G P1: num_recv_tasks=32 → 256 に拡大。
             * pmax=32 (最大 19 piece) + 4 step + dual-rail で 1 rail あたり 40 outstanding
             * となるため、32 を超えてプール枯渇 → DOCA_ERROR_NO_MEMORY を発生させていた。
             * 256 にすれば pmax=64 (32 piece) でも 32×4/2=64 outstanding で十分余裕。 */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv, 0, 0, 0, 256,
                                                     NULL, NULL, NULL, NULL,
                                                     NULL, NULL,
                                                     rdma_recv_comp_cb, rdma_recv_err_cb,
                                                     NULL);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv configure failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_recv, 0);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv export failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Multi-Rail Ring: rail1 recv コンテキスト作成 */
            bool ring_rail1_recv_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, 1);
                if (ret == DOCA_SUCCESS) {
                    /* Phase 12.2-G P1: rail1 も同様に 32 → 256 */
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_rail1, 0, 0, 0, 256,
                                                             NULL, NULL, NULL, NULL,
                                                             NULL, NULL,
                                                             rdma_recv_comp_cb, rdma_recv_err_cb,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_rail1_recv_ok = true;
                        } else { DOCA_LOG_WARN("ring_recv_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("ring_recv_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("ring_recv_rail1 init failed"); }
            }

            /* MPI で接続記述子を交換 */
            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_recv.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_recv.local_conn_desc_len;
            if (ring_rail1_recv_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_recv_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_recv_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("MPI exchange failed"); free(recv_storage); MPI_Barrier(MPI_COMM_WORLD); continue; }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("ring_recv connect failed");
                cw->conn_from_prev = cw->rdma_ring_recv.connections[0];

                /* rail1 接続 */
                if (ring_rail1_recv_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("ring_recv_rail1 connect failed");
                    else cw->conn_from_prev_rail1 = cw->rdma_ring_recv_rail1.connections[0];
                }
            }
            free(recv_storage);
        } else {
            /* rank (i-1) は send 側: rdma_ring_send を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_send, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, 1);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send init failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Phase 12.1 Plan A: num_write_tasks=4 を追加 (RDB barrier の RDMA Write 用)
             * Phase 12.2-G P1: num_send_tasks=32 → 256 (recv 側と対称) */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send, 0, 4, 256, 0,
                                                     NULL, NULL,
                                                     rdma_write_comp_cb, rdma_write_err_cb,
                                                     rdma_send_comp_cb, rdma_send_err_cb,
                                                     NULL, NULL,
                                                     NULL);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send configure failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_send, 0);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send export failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Multi-Rail Ring: rail1 send コンテキスト作成 */
            bool ring_rail1_send_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_send_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, 1);
                if (ret == DOCA_SUCCESS) {
                    /* rail1 は RDB を使わないため Write タスクは 0 のまま
                     * Phase 12.2-G P1: num_send_tasks=32 → 256 */
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_rail1, 0, 0, 256, 0,
                                                             NULL, NULL, NULL, NULL,
                                                             rdma_send_comp_cb, rdma_send_err_cb,
                                                             NULL, NULL,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_send_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_rail1_send_ok = true;
                        } else { DOCA_LOG_WARN("ring_send_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("ring_send_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("ring_send_rail1 init failed"); }
            }

            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_send.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_send.local_conn_desc_len;
            if (ring_rail1_send_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_send_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_send_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("MPI exchange failed"); free(recv_storage); MPI_Barrier(MPI_COMM_WORLD); continue; }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_send, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("ring_send connect failed");
                cw->conn_to_next = cw->rdma_ring_send.connections[0];

                /* rail1 接続 */
                if (ring_rail1_send_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("ring_send_rail1 connect failed");
                    else cw->conn_to_next_rail1 = cw->rdma_ring_send_rail1.connections[0];
                }
            }
            free(recv_storage);
        }
    ring_barrier:
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* ========================================================================
     * Phase 15: Ring 1 (parallel second ring)
     * Ring 0 と完全に同じパターンで 2 本目の Ring を構築する。
     * 重要:
     *   - MPI exchange tag を 0x501 に変更して Ring 0 と衝突しないようにする
     *   - 各 ctx は _r1 suffix を持つ別フィールド
     *   - conn/PE spinlock も独立
     *   - 失敗した場合は ring_r1_enabled=false で Ring 0 のみにフォールバック
     * ======================================================================== */
    cw->ring_r1_enabled = false;
    cw->ring_r1_rail1_enabled = false;
    pe_spin_init(&cw->ring_send_pe_spin_r1);
    pe_spin_init(&cw->ring_send_pe_spin_r1_rail1);
    pe_spin_init(&cw->ring_recv_pe_spin_r1);
    pe_spin_init(&cw->ring_recv_pe_spin_r1_rail1);

    bool ring_r1_setup_ok = true;
    for (size_t i = 0; i < world_size; i++) {
        if (rank != i && rank != ((i - 1 + world_size) % world_size))
            goto ring_r1_barrier;

        if (rank == i) {
            /* rank i は recv 側 (Ring 1): rdma_ring_recv_r1 を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_r1, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, 1);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 init failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Phase 12.2-G P1: num_recv_tasks=256 */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_r1, 0, 0, 0, 256,
                                                     NULL, NULL, NULL, NULL,
                                                     NULL, NULL,
                                                     rdma_recv_comp_cb, rdma_recv_err_cb,
                                                     NULL);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 configure failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_r1, 0);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 export failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Multi-Rail Ring 1: rail1 recv コンテキスト */
            bool ring_r1_rail1_recv_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_r1_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, 1);
                if (ret == DOCA_SUCCESS) {
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_r1_rail1, 0, 0, 0, 256,
                                                             NULL, NULL, NULL, NULL,
                                                             NULL, NULL,
                                                             rdma_recv_comp_cb, rdma_recv_err_cb,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_r1_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_r1_rail1_recv_ok = true;
                        } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 init failed"); }
            }

            /* MPI で接続記述子を交換 (tag 0x501) */
            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_recv_r1.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_recv_r1.local_conn_desc_len;
            if (ring_r1_rail1_recv_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_recv_r1_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_recv_r1_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd_tagged(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len,
                                            0x501);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] MPI exchange failed: %s", doca_error_get_name(ret));
                free(recv_storage); ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_r1, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("[R1] ring_recv_r1 connect failed: %s", doca_error_get_name(ret)); ring_r1_setup_ok = false; }
                else cw->conn_from_prev_r1 = cw->rdma_ring_recv_r1.connections[0];

                if (ring_r1_rail1_recv_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_r1_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 connect failed");
                    else cw->conn_from_prev_r1_rail1 = cw->rdma_ring_recv_r1_rail1.connections[0];
                }
            }
            free(recv_storage);
        } else {
            /* rank (i-1) は send 側 (Ring 1): rdma_ring_send_r1 を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_send_r1, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, 1);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 init failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Ring 1 は RDB を使わないので write tasks は 0 でよい。num_send_tasks=256 */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_r1, 0, 0, 256, 0,
                                                     NULL, NULL, NULL, NULL,
                                                     rdma_send_comp_cb, rdma_send_err_cb,
                                                     NULL, NULL,
                                                     NULL);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 configure failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_send_r1, 0);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 export failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Multi-Rail Ring 1: rail1 send コンテキスト */
            bool ring_r1_rail1_send_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_send_r1_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, 1);
                if (ret == DOCA_SUCCESS) {
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_r1_rail1, 0, 0, 256, 0,
                                                             NULL, NULL, NULL, NULL,
                                                             rdma_send_comp_cb, rdma_send_err_cb,
                                                             NULL, NULL,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_send_r1_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_r1_rail1_send_ok = true;
                        } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 init failed"); }
            }

            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_send_r1.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_send_r1.local_conn_desc_len;
            if (ring_r1_rail1_send_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_send_r1_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_send_r1_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd_tagged(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len,
                                            0x501);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] MPI exchange failed: %s", doca_error_get_name(ret));
                free(recv_storage); ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_r1, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("[R1] ring_send_r1 connect failed: %s", doca_error_get_name(ret)); ring_r1_setup_ok = false; }
                else cw->conn_to_next_r1 = cw->rdma_ring_send_r1.connections[0];

                if (ring_r1_rail1_send_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_r1_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("[R1] ring_send_r1_rail1 connect failed");
                    else cw->conn_to_next_r1_rail1 = cw->rdma_ring_send_r1_rail1.connections[0];
                }
            }
            free(recv_storage);
        }
    ring_r1_barrier:
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Ring 1 の接続が両方向成功していれば有効化 */
    if (ring_r1_setup_ok && cw->conn_to_next_r1 && cw->conn_from_prev_r1) {
        cw->ring_r1_enabled = true;
        if (cw->conn_to_next_r1_rail1 && cw->conn_from_prev_r1_rail1) {
            cw->ring_r1_rail1_enabled = true;
        }
        if (rank == 0) {
            printf("[Phase15] Ring 1 enabled (rail1=%s)\n",
                   cw->ring_r1_rail1_enabled ? "yes" : "no");
            fflush(stdout);
        }
    } else {
        if (rank == 0) {
            printf("[Phase15] Ring 1 setup failed — falling back to Ring 0 only\n");
            fflush(stdout);
        }
    }

    /* ========================================================================
     * Phase 15: Per-ring MPI communicator の作成
     * Ring 0 / Ring 1 それぞれに MPI_Comm_dup で独立した通信子を割り当てる。
     * これにより MPI_Barrier フォールバック時に ring 間で barrier が混線しない。
     *
     * 重要: MPI_Comm_dup は collective 操作なので、全 rank が同じ順序で呼ぶ必要がある。
     * また Ring 1 が無効でも全 rank で Comm_dup を呼ぶ必要がある
     * (一部 rank のみ Comm_dup しないとハングする)。
     * ======================================================================== */
    for (int ri = 0; ri < N_RINGS; ri++) {
        cw->ring_comm[ri] = MPI_COMM_NULL;
        cw->ring_comm_valid[ri] = false;
    }
    {
        int dup_rc = MPI_Comm_dup(MPI_COMM_WORLD, &cw->ring_comm[0]);
        if (dup_rc == MPI_SUCCESS) {
            cw->ring_comm_valid[0] = true;
        } else {
            DOCA_LOG_WARN("[Phase15] MPI_Comm_dup for Ring 0 failed, using MPI_COMM_WORLD fallback");
        }
    }
    {
        int dup_rc = MPI_Comm_dup(MPI_COMM_WORLD, &cw->ring_comm[1]);
        if (dup_rc == MPI_SUCCESS) {
            cw->ring_comm_valid[1] = true;
        } else {
            DOCA_LOG_WARN("[Phase15] MPI_Comm_dup for Ring 1 failed, using MPI_COMM_WORLD fallback");
        }
    }
    if (rank == 0) {
        printf("[Phase15] Per-ring MPI_Comm created (ring0=%s ring1=%s)\n",
               cw->ring_comm_valid[0] ? "ok" : "fallback",
               cw->ring_comm_valid[1] ? "ok" : "fallback");
        fflush(stdout);
    }

    /* Phase 12.2 G: piece config (env-tunable) を初回 1 度だけ初期化・出力 */
    ag_piece_config_init_once();

    /* ---- Phase 12.1 Plan A: RDB barrier 初期化 (Ring 接続確立後) ----
     * MPI_Barrier (avg 325μs / max 281ms) を 2-pass Ring barrier (~40μs) に置換。
     * 失敗時は cw->rdb_enabled=false で MPI_Barrier フォールバック。 */
    cw->rdb_enabled = false;
    cw->rdb_my_seq = 0;
    cw->rdb_local_slot = NULL;
    cw->rdb_send_data = NULL;
    cw->rdb_next_remote_slot_addr = 0;
    cw->rdb_total_count = 0;
    cw->rdb_total_ns = 0;
    cw->rdb_max_ns = 0;
    {
        doca_error_t rdb_ret = rdb_init(cw, (int)rank, (int)world_size);
        if (rdb_ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("rdb_init returned %s, falling back to MPI_Barrier",
                          doca_error_get_name(rdb_ret));
        }
    }

#if AG_TIMING_ENABLED
    /* ---- Phase 7 Step 0-D: DOCA API micro-benchmark (ワーカ起動前に 1 回だけ) ---- */
    {
        static bool microbench_done = false;
        if (!microbench_done) {
            run_doca_api_microbenchmark(cw);
            microbench_done = true;
        }
    }
#endif

    /* ---- 4. ワーカスレッドプール初期化 ---- */
    if (cw->worker_pool == NULL) {
        cw->worker_pool = (struct doca_worker_thread_pool_t *)calloc(1, sizeof(struct doca_worker_thread_pool_t));
        if (cw->worker_pool) {
            int pool_ret = doca_worker_thread_pool_init(cw->worker_pool, cw, (int)rank);
            if (pool_ret != 0) {
                DOCA_LOG_ERR("Failed to init DOCA worker thread pool");
                free(cw->worker_pool);
                cw->worker_pool = NULL;
            } else {
                printf("[Rank%lu] DOCA worker thread pool initialized\n", rank);
            }
        }
    }

    /* ---- 5. RS スレッドプール初期化 ---- */
    if (cw->rs_pool == NULL) {
        ret = rs_thread_pool_create(&cw->rs_pool, g_compute_cores);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create rs_thread_pool");
            cw->rs_pool = NULL;
        } else {
            printf("[Rank%lu] RS aggregation thread pool initialized\n", rank);
        }
    }

    /* ---- 6. 通知送信 ---- */
    struct control_notify send_notify;
    send_notify.type = CONTROL_NOTIFY_UCP_CREATE_RING;
    send_notify.ucp_create_ring.id = recv_cmd->ucp_create_ring.id;
    ret = comch_send_control_notify(&send_notify, sample_objects);
    if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to send ring notify");
    fflush(stdout);
}

/* =====================================================
 * Phase 14: GPU flag pool init handler
 *   ホストの GpuFlagPool 用 RDMA export を受け取り、DPU 側で remote_mem として保持する。
 *   1 度だけ呼ばれる (起動時の Python init)。
 *   AG 完了時に flag_pool_rmem を target に inline RDMA Write を発行する。
 * ===================================================== */
void execute_doca_init_flag_pool_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t base_addr = recv_cmd->ucp_init_flag_pool.base_addr;
    uint64_t length    = recv_cmd->ucp_init_flag_pool.length;
    void    *rkey_buf  = recv_cmd->ucp_init_flag_pool.rkey_buf;
    uint64_t rkey_len  = recv_cmd->ucp_init_flag_pool.rkey_buf_len;

    if (cw->flag_pool_enabled) {
        DOCA_LOG_WARN("flag pool already enabled, ignoring re-init");
        return;
    }
    if (base_addr == 0 || length == 0 || rkey_len == 0) {
        DOCA_LOG_ERR("invalid INIT_FLAG_POOL: addr=0x%lx len=%lu rkey_len=%lu",
                     base_addr, length, rkey_len);
        return;
    }

    /* rail0: DPU の RMA RDMA dev (= host_conn 経由で host GPU に RDMA Write) で remote_mem を作成 */
    doca_error_t ret = doca_remote_mem_create(&cw->flag_pool_rmem, cw->rdma_dev,
                                              rkey_buf, rkey_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("flag pool remote_mem create rail0 failed: %s", doca_error_get_descr(ret));
        return;
    }

    /* rail1 (best effort) — host が rail1 export desc を送ってきていれば */
    if (cw->dual_rail && cw->rdma_dev_rail1 &&
        recv_cmd->ucp_init_flag_pool.rkey_buf_rail1 &&
        recv_cmd->ucp_init_flag_pool.rkey_buf_len_rail1 > 0) {
        doca_error_t r1 = doca_remote_mem_create(&cw->flag_pool_rmem_rail1, cw->rdma_dev_rail1,
                                                  recv_cmd->ucp_init_flag_pool.rkey_buf_rail1,
                                                  recv_cmd->ucp_init_flag_pool.rkey_buf_len_rail1);
        cw->flag_pool_rail1_enabled = (r1 == DOCA_SUCCESS);
        if (!cw->flag_pool_rail1_enabled) {
            DOCA_LOG_WARN("flag pool remote_mem create rail1 failed");
        }
    }

    /* ---- Phase 15: Cross-GVMI PCI import (optional) ----
     *   host が pci_export_buf を送ってきていれば、そちらを local mmap として
     *   import する。phase14_write_flag はこれを local_mmap_override として
     *   使い、GPU memory に PCIe 経由で直接 Write する。
     *
     *   重要: ring_dev と rdma_dev のどちらを使うかは用途によるが、既存の
     *   GPU dst PCI import は ring_dev を使っている。同じパターンに合わせる。
     *   ただしここで問題は: rdma_rma ctx は rdma_dev で作られており、
     *   ring_dev の local mmap とはデバイスが違う。
     *
     *   → 最小変更方針: rdma_dev で import する (rdma_rma ctx と整合)。
     *      ring_dev import は不要 (phase14_write_flag は rdma_rma を使うため)。 */
    cw->flag_pool_pci_mmap = NULL;
    cw->flag_pool_pci_mmap_rail1 = NULL;
    cw->flag_pool_pci_enabled = false;
    if (recv_cmd->ucp_init_flag_pool.pci_export_buf &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_len > 0) {
        doca_error_t pret = doca_mmap_create_from_export(NULL,
            recv_cmd->ucp_init_flag_pool.pci_export_buf,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_len,
            cw->rdma_dev, &cw->flag_pool_pci_mmap);
        if (pret == DOCA_SUCCESS) {
            cw->flag_pool_pci_enabled = true;
            if (sample_objects->rank == 0) {
                printf("[Phase15 DPU] flag pool PCI import rail0 SUCCESS (len=%lu)\n",
                       recv_cmd->ucp_init_flag_pool.pci_export_buf_len);
                fflush(stdout);
            }
        } else {
            cw->flag_pool_pci_mmap = NULL;
            DOCA_LOG_WARN("flag pool PCI import rail0 failed: %s (fallback to rdma remote_mem)",
                          doca_error_get_descr(pret));
        }
    }
    if (cw->flag_pool_pci_enabled &&
        cw->rdma_dev_rail1 &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_rail1 &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1 > 0) {
        doca_error_t pret = doca_mmap_create_from_export(NULL,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_rail1,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1,
            cw->rdma_dev_rail1, &cw->flag_pool_pci_mmap_rail1);
        if (pret != DOCA_SUCCESS) {
            cw->flag_pool_pci_mmap_rail1 = NULL;
            DOCA_LOG_WARN("flag pool PCI import rail1 failed: %s",
                          doca_error_get_descr(pret));
        }
    }

    /* DPU 側送信バッファ: 各 AG が 4 バイトの flag_value を inline でなく
     * バッファ経由で書き込む際に使用するリング状プール (256 エントリ)。
     * 256 個までの並行 AG が衝突せず使える。
     *
     * 重要: RDMA Write の local source は **DPU の登録済み mmap 内** に
     * なければならない。`aligned_alloc` で確保した plain heap は登録されておらず、
     * NIC が読めないため "Input/Output Operation Failed" となる。
     * → working_buf 末尾の RDB 領域 (192B) より手前 4KB を流用する。
     *   working_buf は rdma_rma の local mmap として登録済みなので NIC が読める。 */
    const size_t pool_entries = 256;
    const size_t flag_pool_bytes = pool_entries * 16;  /* 16 byte/entry (4B data + padding) = 4KB */
    if (cw->working_buf_size < (size_t)RDB_SLOT_REGION_SIZE + flag_pool_bytes) {
        DOCA_LOG_ERR("working_buf too small for flag pool: %zu < %zu",
                     cw->working_buf_size,
                     (size_t)RDB_SLOT_REGION_SIZE + flag_pool_bytes);
        doca_remote_mem_destroy(&cw->flag_pool_rmem);
        return;
    }
    /* Layout (working_buf 末尾):
     *   [working_buf - RDB_SLOT_REGION_SIZE - flag_pool_bytes ... -RDB_SLOT_REGION_SIZE)  ← flag pool
     *   [working_buf - RDB_SLOT_REGION_SIZE ... -1]                                       ← RDB
     */
    size_t flag_pool_offset = cw->working_buf_size - (size_t)RDB_SLOT_REGION_SIZE - flag_pool_bytes;
    cw->flag_send_buf_pool = (char *)cw->working_buf + flag_pool_offset;
    cw->flag_send_buf_pool_len = flag_pool_bytes;
    memset(cw->flag_send_buf_pool, 0, cw->flag_send_buf_pool_len);
    atomic_store(&cw->flag_send_buf_idx, 0);

    cw->flag_pool_enabled = true;
    cw->flag_write_total_count = 0;
    cw->flag_write_total_ns    = 0;
    cw->flag_write_max_ns      = 0;
    /* Phase 15: Ring 0 / Ring 1 の両 thread が同時に flag write を発行する可能性が
     * あるため、rdma_rma ctx への書き込みを mutex で直列化する。 */
    pthread_mutex_init(&cw->flag_write_mtx, NULL);

    if (sample_objects->rank == 0) {
        printf("[Phase15 DPU] flag pool initialized: addr=0x%lx len=%lu rail1=%s pool_entries=%zu pci_path=%s\n",
               base_addr, length,
               cw->flag_pool_rail1_enabled ? "yes" : "no", pool_entries,
               cw->flag_pool_pci_enabled ? "YES (Cross-GVMI)" : "no (legacy RDMA rkey)");
        fflush(stdout);
    }
}

/* Phase 14/15: AG 完了時に flag_value を flag_gpu_addr に書き込む。
 * RMA RDMA Write を発行 (ホスト GPU 上の int32 スロットを直接更新)。
 *
 * Phase 14 (legacy): flag は pinned host memory 上にあり、DPU は rdma_rma ctx 経由で
 *   host memory に RDMA Write する。GPU は cuStreamWaitValue32 で pinned host を poll
 *   する (~1.8ms/AG のストール → fwd に蓄積)。
 *
 * Phase 15: flag は GPU memory 上にあり、host 側 mmap は PCI_READ_WRITE permission
 *   付きで export されている (AG dst と同じパターン)。doca_remote_mem_create から
 *   得た flag_pool_rmem に対する RDMA Write が Cross-GVMI で GPU memory に直接着地する。
 *   GPU 側 cuStreamWaitValue32 は GPU-local な semaphore として即座に解放される。
 *   (fwd の ~168ms ストールが消える想定)
 *
 * 順序保証: 直前の AG データ書き込みは PCIe 書き込みとして既に submit 済み。
 * doca_task_submit は同 ctx 内で submit 順序を保つ。
 * よって compute stream が flag を見た時には AG データが GPU 上に揃っている。
 */
static inline void phase14_write_flag(struct collective_worker_t *cw,
                                       uint64_t flag_gpu_addr, uint32_t flag_value)
{
    if (!cw->flag_pool_enabled || flag_gpu_addr == 0) return;

    /* Phase 15: 2 つの ring proc thread が同時に flag write するケースに備え、
     * rdma_rma ctx への submit を mutex で直列化する。critical section は μs 規模なので
     * throughput への影響は小さい。 */
    pthread_mutex_lock(&cw->flag_write_mtx);

    uint64_t _t0 = now_monotonic_ns();

    /* リング状送信バッファプールから次エントリを取得 (16 byte/entry, 256 エントリ) */
    unsigned idx = atomic_fetch_add_explicit(&cw->flag_send_buf_idx, 1, memory_order_relaxed);
    idx = idx & 0xFF;  /* 256 エントリ */
    uint32_t *send_slot = (uint32_t *)((char *)cw->flag_send_buf_pool + (size_t)idx * 16);
    *send_slot = flag_value;

    /* RDMA Write 発行 (rail0 を使用、4 バイト)。
     * 注意: flag_pool_rmem は host 側 mmap (PCI_READ_WRITE 付) から export された
     * rdma rkey で作成されている。GPU memory 上の flag でも Cross-GVMI で着地する。
     * AG dst の host_dst_rmem と同じパターン。 */
    struct doca_task_desc write_task;
    init_write_task(&write_task,
                    send_slot,
                    (void *)flag_gpu_addr,
                    sizeof(uint32_t),
                    &cw->flag_pool_rmem,
                    &cw->rdma_rma, cw->host_conn, 0);
    write_task.pe_spin = &cw->rma_pe_spin;
    submit_and_wait_doca(cw->worker_pool, &write_task);

    uint64_t _t1 = now_monotonic_ns();
    uint64_t ns = _t1 - _t0;
    cw->flag_write_total_count++;
    cw->flag_write_total_ns += ns;
    if (ns > cw->flag_write_max_ns) cw->flag_write_max_ns = ns;
    if ((cw->flag_write_total_count % 5000) == 0) {
        printf("[FLAG WRITE PROF] ops=%lu | avg=%.1fus max=%.1fus\n",
               cw->flag_write_total_count,
               (double)cw->flag_write_total_ns / cw->flag_write_total_count / 1000.0,
               (double)cw->flag_write_max_ns / 1000.0);
        fflush(stdout);
    }

    pthread_mutex_unlock(&cw->flag_write_mtx);
}

void execute_doca_collective_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects, uint64_t t_recv_ns, int ring_id)
{
    doca_error_t result;
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t id = recv_cmd->ucp_collective.id;
    /* Phase 15: ring_id は AG のみに影響。RS は常に Ring 0 を使う (collective_reduce_scatter は
     * ring_id を取らず内部で cw->rdma_ring_send/recv を直接参照する)。 */

    /* ---- Timing instrumentation ---- */
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t t_entry_ns = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;

    /* Phase 17: inflight counter を inc + 入口時点の値を capture。
     * 注意: これより後の全 return path で dec する必要がある。
     * safer pattern: cleanup ブロックで dec する (下記) */
    int inflight_at_entry = atomic_fetch_add_explicit(&g_dpu_ag_inflight, 1, memory_order_relaxed) + 1;

    /* Phase 12 Calc C: per-op timing — barrier 計測用 TLS をリセット */
    g_per_op_barrier_ns = 0;
    /* Phase 16: ring 内部 breakdown TLS を初期化 (RS 時は AG 由来の古い値が残らないように) */
    g_per_op_recv_post_ns = 0;
    g_per_op_step_submit_ns = 0;
    g_per_op_step_wait_ns = 0;
    g_per_op_num_pieces = 0;
    g_per_op_num_steps = 0;

    /* ---- Phase 8/9/12.2: Cross-GVMI PCI import with caching (dst + src, rail0 + rail1) ---- */
    struct doca_mmap *gpu_dst_local_mmap = NULL;
    struct doca_mmap *gpu_src_local_mmap = NULL;
    struct doca_mmap *gpu_dst_local_mmap_rail1 = NULL;
    struct doca_mmap *gpu_src_local_mmap_rail1 = NULL;

    /* dst PCI import (rail0 = ring_dev) */
    if (recv_cmd->ucp_collective.dst_pci_export_buf_len > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
        gpu_dst_local_mmap = pci_mmap_cache_find(cw->gpu_dst_pci_cache, addr, len);
        if (!gpu_dst_local_mmap) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.dst_pci_export_buf,
                recv_cmd->ucp_collective.dst_pci_export_buf_len,
                cw->ring_dev, &gpu_dst_local_mmap);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_dst_pci_cache, addr, len, gpu_dst_local_mmap);
            } else { gpu_dst_local_mmap = NULL; }
        }
    }

    /* src PCI import (rail0) */
    if (recv_cmd->ucp_collective.src_pci_export_buf_len > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_src_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_src_buffer_len;
        gpu_src_local_mmap = pci_mmap_cache_find(cw->gpu_src_pci_cache, addr, len);
        if (!gpu_src_local_mmap) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.src_pci_export_buf,
                recv_cmd->ucp_collective.src_pci_export_buf_len,
                cw->ring_dev, &gpu_src_local_mmap);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_src_pci_cache, addr, len, gpu_src_local_mmap);
            } else { gpu_src_local_mmap = NULL; }
        }
    }

    /* Phase 12.2: dst PCI import (rail1 = ring_dev_rail1) */
    if (cw->ring_dev_rail1 && recv_cmd->ucp_collective.dst_pci_export_buf_len_rail1 > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
        gpu_dst_local_mmap_rail1 = pci_mmap_cache_find(cw->gpu_dst_pci_cache_rail1, addr, len);
        if (!gpu_dst_local_mmap_rail1) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.dst_pci_export_buf_rail1,
                recv_cmd->ucp_collective.dst_pci_export_buf_len_rail1,
                cw->ring_dev_rail1, &gpu_dst_local_mmap_rail1);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_dst_pci_cache_rail1, addr, len, gpu_dst_local_mmap_rail1);
            } else {
                gpu_dst_local_mmap_rail1 = NULL;
                /* Phase 12.2 H-3: rail1 import 失敗の throttled ログ */
                static atomic_int g_dst_r1_fail;
                int n = atomic_fetch_add(&g_dst_r1_fail, 1) + 1;
                if (n <= 5 || n % 1000 == 0) {
                    DOCA_LOG_WARN("dst PCI import rail1 failed #%d: %s (addr=0x%lx len=%zu) — falling back to single-rail for this AG",
                                  n, doca_error_get_descr(pci_ret), addr, len);
                }
            }
        }
    }

    /* Phase 12.2: src PCI import (rail1) */
    if (cw->ring_dev_rail1 && recv_cmd->ucp_collective.src_pci_export_buf_len_rail1 > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_src_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_src_buffer_len;
        gpu_src_local_mmap_rail1 = pci_mmap_cache_find(cw->gpu_src_pci_cache_rail1, addr, len);
        if (!gpu_src_local_mmap_rail1) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.src_pci_export_buf_rail1,
                recv_cmd->ucp_collective.src_pci_export_buf_len_rail1,
                cw->ring_dev_rail1, &gpu_src_local_mmap_rail1);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_src_pci_cache_rail1, addr, len, gpu_src_local_mmap_rail1);
            } else {
                gpu_src_local_mmap_rail1 = NULL;
                /* Phase 12.2 H-3: rail1 import 失敗の throttled ログ */
                static atomic_int g_src_r1_fail;
                int n = atomic_fetch_add(&g_src_r1_fail, 1) + 1;
                if (n <= 5 || n % 1000 == 0) {
                    DOCA_LOG_WARN("src PCI import rail1 failed #%d: %s (addr=0x%lx len=%zu) — falling back to single-rail for this AG",
                                  n, doca_error_get_descr(pci_ret), addr, len);
                }
            }
        }
    }

    /* GPU Direct AG かどうかを判定 (PCI import が両方成功 かつ AG) */
    bool is_ag = (recv_cmd->ucp_collective.collective_request.collective_op == COLLECTIVE_ALL_GATHER);
    /* CrossGVMI アブレーション: FORCE_STAGING=1 かつ AG なら PCI import 済み mmap を捨て、
     * gpu_direct を強制 off → host_dst_rmem/host_src_rmem 経由の RDMA staging 経路に落とす。
     * （RS は対象外。mmap キャッシュ自体は保持し、この op だけ staging にする。） */
    if (g_force_staging && is_ag) {
        gpu_dst_local_mmap = NULL;
        gpu_src_local_mmap = NULL;
        gpu_dst_local_mmap_rail1 = NULL;
        gpu_src_local_mmap_rail1 = NULL;
    }
    bool gpu_direct_ag = is_ag && gpu_dst_local_mmap && gpu_src_local_mmap;

    /* ---- 1. リモートメモリ作成 (キャッシュ付き) ---- */
    /* GPU Direct AG では RDMA Read/Write を使わないため rmem 作成をスキップ */
    if (!gpu_direct_ag) {
        /* src remote mmap: キャッシュ検索 */
        {
            uint64_t src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
            size_t   src_len  = recv_cmd->ucp_collective.remote_src_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->src_rmem_cache, src_addr, src_len);
            if (cached) {
                cw->host_src_rmem = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_src_rmem, cw->rdma_dev,
                                                recv_cmd->ucp_collective.src_rkey_buf,
                                                recv_cmd->ucp_collective.src_rkey_buf_len);
                if (result != DOCA_SUCCESS) { DOCA_LOG_ERR("src remote_mem create failed: %s", doca_error_get_name(result)); goto out_dec_inflight; }
                rmem_cache_store(cw->src_rmem_cache, cw->host_src_rmem.remote_addr, cw->host_src_rmem.remote_len, &cw->host_src_rmem);
            }
        }

        /* dst remote mmap: キャッシュ検索 */
        {
            uint64_t dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
            size_t   dst_len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->dst_rmem_cache, dst_addr, dst_len);
            if (cached) {
                cw->host_dst_rmem = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_dst_rmem, cw->rdma_dev,
                                                recv_cmd->ucp_collective.dst_rkey_buf,
                                                recv_cmd->ucp_collective.dst_rkey_buf_len);
                if (result != DOCA_SUCCESS) { DOCA_LOG_ERR("dst remote_mem create failed: %s", doca_error_get_name(result)); goto out_dec_inflight; }
                rmem_cache_store(cw->dst_rmem_cache, cw->host_dst_rmem.remote_addr, cw->host_dst_rmem.remote_len, &cw->host_dst_rmem);
            }
        }

        /* ---- Multi-Rail: rail1 用リモートメモリ作成 ---- */
        if (cw->dual_rail && recv_cmd->ucp_collective.src_rkey_buf_len_rail1 > 0) {
            uint64_t src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
            size_t   src_len  = recv_cmd->ucp_collective.remote_src_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->src_rmem_cache_rail1, src_addr, src_len);
            if (cached) {
                cw->host_src_rmem_rail1 = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_src_rmem_rail1, cw->rdma_dev_rail1,
                                                recv_cmd->ucp_collective.src_rkey_buf_rail1,
                                                recv_cmd->ucp_collective.src_rkey_buf_len_rail1);
                if (result != DOCA_SUCCESS) { DOCA_LOG_WARN("rail1 src remote_mem create failed"); }
                else { rmem_cache_store(cw->src_rmem_cache_rail1, cw->host_src_rmem_rail1.remote_addr, cw->host_src_rmem_rail1.remote_len, &cw->host_src_rmem_rail1); }
            }

            uint64_t dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
            size_t   dst_len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
            cached = rmem_cache_find(cw->dst_rmem_cache_rail1, dst_addr, dst_len);
            if (cached) {
                cw->host_dst_rmem_rail1 = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_dst_rmem_rail1, cw->rdma_dev_rail1,
                                                recv_cmd->ucp_collective.dst_rkey_buf_rail1,
                                                recv_cmd->ucp_collective.dst_rkey_buf_len_rail1);
                if (result != DOCA_SUCCESS) { DOCA_LOG_WARN("rail1 dst remote_mem create failed"); }
                else { rmem_cache_store(cw->dst_rmem_cache_rail1, cw->host_dst_rmem_rail1.remote_addr, cw->host_dst_rmem_rail1.remote_len, &cw->host_dst_rmem_rail1); }
            }
        }
    }

    /* ---- 2. バッファプールスロット取得 ---- */
    /* GPU Direct AG では DPU 中間バッファ不要 → スキップ */
    size_t working_buffer_len = (size_t)max_u64(recv_cmd->ucp_collective.remote_src_buffer_len,
                                                 recv_cmd->ucp_collective.remote_dst_buffer_len);
    size_t elem_size = sizeof(fp16_t);
    size_t num_elems = (working_buffer_len + elem_size - 1) / elem_size;
    working_buffer_len = elem_size * num_elems;

    void *send_buf = NULL;
    size_t send_cap = 0;
    int send_slot = -1;
    void *recv_buf = NULL;
    size_t recv_cap = 0;
    int recv_slot = -1;

    if (gpu_direct_ag) {
        /* GPU Direct AG: バッファプール不要 (GPU メモリに直接 Send/Recv) */
        recv_buf = NULL;
    } else {
        send_slot = cw_buffer_pool_acquire(&cw->send_pool, working_buffer_len, &send_buf, &send_cap);
        if (send_slot < 0) {
            DOCA_LOG_ERR("Failed to acquire send_pool slot");
            goto out_dec_inflight;
        }

        if (is_ag) {
            recv_buf = send_buf;
        } else {
            recv_slot = cw_buffer_pool_acquire(&cw->recv_pool, working_buffer_len, &recv_buf, &recv_cap);
            if (recv_slot < 0) {
                DOCA_LOG_ERR("Failed to acquire recv_pool slot");
                cw_buffer_pool_release(&cw->send_pool, send_slot);
                goto out_dec_inflight;
            }
        }
    }

    /* ---- 3. 集合通信の実行 ---- */
    uint64_t remote_src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
    uint64_t remote_dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
    /* uint64_t data_len = recv_cmd->ucp_collective.remote_src_buffer_len; */  /* DPU profiling 用 (現在無効) */

    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t t_pre_pause_ns = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
    /* uint64_t t_setup_done_ns = t_pre_pause_ns; */  /* DPU profiling 用 (現在無効) */

    /* Phase 6 Step E: ワーカスレッドは停止しない (常時 PE を回し続ける) */

    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t t_post_pause_ns = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;

    /* DPU Union Tracker は collective_*() 内部の "ring-only" 区間 (barrier 後〜ring 完了) で
     * 直接呼ぶため、ここでは wrap しない。 */

    switch (recv_cmd->ucp_collective.collective_request.collective_op) {
    case COLLECTIVE_REDUCE_SCATTER:
        result = collective_reduce_scatter(sample_objects->rank, sample_objects->world_size,
                                           cw, send_buf, recv_buf,
                                           working_buffer_len, remote_src_addr, remote_dst_addr, id);
        break;
    case COLLECTIVE_ALL_GATHER:
        result = collective_all_gather(sample_objects->rank, sample_objects->world_size,
                                       cw, send_buf,
                                       working_buffer_len, remote_src_addr, remote_dst_addr,
                                       gpu_src_local_mmap, id,
                                       gpu_dst_local_mmap,
                                       /* Phase 12.2: rail1 mmaps */
                                       gpu_src_local_mmap_rail1,
                                       gpu_dst_local_mmap_rail1,
                                       /* Phase 15: ring_id (0 or 1) */
                                       ring_id);
        break;
    default:
        DOCA_LOG_ERR("未実装 collective op: %u", recv_cmd->ucp_collective.collective_request.collective_op);
        result = DOCA_ERROR_INVALID_VALUE;
        break;
    }

    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t t_post_collective_ns = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;

    /* Phase 14: AG 完了時に GPU flag を書き込む (doorbell より先)。
     * data → flag → doorbell の順序を厳守:
     *   PCIe writes within an endpoint are ordered, かつ doca_task_submit は同 ctx で
     *   submit 順序を保つので、Ring の最終 Recv → flag write → doorbell write の順で
     *   ホスト GPU/CPU 側に観測される。これにより compute stream は flag を見た時点で
     *   AG データが揃っていることを保証できる。 */
    if (recv_cmd->ucp_collective.flag_gpu_addr != 0 && cw->flag_pool_enabled) {
        phase14_write_flag(cw,
                           recv_cmd->ucp_collective.flag_gpu_addr,
                           recv_cmd->ucp_collective.flag_value);
    }

    /* ---- 4. 通知送信 (Phase 5 Step 3: RDMA Doorbell or ComCh fallback) ---- */
    if (cw->doorbell_enabled) {
        /* send_buf が NULL (GPU Direct AG) の場合は working_buf の先頭を使う */
        volatile uint64_t *db_staging = (volatile uint64_t *)(send_buf ? send_buf : cw->working_buf);
        cw->doorbell_seq++;
        *db_staging = cw->doorbell_seq;  /* monotonic completion counter */
        struct doca_task_desc db_task;
        init_write_task(&db_task,
                        (void *)db_staging,
                        (void *)cw->host_doorbell_rmem.remote_addr,
                        sizeof(uint64_t),
                        &cw->host_doorbell_rmem,
                        &cw->rdma_rma, cw->host_conn, 0);
        db_task.pe_spin = &cw->rma_pe_spin;
        submit_and_wait_doca(cw->worker_pool, &db_task);
    } else {
        /* Fallback: ComCh notify */
        struct control_notify send_notify;
        send_notify.type = CONTROL_NOTIFY_UCP_COLLECTIVE;
        send_notify.ucp_collective.id = id;
        result = comch_send_control_notify(&send_notify, sample_objects);
        if (result != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to send collective notify");
    }

    /* ---- 5. local_cpu_buffer 処理 ---- */
    if (recv_cmd->ucp_collective.local_cpu_buffer_address != 0) {
        struct doca_remote_mem_t cpu_rmem;
        result = doca_remote_mem_create(&cpu_rmem, cw->rdma_dev,
                                         recv_cmd->ucp_collective.local_cpu_rkey_buf,
                                         recv_cmd->ucp_collective.local_cpu_rkey_buf_len);
        if (result == DOCA_SUCCESS) {
            struct doca_task_desc put_task;
            init_write_task(&put_task, send_buf,
                            (void *)recv_cmd->ucp_collective.local_cpu_buffer_address,
                            recv_cmd->ucp_collective.local_cpu_buffer_len,
                            &cpu_rmem, &cw->rdma_rma, cw->host_conn, 0);
            put_task.pe_spin = &cw->rma_pe_spin;
            submit_and_wait_doca(cw->worker_pool, &put_task);

            /* フラグ書き込み */
            if (recv_cmd->ucp_collective.local_cpu_flag_address != 0) {
                struct doca_remote_mem_t flag_rmem;
                result = doca_remote_mem_create(&flag_rmem, cw->rdma_dev,
                                                 recv_cmd->ucp_collective.local_cpu_flag_rkey_buf,
                                                 recv_cmd->ucp_collective.local_cpu_flag_rkey_buf_len);
                if (result == DOCA_SUCCESS) {
                    const int num_flags = recv_cmd->ucp_collective.local_cpu_flag_len / 4;
                    uint32_t *ones = (uint32_t *)send_buf;  /* 一時的に send_buf の先頭を使う */
                    for (int i = 0; i < num_flags; i++) ones[i] = 1;

                    struct doca_task_desc flag_task;
                    init_write_task(&flag_task, ones,
                                    (void *)recv_cmd->ucp_collective.local_cpu_flag_address,
                                    recv_cmd->ucp_collective.local_cpu_flag_len,
                                    &flag_rmem, &cw->rdma_rma, cw->host_conn, 0);
                    flag_task.pe_spin = &cw->rma_pe_spin;
                    submit_and_wait_doca(cw->worker_pool, &flag_task);
                    doca_remote_mem_destroy(&flag_rmem);
                }
            }
            doca_remote_mem_destroy(&cpu_rmem);
        }
    }

    /* ---- Phase 7 Step 0: Detailed timing activation + output ---- */
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t t_done_ns = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;

    /* ---- Phase 12 Calc C: per-op record ---- */
    {
        int slot = atomic_fetch_add_explicit(&g_dpu_per_op_count, 1, memory_order_relaxed);
        if (slot < DPU_PER_OP_TIMING_MAX) {
            struct dpu_per_op_record *r = &g_dpu_per_op[slot];
            uint64_t setup_ns    = t_pre_pause_ns - t_entry_ns;
            uint64_t coll_all_ns = t_post_collective_ns - t_post_pause_ns;
            uint64_t bar_ns      = g_per_op_barrier_ns;
            uint64_t ring_ns     = (coll_all_ns > bar_ns) ? (coll_all_ns - bar_ns) : 0;
            uint64_t notify_ns   = t_done_ns - t_post_collective_ns;
            uint64_t data_len    = recv_cmd->ucp_collective.remote_src_buffer_len;
            uint64_t queue_wait_ns = (t_entry_ns > t_recv_ns) ? (t_entry_ns - t_recv_ns) : 0;
            r->id         = id;
            r->size_kb    = (uint32_t)(data_len / 1024);
            r->setup_us   = (uint32_t)(setup_ns / 1000);
            r->barrier_us = (uint32_t)(bar_ns   / 1000);
            r->ring_us    = (uint32_t)(ring_ns  / 1000);
            r->notify_us  = (uint32_t)(notify_ns/ 1000);
            /* Phase 16: ring 内部 breakdown (AG のみ、RS は 0) */
            r->recv_post_us   = (uint32_t)(g_per_op_recv_post_ns / 1000);
            r->step_submit_us = (uint32_t)(g_per_op_step_submit_ns / 1000);
            r->step_wait_us   = (uint32_t)(g_per_op_step_wait_ns / 1000);
            r->num_pieces     = g_per_op_num_pieces;
            r->num_steps      = g_per_op_num_steps;
            /* Phase 17: training-specific overhead 切り分け */
            r->queue_wait_us    = (uint32_t)(queue_wait_ns / 1000);
            r->inflight_at_entry = (uint16_t)(inflight_at_entry > 65535 ? 65535 : inflight_at_entry);
            r->is_ag          = is_ag ? 1 : 0;
        }

        /* Phase 13-A: 全 rank で (id, barrier_us) を記録 */
        int slot13a = atomic_fetch_add_explicit(&g_phase13a_count, 1, memory_order_relaxed);
        if (slot13a < PHASE13A_MAX_OPS) {
            g_phase13a_records[slot13a].id         = (uint32_t)id;
            g_phase13a_records[slot13a].barrier_us = (uint32_t)(g_per_op_barrier_ns / 1000);
        }

#if 0 /* 計測ダンプ [DPU PER-OP] / [Rank Skew] / Phase16/17 を無効化 (2026-07-27)。
       * 5000 op 到達時に 1 回だけ発火する集計出力。phase13a_gather_and_print は
       * MPI_Gather を含むが全 rank でこの経路を通るため、丸ごと無効化しても collective
       * の不整合は起きない (全 rank が一様に skip)。 */
        /* Phase 16: 5000 ops 蓄積したら rank 0 で 1 回だけ top-N 出力 */
        if (sample_objects->rank == 0) {
            static int _printed = 0;
            if (!_printed && (slot + 1) >= 5000) {
                _printed = 1;
                int n_now = atomic_load_explicit(&g_dpu_per_op_count, memory_order_relaxed);
                if (n_now > DPU_PER_OP_TIMING_MAX) n_now = DPU_PER_OP_TIMING_MAX;
                dpu_per_op_print_summary(0, n_now);
            }
        }

        /* Phase 13-A: 5000 op 達成時に全 rank で MPI_Gather → rank 0 で出力 */
        if ((slot13a + 1) >= 5000 && !g_phase13a_gathered) {
            phase13a_gather_and_print();
        }
#endif
    }

    {
        static int _timing_count = 0;
        int phase = _timing_count % 200;  /* 200 = warmup(100) + iters(100) */

#if AG_TIMING_ENABLED
        /* Activate detailed timing for last AG_TIMING_MAX_ITERS of measurement */
        if (phase == 200 - AG_TIMING_MAX_ITERS) {
            g_ag_timing.count = 0;
            g_ag_timing.active = true;
        }

        /* Print detailed timing summary at end of measurement iterations */
        if (phase == 199 && sample_objects->rank == 0) {
            g_ag_timing.active = false;
            uint64_t buflen = recv_cmd->ucp_collective.remote_src_buffer_len;
            print_ag_timing_summary(sample_objects->rank, buflen, sample_objects->world_size);
        }
#endif

#if AG_TIMING_ENABLED
        /* Legacy per-iteration timing (keep first 2 of measurement iters) */
        if (phase >= 100 && phase < 104 && sample_objects->rank == 0) {
            uint64_t len = recv_cmd->ucp_collective.remote_src_buffer_len;
            printf("[TIMING rank0 N=%lu] recv->entry=%.3fms entry->pre_pause=%.3fms "
                    "pause=%.3fms collective=%.3fms notify+cleanup=%.3fms TOTAL=%.3fms\n",
                    len / 2,
                    (double)(t_entry_ns - t_recv_ns) / 1e6,
                    (double)(t_pre_pause_ns - t_entry_ns) / 1e6,
                    (double)(t_post_pause_ns - t_pre_pause_ns) / 1e6,
                    (double)(t_post_collective_ns - t_post_pause_ns) / 1e6,
                    (double)(t_done_ns - t_post_collective_ns) / 1e6,
                    (double)(t_done_ns - t_recv_ns) / 1e6);
        }
#endif
        _timing_count++;
    }

    /* ---- 6. クリーンアップ (キャッシュ済み rmem は破棄しない) ---- */
    /* gpu_dst_local_mmap はキャッシュ済み — 破棄しない (DPA に倣う) */
    cw_buffer_pool_release(&cw->send_pool, send_slot);
    if (recv_slot >= 0) cw_buffer_pool_release(&cw->recv_pool, recv_slot);

out_dec_inflight:
    /* Phase 17: 全 return path でここに到達して inflight counter を decrement する。
     * normal exit path は上の cleanup を通過後に fall through で到達する。 */
    atomic_fetch_sub_explicit(&g_dpu_ag_inflight, 1, memory_order_relaxed);
}

/* =====================================================
 * ComCh コールバック — UNCHANGED
 * ===================================================== */

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
     * 短い run (ViT ag_smartnic 等、5000 ops 未達) でも aggregate を確実に得られるようにする。 */
    dpu_union_tracker_print(sample_objects->rank);
    sample_objects->finish = true;
}

/* =====================================================
 * Phase 15: Per-ring job queue and processing thread
 *
 * Ring 0 / Ring 1 それぞれが自分の job queue を持ち、独立したスレッドで処理する。
 * これにより Ring 0 と Ring 1 が DPU 上で並列実行される。
 *
 * 順序保証: 各 queue は FIFO。Ring assignment は ag_seq_counter で deterministic。
 * 全 rank で同じ AG は同じ ring に割り当てられるため、Ring algorithm は正しく動作する。
 * ===================================================== */

/* Per-ring queue 初期化 */
static int ring_queues_init(struct collective_worker_t *cw)
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

/* Ring queue に job を追加 (FIFO, Phase 18: cond_signal で worker を wake)
 *
 * Phase 18: 3-tier adaptive wait の worker を確実に起こすため、push 時に必ず
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

/* Ring queue から 1 job を取り出す (Phase 18: 3-tier adaptive wait)
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

    /* CPU affinity: 既存の worker thread が core (mpi_rank % 2) * 3 + [0,1,2] を使っている。
     * ring proc thread は別コアに配置したいが、DPU は 16 コアで余裕が少ない。
     * 明示的には pinning せず OS に任せる (kernel が適切なコアに配置する)。 */

    while (!atomic_load(&cw->ring_queue_stop[ring_id])) {
        struct ring_job *job = ring_queue_pop_blocking(cw, ring_id);
        if (!job) break;  /* stopped */

        /* Execute collective on this ring */
        execute_doca_collective_cmd(job->cmd, job->sample_objects, job->t_recv_ns, ring_id);

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
        printf("[Phase15] %d ring processing thread(s) started\n", n);
        fflush(stdout);
    }
    return 0;
}

/* Per-ring 処理スレッド停止 (cleanup で呼ぶ)
 * Phase 18: stop flag セットに加えて cond_broadcast で sleeping worker を起こす。
 * これがないと Tier 3 cond_wait 中の worker が shutdown を検出できず hang する。 */
static void ring_proc_threads_stop(struct collective_worker_t *cw)
{
    int n = cw->ring_r1_enabled ? N_RINGS : 1;
    for (int i = 0; i < n; i++) {
        atomic_store(&cw->ring_queue_stop[i], true);
        /* Phase 18: sleeping worker を起こす (Tier 3 cond_wait から抜けさせる) */
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

/* =====================================================
 * メッセージハンドラ (関数名のみ更新)
 * ===================================================== */

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
            /* Phase 15 infrastructure, Phase 18 refined (env var PHASE15_ENABLE_RING_QUEUES=1):
             *   Ring 0 / Ring 1 の 2 worker thread を起動し、AG を round-robin で並列実行する。
             *
             *   Phase 15 original (busy-poll): unpinned thread × 4 が critical pinned thread
             *     を阻害して全体悪化 → 失敗。
             *   Phase 18 (adaptive hybrid): Tier 1 (fast path) + Tier 2 (short spin 50us) +
             *     Tier 3 (cond_wait)。idle 時は CPU 消費ゼロで pinned thread と競合しない。
             *
             *   デフォルトは無効 (安全側)。PHASE15_ENABLE_RING_QUEUES=1 で opt-in 有効化。
             *   期待効果: DPU AG inflight 1 → 2、fwd 651ms → ~450ms (NCCL level) */
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
                        printf("[Phase15] ring_proc threads NOT started (PHASE15_ENABLE_RING_QUEUES unset), using inline path\n");
                        fflush(stdout);
                    }
                }
            }
            free(recv_cmd);  /* = free(buf_copy) */
            free(w);
            return NULL;
        case CONTROL_CMD_UCP_COLLECTIVE:
            /* Phase 15 infrastructure, Phase 18 refined dispatch (env PHASE15_ENABLE_RING_QUEUES=1):
             *   - AG: ag_seq % N_RINGS で Ring 0/1 に round-robin 配分 → 2-way 並列
             *   - RS: 常に Ring 0 (既存 semantics 維持)
             *   - デフォルト (env unset): 旧来の inline 実行パス (serial)
             *
             *   Phase 17 の per-op 計測で DPU inflight=1 が判明 (msg_pool = 1 thread の結果)。
             *   Phase 18 は adaptive hybrid wait により parallel dispatch を安全に enable する。 */
            {
                static int use_ring_queues = -1;
                if (use_ring_queues < 0) {
                    const char *e = getenv("PHASE15_ENABLE_RING_QUEUES");
                    use_ring_queues = (e && atoi(e) != 0) ? 1 : 0;
                    if (sample_objects->rank == 0) {
                        printf("[Phase18] PHASE15_ENABLE_RING_QUEUES=%d — %s\n",
                               use_ring_queues,
                               use_ring_queues ? "parallel ring queue (2 workers)" : "inline serial (default)");
                        fflush(stdout);
                    }
                }

                if (!use_ring_queues) {
                    /* デフォルト: inline 実行 (Phase 17 までと同じ)。
                     * ring_id は常に 0。RS も AG もこのパスを通る。 */
                    execute_doca_collective_cmd(recv_cmd, sample_objects, w->t_recv_ns, 0);
                    break;
                }

                /* Phase 15 queue path (opt-in only) */
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
                    execute_doca_collective_cmd(recv_cmd, sample_objects, w->t_recv_ns, 0);
                    free(recv_cmd);  /* = free(buf_copy) */
                    free(w);
                    return NULL;
                }
                job->cmd = recv_cmd;
                job->buf_copy = w->buf_copy;  /* same as recv_cmd */
                job->t_recv_ns = w->t_recv_ns;
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
    { struct timespec _rts; clock_gettime(CLOCK_MONOTONIC, &_rts);
      w->t_recv_ns = (uint64_t)_rts.tv_sec * 1000000000ULL + (uint64_t)_rts.tv_nsec; }

    if (!sample_objects->msg_pool || msg_pool_try_submit(sample_objects->msg_pool, w) != 0) {
        /* msg_pool 満杯または無効: drop して解放
         * 注意: recv_cmd は buf_copy 内部を指すポインタなので free(recv_cmd) してはいけない */
        if (sample_objects->msg_pool)
            DOCA_LOG_WARN("msg_pool queue full, dropping collective command");
        free(w); free(buf_copy);
        return;
    }
}

/* =====================================================
 * クリーンアップ
 * ===================================================== */

static void clean_comch_sample_objects(struct comch_ctrl_path_server_objects *sample_objects)
{
    static int cleaned = 0;
    if (cleaned) return;
    cleaned = 1;
    doca_error_t result;

    /* Phase 15: Per-ring 処理スレッドを先に停止 (RDMA ctx 破棄前) */
    ring_proc_threads_stop(&sample_objects->collective_worker);

    /* Phase 15: Per-ring MPI_Comm を解放 (MPI_Finalize より先に)
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

    /* Phase 12.1 Plan A: RDB の remote mmap を破棄 (RDMA ctx より先に) */
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
    /* Phase 15: Ring 1 コンテキスト破棄 */
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
    /* Phase 12.2: rail1 PCI mmap キャッシュ破棄 */
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_dst_pci_cache_rail1);
    sample_objects->collective_worker.gpu_dst_pci_cache_rail1 = NULL;
    pci_mmap_cache_destroy(sample_objects->collective_worker.gpu_src_pci_cache_rail1);
    sample_objects->collective_worker.gpu_src_pci_cache_rail1 = NULL;
    if (sample_objects->msg_pool) { msg_pool_destroy(sample_objects->msg_pool); free(sample_objects->msg_pool); sample_objects->msg_pool = NULL; }

    /* Phase 5 Step 4: cmd_slot 解放 */
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

/* =====================================================
 * ComCh サーバ状態変更コールバック — UNCHANGED
 * ===================================================== */

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

/* =====================================================
 * ComCh サーバ初期化 — UNCHANGED
 * ===================================================== */

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

/* =====================================================
 * main
 * ===================================================== */

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

    /* Phase 7 Step 2: MPI_THREAD_MULTIPLE 必須 (msg_pool_worker スレッドから MPI_Barrier を呼ぶため) */
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
