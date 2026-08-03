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
#include "dpu_config.h"

DOCA_LOG_REGISTER(RDMA_TASK);


/* DOCA RDMA コールバックデータ + コールバック関数 */

struct doca_task_cb_data {
    int stride_id;
    struct doca_buf *buf1;
    struct doca_buf *buf2;
    atomic_int *inline_pending;              /* decrement on completion */
    /* 診断用情報 (RDMA op エラー時に出力) */
    uint32_t  op_type;        /* doca_rdma_task_type_t (READ/WRITE/SEND/RECV) */
    uint32_t  length;
    uint64_t  local_addr;
    uint64_t  remote_addr;
    void     *rdma_ctx_ptr;   /* どの RDMA context か (rail0/rail1 識別用) */
};

/* エラーログ throttling — 最初の N 件は full detail、以降はカウントのみ */
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
    (void)status;
    if (cb->inline_pending) {
        atomic_fetch_sub(cb->inline_pending, 1);
    }
    free(cb);
}

/* 共通エラーログヘルパー */
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

/* Read 完了/エラー */
void rdma_read_comp_cb(struct doca_rdma_task_read *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_read_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

void rdma_read_err_cb(struct doca_rdma_task_read *task,
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

/* Write 完了/エラー */
void rdma_write_comp_cb(struct doca_rdma_task_write *task,
                                union doca_data task_user_data,
                                union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    if (cb->buf2) doca_buf_dec_refcount(cb->buf2, NULL);
    doca_task_free(doca_rdma_task_write_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

void rdma_write_err_cb(struct doca_rdma_task_write *task,
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

/* Send 完了/エラー */
void rdma_send_comp_cb(struct doca_rdma_task_send *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_send_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

void rdma_send_err_cb(struct doca_rdma_task_send *task,
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

/* Receive 完了/エラー */
void rdma_recv_comp_cb(struct doca_rdma_task_receive *task,
                               union doca_data task_user_data,
                               union doca_data ctx_user_data)
{
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)task_user_data.ptr;
    if (cb->buf1) doca_buf_dec_refcount(cb->buf1, NULL);
    doca_task_free(doca_rdma_task_receive_as_task(task));
    task_cb_finish(cb, COMP_SUCCESS);
}

void rdma_recv_err_cb(struct doca_rdma_task_receive *task,
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

/* * DOCA タスク提出 (desc → DOCA API 呼び出し) */

void submit_doca_task_from_desc(struct doca_rdma_ctx_t *rdma_ctx,
                                        const struct doca_task_desc *desc)
{
    doca_error_t ret = DOCA_ERROR_INVALID_VALUE;
    struct doca_buf *local_buf = NULL, *remote_buf = NULL;
    struct doca_task_cb_data *cb = (struct doca_task_cb_data *)calloc(1, sizeof(*cb));
    if (!cb) {
        if (desc->inline_pending) {
            atomic_fetch_sub(desc->inline_pending, 1);
        }
        return;
    }
    cb->stride_id = desc->stride_id;
    cb->inline_pending = desc->inline_pending;
    /* 診断用情報をコピー */
    cb->op_type      = (uint32_t)desc->type;
    cb->length       = (uint32_t)desc->length;
    cb->local_addr   = (uint64_t)(uintptr_t)desc->local_addr;
    cb->remote_addr  = (uint64_t)(uintptr_t)desc->remote_addr;
    cb->rdma_ctx_ptr = (void *)rdma_ctx;

    union doca_data ud;
    ud.ptr = cb;

    /* PE spinlock で buf/task alloc + submit + PE progress を排他 */
    struct pe_spin_t *spin = desc->pe_spin;
    if (spin) pe_spin_lock(spin);

    switch (desc->type) {
    case DOCA_RDMA_TASK_READ: {
        ret = doca_rdma_get_remote_buf(rdma_ctx, desc->remote_mem, desc->remote_addr, desc->length, &remote_buf);
        if (ret != DOCA_SUCCESS) goto fail;
        ret = doca_buf_set_data(remote_buf, desc->remote_addr, desc->length);
        if (ret != DOCA_SUCCESS) { doca_buf_dec_refcount(remote_buf, NULL); goto fail; }

        /* local_mmap_override → Read の dst を GPU dst (PCI import mmap) に */
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
        /* local_mmap_override → Cross-GVMI PCI import した GPU メモリを使用 */
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
        /* local_mmap_override → Cross-GVMI PCI import した GPU メモリを使用 */
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
    /* 提出失敗の詳細ログ (throttled) */
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
    }
}

/* * DOCA ワーカスレッド メインループ */

static void *doca_worker_thread_main(void *arg)
{
    struct doca_worker_thread_ctx *ctx = (struct doca_worker_thread_ctx *)arg;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ctx->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (atomic_load(&ctx->running)) {
        bool did_work = false;

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

/* * DOCA ワーカスレッドプール */

int doca_worker_thread_pool_init(
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

    /* PE spinlock 初期化 */
    pe_spin_init(&cw->rma_pe_spin);
    pe_spin_init(&cw->rma_pe_spin_rail1);
    pe_spin_init(&cw->ring_send_pe_spin);
    pe_spin_init(&cw->ring_send_pe_spin_rail1);
    pe_spin_init(&cw->ring_recv_pe_spin);
    pe_spin_init(&cw->ring_recv_pe_spin_rail1);

    /* PE spinlock マッピングテーブル */
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
        atomic_init(&wk->running, true);

        DOCA_LOG_INFO("comm worker %d: core=%d n_pe=%d", w, wk->core_id, wk->n_pe);
        if (pthread_create(&wk->thread, NULL, doca_worker_thread_main, wk) != 0) {
            atomic_store(&wk->running, false);
            return -1;
        }
    }
    atomic_store(&pool->initialized, true);
    return 0;
}

void doca_worker_thread_pool_destroy(struct doca_worker_thread_pool_t *pool)
{
    if (!pool || !atomic_load(&pool->initialized)) return;

    for (int i = 0; i < pool->n_workers; i++) {
        atomic_store(&pool->workers[i].running, false);
    }
    for (int i = 0; i < pool->n_workers; i++) {
        struct doca_worker_thread_ctx *w = &pool->workers[i];
        if (w->thread) pthread_join(w->thread, NULL);
    }
    atomic_store(&pool->initialized, false);
}

/* * タスク提出・完了待機ヘルパー */

void submit_and_wait_doca(
    struct doca_worker_thread_pool_t *pool,
    struct doca_task_desc *task)
{
    /* inline submit + PE progress from main thread */
    atomic_int pending;
    atomic_init(&pending, 1);
    task->inline_pending = &pending;

    struct doca_rdma_ctx_t *ctx = task->rdma_ctx;
    struct pe_spin_t *spin = task->pe_spin;

    /* Direct submit (bypass SPSC queue) */
    submit_doca_task_from_desc(ctx, task);

    /* Main thread PE progress to drive completion */
    while (atomic_load_explicit(&pending, memory_order_acquire) > 0) {
        if (spin && ctx && ctx->pe)
            try_pe_progress(ctx->pe, spin);
        else if (ctx && ctx->pe)
            doca_pe_progress(ctx->pe);
    }
}
