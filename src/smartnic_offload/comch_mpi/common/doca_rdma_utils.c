#include "doca_rdma_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

DOCA_LOG_REGISTER(DOCA_RDMA_UTILS);

/* ---- 内部ヘルパー: PE を回してコンテキストが RUNNING になるまで待つ ---- */
static doca_error_t wait_for_ctx_running(struct doca_pe *pe, struct doca_ctx *ctx)
{
    enum doca_ctx_states state;
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000};

    for (int i = 0; i < 1000000; i++) {
        (void)doca_pe_progress(pe);

        doca_error_t ret = doca_ctx_get_state(ctx, &state);
        if (ret != DOCA_SUCCESS)
            return ret;
        if (state == DOCA_CTX_STATE_RUNNING)
            return DOCA_SUCCESS;
        if (state == DOCA_CTX_STATE_IDLE)
            return DOCA_ERROR_BAD_STATE;

        nanosleep(&ts, NULL);
    }
    return DOCA_ERROR_TIME_OUT;
}

/* ---- コンテキスト状態変更コールバック ---- */
static void rdma_state_change_cb(const union doca_data user_data,
                                  struct doca_ctx *ctx,
                                  enum doca_ctx_states prev_state,
                                  enum doca_ctx_states next_state)
{
    (void)user_data;
    (void)ctx;
    (void)prev_state;
    (void)next_state;
    /* 状態遷移は wait_for_ctx_running でポーリングするため、ここでは何もしない */
}

/* ---- 初期化 ---- */

doca_error_t doca_rdma_ctx_init(
    struct doca_rdma_ctx_t *ctx,
    struct doca_dev *dev,
    struct doca_pe *pe,
    void *local_addr,
    size_t local_size,
    uint32_t permissions,
    uint32_t send_q_size,
    uint32_t recv_q_size,
    uint32_t gid_index)
{
    doca_error_t ret;

    memset(ctx, 0, sizeof(*ctx));
    ctx->dev = dev;

    /* PE: 外部から渡されなければ新規作成 */
    if (pe != NULL) {
        ctx->pe = pe;
    } else {
        ret = doca_pe_create(&ctx->pe);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_pe_create failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }

    /* RDMA インスタンス作成 */
    ret = doca_rdma_create(dev, &ctx->rdma);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_rdma_create failed: %s", doca_error_get_name(ret));
        return ret;
    }

    /* キューサイズ設定 */
    if (send_q_size > 0) {
        ret = doca_rdma_set_send_queue_size(ctx->rdma, send_q_size);
        if (ret != DOCA_SUCCESS)
            DOCA_LOG_WARN("doca_rdma_set_send_queue_size failed: %s", doca_error_get_name(ret));
    }
    if (recv_q_size > 0) {
        ret = doca_rdma_set_recv_queue_size(ctx->rdma, recv_q_size);
        if (ret != DOCA_SUCCESS && ret != DOCA_ERROR_NOT_SUPPORTED)
            DOCA_LOG_WARN("doca_rdma_set_recv_queue_size failed: %s", doca_error_get_name(ret));
        /* DOCA_ERROR_NOT_SUPPORTED は CPU data-path では正常 (DPA/GPU 専用 API) */
    }

    /* パーミッション設定 */
    if (permissions != 0) {
        ret = doca_rdma_set_permissions(ctx->rdma, permissions);
        if (ret != DOCA_SUCCESS)
            DOCA_LOG_WARN("doca_rdma_set_permissions failed: %s", doca_error_get_name(ret));
    }

    /* 接続数: Ring で 2 つ必要なケースに対応 */
    ret = doca_rdma_set_max_num_connections(ctx->rdma, 4);
    if (ret != DOCA_SUCCESS)
        DOCA_LOG_WARN("doca_rdma_set_max_num_connections failed: %s", doca_error_get_name(ret));

    /* RC トランスポート */
    ret = doca_rdma_set_transport_type(ctx->rdma, DOCA_RDMA_TRANSPORT_TYPE_RC);
    if (ret != DOCA_SUCCESS)
        DOCA_LOG_WARN("doca_rdma_set_transport_type failed: %s", doca_error_get_name(ret));

    /* GID インデックス設定 (RoCE バージョンを Host-DPU 間で一致させる) */
    ret = doca_rdma_set_gid_index(ctx->rdma, gid_index);
    if (ret != DOCA_SUCCESS)
        DOCA_LOG_WARN("doca_rdma_set_gid_index(%u) failed: %s", gid_index, doca_error_get_name(ret));

    /* RNR リトライ回数 (7 = 無限リトライ) — Recv が未 post でも Send が即失敗しない */
    ret = doca_rdma_set_rnr_retry_count(ctx->rdma, 7);
    if (ret != DOCA_SUCCESS)
        DOCA_LOG_WARN("doca_rdma_set_rnr_retry_count failed: %s", doca_error_get_name(ret));

    /* ローカル mmap */
    if (local_addr != NULL && local_size > 0) {
        ret = doca_mmap_create(&ctx->local_mmap);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_mmap_create failed: %s", doca_error_get_name(ret));
            return ret;
        }

        ret = doca_mmap_add_dev(ctx->local_mmap, dev);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_mmap_add_dev failed: %s", doca_error_get_name(ret));
            return ret;
        }

        ret = doca_mmap_set_memrange(ctx->local_mmap, local_addr, local_size);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_mmap_set_memrange failed: %s", doca_error_get_name(ret));
            return ret;
        }

        ret = doca_mmap_set_permissions(ctx->local_mmap,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_mmap_set_permissions failed: %s", doca_error_get_name(ret));
            return ret;
        }

        ret = doca_mmap_start(ctx->local_mmap);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_mmap_start failed: %s", doca_error_get_name(ret));
            return ret;
        }

        ctx->local_mmap_addr = local_addr;
        ctx->local_mmap_size = local_size;
    }

    /* コンテキスト取得 */
    ctx->ctx = doca_rdma_as_ctx(ctx->rdma);
    if (ctx->ctx == NULL) {
        DOCA_LOG_ERR("doca_rdma_as_ctx failed");
        return DOCA_ERROR_INITIALIZATION;
    }

    /* PE にコンテキストを接続 */
    ret = doca_pe_connect_ctx(ctx->pe, ctx->ctx);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_pe_connect_ctx failed: %s", doca_error_get_name(ret));
        return ret;
    }

    /* 状態変更コールバック */
    ret = doca_ctx_set_state_changed_cb(ctx->ctx, rdma_state_change_cb);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_ctx_set_state_changed_cb failed: %s", doca_error_get_name(ret));
        return ret;
    }

    return DOCA_SUCCESS;
}

/* ---- タスク設定 + 開始 ---- */

doca_error_t doca_rdma_ctx_configure_and_start(
    struct doca_rdma_ctx_t *ctx,
    uint32_t num_read_tasks,
    uint32_t num_write_tasks,
    uint32_t num_send_tasks,
    uint32_t num_recv_tasks,
    doca_rdma_task_read_completion_cb_t   read_comp_cb,
    doca_rdma_task_read_completion_cb_t   read_err_cb,
    doca_rdma_task_write_completion_cb_t  write_comp_cb,
    doca_rdma_task_write_completion_cb_t  write_err_cb,
    doca_rdma_task_send_completion_cb_t   send_comp_cb,
    doca_rdma_task_send_completion_cb_t   send_err_cb,
    doca_rdma_task_receive_completion_cb_t recv_comp_cb,
    doca_rdma_task_receive_completion_cb_t recv_err_cb,
    void *ctx_user_data)
{
    doca_error_t ret;

    /* タスクコールバック設定 */
    if (num_read_tasks > 0 && read_comp_cb != NULL) {
        ret = doca_rdma_task_read_set_conf(ctx->rdma, read_comp_cb, read_err_cb, num_read_tasks);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_rdma_task_read_set_conf failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }
    if (num_write_tasks > 0 && write_comp_cb != NULL) {
        ret = doca_rdma_task_write_set_conf(ctx->rdma, write_comp_cb, write_err_cb, num_write_tasks);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_rdma_task_write_set_conf failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }
    if (num_send_tasks > 0 && send_comp_cb != NULL) {
        ret = doca_rdma_task_send_set_conf(ctx->rdma, send_comp_cb, send_err_cb, num_send_tasks);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_rdma_task_send_set_conf failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }
    if (num_recv_tasks > 0 && recv_comp_cb != NULL) {
        ret = doca_rdma_task_receive_set_conf(ctx->rdma, recv_comp_cb, recv_err_cb, num_recv_tasks);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_rdma_task_receive_set_conf failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }

    /* buf_inventory 作成 */
    uint32_t total_bufs = (num_read_tasks + num_write_tasks + num_send_tasks + num_recv_tasks) * 2 + 16;
    if (total_bufs < DOCA_RDMA_BUF_INVENTORY_SIZE)
        total_bufs = DOCA_RDMA_BUF_INVENTORY_SIZE;

    ret = doca_buf_inventory_create(total_bufs, &ctx->buf_inv);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_buf_inventory_create failed: %s", doca_error_get_name(ret));
        return ret;
    }

    ret = doca_buf_inventory_start(ctx->buf_inv);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_buf_inventory_start failed: %s", doca_error_get_name(ret));
        return ret;
    }

    /* ctx_user_data を start 前に設定 */
    if (ctx_user_data != NULL) {
        union doca_data ud;
        ud.ptr = ctx_user_data;
        ret = doca_ctx_set_user_data(ctx->ctx, ud);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_ctx_set_user_data failed: %s", doca_error_get_name(ret));
            return ret;
        }
    }

    /* コンテキスト開始 */
    ret = doca_ctx_start(ctx->ctx);
    if (ret != DOCA_SUCCESS && ret != DOCA_ERROR_IN_PROGRESS) {
        DOCA_LOG_ERR("doca_ctx_start failed: %s", doca_error_get_name(ret));
        return ret;
    }

    ctx->ctx_started = true;
    return DOCA_SUCCESS;
}

/* ---- 破棄 ---- */

doca_error_t doca_rdma_ctx_destroy(struct doca_rdma_ctx_t *ctx)
{
    if (ctx == NULL)
        return DOCA_SUCCESS;

    if (ctx->ctx_started && ctx->ctx != NULL) {
        doca_ctx_stop(ctx->ctx);
        /* PE を回して IDLE になるまで待つ */
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000};
        for (int i = 0; i < 100000; i++) {
            doca_pe_progress(ctx->pe);
            enum doca_ctx_states state;
            if (doca_ctx_get_state(ctx->ctx, &state) == DOCA_SUCCESS && state == DOCA_CTX_STATE_IDLE)
                break;
            nanosleep(&ts, NULL);
        }
    }

    if (ctx->buf_inv != NULL) {
        doca_buf_inventory_stop(ctx->buf_inv);
        doca_buf_inventory_destroy(ctx->buf_inv);
        ctx->buf_inv = NULL;
    }

    if (ctx->rdma != NULL) {
        doca_rdma_destroy(ctx->rdma);
        ctx->rdma = NULL;
        ctx->ctx = NULL;
    }

    if (ctx->local_mmap != NULL) {
        doca_mmap_stop(ctx->local_mmap);
        doca_mmap_destroy(ctx->local_mmap);
        ctx->local_mmap = NULL;
    }

    /* PE は外部管理の可能性があるため、ここでは destroy しない */

    return DOCA_SUCCESS;
}

/* ---- 接続管理 ---- */

doca_error_t doca_rdma_ctx_export(
    struct doca_rdma_ctx_t *ctx,
    int conn_index)
{
    if (conn_index < 0 || conn_index >= 8)
        return DOCA_ERROR_INVALID_VALUE;

    doca_error_t ret = doca_rdma_export(ctx->rdma,
        &ctx->local_conn_desc,
        &ctx->local_conn_desc_len,
        &ctx->connections[conn_index]);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_rdma_export failed: %s", doca_error_get_name(ret));
        return ret;
    }
    if (conn_index >= ctx->num_connections)
        ctx->num_connections = conn_index + 1;

    return DOCA_SUCCESS;
}

doca_error_t doca_rdma_ctx_connect(
    struct doca_rdma_ctx_t *ctx,
    int conn_index,
    const void *remote_conn_desc,
    size_t remote_conn_desc_len)
{
    if (conn_index < 0 || conn_index >= 8)
        return DOCA_ERROR_INVALID_VALUE;
    if (ctx->connections[conn_index] == NULL)
        return DOCA_ERROR_BAD_STATE;

    doca_error_t ret = doca_rdma_connect(ctx->rdma,
        remote_conn_desc,
        remote_conn_desc_len,
        ctx->connections[conn_index]);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_rdma_connect failed: %s", doca_error_get_name(ret));
        return ret;
    }

    /* PE を回して RUNNING になるまで待つ */
    ret = wait_for_ctx_running(ctx->pe, ctx->ctx);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("wait_for_ctx_running failed after connect: %s", doca_error_get_name(ret));
        return ret;
    }

    return DOCA_SUCCESS;
}

/* ---- リモートメモリ ---- */

doca_error_t doca_remote_mem_create(
    struct doca_remote_mem_t *rmem,
    struct doca_dev *dev,
    const void *export_desc,
    size_t export_desc_len)
{
    memset(rmem, 0, sizeof(*rmem));

    doca_error_t ret = doca_mmap_create_from_export(NULL,
        export_desc, export_desc_len,
        dev, &rmem->remote_mmap);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_mmap_create_from_export failed: %s", doca_error_get_name(ret));
        return ret;
    }

    /* リモート mmap のメモリ範囲を取得 */
    ret = doca_mmap_get_memrange(rmem->remote_mmap,
        (void **)&rmem->remote_addr, &rmem->remote_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_WARN("doca_mmap_get_memrange failed: %s", doca_error_get_name(ret));
        /* アドレスとサイズは呼び出し元が control_cmd から設定する */
    }

    rmem->valid = true;
    return DOCA_SUCCESS;
}

void doca_remote_mem_destroy(struct doca_remote_mem_t *rmem)
{
    if (rmem == NULL || !rmem->valid)
        return;
    if (rmem->remote_mmap != NULL) {
        doca_mmap_destroy(rmem->remote_mmap);
        rmem->remote_mmap = NULL;
    }
    rmem->valid = false;
}

/* ---- ローカル mmap エクスポート ---- */

doca_error_t doca_rdma_ctx_export_mmap(
    struct doca_rdma_ctx_t *ctx,
    const void **export_desc,
    size_t *export_desc_len)
{
    if (ctx->local_mmap == NULL)
        return DOCA_ERROR_BAD_STATE;

    return doca_mmap_export_rdma(ctx->local_mmap, ctx->dev,
        export_desc, export_desc_len);
}
