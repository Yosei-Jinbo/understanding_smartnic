#ifndef DOCA_RDMA_UTILS_H
#define DOCA_RDMA_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
#include <atomic>
using std::atomic_int;
using std::atomic_bool;
using std::atomic_store;
using std::atomic_load;
using std::atomic_store_explicit;
using std::atomic_load_explicit;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_relaxed;
#else
#include <stdatomic.h>
#endif

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_ctx.h>
#include <doca_log.h>

/* ---- 定数 ---- */
#define DOCA_RDMA_BUF_INVENTORY_SIZE   256

#ifdef __cplusplus
extern "C" {
#endif

/* ---- DOCA RDMA データコンテナ (ucp_data_t の置換) ---- */
struct doca_rdma_ctx_t {
    struct doca_dev           *dev;
    struct doca_pe            *pe;         /* Progress Engine (共有可能) */
    struct doca_rdma          *rdma;
    struct doca_ctx           *ctx;

    struct doca_mmap          *local_mmap;
    void                      *local_mmap_addr;
    size_t                     local_mmap_size;

    struct doca_buf_inventory *buf_inv;

    /* export/connect 用 */
    const void                *local_conn_desc;
    size_t                     local_conn_desc_len;
    struct doca_rdma_connection *connections[8];  /* 最大 8 接続 */
    int                        num_connections;

    bool                       ctx_started;
};

/* ---- Host GPU メモリのリモートアクセス情報 ---- */
struct doca_remote_mem_t {
    struct doca_mmap *remote_mmap;       /* doca_mmap_create_from_export で作成 */
    uint64_t          remote_addr;       /* リモートバッファのベースアドレス */
    size_t            remote_len;        /* リモートバッファのサイズ */
    bool              valid;
};

/* ---- 初期化・破棄 ---- */

/*
 * DOCA RDMA コンテキストを初期化する。
 *   dev: 既にオープン済みの doca_dev (外部管理)
 *   pe:  既にある PE を使う場合は非 NULL、NULL なら新規作成
 *   local_addr, local_size: ローカル mmap に登録する領域 (NULL なら登録しない)
 *   permissions: DOCA_ACCESS_FLAG_* の組み合わせ
 *   send_q_size, recv_q_size: RDMA キューサイズ
 *   gid_index: RoCE GID インデックス (0 = RoCE v1, 3 = 通常 RoCE v2)
 *              Host-DPU 間で一致させる必要がある
 */
doca_error_t doca_rdma_ctx_init(
    struct doca_rdma_ctx_t *ctx,
    struct doca_dev *dev,
    struct doca_pe *pe,           /* NULL で新規 PE 作成 */
    void *local_addr,
    size_t local_size,
    uint32_t permissions,
    uint32_t send_q_size,
    uint32_t recv_q_size,
    uint32_t gid_index);

/* タスクコールバックを設定し、コンテキストを開始する。
 * ctx_user_data: コールバックの ctx_user_data に渡す値 (NULL なら設定しない) */
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
    void *ctx_user_data);

/* DOCA RDMA コンテキストを破棄する */
doca_error_t doca_rdma_ctx_destroy(struct doca_rdma_ctx_t *ctx);

/* ---- 接続管理 ---- */

/*
 * 接続記述子をエクスポートし、新しい connection を取得する。
 * 呼び出し後、local_conn_desc / local_conn_desc_len が設定される。
 * connections[conn_index] に格納。
 */
doca_error_t doca_rdma_ctx_export(
    struct doca_rdma_ctx_t *ctx,
    int conn_index);

/*
 * リモートの接続記述子で接続を確立する。
 * connections[conn_index] に対して connect する。
 */
doca_error_t doca_rdma_ctx_connect(
    struct doca_rdma_ctx_t *ctx,
    int conn_index,
    const void *remote_conn_desc,
    size_t remote_conn_desc_len);

/* ---- リモートメモリ ---- */

/*
 * mmap export descriptor からリモートメモリを作成する。
 */
doca_error_t doca_remote_mem_create(
    struct doca_remote_mem_t *rmem,
    struct doca_dev *dev,
    const void *export_desc,
    size_t export_desc_len);

void doca_remote_mem_destroy(struct doca_remote_mem_t *rmem);

/* ---- ローカル mmap エクスポート ---- */

/*
 * ローカル mmap を RDMA 用にエクスポートする。
 * export_desc, export_desc_len に結果が格納される。
 */
doca_error_t doca_rdma_ctx_export_mmap(
    struct doca_rdma_ctx_t *ctx,
    const void **export_desc,
    size_t *export_desc_len);

/* ---- Progress ---- */

static inline int doca_rdma_progress(struct doca_rdma_ctx_t *ctx)
{
    return (int)doca_pe_progress(ctx->pe);
}

/* ---- doca_buf ヘルパー ---- */

/*
 * ローカル mmap から doca_buf を取得する。
 * addr は local_mmap 内のアドレス、len はサイズ。
 */
static inline doca_error_t doca_rdma_get_local_buf(
    struct doca_rdma_ctx_t *ctx,
    void *addr,
    size_t len,
    struct doca_buf **buf)
{
    return doca_buf_inventory_buf_get_by_addr(
        ctx->buf_inv, ctx->local_mmap, addr, len, buf);
}

/*
 * リモート mmap から doca_buf を取得する。
 * addr はリモートのアドレス、len はサイズ。
 */
static inline doca_error_t doca_rdma_get_remote_buf(
    struct doca_rdma_ctx_t *ctx,
    struct doca_remote_mem_t *rmem,
    void *addr,
    size_t len,
    struct doca_buf **buf)
{
    return doca_buf_inventory_buf_get_by_addr(
        ctx->buf_inv, rmem->remote_mmap, addr, len, buf);
}

#ifdef __cplusplus
}
#endif

#endif /* DOCA_RDMA_UTILS_H */
