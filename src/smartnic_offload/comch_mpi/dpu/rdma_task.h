#ifndef RDMA_TASK_H
#define RDMA_TASK_H

#include "comch_server.h"

/* RDMA タスク発行層 (rdma_task.c)
 *   - DOCA RDMA タスクのコールバック / inline submit
 *   - PE progress 駆動ワーカスレッドプール */

void rdma_read_comp_cb(struct doca_rdma_task_read *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_read_err_cb(struct doca_rdma_task_read *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_write_comp_cb(struct doca_rdma_task_write *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_write_err_cb(struct doca_rdma_task_write *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_send_comp_cb(struct doca_rdma_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_send_err_cb(struct doca_rdma_task_send *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_recv_comp_cb(struct doca_rdma_task_receive *task, union doca_data task_user_data, union doca_data ctx_user_data);
void rdma_recv_err_cb(struct doca_rdma_task_receive *task, union doca_data task_user_data, union doca_data ctx_user_data);

void submit_doca_task_from_desc(struct doca_rdma_ctx_t *rdma_ctx, const struct doca_task_desc *desc);
int  doca_worker_thread_pool_init(struct doca_worker_thread_pool_t *pool, struct collective_worker_t *cw, int mpi_rank);
void doca_worker_thread_pool_destroy(struct doca_worker_thread_pool_t *pool);
void submit_and_wait_doca(struct doca_worker_thread_pool_t *pool, struct doca_task_desc *task);

/* * DOCA タスク初期化ヘルパー */

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

#endif /* RDMA_TASK_H */
