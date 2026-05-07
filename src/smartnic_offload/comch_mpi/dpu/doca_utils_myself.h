#ifndef DOCA_UTILS_MYSELF_H_
#define DOCA_UTILS_MYSELF_H_

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_sync_event.h>
#include <doca_comch.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>
#include <ucp/api/ucp.h>
#include <mpi.h>

/*-------------------ユーティリティ関数 --------------------*/
/* ---- hexdump helper ---- */
static void hexdump(const void *data, size_t size)
{
    const unsigned char *p = (const unsigned char *)data;

    for (size_t i = 0; i < size; i++) {
        printf("%02x ", p[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    if (size % 16 != 0)
        printf("\n");
}

/*このマクロによりDOCA_CHECK(関数)とすると関数がエラーの時はメッセージを出してそのまま帰ってくれる*/
#define DOCA_CHECK(expr)                                     \
    do {                                                     \
        doca_error_t __ret = (expr);                         \
        if (__ret != DOCA_SUCCESS) {                         \
            DOCA_LOG_ERR("Failed to %s", #expr);             \
            return __ret;                                    \
        }                                                    \
    } while (0)

/*==================== RDMA ユーティリティ / コールバック ====================*/

/* device_name から rdma_config を初期化するユーティリティ */
doca_error_t config_init_with_device_name(struct rdma_config *cfg,
                                          const void *device_name_void,
                                          size_t device_name_len);

/* RDMA send task 完了コールバック */
void rdma_send_completed_callback(struct doca_rdma_task_send *rdma_send_task,
                                  union doca_data task_user_data,
                                  union doca_data ctx_user_data);

/* RDMA send task エラーコールバック */
void rdma_send_error_callback(struct doca_rdma_task_send *rdma_send_task,
                              union doca_data task_user_data,
                              union doca_data ctx_user_data);

/* RDMA send 用コンテキスト状態変化コールバック */
void rdma_send_state_change_callback(const union doca_data user_data,
                                     struct doca_ctx *ctx,
                                     enum doca_ctx_states prev_state,
                                     enum doca_ctx_states next_state);

/* RDMA receive task 完了コールバック */
void rdma_receive_completed_callback(struct doca_rdma_task_receive *rdma_receive_task,
                                     union doca_data task_user_data,
                                     union doca_data ctx_user_data);

/* RDMA receive task エラーコールバック */
void rdma_receive_error_callback(struct doca_rdma_task_receive *rdma_receive_task,
                                 union doca_data task_user_data,
                                 union doca_data ctx_user_data);

/* RDMA receive 用コンテキスト状態変化コールバック */
void rdma_receive_state_change_callback(const union doca_data user_data,
                                        struct doca_ctx *ctx,
                                        enum doca_ctx_states prev_state,
                                        enum doca_ctx_states next_state);

/* RDMA リソース生成ユーティリティ（send/recv 共通） */
doca_error_t create_rdma_resources(const void *device_name_void,
                                   size_t device_name_len,
                                   struct rdma_resources *resources,
                                   bool is_send,
                                   size_t working_buffer_size);

/* RDMA コンテキストに user_data を設定し、RUNNING まで起動するユーティリティ */
doca_error_t rdma_ctx_set_user_data_and_start(struct rdma_resources *resources);

#endif