#ifndef UCP_UTILS_H
#define UCP_UTILS_H

#include "comch_mpi_common.h"
#include<doca_log.h>
#include<doca_error.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/spinlock.h>
#include <ucs/datastruct/mpool.h>

KHASH_MAP_INIT_INT64(ep, ucp_ep_h);

struct ucp_data_t {
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_address_t *worker_address;
    size_t ucp_addrlen;        //UCPワーカのアドレス長
    khash_t(ep) * eps;	       /* Worker endpoints map */

    ucs_list_link_t completed_reqs;
};

doca_error_t create_ucp_worker(struct ucp_data_t *ucp_data);
doca_error_t register_ucp_worker_addresses(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size);
ucs_status_t ucx_wait(ucp_worker_h worker, void *request, const char *op_name);

//コマンドを送受信するための関数, これによりRDMAの記述子を交換するよ
doca_error_t run_ucp_tag_exchange_cmd(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size, uint64_t src_rank, uint64_t dst_rank,
                         struct control_cmd *send_cmd, struct control_cmd **recv_cmd, 
                         /*malloc した受信バッファ*/ void **recv_storage, /*そのサイズ (≒CONTROL_CMD_MAX_SIZE)*/ size_t *recv_storage_len);

#endif