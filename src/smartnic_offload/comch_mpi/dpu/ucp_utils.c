#include <signal.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <doca_comch.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>
#include <ucp/api/ucp.h>
#include <mpi.h>

#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "ucp_utils.h"
#include "../common/comch_mpi_common.h"

DOCA_LOG_REGISTER(UCP_UTILS);

doca_error_t create_ucp_worker(struct ucp_data_t *ucp_data)
{
	ucs_status_t status;
    ucp_params_t ucp_params;
    ucp_config_t *ucp_config;
	ucp_context_h ucp_context;
    ucp_worker_params_t worker_params;

	status = ucp_config_read(NULL, NULL, &ucp_config);
	if(status != UCS_OK) {
		DOCA_LOG_ERR("Failed to read UCP config");
		return DOCA_ERROR_NOT_FOUND;
	}
	status = ucp_config_modify(ucp_config, "TLS", "^sm");
	if(status != UCS_OK)
		return DOCA_ERROR_NOT_FOUND;
#if UCP_API_VERSION >= UCP_VERSION(1, 17)
	status = ucp_config_modify(ucp_config, "TCP_PUT_ENABLE", "n");
#else
	status = ucp_config_modify(ucp_config, "PUT_ENABLE", "n");
#endif
	if (status != UCS_OK) {
        DOCA_LOG_ERR("Failed to read UCP config");
        ucp_config_release(ucp_config);
        return DOCA_ERROR_NOT_FOUND;
    }

	ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
    ucp_params.features |= UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = 1;
    status = ucp_init(&ucp_params, ucp_config, &ucp_context);
	ucp_config_release(ucp_config);
	if (status != UCS_OK)
		return DOCA_ERROR_NOT_FOUND;
	//複数スレッドに拡張するときにまた来てね!!
	worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    //worker_params.thread_mode  = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(ucp_context, &worker_params, &(ucp_data->ucp_worker));
    if(status != UCS_OK) {
        DOCA_LOG_ERR("Unable to create ucp worker");
        goto err_worker_create;
    }
	status = ucp_worker_get_address(ucp_data->ucp_worker, &(ucp_data->worker_address), &(ucp_data->ucp_addrlen));
    if(status != UCS_OK) {
        DOCA_LOG_ERR("Unable to get ucp worker address");
        goto err_worker_address;
    }
    ucp_data->ucp_context = ucp_context;
	ucp_data->eps = kh_init(ep);
	if(!ucp_data->eps)
        goto err_eps;

	ucs_list_head_init(&ucp_data->completed_reqs); //完了通知キューの初期化, この部分しかないのでじかに触れていい気がするよ
    return DOCA_SUCCESS;

err_eps:
	ucp_worker_release_address(ucp_data->ucp_worker, ucp_data->worker_address);
err_worker_address:
	ucp_worker_destroy(ucp_data->ucp_worker);
err_worker_create:
	ucp_cleanup(ucp_context);
	return DOCA_ERROR_NOT_FOUND;
}

doca_error_t register_ucp_worker_addresses(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size)
{
    doca_error_t doca_status = DOCA_SUCCESS;
    ucs_status_t ucs_status;

    /* --- Step 1: 各ランクのワーカアドレス長を交換 (Allgather) --- */
    size_t my_addrlen = ucp_data->ucp_addrlen;
    size_t *all_addrlen = malloc(sizeof(size_t) * world_size);
    if (all_addrlen == NULL)
        return DOCA_ERROR_NO_MEMORY;
    MPI_Allgather(&my_addrlen, sizeof(size_t), MPI_BYTE, all_addrlen, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    /* --- Step 2: 全ランクのアドレスを入れるバッファを確保 --- */
    size_t total_size = 0;
    for (size_t i = 0; i < world_size; i++)
        total_size += all_addrlen[i];
    uint8_t *all_addrs = malloc(total_size);
    if (all_addrs == NULL) {
        free(all_addrlen);
        return DOCA_ERROR_NO_MEMORY;
    }
    /* --- Step 3: 各 rank のアドレスのオフセットを計算 --- */
    int *recvcounts = malloc(sizeof(int) * world_size);
    int *displs     = malloc(sizeof(int) * world_size);
    if (!recvcounts || !displs) {
        free(all_addrlen);
        free(all_addrs);
        free(recvcounts);
        free(displs);
        return DOCA_ERROR_NO_MEMORY;
    }
    size_t offset = 0;
    for (size_t i = 0; i < world_size; i++) {
        recvcounts[i] = all_addrlen[i];
        displs[i]     = offset;
        offset       += all_addrlen[i];
    }

    /* --- Step 4: アドレス本体を Allgatherv で交換 --- */
    MPI_Allgatherv(ucp_data->worker_address, my_addrlen, MPI_BYTE, all_addrs, recvcounts, displs, MPI_BYTE, MPI_COMM_WORLD);

    /* --- Step 5: 各 rank のアドレスで UCP EP を作成し、ハッシュへ登録 --- */
    for (size_t peer = 0; peer < world_size; peer++) {
        if (peer == rank)
            continue;  /* 自分自身へ EP は不要 */
        void *peer_addr = all_addrs + displs[peer];
        size_t peer_len = all_addrlen[peer];
        ucp_ep_params_t ep_params = {0};
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = peer_addr;
        ucp_ep_h ep;
        ucs_status = ucp_ep_create(ucp_data->ucp_worker, &ep_params, &ep);
        if (ucs_status != UCS_OK) {
            DOCA_LOG_ERR("Failed to create EP for peer %lu: %s", peer, ucs_status_string(ucs_status));
            doca_status = DOCA_ERROR_DRIVER;
            break;
        }
        /* --- ハッシュテーブルに保存 --- */
        int ret;
        khint_t k = kh_put(ep, ucp_data->eps, peer, &ret);
        if (ret <= 0) {
            DOCA_LOG_ERR("Failed to insert EP entry into hash map");
            ucp_ep_destroy(ep);
            doca_status = DOCA_ERROR_NO_MEMORY;
            break;
        }
        kh_value(ucp_data->eps, k) = ep;

        /* --- ログ: peer の UCP アドレスを hex で出力 --- */
        char *hex_str = malloc(peer_len * 2 + 1);
        if (hex_str != NULL) {
            for (size_t bi = 0; bi < peer_len; bi++)
                sprintf(&hex_str[bi * 2], "%02X", ((uint8_t*)peer_addr)[bi]);
            DOCA_LOG_INFO("[LOG][rank:%lu] Registered EP for peer %lu | addr_len=%lu | addr_hex=%s",rank, peer, peer_len, hex_str);
            free(hex_str);
        } else {
            DOCA_LOG_WARN("[LOG][rank:%lu] Registered EP for peer %lu (addr_len=%lu, hex_dump_failed)", rank, peer, peer_len);
        }
    }
    free(all_addrlen);
    free(all_addrs);
    free(recvcounts);
    free(displs);
    return doca_status;
}

ucs_status_t ucx_wait(ucp_worker_h worker, void *request, const char *op_name)
{
    ucs_status_t status;
    /* NULL なら即完了 */
    if (request == NULL)
        return UCS_OK;
    /* エラー付きポインタかどうか */
    if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
        DOCA_LOG_ERR("%s request error: %s", op_name, ucs_status_string(status));
        return status;
    }
    /* INPROGRESS の間は progress を回し続ける */
    do {
        ucp_worker_progress(worker);
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
    ucp_request_free(request);
    if (status != UCS_OK) 
        DOCA_LOG_ERR("%s completed with %s", op_name, ucs_status_string(status));
    return status;
}

//タグ付きメッセージをsrc_rank, dst_rank間で交換
doca_error_t run_ucp_tag_exchange_cmd(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size, uint64_t src_rank, uint64_t dst_rank,
                         struct control_cmd *send_cmd, struct control_cmd **recv_cmd, /*malloc した受信バッファ*/ void **recv_storage, /*そのサイズ (≒CONTROL_CMD_MAX_SIZE)*/ size_t *recv_storage_len)
{
    const ucp_tag_t TAG_FWD   = 0x500; /* src -> dst: control_cmd */
    const ucp_tag_t TAG_REPLY = 0x501; /* dst -> src: control_cmd */
    const ucp_tag_t TAG_MASK  = (ucp_tag_t)-1;

    ucp_request_param_t param;
    void *req;
    ucs_status_t ucs_status;
    if (recv_cmd)
        *recv_cmd = NULL;
    if (recv_storage)
        *recv_storage = NULL;
    if (recv_storage_len)
        *recv_storage_len = 0;

    if (world_size <= src_rank || world_size <= dst_rank) {
        DOCA_LOG_ERR("world_size (%lu) is too small for src=%lu dst=%lu", world_size, src_rank, dst_rank);
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (src_rank == dst_rank) {
        DOCA_LOG_ERR("src_rank (%lu) must be different from dst_rank (%lu)", src_rank, dst_rank);
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* 全ランクの足並みを揃える */
    if (rank == src_rank) {
        /* === src_rank 側 === */
        khint_t k = kh_get(ep, ucp_data->eps, dst_rank);
        if (k == kh_end(ucp_data->eps)) {
            DOCA_LOG_ERR("EP for peer %lu not found in hash", dst_rank);
            return DOCA_ERROR_NOT_CONNECTED;
        }
        ucp_ep_h ep = kh_value(ucp_data->eps, k);

        /* 送信用に control_cmd をパック */
        uint8_t send_buf[CONTROL_CMD_MAX_SIZE];
        size_t  send_cap = CONTROL_CMD_MAX_SIZE;
        doca_error_t st = control_cmd_pack(send_cmd, &send_cap, send_buf);
        if (st != DOCA_SUCCESS) {
            DOCA_LOG_ERR("control_cmd_pack failed");
            return st;
        }

        /* 1) src -> dst にコマンドを送信（長さメッセージなし） */
        memset(&param, 0, sizeof(param));
        req = ucp_tag_send_nbx(ep, send_buf, send_cap, TAG_FWD, &param);
        ucs_status = ucx_wait(ucp_data->ucp_worker, req, "cmd_send src->dst");
        if (ucs_status != UCS_OK)
            return DOCA_ERROR_DRIVER;

        /* 2) dst -> src からの返信コマンドを受信 */
        void *recv_buf = malloc(CONTROL_CMD_MAX_SIZE);
        if (!recv_buf)
            return DOCA_ERROR_NO_MEMORY;

        memset(&param, 0, sizeof(param));
        req = ucp_tag_recv_nbx(ucp_data->ucp_worker, recv_buf, CONTROL_CMD_MAX_SIZE, TAG_REPLY, TAG_MASK, &param);
        ucs_status = ucx_wait(ucp_data->ucp_worker, req, "cmd_recv dst->src");
        if (ucs_status != UCS_OK) {
            free(recv_buf);
            return DOCA_ERROR_DRIVER;
        }

        /* 3) in-place アンパックして control_cmd* を得る */
        struct control_cmd *rcmd = NULL;
        st = control_cmd_unpack(recv_buf, CONTROL_CMD_MAX_SIZE, &rcmd);
        if (st != DOCA_SUCCESS) {
            free(recv_buf);
            return st;
        }
        if (recv_cmd)
            *recv_cmd = rcmd;
        if (recv_storage) {
            *recv_storage = recv_buf;
        } else {
            /* 呼び出し側が要らないならここで解放してもよい */
            free(recv_buf);
        }
        if (recv_storage_len)
            *recv_storage_len = CONTROL_CMD_MAX_SIZE;
        DOCA_LOG_INFO("[UCX][rank %lu] exchanged control_cmd with rank %lu", rank, dst_rank);
    } else if (rank == dst_rank) {
        /* === dst_rank 側 === */
        khint_t k = kh_get(ep, ucp_data->eps, src_rank);
        if (k == kh_end(ucp_data->eps)) {
            DOCA_LOG_ERR("EP for peer %lu not found in hash", src_rank);
            return DOCA_ERROR_NOT_CONNECTED;
        }
        ucp_ep_h ep = kh_value(ucp_data->eps, k);

        /* 1) src からのコマンドを受信 */
        void *recv_buf = malloc(CONTROL_CMD_MAX_SIZE);
        if (!recv_buf)
            return DOCA_ERROR_NO_MEMORY;

        memset(&param, 0, sizeof(param));
        req = ucp_tag_recv_nbx(ucp_data->ucp_worker, recv_buf, CONTROL_CMD_MAX_SIZE, TAG_FWD, TAG_MASK, &param);
        ucs_status = ucx_wait(ucp_data->ucp_worker, req, "cmd_recv src->dst");
        if (ucs_status != UCS_OK) {
            free(recv_buf);
            return DOCA_ERROR_DRIVER;
        }

        struct control_cmd *rcmd = NULL;
        doca_error_t st = control_cmd_unpack(recv_buf, CONTROL_CMD_MAX_SIZE, &rcmd);
        if (st != DOCA_SUCCESS) {
            free(recv_buf);
            return st;
        }

        if (recv_cmd)
            *recv_cmd = rcmd;
        if (recv_storage)
            *recv_storage = recv_buf;
        if (recv_storage_len)
            *recv_storage_len = CONTROL_CMD_MAX_SIZE;

        /* 2) 自分の返信コマンドをパックして送信 */
        uint8_t send_buf[CONTROL_CMD_MAX_SIZE];
        size_t  send_cap = CONTROL_CMD_MAX_SIZE;
        st = control_cmd_pack(send_cmd, &send_cap, send_buf);
        if (st != DOCA_SUCCESS) {
            DOCA_LOG_ERR("control_cmd_pack (reply) failed");
            return st;
        }

        memset(&param, 0, sizeof(param));
        req = ucp_tag_send_nbx(ep, send_buf, send_cap, TAG_REPLY, &param);
        ucs_status = ucx_wait(ucp_data->ucp_worker, req, "cmd_send dst->src");
        if (ucs_status != UCS_OK)
            return DOCA_ERROR_DRIVER;
        DOCA_LOG_INFO("[UCX][rank %lu] exchanged control_cmd with rank %lu", rank, src_rank);
    } else {
        /* その他のランク: 何もしないで Barrier だけ合わせる */
        DOCA_LOG_INFO("[UCX][rank %lu] skip control_cmd exchange (only %lu <-> %lu)", rank, src_rank, dst_rank);
    }

    return DOCA_SUCCESS;
}