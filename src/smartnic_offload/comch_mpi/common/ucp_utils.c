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

/* 
 * MPI を使って control_cmd を src_rank <-> dst_rank で「双方向」に交換する。
 * - 各ランクは自分の send_cmd を送信し、
 * - 相手ランクの send_cmd が、自分側の recv_cmd として返ってくる。
 *
 * UCP 版 run_ucp_tag_exchange_cmd と似たインターフェースにしてある。
 */
doca_error_t run_mpi_tag_exchange_cmd(uint64_t rank, uint64_t world_size,
                                      uint64_t src_rank, uint64_t dst_rank,
                                      struct control_cmd *send_cmd,
                                      struct control_cmd **recv_cmd,
                                      void **recv_storage,
                                      size_t *recv_storage_len)
{
    const int TAG_EXCHANGE = 0x500; /* 適当なタグ。必要なら id を足してもよい */

    if (recv_cmd)
        *recv_cmd = NULL;
    if (recv_storage)
        *recv_storage = NULL;
    if (recv_storage_len)
        *recv_storage_len = 0;

    /* rank の範囲チェック */
    if (world_size <= src_rank || world_size <= dst_rank)
        return DOCA_ERROR_INVALID_VALUE;
    if (src_rank == dst_rank)
        return DOCA_ERROR_INVALID_VALUE;

    /* src/dst 以外のランクは何もしない（UCP 版と同じ挙動） */
    if (rank != src_rank && rank != dst_rank) {
        return DOCA_SUCCESS;
    }

    /* 通信相手の rank */
    int partner = (rank == src_rank) ? (int)dst_rank : (int)src_rank;

    /* === 自分の send_cmd を pack === */
    uint8_t send_buf[CONTROL_CMD_MAX_SIZE];
    size_t  send_cap = CONTROL_CMD_MAX_SIZE;

    doca_error_t st = control_cmd_pack(send_cmd, &send_cap, send_buf);
    if (st != DOCA_SUCCESS) {
        printf("control_cmd_pack failed in run_mpi_tag_exchange_cmd\n");
        return st;
    }

    /* === 相手からの cmd を受信するバッファを用意 ===
     * control_cmd_unpack は packed バッファの中にポインタを張るので、
     * ここで確保した recv_buf を *free してはいけない*。
     * 呼び出し側で free できるように recv_storage に渡す。
     */
    void *recv_buf = malloc(CONTROL_CMD_MAX_SIZE);
    if (!recv_buf)
        return DOCA_ERROR_NO_MEMORY;

    MPI_Status status;
    int rc = MPI_Sendrecv(send_buf, (int)send_cap, MPI_BYTE,
                          partner, TAG_EXCHANGE,
                          recv_buf, CONTROL_CMD_MAX_SIZE, MPI_BYTE,
                          partner, TAG_EXCHANGE,
                          MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS) {
        printf("MPI_Sendrecv failed in run_mpi_tag_exchange_cmd, rc=%d\n", rc);
        free(recv_buf);
        return DOCA_ERROR_DRIVER;
    }

    int recv_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &recv_count);

    /* === 受信したバイト列から control_cmd 構造体を復元 === */
    struct control_cmd *rcmd = NULL;
    st = control_cmd_unpack(recv_buf, (size_t)recv_count, &rcmd);
    if (st != DOCA_SUCCESS) {
        printf("control_cmd_unpack failed in run_mpi_tag_exchange_cmd\n");
        free(recv_buf);
        return st;
    }

    /* ここで rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address などの
       ポインタは recv_buf の中を指しているので、recv_buf を保持しておく必要がある */

    if (recv_cmd)
        *recv_cmd = rcmd;
    if (recv_storage)
        *recv_storage = recv_buf;
    else
        free(recv_buf);  /* 呼び出し側が要らないならここで解放 */
    if (recv_storage_len)
        *recv_storage_len = (size_t)recv_count;

    return DOCA_SUCCESS;
}

/* create_ucp_worker:
 * 呼び出し前に ucp_data->ucp_context を共有コンテキストへのポインタにセットしておくこと。
 * この関数はコンテキストを作成しない。ワーカのみを作成する。
 */
doca_error_t create_ucp_worker(struct ucp_data_t *ucp_data)
{
    ucs_status_t status;
    ucp_worker_params_t worker_params;

    if (ucp_data->ucp_context == NULL) {
        DOCA_LOG_ERR("ucp_data->ucp_context is NULL; set shared context pointer before calling create_ucp_worker");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* 複数スレッドへ拡張する際は thread_mode を見直すこと */
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(*(ucp_data->ucp_context), &worker_params, &(ucp_data->ucp_worker));
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Unable to create ucp worker: %d %s", (int)status, ucs_status_string(status));
        goto err_worker_create;
    }
    status = ucp_worker_get_address(ucp_data->ucp_worker, &(ucp_data->worker_address), &(ucp_data->ucp_addrlen));
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Unable to get ucp worker address");
        goto err_worker_address;
    }
    printf("!!! Worker addr length: %lu\n", ucp_data->ucp_addrlen);
    ucp_data->eps = kh_init(ep);
    if (!ucp_data->eps)
        goto err_eps;

    /* 完了通知キューの初期化 */
    ucs_list_head_init(&ucp_data->completed_reqs);
    return DOCA_SUCCESS;

err_eps:
    ucp_worker_release_address(ucp_data->ucp_worker, ucp_data->worker_address);
err_worker_address:
    ucp_worker_destroy(ucp_data->ucp_worker);
err_worker_create:
    return DOCA_ERROR_NOT_FOUND;
}

doca_error_t register_ucp_worker_addresses(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size)
{
    doca_error_t doca_status = DOCA_SUCCESS;
    ucs_status_t ucs_status;

    /* --- Step 1: 各ランクのワーカアドレス長を交換 (Allgather) --- */
    size_t my_addrlen = ucp_data->ucp_addrlen;
    size_t *all_addrlen = (size_t *)malloc(sizeof(size_t) * world_size);
    if (all_addrlen == NULL)
        return DOCA_ERROR_NO_MEMORY;
    MPI_Allgather(&my_addrlen, sizeof(size_t), MPI_BYTE,
                  all_addrlen, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    /* --- Step 2: 全ランクのアドレスを入れるバッファを確保 --- */
    size_t total_size = 0;
    for (size_t i = 0; i < world_size; i++)
        total_size += all_addrlen[i];
    uint8_t *all_addrs = (uint8_t *)malloc(total_size);
    if (all_addrs == NULL) {
        free(all_addrlen);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* --- Step 3: 各 rank のアドレスのオフセットを計算 --- */
    int *recvcounts = (int *)malloc(sizeof(int) * world_size);
    int *displs     = (int *)malloc(sizeof(int) * world_size);
    if (!recvcounts || !displs) {
        free(all_addrlen);
        free(all_addrs);
        free(recvcounts);
        free(displs);
        return DOCA_ERROR_NO_MEMORY;
    }
    size_t offset = 0;
    for (size_t i = 0; i < world_size; i++) {
        recvcounts[i] = (int)all_addrlen[i];
        displs[i]     = (int)offset;
        offset       += all_addrlen[i];
    }

    /* --- Step 4: アドレス本体を Allgatherv で交換 --- */
    MPI_Allgatherv(ucp_data->worker_address, my_addrlen, MPI_BYTE,
                   all_addrs, recvcounts, displs, MPI_BYTE, MPI_COMM_WORLD);

    /* --- Step 5: 各 rank のアドレスで UCP EP を作成し、ハッシュへ登録 --- */
    for (size_t peer = 0; peer < world_size; peer++) {
        if (peer == rank)
            continue;  /* 自分自身へ EP は不要 */

        void *peer_addr = all_addrs + displs[peer];
        size_t peer_len = all_addrlen[peer];

        ucp_ep_params_t ep_params;
        memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        /* C++ では void* から const ucp_address_t* への代入にキャストが必要 */
        ep_params.address    = (const ucp_address_t *)peer_addr;

        ucp_ep_h ep;
        ucs_status = ucp_ep_create(ucp_data->ucp_worker, &ep_params, &ep);
        if (ucs_status != UCS_OK) {
            DOCA_LOG_ERR("Failed to create EP for peer %lu: %s",
                         (unsigned long)peer, ucs_status_string(ucs_status));
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
        char *hex_str = (char *)malloc(peer_len * 2 + 1);
        if (hex_str != NULL) {
            for (size_t bi = 0; bi < peer_len; bi++)
                sprintf(&hex_str[bi * 2], "%02X", ((uint8_t *)peer_addr)[bi]);
            hex_str[peer_len * 2] = '\0';
            DOCA_LOG_INFO("[LOG][rank:%lu] Registered EP for peer %lu | addr_len=%lu | addr_hex=%s",
                          (unsigned long)rank, (unsigned long)peer,
                          (unsigned long)peer_len, hex_str);
            free(hex_str);
        } else {
            DOCA_LOG_WARN("[LOG][rank:%lu] Registered EP for peer %lu (addr_len=%lu, hex_dump_failed)",
                          (unsigned long)rank, (unsigned long)peer,
                          (unsigned long)peer_len);
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

/* タグ付きメッセージをsrc_rank, dst_rank間で交換 */
doca_error_t run_ucp_tag_exchange_cmd(struct ucp_data_t *ucp_data,
                                      uint64_t rank, uint64_t world_size,
                                      uint64_t src_rank, uint64_t dst_rank,
                                      struct control_cmd *send_cmd,
                                      struct control_cmd **recv_cmd,
                                      /* malloc した受信バッファ */ void **recv_storage,
                                      /* そのサイズ (≒CONTROL_CMD_MAX_SIZE) */ size_t *recv_storage_len)
{
    printf("Start run ucp tag exchange cmd => rank:%lu, world_size:%lu, src_rank:%lu, dst_rank:%lu\n",
           (unsigned long)rank, (unsigned long)world_size,
           (unsigned long)src_rank, (unsigned long)dst_rank);
    fflush(stdout);

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
        DOCA_LOG_ERR("world_size (%lu) is too small for src=%lu dst=%lu",
                     (unsigned long)world_size,
                     (unsigned long)src_rank, (unsigned long)dst_rank);
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (src_rank == dst_rank) {
        DOCA_LOG_ERR("src_rank (%lu) must be different from dst_rank (%lu)",
                     (unsigned long)src_rank, (unsigned long)dst_rank);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (rank == src_rank) {
        printf("In run ucp tag exhange cmd => src_rank:%lu, before key-hash get\n",
               (unsigned long)src_rank);
        fflush(stdout);
        /* === src_rank 側 === */
        khint_t k = kh_get(ep, ucp_data->eps, dst_rank);
        if (k == kh_end(ucp_data->eps)) {
            DOCA_LOG_ERR("EP for peer %lu not found in hash", (unsigned long)dst_rank);
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

        printf("before ucp_tag send nbx\n");
        fflush(stdout);

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
        req = ucp_tag_recv_nbx(ucp_data->ucp_worker,
                               recv_buf, CONTROL_CMD_MAX_SIZE,
                               TAG_REPLY, TAG_MASK, &param);
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
            free(recv_buf);
        }
        if (recv_storage_len)
            *recv_storage_len = CONTROL_CMD_MAX_SIZE;

        DOCA_LOG_INFO("[UCX][rank %lu] exchanged control_cmd with rank %lu",
                      (unsigned long)rank, (unsigned long)dst_rank);

    } else if (rank == dst_rank) {
        printf("In run ucp tag exhange cmd => dst_rank:%lu, before key-hash get\n",
               (unsigned long)dst_rank);
        fflush(stdout);

        /* === dst_rank 側 === */
        khint_t k = kh_get(ep, ucp_data->eps, src_rank);
        if (k == kh_end(ucp_data->eps)) {
            DOCA_LOG_ERR("EP for peer %lu not found in hash", (unsigned long)src_rank);
            return DOCA_ERROR_NOT_CONNECTED;
        }
        ucp_ep_h ep = kh_value(ucp_data->eps, k);

        /* 1) src からのコマンドを受信 */
        void *recv_buf = malloc(CONTROL_CMD_MAX_SIZE);
        if (!recv_buf)
            return DOCA_ERROR_NO_MEMORY;

        memset(&param, 0, sizeof(param));
        req = ucp_tag_recv_nbx(ucp_data->ucp_worker,
                               recv_buf, CONTROL_CMD_MAX_SIZE,
                               TAG_FWD, TAG_MASK, &param);
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

        DOCA_LOG_INFO("[UCX][rank %lu] exchanged control_cmd with rank %lu",
                      (unsigned long)rank, (unsigned long)src_rank);

    } else {
        /* その他のランク: 何もしないでメッセージ交換はスキップ（Barrier 等は上位で調整） */
        DOCA_LOG_INFO("[UCX][rank %lu] skip control_cmd exchange (only %lu <-> %lu)",
                      (unsigned long)rank,
                      (unsigned long)src_rank, (unsigned long)dst_rank);
    }

    return DOCA_SUCCESS;
}

/*--------------------ハッシュによるrkeyの登録--------------------*/
/* addr,len から 64bit キーを作るヘルパ */
static inline uint64_t
make_addr_len_key(uint64_t addr, uint64_t len)
{
    uint64_t x = addr;
    x ^= len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x;
}

/*--------------------ハッシュ周りのユーティリティ関数ここから--------------------*/
/* ========= remote_ucp_t: remote_ucp_buffer_infos (value = remote_buffer_info*) ========= */

void remote_ucp_init(struct remote_ucp_t *remote)
{
    if (!remote)
        return;

    memset(remote, 0, sizeof(*remote));
    remote->remote_ucp_buffer_infos = kh_init(remote_ucp_buffer_info);
    ucs_spinlock_init(&remote->remote_ucp_lock, 0);

    remote->rkey_cache = kh_init(remote_rkey_cache);
    ucs_spinlock_init(&remote->rkey_cache_lock, 0);
}

void remote_ucp_destroy(struct remote_ucp_t *remote)
{
    if (!remote)
        return;

    /* まず remote_ucp_buffer_infos から remote_buffer_info* を free */
    if (remote->remote_ucp_buffer_infos) {
        khash_t(remote_ucp_buffer_info) *h = remote->remote_ucp_buffer_infos;
        khiter_t it;
        for (it = kh_begin(h); it != kh_end(h); ++it) {
            if (!kh_exist(h, it))
                continue;

            struct remote_buffer_info *info = kh_val(h, it);
            if (!info)
                continue;

            /* info->rkey は rkey_cache が管理するので destroy しない */
            free(info);
        }
        kh_destroy(remote_ucp_buffer_info, h);
        remote->remote_ucp_buffer_infos = NULL;
    }
    ucs_spinlock_destroy(&remote->remote_ucp_lock);

    /* rkey_cache から rkey を全部破棄 */
    if (remote->rkey_cache) {
        khash_t(remote_rkey_cache) *h = remote->rkey_cache;
        khiter_t it;
        for (it = kh_begin(h); it != kh_end(h); ++it) {
            if (!kh_exist(h, it))
                continue;

            ucp_rkey_h rkey = kh_val(h, it);
            if (rkey)
                ucp_rkey_destroy(rkey);
        }
        kh_destroy(remote_rkey_cache, h);
        remote->rkey_cache = NULL;
    }
    ucs_spinlock_destroy(&remote->rkey_cache_lock);

    /* remote_worker_address 周りはこれまで通り */
    remote->remote_worker_address = NULL;
    remote->remote_worker_address_len = 0;
}

doca_error_t
remote_ucp_get_or_create_rkey(struct remote_ucp_t *remote,
                              ucp_ep_h ep,
                              uint64_t addr,
                              uint64_t len,
                              const void *rkey_buf,
                              size_t rkey_buf_len,
                              ucp_rkey_h *rkey_out)
{
    if (!remote || !rkey_out)
        return DOCA_ERROR_INVALID_VALUE;

    uint64_t key = make_addr_len_key(addr, len);

    ucs_spin_lock(&remote->rkey_cache_lock);

    khiter_t it = kh_get(remote_rkey_cache, remote->rkey_cache, key);
    if (it != kh_end(remote->rkey_cache)) {
        /* 既にキャッシュ済み */
        *rkey_out = kh_val(remote->rkey_cache, it);
        ucs_spin_unlock(&remote->rkey_cache_lock);
        return DOCA_SUCCESS;
    }

    /* 未登録: rkey_buf から unpack */
    ucp_rkey_h rkey = NULL;
    ucs_status_t status = ucp_ep_rkey_unpack(ep, rkey_buf, &rkey);
    if (status != UCS_OK) {
        ucs_spin_unlock(&remote->rkey_cache_lock);
        DOCA_LOG_ERR("ucp_ep_rkey_unpack failed: %s",
                     ucs_status_string(status));
        return DOCA_ERROR_IO_FAILED;
    }

    int ret;
    it = kh_put(remote_rkey_cache, remote->rkey_cache, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&remote->rkey_cache_lock);
        ucp_rkey_destroy(rkey);
        return DOCA_ERROR_NO_MEMORY;
    }
    kh_val(remote->rkey_cache, it) = rkey;

    *rkey_out = rkey;

    ucs_spin_unlock(&remote->rkey_cache_lock);
    return DOCA_SUCCESS;
}

doca_error_t remote_ucp_buffer_info_put(struct remote_ucp_t *remote,
                                        uint64_t key,
                                        const struct remote_buffer_info *info_in)
{
    if (!remote || !remote->remote_ucp_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct remote_buffer_info *info =
        (struct remote_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in; /* 値をコピー */

    ucs_spin_lock(&remote->remote_ucp_lock);

    int ret;
    khiter_t it = kh_put(remote_ucp_buffer_info,
                         remote->remote_ucp_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&remote->remote_ucp_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        /* 既存キーがある → ポリシーとしてはエラー */
        ucs_spin_unlock(&remote->remote_ucp_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(remote->remote_ucp_buffer_infos, it) = info;
    ucs_spin_unlock(&remote->remote_ucp_lock);
    return DOCA_SUCCESS;
}

doca_error_t remote_ucp_buffer_info_get(struct remote_ucp_t *remote,
                                        uint64_t key,
                                        struct remote_buffer_info **info_out)
{
    if (!remote || !remote->remote_ucp_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&remote->remote_ucp_lock);

    khiter_t it = kh_get(remote_ucp_buffer_info,
                         remote->remote_ucp_buffer_infos, key);
    if (it == kh_end(remote->remote_ucp_buffer_infos)) {
        ucs_spin_unlock(&remote->remote_ucp_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(remote->remote_ucp_buffer_infos, it); /* struct* を渡す */
    ucs_spin_unlock(&remote->remote_ucp_lock);
    return DOCA_SUCCESS;
}

doca_error_t remote_ucp_buffer_info_del(struct remote_ucp_t *remote,
                                        uint64_t key)
{
    if (!remote || !remote->remote_ucp_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&remote->remote_ucp_lock);

    khiter_t it = kh_get(remote_ucp_buffer_info,
                         remote->remote_ucp_buffer_infos, key);
    if (it == kh_end(remote->remote_ucp_buffer_infos)) {
        ucs_spin_unlock(&remote->remote_ucp_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct remote_buffer_info *info =
        kh_val(remote->remote_ucp_buffer_infos, it);
    if (info) {
        /*if (info->rkey) {
            ucp_rkey_destroy(info->rkey);
            info->rkey = NULL;
        }*/
        /* buffer_address の実体メモリを free する場合はここで */
        free(info);
    }

    kh_del(remote_ucp_buffer_info, remote->remote_ucp_buffer_infos, it);
    ucs_spin_unlock(&remote->remote_ucp_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_put_get_data_t: ローカルマップ (value = local_buffer_info*) ========= */

void ucp_put_get_data_maps_init(struct ucp_put_get_data_t *d)
{
    if (!d)
        return;

    memset(d, 0, sizeof(*d));

    d->local_ucp_put_get_data_buffer_infos =
        kh_init(local_ucp_put_get_data_buffer_info);
    ucs_spinlock_init(&d->local_put_get_lock, 0);

    //remote_ucp_init(&d->remote_ucp);
    remote_ucp_init(&d->host_src_ucp);
    remote_ucp_init(&d->host_dst_ucp);
}

void ucp_put_get_data_maps_destroy(struct ucp_put_get_data_t *d)
{
    if (!d)
        return;

    if (d->local_ucp_put_get_data_buffer_infos) {
        khash_t(local_ucp_put_get_data_buffer_info) *h =
            d->local_ucp_put_get_data_buffer_infos;
        khiter_t it;
        for (it = kh_begin(h); it != kh_end(h); ++it) {
            if (!kh_exist(h, it))
                continue;

            struct local_buffer_info *info = kh_val(h, it);
            if (!info)
                continue;

            if (info->memh) {
                ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
                info->memh = NULL;
            }
            free(info);
        }
        kh_destroy(local_ucp_put_get_data_buffer_info, h);
        d->local_ucp_put_get_data_buffer_infos = NULL;
    }

    ucs_spinlock_destroy(&d->local_put_get_lock);
    //remote_ucp_destroy(&d->remote_ucp);
    remote_ucp_destroy(&d->host_src_ucp);
    remote_ucp_destroy(&d->host_dst_ucp);
}

doca_error_t ucp_put_get_local_buffer_put(struct ucp_put_get_data_t *d,
                                          uint64_t key,
                                          const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_put_get_data_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->local_put_get_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_put_get_data_buffer_info,
                         d->local_ucp_put_get_data_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->local_put_get_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->local_put_get_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_put_get_data_buffer_infos, it) = info;
    ucs_spin_unlock(&d->local_put_get_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_put_get_local_buffer_get(struct ucp_put_get_data_t *d,
                                          uint64_t key,
                                          struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_put_get_data_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_put_get_lock);

    khiter_t it = kh_get(local_ucp_put_get_data_buffer_info,
                         d->local_ucp_put_get_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_put_get_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_put_get_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_put_get_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_put_get_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_put_get_local_buffer_del(struct ucp_put_get_data_t *d,
                                          uint64_t key)
{
    if (!d || !d->local_ucp_put_get_data_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_put_get_lock);

    khiter_t it = kh_get(local_ucp_put_get_data_buffer_info,
                         d->local_ucp_put_get_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_put_get_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_put_get_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_put_get_data_buffer_infos, it);
    if (info) {
        if (info->memh) {
            ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
            info->memh = NULL;
        }
        free(info);
    }

    kh_del(local_ucp_put_get_data_buffer_info,
           d->local_ucp_put_get_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_put_get_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_host_data_t ========= */

void ucp_host_data_maps_init(struct ucp_host_data_t *d)
{
    if (!d)
        return;

    memset(d, 0, sizeof(*d));

    d->local_ucp_host_data_src_buffer_infos =
        kh_init(local_ucp_host_data_src_buffer_info);
    d->local_ucp_host_data_dst_buffer_infos =
        kh_init(local_ucp_host_data_dst_buffer_info);
    d->local_ucp_host_data_local_buffer_infos =
        kh_init(local_ucp_host_data_local_buffer_info);
    d->local_ucp_host_data_local_flag_buffer_infos =
        kh_init(local_ucp_host_data_local_flag_buffer_info);

    ucs_spinlock_init(&d->host_src_lock, 0);
    ucs_spinlock_init(&d->host_dst_lock, 0);
    ucs_spinlock_init(&d->host_local_lock, 0);
    ucs_spinlock_init(&d->host_local_flag_lock, 0);

    remote_ucp_init(&d->remote_ucp);
}

static void destroy_local_buffer_map_with_unmap(struct ucp_data_t *ucp_data,
                                                khash_t(local_ucp_host_data_src_buffer_info) *h)
{
    if (!h)
        return;

    khiter_t it;
    for (it = kh_begin(h); it != kh_end(h); ++it) {
        if (!kh_exist(h, it))
            continue;
        struct local_buffer_info *info = kh_val(h, it);
        if (!info)
            continue;

        if (info->memh) {
            ucp_mem_unmap(*(ucp_data->ucp_context), info->memh);
            info->memh = NULL;
        }
        free(info);
    }
}

void ucp_host_data_maps_destroy(struct ucp_host_data_t *d)
{
    if (!d)
        return;

    if (d->local_ucp_host_data_src_buffer_infos) {
        destroy_local_buffer_map_with_unmap(
            &d->ucp_data,
            (khash_t(local_ucp_host_data_src_buffer_info) *)d->local_ucp_host_data_src_buffer_infos);
        kh_destroy(local_ucp_host_data_src_buffer_info,
                   d->local_ucp_host_data_src_buffer_infos);
        d->local_ucp_host_data_src_buffer_infos = NULL;
    }

    if (d->local_ucp_host_data_dst_buffer_infos) {
        destroy_local_buffer_map_with_unmap(
            &d->ucp_data,
            (khash_t(local_ucp_host_data_src_buffer_info) *)d->local_ucp_host_data_dst_buffer_infos);
        kh_destroy(local_ucp_host_data_dst_buffer_info,
                   d->local_ucp_host_data_dst_buffer_infos);
        d->local_ucp_host_data_dst_buffer_infos = NULL;
    }

    if (d->local_ucp_host_data_local_buffer_infos) {
        destroy_local_buffer_map_with_unmap(
            &d->ucp_data,
            (khash_t(local_ucp_host_data_src_buffer_info) *)d->local_ucp_host_data_local_buffer_infos);
        kh_destroy(local_ucp_host_data_local_buffer_info,
                   d->local_ucp_host_data_local_buffer_infos);
        d->local_ucp_host_data_local_buffer_infos = NULL;
    }

    if (d->local_ucp_host_data_local_flag_buffer_infos) {
        destroy_local_buffer_map_with_unmap(
            &d->ucp_data,
            (khash_t(local_ucp_host_data_src_buffer_info) *)d->local_ucp_host_data_local_flag_buffer_infos);
        kh_destroy(local_ucp_host_data_local_flag_buffer_info,
                   d->local_ucp_host_data_local_flag_buffer_infos);
        d->local_ucp_host_data_local_flag_buffer_infos = NULL;
    }

    ucs_spinlock_destroy(&d->host_src_lock);
    ucs_spinlock_destroy(&d->host_dst_lock);
    ucs_spinlock_destroy(&d->host_local_lock);
    ucs_spinlock_destroy(&d->host_local_flag_lock);

    remote_ucp_destroy(&d->remote_ucp);
}

doca_error_t ucp_host_src_buffer_put(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_host_data_src_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->host_src_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_host_data_src_buffer_info,
                         d->local_ucp_host_data_src_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->host_src_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->host_src_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_host_data_src_buffer_infos, it) = info;
    ucs_spin_unlock(&d->host_src_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_src_buffer_get(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_host_data_src_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_src_lock);

    khiter_t it = kh_get(local_ucp_host_data_src_buffer_info,
                         d->local_ucp_host_data_src_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_src_buffer_infos)) {
        ucs_spin_unlock(&d->host_src_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_host_data_src_buffer_infos, it);
    ucs_spin_unlock(&d->host_src_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_src_buffer_del(struct ucp_host_data_t *d,
                                     uint64_t key)
{
    if (!d || !d->local_ucp_host_data_src_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_src_lock);

    khiter_t it = kh_get(local_ucp_host_data_src_buffer_info,
                         d->local_ucp_host_data_src_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_src_buffer_infos)) {
        ucs_spin_unlock(&d->host_src_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_host_data_src_buffer_infos, it);
    if (info) {
        /*if (info->memh) {
            ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
            info->memh = NULL;
        }*/
        free(info);
    }

    kh_del(local_ucp_host_data_src_buffer_info,
           d->local_ucp_host_data_src_buffer_infos, it);
    ucs_spin_unlock(&d->host_src_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_dst_buffer_put(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_host_data_dst_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->host_dst_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_host_data_dst_buffer_info,
                         d->local_ucp_host_data_dst_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->host_dst_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->host_dst_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_host_data_dst_buffer_infos, it) = info;
    ucs_spin_unlock(&d->host_dst_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_dst_buffer_get(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_host_data_dst_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_dst_lock);

    khiter_t it = kh_get(local_ucp_host_data_dst_buffer_info,
                         d->local_ucp_host_data_dst_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_dst_buffer_infos)) {
        ucs_spin_unlock(&d->host_dst_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_host_data_dst_buffer_infos, it);
    ucs_spin_unlock(&d->host_dst_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_dst_buffer_del(struct ucp_host_data_t *d,
                                     uint64_t key)
{
    if (!d || !d->local_ucp_host_data_dst_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_dst_lock);

    khiter_t it = kh_get(local_ucp_host_data_dst_buffer_info,
                         d->local_ucp_host_data_dst_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_dst_buffer_infos)) {
        ucs_spin_unlock(&d->host_dst_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_host_data_dst_buffer_infos, it);
    if (info) {
        /*if (info->memh) {
            ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
            info->memh = NULL;
        }*/
        free(info);
    }

    kh_del(local_ucp_host_data_dst_buffer_info,
           d->local_ucp_host_data_dst_buffer_infos, it);
    ucs_spin_unlock(&d->host_dst_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_host_data_t: local ========= */

doca_error_t ucp_host_local_buffer_put(struct ucp_host_data_t *d,
                                       uint64_t key,
                                       const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_host_data_local_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->host_local_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_host_data_local_buffer_info,
                         d->local_ucp_host_data_local_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->host_local_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->host_local_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_host_data_local_buffer_infos, it) = info;
    ucs_spin_unlock(&d->host_local_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_local_buffer_get(struct ucp_host_data_t *d,
                                       uint64_t key,
                                       struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_host_data_local_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_local_lock);

    khiter_t it = kh_get(local_ucp_host_data_local_buffer_info,
                         d->local_ucp_host_data_local_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_local_buffer_infos)) {
        ucs_spin_unlock(&d->host_local_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_host_data_local_buffer_infos, it);
    ucs_spin_unlock(&d->host_local_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_local_buffer_del(struct ucp_host_data_t *d,
                                       uint64_t key)
{
    if (!d || !d->local_ucp_host_data_local_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_local_lock);

    khiter_t it = kh_get(local_ucp_host_data_local_buffer_info,
                         d->local_ucp_host_data_local_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_local_buffer_infos)) {
        ucs_spin_unlock(&d->host_local_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_host_data_local_buffer_infos, it);
    if (info) {
        free(info);
    }

    kh_del(local_ucp_host_data_local_buffer_info,
           d->local_ucp_host_data_local_buffer_infos, it);
    ucs_spin_unlock(&d->host_local_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_host_data_t: local_flag ========= */

doca_error_t ucp_host_local_flag_buffer_put(struct ucp_host_data_t *d,
                                            uint64_t key,
                                            const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_host_data_local_flag_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->host_local_flag_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_host_data_local_flag_buffer_info,
                         d->local_ucp_host_data_local_flag_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->host_local_flag_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->host_local_flag_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_host_data_local_flag_buffer_infos, it) = info;
    ucs_spin_unlock(&d->host_local_flag_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_local_flag_buffer_get(struct ucp_host_data_t *d,
                                            uint64_t key,
                                            struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_host_data_local_flag_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_local_flag_lock);

    khiter_t it = kh_get(local_ucp_host_data_local_flag_buffer_info,
                         d->local_ucp_host_data_local_flag_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_local_flag_buffer_infos)) {
        ucs_spin_unlock(&d->host_local_flag_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_host_data_local_flag_buffer_infos, it);
    ucs_spin_unlock(&d->host_local_flag_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_host_local_flag_buffer_del(struct ucp_host_data_t *d,
                                            uint64_t key)
{
    if (!d || !d->local_ucp_host_data_local_flag_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->host_local_flag_lock);

    khiter_t it = kh_get(local_ucp_host_data_local_flag_buffer_info,
                         d->local_ucp_host_data_local_flag_buffer_infos, key);
    if (it == kh_end(d->local_ucp_host_data_local_flag_buffer_infos)) {
        ucs_spin_unlock(&d->host_local_flag_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_host_data_local_flag_buffer_infos, it);
    if (info) {
        free(info);
    }

    kh_del(local_ucp_host_data_local_flag_buffer_info,
           d->local_ucp_host_data_local_flag_buffer_infos, it);
    ucs_spin_unlock(&d->host_local_flag_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_receive_data_t ========= */

void ucp_receive_data_maps_init(struct ucp_receive_data_t *d)
{
    if (!d)
        return;

    memset(d, 0, sizeof(*d));

    d->local_ucp_receive_data_buffer_infos =
        kh_init(local_ucp_receive_data_buffer_info);
    ucs_spinlock_init(&d->local_receive_lock, 0);

    remote_ucp_init(&d->prev_rank_send_ucp);
}

void ucp_receive_data_maps_destroy(struct ucp_receive_data_t *d)
{
    if (!d)
        return;

    if (d->local_ucp_receive_data_buffer_infos) {
        khash_t(local_ucp_receive_data_buffer_info) *h =
            d->local_ucp_receive_data_buffer_infos;
        khiter_t it;
        for (it = kh_begin(h); it != kh_end(h); ++it) {
            if (!kh_exist(h, it))
                continue;

            struct local_buffer_info *info = kh_val(h, it);
            if (!info)
                continue;

            if (info->memh) {
                ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
                info->memh = NULL;
            }
            free(info);
        }
        kh_destroy(local_ucp_receive_data_buffer_info, h);
        d->local_ucp_receive_data_buffer_infos = NULL;
    }

    ucs_spinlock_destroy(&d->local_receive_lock);
    remote_ucp_destroy(&d->prev_rank_send_ucp);
}

doca_error_t ucp_receive_local_buffer_put(struct ucp_receive_data_t *d,
                                          uint64_t key,
                                          const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_receive_data_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->local_receive_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_receive_data_buffer_info,
                         d->local_ucp_receive_data_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->local_receive_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->local_receive_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_receive_data_buffer_infos, it) = info;
    ucs_spin_unlock(&d->local_receive_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_receive_local_buffer_get(struct ucp_receive_data_t *d,
                                          uint64_t key,
                                          struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_receive_data_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_receive_lock);

    khiter_t it = kh_get(local_ucp_receive_data_buffer_info,
                         d->local_ucp_receive_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_receive_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_receive_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_receive_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_receive_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_receive_local_buffer_del(struct ucp_receive_data_t *d,
                                          uint64_t key)
{
    if (!d || !d->local_ucp_receive_data_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_receive_lock);

    khiter_t it = kh_get(local_ucp_receive_data_buffer_info,
                         d->local_ucp_receive_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_receive_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_receive_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_receive_data_buffer_infos, it);
    if (info) {
        if (info->memh) {
            ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
            info->memh = NULL;
        }
        free(info);
    }

    kh_del(local_ucp_receive_data_buffer_info,
           d->local_ucp_receive_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_receive_lock);
    return DOCA_SUCCESS;
}

/* ========= ucp_send_data_t ========= */

void ucp_send_data_maps_init(struct ucp_send_data_t *d)
{
    if (!d)
        return;

    memset(d, 0, sizeof(*d));

    d->local_ucp_send_data_buffer_infos =
        kh_init(local_ucp_send_data_buffer_info);
    ucs_spinlock_init(&d->local_send_lock, 0);

    //remote_ucp_init(&d->host_src_ucp);
    //remote_ucp_init(&d->host_dst_ucp);
    remote_ucp_init(&d->next_rank_receive_ucp);
}

void ucp_send_data_maps_destroy(struct ucp_send_data_t *d)
{
    if (!d)
        return;

    if (d->local_ucp_send_data_buffer_infos) {
        khash_t(local_ucp_send_data_buffer_info) *h =
            d->local_ucp_send_data_buffer_infos;
        khiter_t it;
        for (it = kh_begin(h); it != kh_end(h); ++it) {
            if (!kh_exist(h, it))
                continue;

            struct local_buffer_info *info = kh_val(h, it);
            if (!info)
                continue;

            if (info->memh) {
                ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
                info->memh = NULL;
            }
            free(info);
        }
        kh_destroy(local_ucp_send_data_buffer_info, h);
        d->local_ucp_send_data_buffer_infos = NULL;
    }

    ucs_spinlock_destroy(&d->local_send_lock);

    //remote_ucp_destroy(&d->host_src_ucp);
    //remote_ucp_destroy(&d->host_dst_ucp);
    remote_ucp_destroy(&d->next_rank_receive_ucp);
}

doca_error_t ucp_send_local_buffer_put(struct ucp_send_data_t *d,
                                       uint64_t key,
                                       const struct local_buffer_info *info_in)
{
    if (!d || !d->local_ucp_send_data_buffer_infos || !info_in)
        return DOCA_ERROR_INVALID_VALUE;

    struct local_buffer_info *info =
        (struct local_buffer_info *)malloc(sizeof(*info));
    if (!info)
        return DOCA_ERROR_NO_MEMORY;
    *info = *info_in;

    ucs_spin_lock(&d->local_send_lock);

    int ret;
    khiter_t it = kh_put(local_ucp_send_data_buffer_info,
                         d->local_ucp_send_data_buffer_infos, key, &ret);
    if (ret < 0) {
        ucs_spin_unlock(&d->local_send_lock);
        free(info);
        return DOCA_ERROR_NO_MEMORY;
    }
    if (ret == 0) {
        ucs_spin_unlock(&d->local_send_lock);
        free(info);
        return DOCA_ERROR_INVALID_VALUE;
    }

    kh_val(d->local_ucp_send_data_buffer_infos, it) = info;
    ucs_spin_unlock(&d->local_send_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_send_local_buffer_get(struct ucp_send_data_t *d,
                                       uint64_t key,
                                       struct local_buffer_info **info_out)
{
    if (!d || !d->local_ucp_send_data_buffer_infos || !info_out)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_send_lock);

    khiter_t it = kh_get(local_ucp_send_data_buffer_info,
                         d->local_ucp_send_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_send_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_send_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    *info_out = kh_val(d->local_ucp_send_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_send_lock);
    return DOCA_SUCCESS;
}

doca_error_t ucp_send_local_buffer_del(struct ucp_send_data_t *d,
                                       uint64_t key)
{
    if (!d || !d->local_ucp_send_data_buffer_infos)
        return DOCA_ERROR_INVALID_VALUE;

    ucs_spin_lock(&d->local_send_lock);

    khiter_t it = kh_get(local_ucp_send_data_buffer_info,
                         d->local_ucp_send_data_buffer_infos, key);
    if (it == kh_end(d->local_ucp_send_data_buffer_infos)) {
        ucs_spin_unlock(&d->local_send_lock);
        return DOCA_ERROR_NOT_FOUND;
    }

    struct local_buffer_info *info =
        kh_val(d->local_ucp_send_data_buffer_infos, it);
    if (info) {
        if (info->memh) {
            ucp_mem_unmap(*(d->ucp_data.ucp_context), info->memh);
            info->memh = NULL;
        }
        free(info);
    }

    kh_del(local_ucp_send_data_buffer_info,
           d->local_ucp_send_data_buffer_infos, it);
    ucs_spin_unlock(&d->local_send_lock);
    return DOCA_SUCCESS;
}
/*--------------------ハッシュ周りのユーティリティ関数ここまで--------------------*/