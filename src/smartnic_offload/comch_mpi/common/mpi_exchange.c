#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <doca_error.h>

#include "comch_mpi_common.h"

/*
 * 内部実装 — 指定 MPI tag で control_cmd を交換する。
 */
static doca_error_t _mpi_exchange_cmd_with_tag(uint64_t rank, uint64_t world_size,
                                                uint64_t src_rank, uint64_t dst_rank,
                                                struct control_cmd *send_cmd,
                                                struct control_cmd **recv_cmd,
                                                void **recv_storage,
                                                size_t *recv_storage_len,
                                                int tag)
{
    if (recv_cmd)
        *recv_cmd = NULL;
    if (recv_storage)
        *recv_storage = NULL;
    if (recv_storage_len)
        *recv_storage_len = 0;

    if (world_size <= src_rank || world_size <= dst_rank)
        return DOCA_ERROR_INVALID_VALUE;
    if (src_rank == dst_rank)
        return DOCA_ERROR_INVALID_VALUE;

    if (rank != src_rank && rank != dst_rank) {
        return DOCA_SUCCESS;
    }

    int partner = (rank == src_rank) ? (int)dst_rank : (int)src_rank;

    uint8_t send_buf[CONTROL_CMD_MAX_SIZE];
    size_t  send_cap = CONTROL_CMD_MAX_SIZE;

    doca_error_t st = control_cmd_pack(send_cmd, &send_cap, send_buf);
    if (st != DOCA_SUCCESS) {
        printf("control_cmd_pack failed in _mpi_exchange_cmd_with_tag\n");
        return st;
    }

    void *recv_buf = malloc(CONTROL_CMD_MAX_SIZE);
    if (!recv_buf)
        return DOCA_ERROR_NO_MEMORY;

    MPI_Status status;
    int rc = MPI_Sendrecv(send_buf, (int)send_cap, MPI_BYTE,
                          partner, tag,
                          recv_buf, CONTROL_CMD_MAX_SIZE, MPI_BYTE,
                          partner, tag,
                          MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS) {
        printf("MPI_Sendrecv failed in _mpi_exchange_cmd_with_tag, rc=%d\n", rc);
        free(recv_buf);
        return DOCA_ERROR_DRIVER;
    }

    int recv_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &recv_count);

    struct control_cmd *rcmd = NULL;
    st = control_cmd_unpack(recv_buf, (size_t)recv_count, &rcmd);
    if (st != DOCA_SUCCESS) {
        printf("control_cmd_unpack failed in _mpi_exchange_cmd_with_tag\n");
        free(recv_buf);
        return st;
    }

    if (recv_cmd)
        *recv_cmd = rcmd;
    if (recv_storage)
        *recv_storage = recv_buf;
    else
        free(recv_buf);
    if (recv_storage_len)
        *recv_storage_len = (size_t)recv_count;

    return DOCA_SUCCESS;
}

/*
 * MPI を使って control_cmd を src_rank <-> dst_rank で「双方向」に交換する。
 * - 各ランクは自分の send_cmd を送信し、
 * - 相手ランクの send_cmd が、自分側の recv_cmd として返ってくる。
 *
 * デフォルトタグ TAG_EXCHANGE=0x500 を使用する。
 */
doca_error_t run_mpi_tag_exchange_cmd(uint64_t rank, uint64_t world_size,
                                      uint64_t src_rank, uint64_t dst_rank,
                                      struct control_cmd *send_cmd,
                                      struct control_cmd **recv_cmd,
                                      void **recv_storage,
                                      size_t *recv_storage_len)
{
    const int TAG_EXCHANGE = 0x500;
    return _mpi_exchange_cmd_with_tag(rank, world_size, src_rank, dst_rank,
                                       send_cmd, recv_cmd, recv_storage, recv_storage_len,
                                       TAG_EXCHANGE);
}

/*
 * Phase 15: 異なる MPI タグで交換する variant。
 * Ring 1 の接続確立など、既存の Ring 0 用交換 (tag 0x500) と区別するために使う。
 */
doca_error_t run_mpi_tag_exchange_cmd_tagged(uint64_t rank, uint64_t world_size,
                                             uint64_t src_rank, uint64_t dst_rank,
                                             struct control_cmd *send_cmd,
                                             struct control_cmd **recv_cmd,
                                             void **recv_storage,
                                             size_t *recv_storage_len,
                                             int mpi_tag)
{
    return _mpi_exchange_cmd_with_tag(rank, world_size, src_rank, dst_rank,
                                       send_cmd, recv_cmd, recv_storage, recv_storage_len,
                                       mpi_tag);
}
