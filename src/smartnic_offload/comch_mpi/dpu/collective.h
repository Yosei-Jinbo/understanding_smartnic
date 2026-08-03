#ifndef COLLECTIVE_H
#define COLLECTIVE_H

#include "comch_server.h"

/* 集合通信 (collective.c)
 *   AG/RS Ring アルゴリズム + piece 分割 + RDB (RDMA Doorbell Barrier) */

/* RDB slot 領域: working_buf 末尾の 192 bytes (レイアウトは collective.c 参照)。
 * handlers.c の flag pool 配置計算もこの値を使い、領域の衝突を避ける。 */
#define RDB_SLOT_REGION_SIZE  192

/* RDB 初期化 (Ring 接続確立後に 1 回だけ呼ぶ。失敗時は MPI_Barrier フォールバック) */
doca_error_t rdb_init(struct collective_worker_t *cw, int rank, int world_size);

/* AG piece 分割の env 設定 (AG_PIECE_MAX / AG_PIECE_TARGET) を初回 1 度だけ反映 */
void ag_piece_config_init_once(void);

doca_error_t collective_reduce_scatter(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf, void *recv_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    uint64_t id);

doca_error_t collective_all_gather(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    struct doca_mmap *gpu_src_local_mmap,
    uint64_t id,
    struct doca_mmap *gpu_dst_local_mmap,
    /* dual-rail GPU Direct (rail1 mmaps; NULL ならシングルレール) */
    struct doca_mmap *gpu_src_local_mmap_rail1,
    struct doca_mmap *gpu_dst_local_mmap_rail1,
    /* ring_id (0 or 1) で Ring instance を選ぶ */
    int ring_id);

#endif /* COLLECTIVE_H */
