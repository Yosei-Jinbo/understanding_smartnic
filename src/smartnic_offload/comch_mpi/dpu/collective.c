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
#include "rs_pool.h"
#include "union_tracker.h"
#include "collective.h"

DOCA_LOG_REGISTER(COLLECTIVE);

/* ============================================================
 * AG piece 分割 (runtime-tunable)
 *
 * 環境変数:
 *   AG_PIECE_MAX     - chunk あたりの最大 piece 数 (default: 8, hard max: 32)
 *   AG_PIECE_TARGET  - piece size の目標バイト数 (default: 8 MB)
 *
 * num_pieces 計算:
 *   num_pieces = clamp(ceil(chunk_size / AG_PIECE_TARGET), 1, AG_PIECE_MAX)
 *
 * 例 (max=16, target=8MB):
 *   chunk=12.25MB → 2 pieces of 6.1MB
 *   chunk=49MB    → 7 pieces of 7MB → cap 8 で 8 pieces of 6.1MB
 *   chunk=147MB   → 19 pieces → cap 16 で 16 pieces of 9.2MB
 * ============================================================ */
#define AG_PIECE_MIN_SIZE          (256 * 1024)          /* これ未満はパイプラインしない */
#define AG_PIECE_MAX_COUNT_DEFAULT 8                     /* 既定の最大ピース数 */
#define AG_PIECE_HARD_MAX          32                    /* env で指定できる上限 (ビルド時の絶対上限) */
#define AG_PIECE_TARGET_BYTES      (8UL * 1024 * 1024)   /* 目標 piece size = 8MB */

static int  g_ag_piece_max    = AG_PIECE_MAX_COUNT_DEFAULT;
static uint64_t g_ag_piece_target_bytes = AG_PIECE_TARGET_BYTES;
static int  g_ag_piece_inited = 0;

void ag_piece_config_init_once(void)
{
    if (g_ag_piece_inited) return;
    g_ag_piece_inited = 1;

    const char *env_max = getenv("AG_PIECE_MAX");
    if (env_max) {
        int v = atoi(env_max);
        if (v >= 1 && v <= AG_PIECE_HARD_MAX) {
            g_ag_piece_max = v;
        } else {
            DOCA_LOG_WARN("AG_PIECE_MAX=%s out of range [1,%d], using default %d",
                          env_max, AG_PIECE_HARD_MAX, g_ag_piece_max);
        }
    }

    const char *env_target = getenv("AG_PIECE_TARGET");
    if (env_target) {
        long long v = atoll(env_target);
        if (v >= (long long)AG_PIECE_MIN_SIZE && v <= (long long)(1024 * 1024 * 1024)) {
            g_ag_piece_target_bytes = (uint64_t)v;
        } else {
            DOCA_LOG_WARN("AG_PIECE_TARGET=%s out of range, using default %lu",
                          env_target, g_ag_piece_target_bytes);
        }
    }

    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        printf("[AG PIECE CONFIG rank0] max=%d target=%lu bytes (%.1f MB)\n",
               g_ag_piece_max, g_ag_piece_target_bytes,
               (double)g_ag_piece_target_bytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

/* chunk_size から num_pieces を計算 (size-adaptive) */
static inline int ag_compute_num_pieces(uint64_t chunk_size)
{
    /* target に近い piece 数を求める */
    uint64_t n = (chunk_size + g_ag_piece_target_bytes - 1) / g_ag_piece_target_bytes;
    if (n < 1) n = 1;
    if (n > (uint64_t)g_ag_piece_max) n = (uint64_t)g_ag_piece_max;
    /* MIN_SIZE 制約: piece が小さすぎる場合は piece 数を減らす */
    while (n > 1 && (chunk_size / n) < AG_PIECE_MIN_SIZE) {
        n--;
    }
    return (int)n;
}

/* =====================================================
 * RDMA Doorbell Barrier (RDB)
 *
 * MPI_Barrier (avg 325μs, max 281ms 実測) を 2-pass Ring barrier に置換する。
 * 隣接 Ring 接続のみを使用 (フルメッシュ禁止)。
 *
 * 設計原則:
 *   - OS バイパス: RDMA Write inline のみ、カーネル経由なし
 *   - 既存 Ring 接続再利用: rdma_ring_send + conn_to_next
 *   - working_buf 末尾の 192B を slot 領域として使用 (追加メモリ登録不要)
 *   - 隣接 rank の rdb_local_slot のリモートアドレスを起動時に MPI 1 回交換
 *   - monotonic seq でリセット不要、複数 barrier を区別
 *
 * 2-pass プロトコル (barrier #k, k = 1, 2, ...):
 *   if rank == 0:
 *     Write(2k-1) → next.slot      // gather wave start
 *     wait my.slot >= 2k-1          // gather wave returned (= 全 rank が到達)
 *     Write(2k)   → next.slot      // release wave start
 *     wait my.slot >= 2k            // release wave returned
 *   else:
 *     wait my.slot >= 2k-1          // gather wave from prev
 *     Write(2k-1) → next.slot      // forward gather wave
 *     wait my.slot >= 2k            // release wave from prev
 *     Write(2k)   → next.slot      // forward release wave
 *
 * 重要な性質:
 *   - 2-pass にすることで、release wave が始まる前に必ず gather wave が
 *     完走する。よって「ある rank が exit した時、全 rank は少なくとも
 *     barrier に到達済み」を保証 (1-pass では破られる)。
 *   - 想定 latency: N=4 なら 2(N-1)+2 = 8 hops × ~5μs = ~40μs。
 *     MPI_Barrier の avg 325μs / max 281ms に比べ大幅改善。
 *
 * Slot レイアウト (working_buf 末尾 192 bytes):
 *   [working_buf - 192] rdb_send_data_a (8B + padding to 64B)  // gather Write 用
 *   [working_buf - 128] rdb_send_data_b (8B + padding to 64B)  // release Write 用
 *   [working_buf -  64] rdb_local_slot  (8B + padding to 64B)  // 前 rank が書く
 *
 * 2 つの送信バッファを用意することで、gather Write の NIC 読み込みが
 * 完了する前に release Write のバッファを書き換える race を回避。
 * ===================================================== */

#define RDB_OFFSET_SEND_A     192
#define RDB_OFFSET_SEND_B     128
#define RDB_OFFSET_LOCAL_SLOT 64

/* MPI で交換するハンドシェイク構造 (固定サイズ、~120 byte の export desc を 256 byte で余裕を持って格納) */
struct rdb_handshake {
    uint64_t slot_va;          /* 自分の rdb_local_slot のローカル VA */
    uint32_t export_len;       /* working_buf mmap export の長さ */
    uint32_t reserved;
    uint8_t  export_desc[256]; /* working_buf mmap export descriptor */
};

/* gather と release で別々の send buffer を使う */
static inline uint64_t *rdb_get_send_buf_a(struct collective_worker_t *cw) {
    return (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_A);
}
static inline uint64_t *rdb_get_send_buf_b(struct collective_worker_t *cw) {
    return (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_B);
}

/* 1 hop の RDMA Write を発行 (submit_and_wait — inline submit + PE progress 待ち) */
static inline void rdb_write_next(struct collective_worker_t *cw,
                                  uint64_t *local_buf, uint64_t value)
{
    *local_buf = value;

    struct doca_task_desc write_task;
    init_write_task(&write_task,
                    local_buf,
                    (void *)cw->rdb_next_remote_slot_addr,
                    sizeof(uint64_t),
                    &cw->rdb_next_rmem,
                    &cw->rdma_ring_send, cw->conn_to_next, 0);
    write_task.pe_spin = &cw->ring_send_pe_spin;
    submit_and_wait_doca(cw->worker_pool, &write_task);
}

/* 自分の slot を spin で待つ (前 rank が target 以上の値を書き込むまで) */
static inline void rdb_wait_local_slot(struct collective_worker_t *cw, uint64_t target)
{
    while (*cw->rdb_local_slot < target) {
        cpu_relax();
    }
    __sync_synchronize();  /* slot 読み出しが後続のメモリ操作より先に行われることを保証 */
}

/* RDB 初期化 (Ring 接続確立後に 1 回だけ呼ぶ) */
doca_error_t rdb_init(struct collective_worker_t *cw, int rank, int world_size)
{
    /* RDB_DISABLE=1 で無効化 */
    const char *dis = getenv("RDB_DISABLE");
    if (dis && atoi(dis) != 0) {
        DOCA_LOG_INFO("RDB_DISABLE=1, falling back to MPI_Barrier");
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    if (cw->working_buf_size < RDB_SLOT_REGION_SIZE) {
        DOCA_LOG_WARN("RDB: working_buf too small (%zu), falling back to MPI_Barrier",
                      cw->working_buf_size);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    if (!cw->conn_to_next) {
        DOCA_LOG_WARN("RDB: conn_to_next not established, falling back to MPI_Barrier");
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* slot 領域の初期化 (working_buf 末尾) */
    uint64_t *send_a = (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_A);
    uint64_t *send_b = (uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_SEND_B);
    volatile uint64_t *local_slot = (volatile uint64_t *)((char *)cw->working_buf + cw->working_buf_size - RDB_OFFSET_LOCAL_SLOT);
    *send_a = 0;
    *send_b = 0;
    *local_slot = 0;
    cw->rdb_local_slot = local_slot;
    cw->rdb_send_data  = send_a;  /* 互換性のため (legacy フィールド) */

    /* working_buf を rdma_ring_send 経由で export */
    const void *my_export = NULL;
    size_t      my_export_len = 0;
    doca_error_t ret = doca_rdma_ctx_export_mmap(&cw->rdma_ring_send,
                                                  &my_export, &my_export_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_WARN("RDB: doca_rdma_ctx_export_mmap failed: %s, falling back to MPI_Barrier",
                      doca_error_get_name(ret));
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }
    if (my_export_len > sizeof(((struct rdb_handshake *)0)->export_desc)) {
        DOCA_LOG_WARN("RDB: export_desc too large (%zu > 256), falling back to MPI_Barrier",
                      my_export_len);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* 隣接 rank と handshake を MPI_Sendrecv で交換
     *   prev_rank に送信 (PREV にとって自分は NEXT なので、PREV は自分を書きたい)
     *   next_rank から受信 (自分は NEXT を書きたいので、NEXT の情報を得る)
     */
    int prev_rank = (rank - 1 + world_size) % world_size;
    int next_rank = (rank + 1) % world_size;

    struct rdb_handshake my_hs, next_hs;
    memset(&my_hs,   0, sizeof(my_hs));
    memset(&next_hs, 0, sizeof(next_hs));
    my_hs.slot_va    = (uint64_t)(uintptr_t)cw->rdb_local_slot;
    my_hs.export_len = (uint32_t)my_export_len;
    memcpy(my_hs.export_desc, my_export, my_export_len);

    int mret = MPI_Sendrecv(&my_hs,   sizeof(my_hs),   MPI_BYTE, prev_rank, 0x12345,
                            &next_hs, sizeof(next_hs), MPI_BYTE, next_rank, 0x12345,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (mret != MPI_SUCCESS) {
        DOCA_LOG_WARN("RDB: MPI_Sendrecv failed (%d), falling back to MPI_Barrier", mret);
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    /* 受け取った export descriptor を import (next_rank の working_buf を remote_mem 化) */
    ret = doca_remote_mem_create(&cw->rdb_next_rmem, cw->ring_dev,
                                  next_hs.export_desc, next_hs.export_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_WARN("RDB: doca_remote_mem_create failed: %s, falling back to MPI_Barrier",
                      doca_error_get_name(ret));
        cw->rdb_enabled = false;
        return DOCA_SUCCESS;
    }

    cw->rdb_next_remote_slot_addr = next_hs.slot_va;
    cw->rdb_my_seq  = 0;
    cw->rdb_enabled = true;

    DOCA_LOG_INFO("[rank%d] RDB initialized: my_slot_va=0x%lx next_slot_va=0x%lx (prev=%d, next=%d)",
                  rank, (uint64_t)(uintptr_t)cw->rdb_local_slot,
                  cw->rdb_next_remote_slot_addr, prev_rank, next_rank);
    return DOCA_SUCCESS;
}

/* 2-pass Ring barrier (rank と world_size を引数に取る) */
static doca_error_t rdb_barrier(struct collective_worker_t *cw, int rank, int world_size, int ring_id)
{
    (void)world_size;  /* 現在は不使用 (将来 N>16 等で使用予定) */

    if (!cw->rdb_enabled) {
        /* RDB 未初期化 → MPI_Barrier フォールバック。
         * Ring 0 と Ring 1 が同時に barrier を呼ぶ可能性があるため、
         * ring 専用の MPI_Comm を使う (MPI_COMM_WORLD を共有すると barrier が
         * 混線して Ring algorithm が破綻する)。 */
        MPI_Comm comm = (ring_id >= 0 && ring_id < N_RINGS && cw->ring_comm_valid[ring_id])
                        ? cw->ring_comm[ring_id]
                        : MPI_COMM_WORLD;
        MPI_Barrier(comm);
        return DOCA_SUCCESS;
    }

    uint64_t my_seq         = ++cw->rdb_my_seq;
    uint64_t gather_target  = 2 * my_seq - 1;
    uint64_t release_target = 2 * my_seq;

    uint64_t *buf_a = rdb_get_send_buf_a(cw);
    uint64_t *buf_b = rdb_get_send_buf_b(cw);

    if (rank == 0) {
        /* Phase 1: gather wave 開始 */
        rdb_write_next(cw, buf_a, gather_target);
        rdb_wait_local_slot(cw, gather_target);
        /* Phase 2: release wave 開始 */
        rdb_write_next(cw, buf_b, release_target);
        rdb_wait_local_slot(cw, release_target);
    } else {
        /* Phase 1: gather wave 受信 → 転送 */
        rdb_wait_local_slot(cw, gather_target);
        rdb_write_next(cw, buf_a, gather_target);
        /* Phase 2: release wave 受信 → 転送 */
        rdb_wait_local_slot(cw, release_target);
        rdb_write_next(cw, buf_b, release_target);
    }

    return DOCA_SUCCESS;
}

/* ============================================================
 * PE progress・ring context
 * ============================================================ */

/* main thread から RMA (Host-DPU) の PE progress を回す */
static inline void progress_rma_pes(struct collective_worker_t *cw, bool use_dual)
{
    try_pe_progress(cw->rdma_rma.pe, &cw->rma_pe_spin);
    if (use_dual && cw->rdma_rma_rail1.pe)
        try_pe_progress(cw->rdma_rma_rail1.pe, &cw->rma_pe_spin_rail1);
}

/* Ring-aware context bundle. 各 AG は ring_id (0 or 1) を受け、
 * 対応する RDMA context / connection / pe_spin をこの構造体経由でアクセスする。
 * collective_all_gather の開始時に get_ring_ctx() で初期化する。 */
struct ring_ctx_ptrs {
    struct doca_rdma_ctx_t *send;
    struct doca_rdma_ctx_t *recv;
    struct doca_rdma_ctx_t *send_rail1;
    struct doca_rdma_ctx_t *recv_rail1;
    struct doca_rdma_connection *conn_to_next;
    struct doca_rdma_connection *conn_from_prev;
    struct doca_rdma_connection *conn_to_next_rail1;
    struct doca_rdma_connection *conn_from_prev_rail1;
    struct pe_spin_t *send_pe_spin;
    struct pe_spin_t *recv_pe_spin;
    struct pe_spin_t *send_pe_spin_rail1;
    struct pe_spin_t *recv_pe_spin_rail1;
    int ring_id;  /* 0 or 1 (for debug logs) */
};

static inline void get_ring_ctx(struct collective_worker_t *cw, int ring_id,
                                 struct ring_ctx_ptrs *rc)
{
    rc->ring_id = ring_id;
    if (ring_id == 0) {
        rc->send = &cw->rdma_ring_send;
        rc->recv = &cw->rdma_ring_recv;
        rc->send_rail1 = &cw->rdma_ring_send_rail1;
        rc->recv_rail1 = &cw->rdma_ring_recv_rail1;
        rc->conn_to_next = cw->conn_to_next;
        rc->conn_from_prev = cw->conn_from_prev;
        rc->conn_to_next_rail1 = cw->conn_to_next_rail1;
        rc->conn_from_prev_rail1 = cw->conn_from_prev_rail1;
        rc->send_pe_spin = &cw->ring_send_pe_spin;
        rc->recv_pe_spin = &cw->ring_recv_pe_spin;
        rc->send_pe_spin_rail1 = &cw->ring_send_pe_spin_rail1;
        rc->recv_pe_spin_rail1 = &cw->ring_recv_pe_spin_rail1;
    } else {
        /* Ring 1 */
        rc->send = &cw->rdma_ring_send_r1;
        rc->recv = &cw->rdma_ring_recv_r1;
        rc->send_rail1 = &cw->rdma_ring_send_r1_rail1;
        rc->recv_rail1 = &cw->rdma_ring_recv_r1_rail1;
        rc->conn_to_next = cw->conn_to_next_r1;
        rc->conn_from_prev = cw->conn_from_prev_r1;
        rc->conn_to_next_rail1 = cw->conn_to_next_r1_rail1;
        rc->conn_from_prev_rail1 = cw->conn_from_prev_r1_rail1;
        rc->send_pe_spin = &cw->ring_send_pe_spin_r1;
        rc->recv_pe_spin = &cw->ring_recv_pe_spin_r1;
        rc->send_pe_spin_rail1 = &cw->ring_send_pe_spin_r1_rail1;
        rc->recv_pe_spin_rail1 = &cw->ring_recv_pe_spin_r1_rail1;
    }
}

/* Ring 0 / Ring 1 共通の PE progress. ring_ctx_ptrs で対象 ring の PE を直接指定。 */
static inline void progress_ring_pes_rc(struct ring_ctx_ptrs *rc, bool use_dual_ring)
{
    try_pe_progress(rc->send->pe, rc->send_pe_spin);
    try_pe_progress(rc->recv->pe, rc->recv_pe_spin);
    if (use_dual_ring) {
        if (rc->send_rail1->pe)
            try_pe_progress(rc->send_rail1->pe, rc->send_pe_spin_rail1);
        if (rc->recv_rail1->pe)
            try_pe_progress(rc->recv_rail1->pe, rc->recv_pe_spin_rail1);
    }
}

/* collective_reduce_scatter 用: Ring 0 のみ progress */
static inline void progress_ring_pes(struct collective_worker_t *cw, bool use_dual_ring)
{
    try_pe_progress(cw->rdma_ring_send.pe, &cw->ring_send_pe_spin);
    try_pe_progress(cw->rdma_ring_recv.pe, &cw->ring_recv_pe_spin);
    if (use_dual_ring) {
        if (cw->rdma_ring_send_rail1.pe)
            try_pe_progress(cw->rdma_ring_send_rail1.pe, &cw->ring_send_pe_spin_rail1);
        if (cw->rdma_ring_recv_rail1.pe)
            try_pe_progress(cw->rdma_ring_recv_rail1.pe, &cw->ring_recv_pe_spin_rail1);
    }
}

/* ============================================================
 * ReduceScatter (Ring 0 固定)
 * ============================================================ */

doca_error_t collective_reduce_scatter(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf, void *recv_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    uint64_t id)
{
    (void)id;
    fp16_t *local_send_buffer = (fp16_t *)send_buf;
    fp16_t *local_receive_buffer = (fp16_t *)recv_buf;
    uint64_t chunk_size = buffer_len / world_size;
    uint64_t elements_per_chunk = chunk_size / sizeof(fp16_t);
    int steps = (int)world_size - 1;
    struct doca_task_desc task;

    /* RDB (RDMA Doorbell Barrier) で全 rank 同期。
     * RDB 未初期化時は内部で MPI_Barrier フォールバック。
     * RS は常に Ring 0 を使用 (RS は並列化していない) */
    rdb_barrier(cw, (int)rank, (int)world_size, 0);

    /* DPU Union Tracker (ring-only): barrier 後〜ring 完了直前までを wrap。
     * 純 NIC 転送時間のみを wall union で測る。 */
    dpu_union_tracker_enter(false /* is_ag=false for RS */);

    /* ---- 2 つの独立した最適化 ----
     *  (1) 先行 post (pre-post): g_rs_prepost (env RS_PREPOST, 既定 1)
     *      全 step の Recv をループ前に一括 post する。**チャンク分割とは無関係**で、
     *      off にすると受信 post が送信到達に間に合わず RoCE の RNR リトライで
     *      サイズ非依存の固定遅延（実測 ~3ms）が乗る。
     *  (2) チャンク分割 (pipelined): num_pieces > 1 (env AG_PIECE_MAX / AG_PIECE_TARGET)
     *      piece の recv 完了ごとに集約し、集約を後続 piece の通信と overlap する。
     *      chunk_size < AG_PIECE_MIN_SIZE(256KB) では num_pieces=1 に落ちる。
     * RS は集約が要素単位なので piece は要素で割り byte サイズを導出（fp16 整列）。 */
    int num_pieces = ag_compute_num_pieces(chunk_size);
    uint64_t elems_per_piece = elements_per_chunk / (uint64_t)num_pieces;
    if (elems_per_piece == 0) { num_pieces = 1; elems_per_piece = elements_per_chunk; }
    uint64_t last_elems = elements_per_chunk - elems_per_piece * (uint64_t)(num_pieces - 1);
    uint64_t piece_size = elems_per_piece * sizeof(fp16_t);        /* bytes (even) */
    uint64_t last_piece_size = last_elems * sizeof(fp16_t);
    /* 集約ループは piece 数でループするため、num_pieces==1 なら自然に逐次動作になる。
     * （旧実装にあった pipelined フラグは先行 post と束ねていたため廃止した） */

    atomic_int recv_pendings[(steps > 0 ? steps : 1) * num_pieces];

    /* 先行 post: num_pieces（チャンク分割の有無）とは独立に、RS_PREPOST=1 なら常に実行 */
    if (g_rs_prepost) {
        for (int step = 0; step < steps; step++) {
            uint64_t r_idx = (rank + 2*world_size - 2 - step) % world_size;
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                int ri = step * num_pieces + p;
                atomic_init(&recv_pendings[ri], 1);
                init_ring_recv_task(&task,
                    (void *)((uintptr_t)recv_buf + r_idx * chunk_size + p_off),
                    p_len, &cw->rdma_ring_recv, ri);
                task.inline_pending = &recv_pendings[ri];
                task.pe_spin = &cw->ring_recv_pe_spin;
                submit_doca_task_from_desc(&cw->rdma_ring_recv, &task);
            }
        }
    }

    /* Read[0]: chunk[(rank-1)%ws] from GPU src → send_buf (chunk 一括) */
    #define RS_INLINE_READ(idx) do { \
        uint64_t _off = (idx) * chunk_size; \
        atomic_int _p; atomic_init(&_p, 1); \
        init_read_task(&task, (void *)((uintptr_t)send_buf + _off), \
                       (void *)(remote_src_addr + _off), \
                       chunk_size, &cw->host_src_rmem, &cw->rdma_rma, cw->host_conn, 0); \
        task.inline_pending = &_p; task.pe_spin = &cw->rma_pe_spin; \
        submit_doca_task_from_desc(&cw->rdma_rma, &task); \
        while (atomic_load_explicit(&_p, memory_order_acquire) > 0) \
            progress_rma_pes(cw, false); \
    } while(0)

    uint64_t read_index = (rank - 1 + world_size) % world_size;
    RS_INLINE_READ(read_index);

    for (int step = 0; step < steps; step++) {
        uint64_t s_idx = (rank + world_size - 1 - step) % world_size;
        uint64_t agg_index = (rank + world_size - 1 - (step + 1)) % world_size;
        uint64_t r_idx = (rank + 2*world_size - 2 - step) % world_size;

        /* 先行 post 無効時のみ: この step の Recv をここで post する */
        if (!g_rs_prepost) {
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                int ri = step * num_pieces + p;
                atomic_init(&recv_pendings[ri], 1);
                init_ring_recv_task(&task,
                    (void *)((uintptr_t)recv_buf + r_idx * chunk_size + p_off),
                    p_len, &cw->rdma_ring_recv, ri);
                task.inline_pending = &recv_pendings[ri];
                task.pe_spin = &cw->ring_recv_pe_spin;
                submit_doca_task_from_desc(&cw->rdma_ring_recv, &task);
            }
        }

        /* Send を piece ごとに post */
        atomic_int send_p;
        atomic_init(&send_p, num_pieces);
        for (int p = 0; p < num_pieces; p++) {
            uint64_t p_off = (uint64_t)p * piece_size;
            uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
            init_ring_send_task(&task,
                (void *)((uintptr_t)send_buf + s_idx * chunk_size + p_off),
                p_len, &cw->rdma_ring_send, cw->conn_to_next, step);
            task.inline_pending = &send_p; task.pe_spin = &cw->ring_send_pe_spin;
            submit_doca_task_from_desc(&cw->rdma_ring_send, &task);
        }

        /* Read next chunk (chunk 一括) — Send/Recv と並行 */
        read_index = (rank - step - 2 + 2*world_size) % world_size;
        atomic_int read_p;
        atomic_init(&read_p, 1);
        {
            uint64_t _off = read_index * chunk_size;
            init_read_task(&task, (void *)((uintptr_t)send_buf + _off),
                           (void *)(remote_src_addr + _off),
                           chunk_size, &cw->host_src_rmem, &cw->rdma_rma, cw->host_conn, 0);
            task.inline_pending = &read_p; task.pe_spin = &cw->rma_pe_spin;
            submit_doca_task_from_desc(&cw->rdma_rma, &task);
        }

        /* Read(自チャンクの GPU 読み)を先に完了させる: read_index == agg_index で
         * 集約のベース(send_buf[agg_index])になるため、集約より前に完了必須。 */
        while (atomic_load_explicit(&read_p, memory_order_acquire) > 0) {
            progress_ring_pes(cw, false);
            progress_rma_pes(cw, false);
        }

        /* Aggregate: piece p の Recv 完了を待って集約 */
        for (int p = 0; p < num_pieces; p++) {
            int ri = step * num_pieces + p;
            while (atomic_load_explicit(&recv_pendings[ri], memory_order_acquire) > 0) {
                progress_ring_pes(cw, false);
                progress_rma_pes(cw, false);
            }
            uint64_t p_elems = (p == num_pieces - 1) ? last_elems : elems_per_piece;
            uint64_t start = elements_per_chunk * agg_index + (uint64_t)p * elems_per_piece;
            uint64_t end = start + p_elems;
            if (p_elems <= RS_AGGREGATION_THRESHOLD / sizeof(fp16_t)) {
                rs_add_range_neon(local_send_buffer, local_receive_buffer, start, end);
            } else {
                submit_rs_task(cw->rs_pool, local_send_buffer, local_receive_buffer, start, end, ri);
                poll_wait_rs_completion(cw->rs_pool, ri);
            }
        }

        /* Send 完了待ち（Recv/Read は上で完了済み）*/
        while (atomic_load_explicit(&send_p, memory_order_acquire) > 0) {
            progress_ring_pes(cw, false);
            progress_rma_pes(cw, false);
        }
    }

    /* Put: send_buf[rank] → GPU dst */
    {
        atomic_int _p; atomic_init(&_p, 1);
        init_write_task(&task, (void *)((uintptr_t)send_buf + rank * chunk_size),
                        (void *)(remote_dst_addr),
                        chunk_size, &cw->host_dst_rmem, &cw->rdma_rma, cw->host_conn, 0);
        task.inline_pending = &_p; task.pe_spin = &cw->rma_pe_spin;
        submit_doca_task_from_desc(&cw->rdma_rma, &task);
        while (atomic_load_explicit(&_p, memory_order_acquire) > 0)
            progress_rma_pes(cw, false);
    }

    #undef RS_INLINE_READ
    /* DPU Union Tracker (ring-only): RS 終了 */
    dpu_union_tracker_exit(false);
    return DOCA_SUCCESS;
}

/* ============================================================
 * AllGather (ring_id で Ring 0/1 を選択)
 * ============================================================ */

doca_error_t collective_all_gather(
    uint64_t rank, uint64_t world_size,
    struct collective_worker_t *cw,
    void *send_buf,
    uint64_t buffer_len, uint64_t remote_src_addr, uint64_t remote_dst_addr,
    struct doca_mmap *gpu_src_local_mmap,
    uint64_t id,
    struct doca_mmap *gpu_dst_local_mmap,
    struct doca_mmap *gpu_src_local_mmap_rail1,
    struct doca_mmap *gpu_dst_local_mmap_rail1,
    int ring_id)
{
    (void)id;
    uint64_t chunk_size = buffer_len / world_size;

    /* ring_id に対応する context/conn/pe_spin を取得 */
    struct ring_ctx_ptrs rc;
    get_ring_ctx(cw, ring_id, &rc);

    /* ピース分割パラメータ (size-adaptive + runtime tunable)。
     * AG_PIECE_MAX (env) と AG_PIECE_TARGET (env) で実行時調整可能。 */
    int num_pieces = ag_compute_num_pieces(chunk_size);
    uint64_t piece_size = chunk_size / (uint64_t)num_pieces;
    uint64_t last_piece_size = chunk_size - piece_size * (uint64_t)(num_pieces - 1);

    struct doca_task_desc task;

    /* Multi-Rail helpers */
    bool use_dual = cw->dual_rail && cw->host_src_rmem_rail1.valid && cw->host_dst_rmem_rail1.valid;
    bool use_dual_ring = (rc.conn_to_next_rail1 != NULL) && (rc.conn_from_prev_rail1 != NULL);

    /* GPU Direct でも dual-rail を使う条件
     *   - rail0 / rail1 の両方に Ring 接続がある (use_dual_ring == true)
     *   - GPU Direct で必要な mmap が両 rail で揃っている
     * full_gpu_direct では src/dst 両方に mmap が必要、
     * gpu_direct のみでは dst だけで足りる。 */
    bool gpu_direct_dual_ring_dst_ok = (gpu_dst_local_mmap != NULL) && (gpu_dst_local_mmap_rail1 != NULL);
    bool gpu_direct_dual_ring_src_ok = (gpu_src_local_mmap != NULL) && (gpu_src_local_mmap_rail1 != NULL);
    if (gpu_dst_local_mmap) {
        /* GPU Direct 時: rail1 mmap が揃っていなければシングルレールにフォールバック */
        if (!gpu_direct_dual_ring_dst_ok)
            use_dual_ring = false;
        /* full_gpu_direct (src も使う) なら src 側 rail1 も必須 */
        if (gpu_src_local_mmap && !gpu_direct_dual_ring_src_ok)
            use_dual_ring = false;
    }

    /* rail スケーリング比較テスト: FORCE_SINGLE_RAIL=1 で dual-rail を強制 off（AG のみ）。 */
    if (g_force_single_rail) {
        use_dual = false;
        use_dual_ring = false;
    }

    /* GPU Direct — PCI import した mmap が有効なら使用 */
    bool gpu_direct = (gpu_dst_local_mmap != NULL);
    /* src GVMI も有効なら world_size ステップ (Read+Write+Barrier 全廃) */
    bool full_gpu_direct = gpu_direct && (gpu_src_local_mmap != NULL);

    /* full_gpu_direct なら world_size ステップ (自分のチャンクも Ring 経由で受信) */
    int total_steps = full_gpu_direct ? (int)world_size : (int)world_size - 1;

    /* RDB barrier で全 rank 同期。RDB 未初期化時は内部で MPI_Barrier フォールバック。
     * ring_id 専用の MPI_Comm を使って Ring 0/1 の barrier が混線しないようにする */
    rdb_barrier(cw, (int)rank, (int)world_size, ring_id);

    /* DPU Union Tracker (ring-only): barrier 後〜ring 完了直前までを wrap。
     * 純 NIC 転送時間のみを wall union で測る。 */
    dpu_union_tracker_enter(true /* is_ag=true */);

    /* ========================================
     * 全 Ring step の Recv を一括先行 post
     * GPU Direct → gpu_dst_local_mmap に直接 Recv (Put 不要)
     * ======================================== */
    atomic_int recv_pendings[total_steps > 0 ? total_steps : 1];
    for (int step = 0; step < total_steps; step++) {
        uint64_t receive_index = (rank - 1 - (uint64_t)step + world_size * 2) % world_size;
        uint64_t receive_offset = receive_index * chunk_size;
        atomic_init(&recv_pendings[step], num_pieces);
        for (int p = 0; p < num_pieces; p++) {
            uint64_t p_off = (uint64_t)p * piece_size;
            uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
            bool piece_rail1 = use_dual_ring && (p & 1);
            /* ring_id に応じた ctx (rc.recv / rc.recv_rail1) を選択 */
            struct doca_rdma_ctx_t *recv_ctx = piece_rail1 ? rc.recv_rail1 : rc.recv;
            if (gpu_direct) {
                init_ring_recv_task(&task,
                    (void *)(remote_dst_addr + receive_offset + p_off),
                    p_len, recv_ctx, step * num_pieces + p);
                /* rail1 piece は ring_dev_rail1 で import した mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_dst_local_mmap_rail1 : gpu_dst_local_mmap;
            } else {
                init_ring_recv_task(&task,
                    (void *)((uintptr_t)send_buf + receive_offset + p_off),
                    p_len, recv_ctx, step * num_pieces + p);
            }
            task.inline_pending = &recv_pendings[step];
            task.pe_spin = piece_rail1 ? rc.recv_pe_spin_rail1 : rc.recv_pe_spin;
            submit_doca_task_from_desc(recv_ctx, &task);
        }
    }

    /* ========================================
     * INITIAL フェーズ: Read + Write (full_gpu_direct なら全スキップ)
     * full_gpu_direct では world_size ステップ Ring で全データ転送
     * ======================================== */
    if (!full_gpu_direct) {
        uint64_t my_offset = rank * chunk_size;

        {
            /* Legacy: Read → DPU DDR → Write → GPU dst
             * (Read の GPU dst 直書きは PCIe U ターン不可のため使えない) */
            atomic_int read_pending;
            atomic_init(&read_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *src_rmem = (use_dual && (p & 1)) ? &cw->host_src_rmem_rail1 : &cw->host_src_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_read_task(&task,
                               (void *)((uintptr_t)send_buf + my_offset + p_off),
                               (void *)(remote_src_addr + p_off),
                               p_len, src_rmem, rma_ctx, conn, p);
                task.inline_pending = &read_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task);
            }
            while (atomic_load_explicit(&read_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);

            /* Write: DPU send_buf → Host GPU dst[my_rank] */
            atomic_int write_pending;
            atomic_init(&write_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *dst_rmem = (use_dual && (p & 1)) ? &cw->host_dst_rmem_rail1 : &cw->host_dst_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_write_task(&task,
                                (void *)((uintptr_t)send_buf + my_offset + p_off),
                                (void *)(remote_dst_addr + my_offset + p_off),
                                p_len, dst_rmem, rma_ctx, conn, p);
                task.inline_pending = &write_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task);
            }
            while (atomic_load_explicit(&write_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);
        }

        /* barrier は Recv bulk post 前に実行済み */
    }

    /* ========================================
     * Ring フェーズ
     * ======================================== */
    for (int step = 0; step < total_steps; step++) {
        uint64_t send_index = (rank - (uint64_t)step + world_size) % world_size;
        uint64_t receive_index = (rank - 1 - (uint64_t)step + world_size * 2) % world_size;
        uint64_t send_offset = send_index * chunk_size;
        uint64_t receive_offset = receive_index * chunk_size;

        /* Send 投入 */
        atomic_int send_pending;
        atomic_init(&send_pending, num_pieces);
        for (int p = 0; p < num_pieces; p++) {
            uint64_t p_off = (uint64_t)p * piece_size;
            uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
            bool piece_rail1 = use_dual_ring && (p & 1);
            /* ring_id に応じた ctx (rc.send / rc.send_rail1) と conn を選択 */
            struct doca_rdma_ctx_t *send_ctx = piece_rail1 ? rc.send_rail1 : rc.send;
            struct doca_rdma_connection *send_conn = piece_rail1 ? rc.conn_to_next_rail1 : rc.conn_to_next;

            if (full_gpu_direct && step == 0) {
                /* step 0 は GPU src から Send */
                init_ring_send_task(&task,
                    (void *)(remote_src_addr + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
                /* rail1 piece は src rail1 mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_src_local_mmap_rail1 : gpu_src_local_mmap;
            } else if (gpu_direct) {
                /* step 1+ は GPU dst から Send */
                init_ring_send_task(&task,
                    (void *)(remote_dst_addr + send_offset + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
                /* rail1 piece は dst rail1 mmap を使う */
                task.local_mmap_override = piece_rail1 ? gpu_dst_local_mmap_rail1 : gpu_dst_local_mmap;
            } else {
                /* Legacy: DPU send_buf から Send */
                init_ring_send_task(&task,
                    (void *)((uintptr_t)send_buf + send_offset + p_off),
                    p_len, send_ctx, send_conn, step * num_pieces + p);
            }
            task.inline_pending = &send_pending;
            task.pe_spin = piece_rail1 ? rc.send_pe_spin_rail1 : rc.send_pe_spin;
            submit_doca_task_from_desc(send_ctx, &task);
        }

        while (atomic_load_explicit(&recv_pendings[step], memory_order_acquire) > 0 ||
               atomic_load_explicit(&send_pending, memory_order_acquire) > 0)
            progress_ring_pes_rc(&rc, use_dual_ring);

        /* Put: gpu_direct なら不要 (Recv が GPU dst に直接着地済み) */
        if (!gpu_direct) {
            atomic_int put_pending;
            atomic_init(&put_pending, num_pieces);
            for (int p = 0; p < num_pieces; p++) {
                uint64_t p_off = (uint64_t)p * piece_size;
                uint64_t p_len = (p == num_pieces - 1) ? last_piece_size : piece_size;
                struct doca_remote_mem_t *dst_rmem = (use_dual && (p & 1)) ? &cw->host_dst_rmem_rail1 : &cw->host_dst_rmem;
                struct doca_rdma_ctx_t   *rma_ctx  = (use_dual && (p & 1)) ? &cw->rdma_rma_rail1 : &cw->rdma_rma;
                struct doca_rdma_connection *conn   = (use_dual && (p & 1)) ? cw->host_conn_rail1 : cw->host_conn;
                init_write_task(&task,
                                (void *)((uintptr_t)send_buf + receive_offset + p_off),
                                (void *)(remote_dst_addr + receive_offset + p_off),
                                p_len, dst_rmem, rma_ctx, conn, p);
                task.inline_pending = &put_pending;
                task.pe_spin = (use_dual && (p & 1)) ? &cw->rma_pe_spin_rail1 : &cw->rma_pe_spin;
                submit_doca_task_from_desc(rma_ctx, &task);
            }
            while (atomic_load_explicit(&put_pending, memory_order_acquire) > 0)
                progress_rma_pes(cw, use_dual);
        }
    }

    /* DPU Union Tracker (ring-only): AG 終了 */
    dpu_union_tracker_exit(true);
    return DOCA_SUCCESS;
}
