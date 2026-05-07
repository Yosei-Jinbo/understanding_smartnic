#ifndef COMCH_SERVER_H
#define COMCH_SERVER_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include <doca_error.h>
#include <doca_rdma.h>
#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_pe.h>
#include <doca_ctx.h>
#include <doca_dev.h>

#include <mpi.h>
#include "../common/doca_rdma_utils.h"
#include "../common/rdma_common.h"
#include "../common/dma_common.h"

/* Phase 15: N-Ring parallel AG
 * Multiple independent Ring instances run in parallel to reduce DPU-side
 * serial processing. Each ring has its own RDMA contexts and connections.
 * AGs are dispatched to rings by a deterministic counter (ag_seq % N_RINGS)
 * to ensure all ranks assign the same AG to the same ring. */
#define N_RINGS 2

/* Forward decl for job queue */
struct ring_job;

/* ---- FP16 型定義 (本実装は float16 のみをサポート) ---- */
#if defined(__FLT16_MAX__) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
typedef _Float16 fp16_t;
#else
#error "FP16 (_Float16) is not supported by this toolchain/target"
#endif

/*
  コア割り当て表

  コア | Rank 0           | Rank 1           | 備考
  ----|------------------|------------------|------------------
   0  | RMA worker       |                  | RDMA Read/Write (Host-DPU)
   1  | Ring RECV worker |                  | RDMA Receive
   2  | Ring SEND worker |                  | RDMA Send
   3  |                  | RMA worker       |
   4  |                  | Ring RECV worker |
   5  |                  | Ring SEND worker |
   6  | msg_pool_worker  |                  | poll_wait_rs実行
   7  |                  | msg_pool_worker  | poll_wait_rs実行
   8  | RS tid=0         | RS tid=0         | 共通（集約処理）
   9  | RS tid=1         | RS tid=1         | 共通
  10  | RS tid=2         | RS tid=2         | 共通
  11  | RS tid=3         | RS tid=3         | 共通
  12  | RS tid=4         | RS tid=4         | 共通
  13  | RS tid=5         | RS tid=5         | 共通
  14  | メインスレッド    |                  |
  15  |                  | メインスレッド    |
*/

/* ---- SmartNIC-SmartNIC 用のメモリプール ---- */
#define CW_BUFFER_POOL_SLOTS  2
#define CW_SLOT_DEFAULT_BYTES  ((size_t)(1ULL << 30))   /* 1GB per slot (OPT-1.3B等の大規模RS対応) */

struct cw_buffer_slot {
    void   *buf;
    size_t  capacity;
    int     in_use;
};

struct cw_buffer_pool {
    struct cw_buffer_slot slots[CW_BUFFER_POOL_SLOTS];
    pthread_spinlock_t    lock;
    void                 *base;        /* 連続確保ベースアドレス */
    size_t                total_size;   /* 連続確保合計サイズ */
};

/* message_recv_callback から投げるワーカースレッド用の引数 */
struct message_worker_arg {
    struct comch_ctrl_path_server_objects *sample_objects;
    struct control_cmd *recv_cmd;
    void *buf_copy;
    uint64_t t_recv_ns;
};

/* ---- DOCA RDMA タスクタイプ (SPSC キュー用) ---- */
typedef enum {
    DOCA_RDMA_TASK_NONE = 0,
    DOCA_RDMA_TASK_READ,       /* Host GPU src → DPU buf (旧 UCP_TASK_SRC_GET) */
    DOCA_RDMA_TASK_WRITE,      /* DPU buf → Host GPU dst (旧 UCP_TASK_DST_PUT) */
    DOCA_RDMA_TASK_RING_SEND,  /* DPU → 次 DPU (旧 UCP_TASK_SEND) */
    DOCA_RDMA_TASK_RING_RECV,  /* 前 DPU → DPU (旧 UCP_TASK_RECV) */
    DOCA_RDMA_TASK_EXIT,
} doca_rdma_task_type_t;

/* SPSC キューに投入するタスク記述子 */
struct doca_task_desc {
    doca_rdma_task_type_t type;

    void    *local_addr;    /* DPU ローカルバッファアドレス */
    void    *remote_addr;   /* リモートバッファアドレス (Read/Write 用、remote mmap 内) */
    size_t   length;

    int stride_id;
    atomic_bool completed;

    /* Read/Write で使うリモートメモリ情報 */
    struct doca_remote_mem_t *remote_mem;   /* src (Read) or dst (Write) */

    /* どの RDMA コンテキストを使うか (ワーカスレッドが参照) */
    struct doca_rdma_ctx_t   *rdma_ctx;
    struct doca_rdma_connection *connection; /* Send/Recv で使う接続 */

    /* Phase 6 Step E: lightweight completion (NULL = use SPSC completion queue) */
    atomic_int *inline_pending;

    /* Phase 6 Step F: PE spinlock for thread-safe inline submit (NULL = no locking) */
    struct pe_spin_t *pe_spin;

    /* Phase 8: Cross-GVMI local mmap override for GPU Direct Ring Send/Recv */
    struct doca_mmap *local_mmap_override;  /* NULL = use rdma_ctx->local_mmap */
};

struct rs_thread_pool_t;

/* ワーカタイプ */
typedef enum {
    DOCA_WORKER_TYPE_RMA = 0,       /* RDMA Read/Write (Host-DPU) */
    DOCA_WORKER_TYPE_RING_RECV = 1, /* RDMA Receive (DPU-DPU) */
    DOCA_WORKER_TYPE_RING_SEND = 2, /* RDMA Send (DPU-DPU) */
    DOCA_WORKER_TYPE_COUNT = 3,
} doca_worker_type_t;

/* 前方宣言 */
struct spsc_doca_task_queue_t;
struct spsc_completion_queue_t;

/* ワーカスレッドコンテキスト */
struct doca_worker_thread_ctx {
    pthread_t              thread;
    doca_worker_type_t     worker_type;
    int                    core_id;

    /* 担当する DOCA RDMA コンテキスト */
    struct doca_rdma_ctx_t *rdma_ctx;
    struct doca_rdma_ctx_t *rdma_ctx_rail1;  /* Multi-Rail: 2 番目 (NULL ならシングル) */

    void                  *task_queue;        /* spsc_doca_task_queue_t* */
    void                  *completion_queue;  /* spsc_completion_queue_t* */

    atomic_bool            running;
    atomic_bool            paused;         /* inline 実行中は true */
    atomic_bool            paused_ack;     /* pause 確認用 */

    /* Phase 6 Step F: PE spinlocks for thread-safe PE progress */
    struct pe_spin_t      *pe_spin;        /* rail0 PE spinlock */
    struct pe_spin_t      *pe_spin_rail1;  /* rail1 PE spinlock (NULL if single rail) */
};

/* DOCA ワーカスレッドプール */
struct doca_worker_thread_pool_t {
    struct doca_worker_thread_ctx workers[DOCA_WORKER_TYPE_COUNT];
    int mpi_rank;
    atomic_bool initialized;
};

/* ---- AllGather パイプライン定数 ----
 * Phase 12.2 G: 実行時可変化
 *   - AG_PIECE_MAX_COUNT_DEFAULT は環境変数 AG_PIECE_MAX が未設定時の既定値
 *   - AG_PIECE_TARGET_BYTES は piece size の目標値 (chunk が大きいとき piece 数を増やす)
 *   - AG_PIECE_HARD_MAX は環境変数で指定できる上限 (ビルド時の絶対上限)
 */
#define AG_PIECE_MIN_SIZE         (256 * 1024)   /* 256KB: これ未満はパイプラインしない */
#define AG_PIECE_MAX_COUNT_DEFAULT 8             /* 既定の最大ピース数 */
#define AG_PIECE_HARD_MAX         32             /* ビルド時の絶対上限 (alloc サイズ等のため) */
#define AG_PIECE_TARGET_BYTES     (8UL * 1024 * 1024)  /* 目標 piece size = 8MB */
/* 旧 API 互換 */
#define AG_PIECE_MAX_COUNT        AG_PIECE_HARD_MAX

/* ---- Remote mmap キャッシュ (DPU 側, ハッシュ直接マッピング) ---- */
#define RMEM_CACHE_SIZE     256            /* 2 のべき乗 */
#define RMEM_CACHE_MASK     (RMEM_CACHE_SIZE - 1)

struct rmem_cache_entry {
    uint64_t addr;                         /* remote_addr (GPU アドレス) */
    size_t   len;                          /* remote_len */
    struct doca_remote_mem_t rmem;
    bool     valid;
};

/* ---- PCI import mmap キャッシュ (GPU Direct, 大容量直接マッピング) ---- */
#define PCI_MMAP_CACHE_SIZE   (1 << 20)         /* 1048576 = 2^20 */
#define PCI_MMAP_CACHE_MASK   (PCI_MMAP_CACHE_SIZE - 1)

struct pci_mmap_cache_entry {
    uint64_t addr;                         /* GPU アドレス */
    size_t   len;                          /* バッファサイズ */
    struct doca_mmap *mmap;                /* PCI import 済み mmap */
    bool     valid;
};

/* ---- Phase 6 Step F: PE spinlock for thread-safe concurrent access ---- */
struct pe_spin_t {
    atomic_flag lock;
};

static inline void pe_spin_init(struct pe_spin_t *s)
{
    atomic_flag_clear(&s->lock);
}

/* Non-blocking trylock PE progress. Returns true if progress was called. */
static inline bool try_pe_progress(struct doca_pe *pe, struct pe_spin_t *spin)
{
    if (!pe) return false;
    if (atomic_flag_test_and_set(&spin->lock)) return false;  /* busy → skip */
    doca_pe_progress(pe);
    atomic_flag_clear(&spin->lock);
    return true;
}

/* Blocking lock for task submit (must succeed). */
static inline void pe_spin_lock(struct pe_spin_t *spin)
{
    while (atomic_flag_test_and_set(&spin->lock)) {
#if defined(__aarch64__) || defined(__arm__)
        __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause" ::: "memory");
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
}

static inline void pe_spin_unlock(struct pe_spin_t *spin)
{
    atomic_flag_clear(&spin->lock);
}

/* ---- メイン集合通信ワーカー構造体 ---- */
struct collective_worker_t {
    /* DOCA デバイス */
    struct doca_dev *dev;           /* ComCh 用 PCIe デバイス */
    struct doca_dev *rdma_dev;      /* RMA 用ネットワークデバイス (Host ポートに対応: mlx5_2 or mlx5_3) */
    struct doca_dev *ring_dev;      /* Ring 用ネットワークデバイス (常に enp3s0f0s0 = mlx5_2) */

    /* Host-DPU RMA 用 RDMA コンテキスト (Read/Write) — 独自 PE を持つ */
    struct doca_rdma_ctx_t rdma_rma;

    /* Host GPU メモリへのリモートアクセス */
    struct doca_remote_mem_t host_src_rmem;  /* GPU src バッファ */
    struct doca_remote_mem_t host_dst_rmem;  /* GPU dst バッファ */

    /* Host との RDMA 接続 */
    struct doca_rdma_connection *host_conn;

    /* DPU-DPU Ring 用 RDMA コンテキスト (各自独立 PE を持つ) */
    struct doca_rdma_ctx_t rdma_ring_send;  /* Send 専用 */
    struct doca_rdma_ctx_t rdma_ring_recv;  /* Receive 専用 */

    /* Ring 接続 */
    struct doca_rdma_connection *conn_to_next;  /* 次 rank への SEND 用 */
    struct doca_rdma_connection *conn_from_prev; /* 前 rank からの RECV 用 */

    /* ---- Multi-Rail Ring: 2 番目のポート (enp3s0f1s0 = mlx5_3) ---- */
    struct doca_dev *ring_dev_rail1;
    struct doca_rdma_ctx_t rdma_ring_send_rail1;
    struct doca_rdma_ctx_t rdma_ring_recv_rail1;
    struct doca_rdma_connection *conn_to_next_rail1;
    struct doca_rdma_connection *conn_from_prev_rail1;

    /* Host との RDMA 接続記述子 (ComCh 経由で交換) */
    const void *host_rma_conn_desc;
    size_t      host_rma_conn_desc_len;

    size_t memrange_size;

    /* 全 RDMA コンテキストで共有する作業バッファ (連続確保) */
    void *working_buf;
    size_t working_buf_size;

    struct cw_buffer_pool send_pool;
    struct cw_buffer_pool recv_pool;

    struct rs_thread_pool_t  *rs_pool;
    int mpi_rank;

    /* DOCA ワーカスレッドプール */
    struct doca_worker_thread_pool_t *worker_pool;

    /* Remote mmap キャッシュ (制御パス高速化) */
    struct rmem_cache_entry src_rmem_cache[RMEM_CACHE_SIZE];
    struct rmem_cache_entry dst_rmem_cache[RMEM_CACHE_SIZE];

    /* ---- Multi-Rail (Phase 4): 2 番目のポート ---- */
    bool dual_rail;
    struct doca_dev *rdma_dev_rail1;           /* RMA rail1 デバイス (他方のポート) */
    struct doca_rdma_ctx_t rdma_rma_rail1;     /* RMA rail1 RDMA コンテキスト */
    struct doca_rdma_connection *host_conn_rail1;
    const void *host_rma_conn_desc_rail1;
    size_t      host_rma_conn_desc_len_rail1;
    struct doca_remote_mem_t host_src_rmem_rail1;
    struct doca_remote_mem_t host_dst_rmem_rail1;
    struct rmem_cache_entry src_rmem_cache_rail1[RMEM_CACHE_SIZE];
    struct rmem_cache_entry dst_rmem_cache_rail1[RMEM_CACHE_SIZE];

    /* ---- Phase 8: PCI import mmap キャッシュ (GPU Direct) ---- */
    /* 16384 エントリの直接マッピングキャッシュ (衝突時は上書き) */
    struct pci_mmap_cache_entry *gpu_dst_pci_cache;  /* dst rail0 (= ring_dev) */
    struct pci_mmap_cache_entry *gpu_src_pci_cache;  /* src rail0 */

    /* ---- Phase 12.2: Dual-rail PCI import mmap キャッシュ (rail1 = ring_dev_rail1) ---- */
    struct pci_mmap_cache_entry *gpu_dst_pci_cache_rail1;
    struct pci_mmap_cache_entry *gpu_src_pci_cache_rail1;

    /* ---- Phase 6 Step F: PE spinlocks ---- */
    struct pe_spin_t rma_pe_spin;
    struct pe_spin_t rma_pe_spin_rail1;
    struct pe_spin_t ring_send_pe_spin;
    struct pe_spin_t ring_send_pe_spin_rail1;
    struct pe_spin_t ring_recv_pe_spin;
    struct pe_spin_t ring_recv_pe_spin_rail1;

    /* ---- Phase 5 Step 3: RDMA Doorbell (DPU→Host 完了通知) ---- */
    struct doca_remote_mem_t host_doorbell_rmem;   /* Host doorbell の remote mmap */
    bool doorbell_enabled;
    uint64_t doorbell_seq;                         /* monotonic completion counter */

    /* ---- Phase 12.1 v2: RDMA Doorbell Barrier (RDB Write 版)
     * MPI_Barrier の OS jitter (1-200ms) を RDMA Write による
     * 1-pass Ring barrier (~20μs) に置き換える。
     *
     * プロトコル (各 rank):
     *   1. my_seq++
     *   2. RDMA Write my_seq → 次 rank の rdb_local_slot
     *   3. busy spin: 自分の rdb_local_slot >= my_seq まで待つ (前 rank が書く)
     *
     * working_buf の末尾を slot 領域として使うため追加メモリ登録不要。
     * 隣接 rank の rdb_local_slot のリモートアドレスを MPI で交換。 */
    bool rdb_enabled;
    uint64_t rdb_my_seq;                         /* バリア sequence (monotonic) */
    volatile uint64_t *rdb_local_slot;           /* 自分の slot (working_buf 内、前 rank が書く) */
    uint64_t *rdb_send_data;                     /* 送信用バッファ (working_buf 内、自分が書く) */
    uint64_t rdb_next_remote_slot_addr;          /* 次 rank の rdb_local_slot のリモートアドレス */
    struct doca_remote_mem_t rdb_next_rmem;      /* 次 rank の working_buf 全体の remote_mem */
    /* profiling */
    uint64_t rdb_total_count;
    uint64_t rdb_total_ns;
    uint64_t rdb_max_ns;

    /* ---- Phase 5 Step 4: RDMA Command Slot (Host→DPU) ---- */
    void *cmd_slot_buf;                    /* DPU-local command slot buffer */
    struct doca_mmap *cmd_slot_mmap;       /* mmap for cmd_slot */
    const void *cmd_slot_export_desc;      /* export descriptor for Host */
    size_t cmd_slot_export_desc_len;
    volatile uint64_t cmd_expected_seq;    /* next expected sequence number */
    bool cmd_slot_enabled;

    /* ---- Phase 14/15: GPU flag pool (DPU→Host GPU completion sync) ----
     *   Phase 14: ホスト Python の GpuFlagPool が確保した int32 配列の remote_mem。
     *     AG 完了時に flag_value を flag_gpu_addr に inline RDMA Write することで、
     *     GPU の compute stream が cuStreamWaitValue32 で待機している箇所を解放する。
     *   Phase 15: flag pool が GPU memory 上にある場合、host は PCI export
     *     (Cross-GVMI) も送ってくる。DPU 側で doca_mmap_create_from_export すると
     *     local mmap が得られ、phase14_write_flag は local_mmap_override 経由で
     *     GPU memory に直接 Write できる。cuStreamWaitValue32 が pinned host
     *     memory を polling する ~1.8ms/AG のストールを消去できる (fwd -200ms)。 */
    bool flag_pool_enabled;
    struct doca_remote_mem_t flag_pool_rmem;        /* rail0 (host_conn 経由, RDMA rkey 由来) */
    struct doca_remote_mem_t flag_pool_rmem_rail1;  /* rail1 (host_conn_rail1 経由, optional) */
    bool flag_pool_rail1_enabled;
    /* Phase 15: Cross-GVMI PCI-imported mmap (local mmap representing host GPU memory) */
    struct doca_mmap *flag_pool_pci_mmap;           /* rail0 (from export_pci) */
    struct doca_mmap *flag_pool_pci_mmap_rail1;     /* rail1 (from export_pci) */
    bool flag_pool_pci_enabled;                     /* true なら PCI path を使う */
    /* per-AG send buffer (4 bytes) のプール: 並行 AG が複数あっても上書きされないよう
     * リング状に使う。num_recv_tasks 256 と整合する。 */
    void   *flag_send_buf_pool;          /* working_buf 内、または 別 alloc */
    size_t  flag_send_buf_pool_len;
    atomic_uint flag_send_buf_idx;
    /* profiling */
    uint64_t flag_write_total_count;
    uint64_t flag_write_total_ns;
    uint64_t flag_write_max_ns;
    pthread_mutex_t flag_write_mtx;  /* Phase 15: 2 rings が同時に flag write する場合の排他 */

    /* ---- Phase 15: Ring 1 (parallel AG second ring) ----
     *   Ring 0 と完全に独立した RDMA contexts / connections / PE spinlocks。
     *   各 AG は ag_seq % N_RINGS で Ring 0 または Ring 1 に deterministic に割り当てる。
     *   Ring 0 と Ring 1 は物理的に並列に実行できる (NIC hardware が多重化)。 */
    struct doca_rdma_ctx_t rdma_ring_send_r1;
    struct doca_rdma_ctx_t rdma_ring_recv_r1;
    struct doca_rdma_ctx_t rdma_ring_send_r1_rail1;
    struct doca_rdma_ctx_t rdma_ring_recv_r1_rail1;
    struct doca_rdma_connection *conn_to_next_r1;
    struct doca_rdma_connection *conn_from_prev_r1;
    struct doca_rdma_connection *conn_to_next_r1_rail1;
    struct doca_rdma_connection *conn_from_prev_r1_rail1;
    struct pe_spin_t ring_send_pe_spin_r1;
    struct pe_spin_t ring_send_pe_spin_r1_rail1;
    struct pe_spin_t ring_recv_pe_spin_r1;
    struct pe_spin_t ring_recv_pe_spin_r1_rail1;
    bool ring_r1_enabled;        /* Ring 1 の接続が成功したかどうか */
    bool ring_r1_rail1_enabled;  /* Ring 1 rail1 が有効かどうか */

    /* Per-ring job queue (Phase 15) */
    struct ring_job *ring_queue_head[N_RINGS];
    struct ring_job *ring_queue_tail[N_RINGS];
    pthread_mutex_t ring_queue_mtx[N_RINGS];
    pthread_cond_t  ring_queue_cv[N_RINGS];
    atomic_bool ring_queue_stop[N_RINGS];
    pthread_t ring_proc_thread[N_RINGS];
    atomic_bool ring_proc_running[N_RINGS];

    /* Phase 15: Per-ring MPI communicator
     *   Ring 0 / Ring 1 thread が同時に MPI_Barrier を呼ぶ場合、MPI_COMM_WORLD を
     *   共有すると barrier 同士がマッチしてデッドロックや異常な待機を起こす。
     *   各 ring に専用の MPI_Comm を dup することで、各 ring 内で正しく barrier が
     *   全 rank 間で同期する。ring_comm_valid[i] = true のとき有効。 */
    MPI_Comm ring_comm[N_RINGS];
    bool ring_comm_valid[N_RINGS];

    /* Deterministic ring assignment counter */
    atomic_uint_fast64_t ag_seq_counter;
};

/* Phase 15: Per-ring job queue entry (lives in the queue, popped by proc thread) */
struct ring_job {
    struct control_cmd *cmd;  /* unpacked cmd (points into buf_copy) */
    void *buf_copy;           /* malloc'd buffer, owned by this job */
    uint64_t t_recv_ns;
    int ring_id;              /* 0 or 1 */
    struct comch_ctrl_path_server_objects *sample_objects;
    struct ring_job *next;
};


#endif
