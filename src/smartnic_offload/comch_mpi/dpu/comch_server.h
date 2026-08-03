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
#include "mmap_cache.h"
#include "dpu_config.h"

/* N-Ring parallel AG
 * Multiple independent Ring instances run in parallel to reduce DPU-side
 * serial processing. Each ring has its own RDMA contexts and connections.
 * AGs are dispatched to rings by a deterministic counter (ag_seq % N_RINGS)
 * to ensure all ranks assign the same AG to the same ring. */
#define N_RINGS 2

/* Forward decl for job queue */
struct ring_job;

/* FP16 型定義 (本実装は float16 のみをサポート)
 * __INTELLISENSE__ 分岐は _Float16 を解釈できない IDE 解析器用 (実ビルドでは使われない) */
#if defined(__INTELLISENSE__)
typedef uint16_t fp16_t;
#elif defined(__FLT16_MAX__) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
typedef _Float16 fp16_t;
#else
#error "FP16 (_Float16) is not supported by this toolchain/target"
#endif

/* ---- コア割り当て (env 駆動、詳細は dpu_config.h) ----
 * 実体は comch_server.c。core_alloc_init_from_env() が env を反映する。 */
extern int g_comm_cores;        /* COMM_CORES:    通信 progress の rank あたりコア数 C */
extern int g_compute_cores;     /* COMPUTE_CORES: 計算 RS 集約スレッド数 K (rank 共有) */
extern int g_force_staging;     /* FORCE_STAGING:     AG の GPU 直接アクセスを無効化 (ablation) */
extern int g_force_single_rail; /* FORCE_SINGLE_RAIL: AG の dual-rail を無効化 (default 0=dual) */
extern int g_rs_prepost;        /* RS_PREPOST:        RS の Recv 先行 post (default 1) */

/* 3 progress worker を C コアに packing（rank 分離） */
static inline int comm_core(int rank, int i)  { return (rank % 2) * g_comm_cores + (i % g_comm_cores); }
static inline int msg_pool_core(int rank)     { return 2 * g_comm_cores + (rank % 2); }
static inline int compute_core_base(void)     { return 2 * g_comm_cores + 2; }
static inline int compute_core(int i)         { return compute_core_base() + i; }               /* 共有 */
static inline int main_core(int rank)         { return compute_core_base() + g_compute_cores + (rank % 2); }

/* SmartNIC-SmartNIC 用のメモリプール */
#define CW_BUFFER_POOL_SLOTS  2
#define CW_SLOT_DEFAULT_BYTES  ((size_t)(1ULL << 30))   /* 1GB per slot (OPT-1.3B等の大規模RS対応) */

/* RDMA command slot (Host→DPU RDMA-Write コマンド) のバッファサイズ */
#define CMD_SLOT_SIZE 4096

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
};

/* DOCA RDMA タスクタイプ */
typedef enum {
    DOCA_RDMA_TASK_NONE = 0,
    DOCA_RDMA_TASK_READ,       /* Host GPU src → DPU buf (旧 UCP_TASK_SRC_GET) */
    DOCA_RDMA_TASK_WRITE,      /* DPU buf → Host GPU dst (旧 UCP_TASK_DST_PUT) */
    DOCA_RDMA_TASK_RING_SEND,  /* DPU → 次 DPU (旧 UCP_TASK_SEND) */
    DOCA_RDMA_TASK_RING_RECV,  /* 前 DPU → DPU (旧 UCP_TASK_RECV) */
} doca_rdma_task_type_t;

/* タスク記述子 (inline submit 用) */
struct doca_task_desc {
    doca_rdma_task_type_t type;

    void    *local_addr;    /* DPU ローカルバッファアドレス */
    void    *remote_addr;   /* リモートバッファアドレス (Read/Write 用、remote mmap 内) */
    size_t   length;

    int stride_id;

    /* Read/Write で使うリモートメモリ情報 */
    struct doca_remote_mem_t *remote_mem;   /* src (Read) or dst (Write) */

    /* どの RDMA コンテキストを使うか */
    struct doca_rdma_ctx_t   *rdma_ctx;
    struct doca_rdma_connection *connection; /* Send/Recv で使う接続 */

    /* 完了時にデクリメントされるカウンタ */
    atomic_int *inline_pending;

    /* inline submit 排他用 PE spinlock (NULL = ロックなし) */
    struct pe_spin_t *pe_spin;

    /* GPU Direct Ring Send/Recv 用の Cross-GVMI local mmap override */
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

/* ワーカスレッドコンテキスト (PE progress 駆動専用) */
/* Option 2: 1 ワーカが複数 PE を round-robin progress。
 * 1 ワーカあたり最大 = 3 役割 × 2 rail = 6 PE。 */
#define WORKER_MAX_PE  (2 * DOCA_WORKER_TYPE_COUNT)

struct doca_worker_thread_ctx {
    pthread_t              thread;
    doca_worker_type_t     worker_type;   /* 互換用（未使用） */
    int                    core_id;

    /* 担当する PE 群（context + spinlock を round-robin で progress） */
    int                     n_pe;
    struct doca_rdma_ctx_t *pe_ctx[WORKER_MAX_PE];
    struct pe_spin_t       *pe_spins[WORKER_MAX_PE];

    atomic_bool            running;
};

/* DOCA ワーカスレッドプール（N = g_comm_cores 本, 最大 DOCA_WORKER_TYPE_COUNT） */
struct doca_worker_thread_pool_t {
    struct doca_worker_thread_ctx workers[DOCA_WORKER_TYPE_COUNT];
    int n_workers;
    int mpi_rank;
    atomic_bool initialized;
};

/* PE spinlock (複数スレッドからの doca_pe_progress 排他用) */
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

/* メイン集合通信ワーカー構造体 */
struct collective_worker_t {
    /* DOCA デバイス */
    struct doca_dev *dev;           /* ComCh 用 PCIe デバイス */
    struct doca_dev *rdma_dev;      /* RMA 用ネットワークデバイス (Host ポートに対応) */
    struct doca_dev *ring_dev;      /* Ring 用ネットワークデバイス (常に RING_IFACE[0]) */

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

    /* ---- Multi-Rail Ring: 2 番目のポート (RING_IFACE[1]) ---- */
    struct doca_dev *ring_dev_rail1;
    struct doca_rdma_ctx_t rdma_ring_send_rail1;
    struct doca_rdma_ctx_t rdma_ring_recv_rail1;
    struct doca_rdma_connection *conn_to_next_rail1;
    struct doca_rdma_connection *conn_from_prev_rail1;

    /* Host との RDMA 接続記述子 (ComCh 経由で交換) */
    const void *host_rma_conn_desc;
    size_t      host_rma_conn_desc_len;

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

    /* ---- Multi-Rail RMA: 2 番目のポート ---- */
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

    /* PCI import mmap キャッシュ (GPU Direct, 直接マッピング) */
    struct pci_mmap_cache_entry *gpu_dst_pci_cache;  /* dst rail0 (= ring_dev) */
    struct pci_mmap_cache_entry *gpu_src_pci_cache;  /* src rail0 */
    struct pci_mmap_cache_entry *gpu_dst_pci_cache_rail1;  /* rail1 = ring_dev_rail1 */
    struct pci_mmap_cache_entry *gpu_src_pci_cache_rail1;

    /* PE spinlocks */
    struct pe_spin_t rma_pe_spin;
    struct pe_spin_t rma_pe_spin_rail1;
    struct pe_spin_t ring_send_pe_spin;
    struct pe_spin_t ring_send_pe_spin_rail1;
    struct pe_spin_t ring_recv_pe_spin;
    struct pe_spin_t ring_recv_pe_spin_rail1;

    /* RDMA Doorbell (DPU→Host 完了通知) */
    struct doca_remote_mem_t host_doorbell_rmem;   /* Host doorbell の remote mmap */
    bool doorbell_enabled;
    uint64_t doorbell_seq;                         /* monotonic completion counter */

    /* ---- RDMA Doorbell Barrier (RDB)
     * MPI_Barrier の OS jitter (avg 325μs / max 281ms 実測) を RDMA Write による
     * 2-pass Ring barrier (~40μs) に置き換える。
     * working_buf の末尾を slot 領域として使うため追加メモリ登録不要。
     * 隣接 rank の rdb_local_slot のリモートアドレスを MPI で交換。
     * プロトコル詳細は collective.c の rdb_barrier を参照。 */
    bool rdb_enabled;
    uint64_t rdb_my_seq;                         /* バリア sequence (monotonic) */
    volatile uint64_t *rdb_local_slot;           /* 自分の slot (working_buf 内、前 rank が書く) */
    uint64_t *rdb_send_data;                     /* 送信用バッファ (working_buf 内、自分が書く) */
    uint64_t rdb_next_remote_slot_addr;          /* 次 rank の rdb_local_slot のリモートアドレス */
    struct doca_remote_mem_t rdb_next_rmem;      /* 次 rank の working_buf 全体の remote_mem */

    /* ---- RDMA Command Slot (Host→DPU) ---- */
    void *cmd_slot_buf;                    /* DPU-local command slot buffer */
    struct doca_mmap *cmd_slot_mmap;       /* mmap for cmd_slot */
    const void *cmd_slot_export_desc;      /* export descriptor for Host */
    size_t cmd_slot_export_desc_len;
    volatile uint64_t cmd_expected_seq;    /* next expected sequence number */
    bool cmd_slot_enabled;

    /* ---- GPU flag pool (DPU→Host GPU completion sync) ----
     *   ホスト Python の GpuFlagPool が確保した int32 配列の remote_mem。
     *   AG 完了時に flag_value を flag_gpu_addr に inline RDMA Write することで、
     *   GPU の compute stream が cuStreamWaitValue32 で待機している箇所を解放する。
     *   flag pool が GPU memory 上にある場合、host は PCI export (Cross-GVMI) も
     *   送ってくる。DPU 側で doca_mmap_create_from_export すると local mmap が得られ、
     *   GPU memory に PCIe 経由で直接 Write できる。cuStreamWaitValue32 が pinned host
     *   memory を polling する ~1.8ms/AG のストールを消去できる。 */
    bool flag_pool_enabled;
    struct doca_remote_mem_t flag_pool_rmem;        /* rail0 (host_conn 経由, RDMA rkey 由来) */
    struct doca_remote_mem_t flag_pool_rmem_rail1;  /* rail1 (host_conn_rail1 経由, optional) */
    bool flag_pool_rail1_enabled;
    /* Cross-GVMI PCI-imported mmap (local mmap representing host GPU memory) */
    struct doca_mmap *flag_pool_pci_mmap;           /* rail0 (from export_pci) */
    struct doca_mmap *flag_pool_pci_mmap_rail1;     /* rail1 (from export_pci) */
    bool flag_pool_pci_enabled;                     /* true なら PCI path を使う */
    /* per-AG send buffer (4 bytes) のプール: 並行 AG が複数あっても上書きされないよう
     * リング状に使う。num_recv_tasks 256 と整合する。 */
    void   *flag_send_buf_pool;          /* working_buf 内 */
    size_t  flag_send_buf_pool_len;
    atomic_uint flag_send_buf_idx;
    pthread_mutex_t flag_write_mtx;  /* 2 rings が同時に flag write する場合の排他 */

    /* ---- Ring 1 (parallel AG second ring) ----
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

    /* Per-ring job queue */
    struct ring_job *ring_queue_head[N_RINGS];
    struct ring_job *ring_queue_tail[N_RINGS];
    pthread_mutex_t ring_queue_mtx[N_RINGS];
    pthread_cond_t  ring_queue_cv[N_RINGS];
    atomic_bool ring_queue_stop[N_RINGS];
    pthread_t ring_proc_thread[N_RINGS];
    atomic_bool ring_proc_running[N_RINGS];

    /* Per-ring MPI communicator
     *   Ring 0 / Ring 1 thread が同時に MPI_Barrier を呼ぶ場合、MPI_COMM_WORLD を
     *   共有すると barrier 同士がマッチしてデッドロックや異常な待機を起こす。
     *   各 ring に専用の MPI_Comm を dup することで、各 ring 内で正しく barrier が
     *   全 rank 間で同期する。ring_comm_valid[i] = true のとき有効。 */
    MPI_Comm ring_comm[N_RINGS];
    bool ring_comm_valid[N_RINGS];

    /* Deterministic ring assignment counter */
    atomic_uint_fast64_t ag_seq_counter;
};

/* Per-ring job queue entry (lives in the queue, popped by proc thread) */
struct ring_job {
    struct control_cmd *cmd;  /* unpacked cmd (points into buf_copy) */
    void *buf_copy;           /* malloc'd buffer, owned by this job */
    int ring_id;              /* 0 or 1 */
    struct comch_ctrl_path_server_objects *sample_objects;
    struct ring_job *next;
};


#include "uthash.h"

struct msg_thread_pool;

static inline uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }


struct comch_ctrl_path_server_objects {
    struct doca_dev *hw_dev;
    struct doca_dev_rep *rep_dev;
    struct doca_pe *pe;
    struct doca_comch_server *server;
    struct doca_comch_connection *connection;
    struct msg_thread_pool *msg_pool;
    uint32_t num_connected_clients;
    doca_error_t result;
    bool finish;
    const char *dev_pci_addr;
    const char *device_name;
    uint64_t rank;
    uint64_t world_size;
    struct send_notify_entry *send_notify_table;
    struct collective_worker_t collective_worker;
};

struct send_notify_entry {
    uint64_t id;
    atomic_bool finished;
    UT_hash_handle hh;
};

/* バッファプール (pthread_spinlock, 連続確保) — comch_server.c */
void cw_buffer_pool_init(struct cw_buffer_pool *pool, void *base, size_t slot_size);
int  cw_buffer_pool_acquire(struct cw_buffer_pool *pool, size_t size, void **buf_out, size_t *capacity_out);
void cw_buffer_pool_release(struct cw_buffer_pool *pool, int slot_idx);
void cw_buffer_pool_destroy(struct cw_buffer_pool *pool);

struct control_notify;
struct control_cmd;

/* ComCh 通知 (comch_server.c) */
doca_error_t comch_send_control_notify(struct control_notify *send_notify, struct comch_ctrl_path_server_objects *sample_objects);

/* Per-ring job queue 初期化 (comch_server.c、create_ring から呼ぶ) */
int ring_queues_init(struct collective_worker_t *cw);

/* セットアップハンドラ (handlers.c) */
void execute_doca_connect_host_dpu_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects);
void execute_doca_create_ring_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects);
void execute_doca_init_flag_pool_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects);
void execute_doca_collective_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects, int ring_id);

#endif
