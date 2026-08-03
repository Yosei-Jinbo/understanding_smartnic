#ifndef COMCH_MPI_COMMON_H_
#define COMCH_MPI_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_sync_event.h>
#include <doca_error.h>

#include "common.h"

#define DOCA_CHECK(expr)                                     \
    do {                                                     \
        doca_error_t __ret = (expr);                         \
        if (__ret != DOCA_SUCCESS) {                         \
            DOCA_LOG_ERR("Failed to %s", #expr);             \
            return __ret;                                    \
        }                                                    \
    } while (0)

#define CONTROL_CMD_MAX_SIZE  4096
#define CONTROL_NOTIFY_MAX_SIZE 4096

#define rdma_serialize_next_raw(_iter, _type, _offset) \
	({ \
		_type *_result = (_type *)(*(_iter)); \
		*(_iter) = (void *)((uint8_t *)(*(_iter)) + (_offset)); \
		_result; \
	})

//集合通信のEnum
typedef enum {
    COLLECTIVE_SIMPLE_RING    = 0,
    COLLECTIVE_REDUCE_SCATTER = 1,
    COLLECTIVE_ALL_GATHER     = 2,
    COLLECTIVE_ALL_REDUCE     = 3,
    COLLECTIVE_BROADCAST      = 4,
    COLLECTIVE_REDUCE         = 5,
} CollectiveCommunication;

struct CollectiveRequest {
    CollectiveCommunication collective_op;
    uint64_t root; //Reduce, Broadcastの時に利用する
};

//どのようなメッセージが送られるか?
enum control_cmd_type {
    /*----------ホスト-DPUでやり取りされる制御メッセージ----------*/
    CONTROL_CMD_UCP_CONNECT_HOST_DPU, //UCPでホスト-DPU間を接続する, ソース・ディスティネーションともに
    CONTROL_CMD_UCP_CREATE_RING,
    CONTROL_CMD_UCP_COLLECTIVE,

    /*----------DPU-DPUでやり取りされる制御メッセージ----------*/
    CONTROL_CMD_UCP_CONNECT_DPU_DPU, //send->receive, receive->send両方共通で自分のエンドポイント情報を持っておけばOK

    /* Phase 14: GPU flag pool init (ホスト→DPU 一回限り) */
    CONTROL_CMD_UCP_INIT_FLAG_POOL,
};

struct ucp_connect_host_dpu_cmd_msg {
    uint64_t id;
    void *remote_ucp_worker_address;
    uint64_t remote_ucp_worker_address_len;
    /* Multi-Rail: 2 番目のポートの接続記述子 (0 ならシングルレール) */
    void *remote_ucp_worker_address_rail1;
    uint64_t remote_ucp_worker_address_len_rail1;
    /* Phase 5 Step 3: RDMA Doorbell — Host doorbell mmap export desc */
    void *doorbell_export_desc;
    uint64_t doorbell_export_desc_len;
};

//ホスト側から伝えることは特にないよ
struct ucp_create_ring_msg {
    uint64_t id;
};

struct ucp_collective_msg {
    uint64_t id;

    uint64_t remote_src_buffer_address;
    uint64_t remote_src_buffer_len;
    void *src_rkey_buf;
    uint64_t src_rkey_buf_len;

    uint64_t remote_dst_buffer_address;
    uint64_t remote_dst_buffer_len;
    void *dst_rkey_buf;
    uint64_t dst_rkey_buf_len;

    /* ローカル CPU バッファリング用 (使わないときは len=0) */
    uint64_t local_cpu_buffer_address;
    uint64_t local_cpu_buffer_len;
    void *local_cpu_rkey_buf;
    uint64_t local_cpu_rkey_buf_len;

    /* ローカル CPU バッファリング完了通知用フラグ領域 */
    uint64_t local_cpu_flag_address;
    //local_cpu_flag_address_len = 4で必ずこのようになっている そんなことないよ int32_tの領域が複数入るようにする
    uint64_t local_cpu_flag_len;
    void* local_cpu_flag_rkey_buf;
    uint64_t local_cpu_flag_rkey_buf_len;

    struct CollectiveRequest collective_request;

    /* Multi-Rail: 2 番目のポートの export desc (0 ならシングルレール) */
    void *src_rkey_buf_rail1;
    uint64_t src_rkey_buf_len_rail1;
    void *dst_rkey_buf_rail1;
    uint64_t dst_rkey_buf_len_rail1;

    /* Phase 8: Cross-GVMI PCI export of dst (for GPU Direct Ring Send/Recv) */
    void *dst_pci_export_buf;
    uint64_t dst_pci_export_buf_len;

    /* Phase 9: Cross-GVMI PCI export of src (for GPU Direct step 0 Send) */
    void *src_pci_export_buf;
    uint64_t src_pci_export_buf_len;

    /* Phase 12.2: Dual-rail Cross-GVMI PCI export (rail1 = port 1)
     * 0 ならシングルレール GPU Direct (現状互換) */
    void *dst_pci_export_buf_rail1;
    uint64_t dst_pci_export_buf_len_rail1;
    void *src_pci_export_buf_rail1;
    uint64_t src_pci_export_buf_len_rail1;

    /* Phase 14: GPU flag completion sync
     *   flag_gpu_addr  = 0  ならフラグ書き込みなし (legacy パス)
     *   flag_gpu_addr != 0  なら DPU は AG 完了直後にこの GPU アドレスへ
     *                       flag_value (uint32) を RDMA Write で書き込む。
     *   ホスト側 (Python) は cuStreamWaitValue32(EQ, flag_value) を schedule して
     *   compute stream をブロックし、Python thread はブロックしない。
     *   フラグプール本体は CONTROL_CMD_UCP_INIT_FLAG_POOL で先に登録される。 */
    uint64_t flag_gpu_addr;
    uint32_t flag_value;
    uint32_t flag_reserved;  /* keep struct 8-byte aligned */
};

/* Phase 14/15: GPU flag pool init message (ホスト→DPU、一回限り)
 *   flag pool の RDMA export (= ホスト GPU 上の int32 配列の rkey buf) を DPU に送る。
 *   Phase 14: DPU は doca_remote_mem_create で remote_mem として保持し、以降の AG
 *     完了時に flag_gpu_addr (オフセット計算済み) へ inline RDMA Write する。
 *   Phase 15: flag pool が GPU memory 上にある場合、追加で Cross-GVMI PCI export
 *     (doca_mmap_export_pci) を送る。DPU 側で doca_mmap_create_from_export で
 *     import して local_mmap_override 経由で GPU memory に直接 Write できる。 */
struct ucp_init_flag_pool_msg {
    uint64_t base_addr;          /* ホスト GPU 上の flag pool ベースアドレス */
    uint64_t length;             /* flag pool バイト長 */
    void    *rkey_buf;           /* ホスト rdma_dev で export した rkey (export_rdma) */
    uint64_t rkey_buf_len;
    /* rail1 用 (dual-rail でホストが両方の dev で export した場合) */
    void    *rkey_buf_rail1;
    uint64_t rkey_buf_len_rail1;
    /* Phase 15: Cross-GVMI PCI export (optional; 0 なら legacy RDMA rkey のみを使う) */
    void    *pci_export_buf;
    uint64_t pci_export_buf_len;
    void    *pci_export_buf_rail1;
    uint64_t pci_export_buf_len_rail1;
};

//DPU間でUCPワーカ情報を交換する
struct ucp_connect_dpu_dpu_cmd_msg {
    uint64_t id; //id普通に要らないかも, マルチスレッドになったらtagはそもそもスレッドのidで決めるよ
    void *remote_ucp_worker_address;
    uint64_t remote_ucp_worker_address_len;
    /* Multi-Rail: Ring rail1 接続記述子 */
    void *remote_ucp_worker_address_rail1;
    uint64_t remote_ucp_worker_address_len_rail1;
};

//これをメッセージとして送る
struct control_cmd {
    uint64_t type;
    union {
        struct ucp_connect_host_dpu_cmd_msg ucp_connect_host_dpu;
        struct ucp_create_ring_msg ucp_create_ring;
        struct ucp_collective_msg ucp_collective;
        struct ucp_connect_dpu_dpu_cmd_msg ucp_connect_dpu_dpu;
        struct ucp_init_flag_pool_msg ucp_init_flag_pool;
    };
};

/* この control_cmd を on-wire で送るとき必要なバイト数を計算 */
static inline size_t control_cmd_packed_len(const struct control_cmd *cmd)
{
    size_t pack_len;
    pack_len = sizeof(struct control_cmd);
    switch(cmd->type) {
    case CONTROL_CMD_UCP_CONNECT_HOST_DPU:
        pack_len += cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        pack_len += cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        pack_len += cmd->ucp_connect_host_dpu.doorbell_export_desc_len;
        break;
    case CONTROL_CMD_UCP_COLLECTIVE:
        pack_len += cmd->ucp_collective.src_rkey_buf_len;
        pack_len += cmd->ucp_collective.dst_rkey_buf_len;
        pack_len += cmd->ucp_collective.local_cpu_rkey_buf_len;
        pack_len += cmd->ucp_collective.local_cpu_flag_rkey_buf_len;
        pack_len += cmd->ucp_collective.src_rkey_buf_len_rail1;
        pack_len += cmd->ucp_collective.dst_rkey_buf_len_rail1;
        pack_len += cmd->ucp_collective.dst_pci_export_buf_len;
        pack_len += cmd->ucp_collective.src_pci_export_buf_len;
        /* Phase 12.2 dual-rail PCI export */
        pack_len += cmd->ucp_collective.dst_pci_export_buf_len_rail1;
        pack_len += cmd->ucp_collective.src_pci_export_buf_len_rail1;
        break;
    case CONTROL_CMD_UCP_CONNECT_DPU_DPU:
        pack_len += cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len;
        pack_len += cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1;
        break;
    case CONTROL_CMD_UCP_INIT_FLAG_POOL:
        pack_len += cmd->ucp_init_flag_pool.rkey_buf_len;
        pack_len += cmd->ucp_init_flag_pool.rkey_buf_len_rail1;
        /* Phase 15: Cross-GVMI PCI export descriptors */
        pack_len += cmd->ucp_init_flag_pool.pci_export_buf_len;
        pack_len += cmd->ucp_init_flag_pool.pci_export_buf_len_rail1;
        break;
    default: //void *で追加があればここに書き足す
        break;
    }
    return pack_len;
}

//コマンドのパック(送信側が行うよ)
/* cmd を packed_buf（capacity: packed_buf_capacity）にパック
 * 成功すると packed_len_out に実際に詰めたバイト数が入る
 */
static inline doca_error_t  control_cmd_pack(struct control_cmd *cmd, size_t *packed_cmd_len, void *packed_cmd)
{
    void *pack_tail = packed_cmd;
    void *pack_head;
    size_t pack_len;

    pack_len = control_cmd_packed_len(cmd);
    if(pack_len > *packed_cmd_len)
        return DOCA_ERROR_INITIALIZATION;

    //共通のベースコマンドをパックする
    pack_len = sizeof(struct control_cmd);
    pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
    memcpy(pack_head, cmd, pack_len);
    *packed_cmd_len = pack_len;

    switch(cmd->type) {
    case CONTROL_CMD_UCP_CONNECT_HOST_DPU:
        pack_len = cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_connect_host_dpu.remote_ucp_worker_address, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_connect_host_dpu.remote_ucp_worker_address_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }

        pack_len = cmd->ucp_connect_host_dpu.doorbell_export_desc_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_connect_host_dpu.doorbell_export_desc, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    case CONTROL_CMD_UCP_CREATE_RING:
        break; //void *のコマンドは特にないので無視
    case CONTROL_CMD_UCP_COLLECTIVE:
        pack_len = cmd->ucp_collective.src_rkey_buf_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_collective.src_rkey_buf, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_collective.dst_rkey_buf_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_collective.dst_rkey_buf, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_collective.local_cpu_rkey_buf_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_collective.local_cpu_rkey_buf, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_collective.local_cpu_flag_rkey_buf_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_collective.local_cpu_flag_rkey_buf, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_collective.src_rkey_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.src_rkey_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        pack_len = cmd->ucp_collective.dst_rkey_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.dst_rkey_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        /* Phase 8: Cross-GVMI PCI export (dst) */
        pack_len = cmd->ucp_collective.dst_pci_export_buf_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.dst_pci_export_buf, pack_len);
            *packed_cmd_len += pack_len;
        }
        /* Phase 9: Cross-GVMI PCI export (src) */
        pack_len = cmd->ucp_collective.src_pci_export_buf_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.src_pci_export_buf, pack_len);
            *packed_cmd_len += pack_len;
        }
        /* Phase 12.2: Dual-rail PCI export (dst rail1) */
        pack_len = cmd->ucp_collective.dst_pci_export_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.dst_pci_export_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        /* Phase 12.2: Dual-rail PCI export (src rail1) */
        pack_len = cmd->ucp_collective.src_pci_export_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_collective.src_pci_export_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    case CONTROL_CMD_UCP_CONNECT_DPU_DPU:
        pack_len = cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address, pack_len);
        *packed_cmd_len += pack_len;

        pack_len = cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    case CONTROL_CMD_UCP_INIT_FLAG_POOL:
        pack_len = cmd->ucp_init_flag_pool.rkey_buf_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_init_flag_pool.rkey_buf, pack_len);
            *packed_cmd_len += pack_len;
        }
        pack_len = cmd->ucp_init_flag_pool.rkey_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_init_flag_pool.rkey_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        /* Phase 15: PCI export desc */
        pack_len = cmd->ucp_init_flag_pool.pci_export_buf_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_init_flag_pool.pci_export_buf, pack_len);
            *packed_cmd_len += pack_len;
        }
        pack_len = cmd->ucp_init_flag_pool.pci_export_buf_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, cmd->ucp_init_flag_pool.pci_export_buf_rail1, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    }
    return DOCA_SUCCESS;
}

//コマンドのアンパック(受信側で利用する)
static inline doca_error_t control_cmd_unpack(void *packed_cmd, size_t packed_cmd_len, struct control_cmd **cmd)
{
    if(packed_cmd_len < sizeof(struct control_cmd)) {
        printf("Invalid packed command length");
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    void *ptr;
    struct control_cmd *ctrl_cmd;
    uint64_t extended_mem = 0;
    /*
    ホスト側のコマンドパックにより例えばworker_rdmo.cでは
    [ control_cmd (固定部)]
                         └ mr_reg: { va, len, packed_rkey(ポインタに差し替え), packed_rkey_len,
                                     packed_memh(ポインタに差し替え), packed_memh_len }

    [ 可変長データ領域 (固定部の直後に続くバイト列) ]
        ├─ packed_rkey の実体バイト列 (長さ = packed_rkey_len)
        └─ packed_memh の実体バイト列 (長さ = packed_memh_len)
    のように固定長->可変長になるようにコマンドパックしているためこの順序でもよくなっている
    */
    *cmd = (struct control_cmd *)packed_cmd;
    ptr = packed_cmd + sizeof(struct control_cmd);
    ctrl_cmd = (struct control_cmd *)(*cmd);

    switch(ctrl_cmd->type) {
    case CONTROL_CMD_UCP_CONNECT_HOST_DPU:
        ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address = ptr;
        extended_mem += ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        ptr += ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;

        if (ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 > 0) {
            ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
            ptr += ctrl_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        }

        if (ctrl_cmd->ucp_connect_host_dpu.doorbell_export_desc_len > 0) {
            ctrl_cmd->ucp_connect_host_dpu.doorbell_export_desc = ptr;
            extended_mem += ctrl_cmd->ucp_connect_host_dpu.doorbell_export_desc_len;
            ptr += ctrl_cmd->ucp_connect_host_dpu.doorbell_export_desc_len;
        }
        break;
    case CONTROL_CMD_UCP_CREATE_RING:
        break; //void*型なので無視
    case CONTROL_CMD_UCP_COLLECTIVE:
        ctrl_cmd->ucp_collective.src_rkey_buf = ptr;
        extended_mem += ctrl_cmd->ucp_collective.src_rkey_buf_len;
        ptr += ctrl_cmd->ucp_collective.src_rkey_buf_len;

        ctrl_cmd->ucp_collective.dst_rkey_buf = ptr;
        extended_mem += ctrl_cmd->ucp_collective.dst_rkey_buf_len;
        ptr += ctrl_cmd->ucp_collective.dst_rkey_buf_len;
        ctrl_cmd->ucp_collective.local_cpu_rkey_buf = ptr;
        extended_mem += ctrl_cmd->ucp_collective.local_cpu_rkey_buf_len;
        ptr += ctrl_cmd->ucp_collective.local_cpu_rkey_buf_len;

        ctrl_cmd->ucp_collective.local_cpu_flag_rkey_buf = ptr;
        extended_mem += ctrl_cmd->ucp_collective.local_cpu_flag_rkey_buf_len;
        ptr += ctrl_cmd->ucp_collective.local_cpu_flag_rkey_buf_len;

        if (ctrl_cmd->ucp_collective.src_rkey_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_collective.src_rkey_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_collective.src_rkey_buf_len_rail1;
            ptr += ctrl_cmd->ucp_collective.src_rkey_buf_len_rail1;
        }
        if (ctrl_cmd->ucp_collective.dst_rkey_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_collective.dst_rkey_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_collective.dst_rkey_buf_len_rail1;
            ptr += ctrl_cmd->ucp_collective.dst_rkey_buf_len_rail1;
        }
        /* Phase 8: Cross-GVMI PCI export */
        if (ctrl_cmd->ucp_collective.dst_pci_export_buf_len > 0) {
            ctrl_cmd->ucp_collective.dst_pci_export_buf = ptr;
            extended_mem += ctrl_cmd->ucp_collective.dst_pci_export_buf_len;
            ptr += ctrl_cmd->ucp_collective.dst_pci_export_buf_len;
        }
        /* Phase 9: src PCI export */
        if (ctrl_cmd->ucp_collective.src_pci_export_buf_len > 0) {
            ctrl_cmd->ucp_collective.src_pci_export_buf = ptr;
            extended_mem += ctrl_cmd->ucp_collective.src_pci_export_buf_len;
            ptr += ctrl_cmd->ucp_collective.src_pci_export_buf_len;
        }
        /* Phase 12.2: Dual-rail PCI export (dst rail1) */
        if (ctrl_cmd->ucp_collective.dst_pci_export_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_collective.dst_pci_export_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_collective.dst_pci_export_buf_len_rail1;
            ptr += ctrl_cmd->ucp_collective.dst_pci_export_buf_len_rail1;
        }
        /* Phase 12.2: Dual-rail PCI export (src rail1) */
        if (ctrl_cmd->ucp_collective.src_pci_export_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_collective.src_pci_export_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_collective.src_pci_export_buf_len_rail1;
            ptr += ctrl_cmd->ucp_collective.src_pci_export_buf_len_rail1;
        }
        break;
    case CONTROL_CMD_UCP_CONNECT_DPU_DPU:
        ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address = ptr;
        extended_mem += ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len;
        ptr += ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len;

        if (ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
            ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1;
            ptr += ctrl_cmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1;
        }
        break;
    case CONTROL_CMD_UCP_INIT_FLAG_POOL:
        if (ctrl_cmd->ucp_init_flag_pool.rkey_buf_len > 0) {
            ctrl_cmd->ucp_init_flag_pool.rkey_buf = ptr;
            extended_mem += ctrl_cmd->ucp_init_flag_pool.rkey_buf_len;
            ptr += ctrl_cmd->ucp_init_flag_pool.rkey_buf_len;
        }
        if (ctrl_cmd->ucp_init_flag_pool.rkey_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_init_flag_pool.rkey_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_init_flag_pool.rkey_buf_len_rail1;
            ptr += ctrl_cmd->ucp_init_flag_pool.rkey_buf_len_rail1;
        }
        /* Phase 15: PCI export desc */
        if (ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len > 0) {
            ctrl_cmd->ucp_init_flag_pool.pci_export_buf = ptr;
            extended_mem += ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len;
            ptr += ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len;
        }
        if (ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1 > 0) {
            ctrl_cmd->ucp_init_flag_pool.pci_export_buf_rail1 = ptr;
            extended_mem += ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1;
            ptr += ctrl_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1;
        }
        break;
    }

    if (sizeof(struct control_cmd) + extended_mem > packed_cmd_len) {
        printf("Invalid control_cmd length\n");
        return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

//DPU -> ホストに返す用の構造体 (扱いは cmd と同形だが、現在は RDMA で通知を書くため使用箇所は限定)
enum control_notify_type {
    CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU,
    CONTROL_NOTIFY_UCP_CREATE_RING,
    CONTROL_NOTIFY_UCP_COLLECTIVE,
};

//ホストからDPUメモリにアクセスすることがないため、リモートメモリ情報を送る必要はない
struct ucp_connect_host_dpu_notify_msg {
    uint64_t id;
    void *remote_ucp_worker_address;
    uint64_t remote_ucp_worker_address_len;
    /* Multi-Rail: DPU 側 2 番目のポートの接続記述子 */
    void *remote_ucp_worker_address_rail1;
    uint64_t remote_ucp_worker_address_len_rail1;
    /* Phase 5 Step 4: DPU command slot export descriptor */
    void *cmd_slot_export_desc;
    uint64_t cmd_slot_export_desc_len;
};

struct ucp_create_ring_notify_msg {
    uint64_t id;
};

struct ucp_collective_notify_msg {
    uint64_t id;
};

//これをメッセージとして送る
struct control_notify {
    uint64_t type;
    union {
        struct ucp_connect_host_dpu_notify_msg ucp_connect_host_dpu;
        struct ucp_create_ring_notify_msg ucp_create_ring;
        struct ucp_collective_notify_msg ucp_collective;
    };
};

/* この control_notify を on-wire で送るとき必要なバイト数を計算 */
static inline size_t control_notify_packed_len(const struct control_notify *notify)
{
    size_t pack_len;
    pack_len = sizeof(struct control_notify);
    switch(notify->type) {
    case CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU:
        pack_len += notify->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        pack_len += notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        pack_len += notify->ucp_connect_host_dpu.cmd_slot_export_desc_len;
        break;
    default: //void *で追加があればここに書き足す
        break;
    }
    return pack_len;
}

//コマンドのパック(送信側が行うよ)
/* notify を packed_buf（capacity: packed_buf_capacity）にパック
 * 成功すると packed_len_out に実際に詰めたバイト数が入る
 */
static inline doca_error_t  control_notify_pack(struct control_notify *notify, size_t *packed_notify_len, void *packed_notify)
{
    void *pack_tail = packed_notify;
    void *pack_head;
    size_t pack_len;

    pack_len = control_notify_packed_len(notify);
    if(pack_len > *packed_notify_len)
        return DOCA_ERROR_INITIALIZATION;

    //共通のベースコマンドをパックする
    pack_len = sizeof(struct control_notify);
    pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
    memcpy(pack_head, notify, pack_len);
    *packed_notify_len = pack_len;

    switch(notify->type) {
    case CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU:
        pack_len = notify->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, notify->ucp_connect_host_dpu.remote_ucp_worker_address, pack_len);
        *packed_notify_len += pack_len;

        pack_len = notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, notify->ucp_connect_host_dpu.remote_ucp_worker_address_rail1, pack_len);
            *packed_notify_len += pack_len;
        }

        pack_len = notify->ucp_connect_host_dpu.cmd_slot_export_desc_len;
        if (pack_len > 0) {
            pack_head = rdma_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, notify->ucp_connect_host_dpu.cmd_slot_export_desc, pack_len);
            *packed_notify_len += pack_len;
        }
        break;
    default:
        break;
    }
    return DOCA_SUCCESS;
}

//コマンドのアンパック(受信側で利用する)
static inline doca_error_t control_notify_unpack(void *packed_notify, size_t packed_notify_len, struct control_notify **notify)
{
    if(packed_notify_len < sizeof(struct control_notify)) {
        printf("Invalid packed command length");
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    void *ptr;
    struct control_notify *ctrl_notify;
    uint64_t extended_mem = 0;

    *notify = (struct control_notify *)packed_notify;
    ptr = packed_notify + sizeof(struct control_notify);
    ctrl_notify = (struct control_notify *)(*notify);

    switch(ctrl_notify->type) {
    case CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU:
        ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address = ptr;
        extended_mem += ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len;
        ptr += ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len;

        if (ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 > 0) {
            ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = ptr;
            extended_mem += ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
            ptr += ctrl_notify->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
        }

        if (ctrl_notify->ucp_connect_host_dpu.cmd_slot_export_desc_len > 0) {
            ctrl_notify->ucp_connect_host_dpu.cmd_slot_export_desc = ptr;
            extended_mem += ctrl_notify->ucp_connect_host_dpu.cmd_slot_export_desc_len;
            ptr += ctrl_notify->ucp_connect_host_dpu.cmd_slot_export_desc_len;
        }
        break;
    default:
        break;
    }

    if (sizeof(struct control_notify) + extended_mem > packed_notify_len) {
        printf("Invalid control_notify length\n");
        return DOCA_ERROR_INVALID_VALUE;
    }
    return DOCA_SUCCESS;
}

/* MPI を使って control_cmd を双方向に交換する (mpi_exchange.c で実装) */
doca_error_t run_mpi_tag_exchange_cmd(uint64_t rank, uint64_t world_size,
                                      uint64_t src_rank, uint64_t dst_rank,
                                      struct control_cmd *send_cmd,
                                      struct control_cmd **recv_cmd,
                                      void **recv_storage,
                                      size_t *recv_storage_len);

/* Phase 15: 指定タグ版。Ring 1 init で Ring 0 init と異なるタグを使うことで
 * MPI メッセージの衝突を回避する。 */
doca_error_t run_mpi_tag_exchange_cmd_tagged(uint64_t rank, uint64_t world_size,
                                             uint64_t src_rank, uint64_t dst_rank,
                                             struct control_cmd *send_cmd,
                                             struct control_cmd **recv_cmd,
                                             void **recv_storage,
                                             size_t *recv_storage_len,
                                             int mpi_tag);

#endif