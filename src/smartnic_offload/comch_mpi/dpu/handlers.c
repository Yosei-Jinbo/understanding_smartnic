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
#include "mmap_cache.h"
#include "collective.h"
#include "dpu_config.h"

DOCA_LOG_REGISTER(HANDLERS);

/* * 接続セットアップ関数 */

void execute_doca_connect_host_dpu_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    doca_error_t ret;

    cw->dev = sample_objects->hw_dev;
    cw->mpi_rank = (int)sample_objects->rank;

    /* RDMA 用ネットワークデバイスを開く (インターフェース名で指定、GID テーブルを持つデバイスを確実に取得) */
    ret = open_doca_device_with_iface_name(
        (const uint8_t *)sample_objects->device_name,
        strlen(sample_objects->device_name),
        NULL, &cw->rdma_dev);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open RDMA network device (iface='%s'): %s",
                     sample_objects->device_name, doca_error_get_name(ret));
        return;
    }
    DOCA_LOG_INFO("Opened RDMA network device (RMA): iface=%s", sample_objects->device_name);

    /* Ring 用デバイスを開く (常に RING_IFACE[0] = port 0 を使用) */
    const char *ring_iface = RING_IFACE[0];
    if (strcmp(sample_objects->device_name, ring_iface) == 0) {
        /* RMA と同じポートなら共有 */
        cw->ring_dev = cw->rdma_dev;
        DOCA_LOG_INFO("Ring device shares RMA device: iface=%s", ring_iface);
    } else {
        ret = open_doca_device_with_iface_name(
            (const uint8_t *)ring_iface,
            strlen(ring_iface),
            NULL, &cw->ring_dev);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to open Ring network device (iface='%s'): %s",
                         ring_iface, doca_error_get_name(ret));
            return;
        }
        DOCA_LOG_INFO("Opened Ring network device: iface=%s", ring_iface);
    }

    /* Multi-Rail Ring: 2 番目のポート (RING_IFACE[1]) を開く */
    const char *ring_iface_rail1 = RING_IFACE[1];
    /* rdma_dev or rdma_dev_rail1 で既に開いている場合は共有 */
    if (strcmp(sample_objects->device_name, ring_iface_rail1) == 0) {
        cw->ring_dev_rail1 = cw->rdma_dev;
        DOCA_LOG_INFO("Ring rail1 device shares RMA device: iface=%s", ring_iface_rail1);
    } else {
        /* rdma_dev_rail1 が後で開かれるので、ここでは別途開く */
        ret = open_doca_device_with_iface_name(
            (const uint8_t *)ring_iface_rail1,
            strlen(ring_iface_rail1),
            NULL, &cw->ring_dev_rail1);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Multi-Rail Ring: failed to open ring_dev_rail1 (iface='%s'): %s",
                          ring_iface_rail1, doca_error_get_name(ret));
            cw->ring_dev_rail1 = NULL;
        } else {
            DOCA_LOG_INFO("Opened Ring rail1 network device: iface=%s", ring_iface_rail1);
        }
    }

    /* Host からの DOCA RDMA 接続記述子をコピーして保存 */
    size_t desc_len = recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len;
    void *desc_copy = malloc(desc_len);
    if (!desc_copy) { DOCA_LOG_ERR("Failed to alloc host conn_desc copy"); return; }
    memcpy(desc_copy, recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address, desc_len);
    cw->host_rma_conn_desc = desc_copy;
    cw->host_rma_conn_desc_len = desc_len;

    /* Multi-Rail: Host から rail1 接続記述子が来ているか確認 */
    size_t desc_len_rail1 = recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1;
    if (desc_len_rail1 > 0) {
        void *desc_copy_rail1 = malloc(desc_len_rail1);
        if (desc_copy_rail1) {
            memcpy(desc_copy_rail1, recv_cmd->ucp_connect_host_dpu.remote_ucp_worker_address_rail1, desc_len_rail1);
            cw->host_rma_conn_desc_rail1 = desc_copy_rail1;
            cw->host_rma_conn_desc_len_rail1 = desc_len_rail1;
        }
    }

    /* ---- Working buffer 確保 (create_ring で使うバッファプールの元) ---- */
    /* doca_mmap は 64B アライメントを要求するため aligned_alloc を使用 */
    size_t slot_size = CW_SLOT_DEFAULT_BYTES;
    size_t pool_total = slot_size * CW_BUFFER_POOL_SLOTS * 2;  /* send + recv */
    /* working_buf 末尾に RDB barrier の slot 領域を 4KB 確保。
     * これにより buffer pool と RDB slot が衝突しないことを保証する。 */
    size_t total_size = pool_total + 4096;
    void *working_buf = aligned_alloc(64, total_size);
    if (!working_buf) { DOCA_LOG_ERR("Failed to allocate working buffer"); free(desc_copy); return; }
    memset(working_buf, 0, total_size);
    cw->working_buf = working_buf;
    cw->working_buf_size = total_size;

    /* ---- PCI mmap キャッシュ初期化 (rail0 / rail1) ---- */
    if (!cw->gpu_dst_pci_cache) cw->gpu_dst_pci_cache = pci_mmap_cache_alloc();
    if (!cw->gpu_src_pci_cache) cw->gpu_src_pci_cache = pci_mmap_cache_alloc();
    if (!cw->gpu_dst_pci_cache_rail1) cw->gpu_dst_pci_cache_rail1 = pci_mmap_cache_alloc();
    if (!cw->gpu_src_pci_cache_rail1) cw->gpu_src_pci_cache_rail1 = pci_mmap_cache_alloc();

    /* ---- RMA (Host-DPU Read/Write) RDMA コンテキスト作成 (永続) ---- */
    ret = doca_rdma_ctx_init(&cw->rdma_rma, cw->rdma_dev, NULL,
                              working_buf, total_size,
                              DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                              64, 0,   /* recv_q_size=0: RMA は Read/Write のみ */
                              DPU_GID_INDEX);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma init failed: %s", doca_error_get_name(ret)); return; }

    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_rma, 32, 32, 0, 0,
                                             rdma_read_comp_cb, rdma_read_err_cb,
                                             rdma_write_comp_cb, rdma_write_err_cb,
                                             NULL, NULL, NULL, NULL,
                                             NULL);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma configure failed"); return; }

    /* Export + Connect */
    ret = doca_rdma_ctx_export(&cw->rdma_rma, 0);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma export failed"); return; }

    ret = doca_rdma_ctx_connect(&cw->rdma_rma, 0,
                                 cw->host_rma_conn_desc, cw->host_rma_conn_desc_len);
    if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("rdma_rma connect to Host failed: %s", doca_error_get_name(ret)); return; }
    cw->host_conn = cw->rdma_rma.connections[0];

    /* ---- Multi-Rail: 2 番目のポートの RMA RDMA コンテキスト作成 ---- */
    struct control_notify send_notify;
    send_notify.type = CONTROL_NOTIFY_UCP_CONNECT_HOST_DPU;
    send_notify.ucp_connect_host_dpu.id = recv_cmd->ucp_connect_host_dpu.id;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len = cw->rdma_rma.local_conn_desc_len;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address = (void *)cw->rdma_rma.local_conn_desc;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 = NULL;
    send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 = 0;

    if (desc_len_rail1 > 0) {
        /* 他方のポートを特定して開く */
        const char *other_iface = NULL;
        if (strcmp(sample_objects->device_name, RING_IFACE[0]) == 0)
            other_iface = RING_IFACE[1];
        else
            other_iface = RING_IFACE[0];

        ret = open_doca_device_with_iface_name(
            (const uint8_t *)other_iface, strlen(other_iface),
            NULL, &cw->rdma_dev_rail1);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("Multi-Rail: failed to open rail1 device (iface='%s'): %s, continuing single-rail",
                          other_iface, doca_error_get_name(ret));
        } else {
            DOCA_LOG_INFO("Multi-Rail: opened rail1 device: iface=%s", other_iface);

            /* rail1 RMA RDMA コンテキスト作成 (working_buf を共有) */
            ret = doca_rdma_ctx_init(&cw->rdma_rma_rail1, cw->rdma_dev_rail1, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, DPU_GID_INDEX);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 init failed: %s", doca_error_get_name(ret));
            } else {
                ret = doca_rdma_ctx_configure_and_start(&cw->rdma_rma_rail1, 32, 32, 0, 0,
                                                         rdma_read_comp_cb, rdma_read_err_cb,
                                                         rdma_write_comp_cb, rdma_write_err_cb,
                                                         NULL, NULL, NULL, NULL, NULL);
                if (ret != DOCA_SUCCESS) {
                    DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 configure failed");
                } else {
                    ret = doca_rdma_ctx_export(&cw->rdma_rma_rail1, 0);
                    if (ret != DOCA_SUCCESS) {
                        DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 export failed");
                    } else {
                        ret = doca_rdma_ctx_connect(&cw->rdma_rma_rail1, 0,
                                                     cw->host_rma_conn_desc_rail1,
                                                     cw->host_rma_conn_desc_len_rail1);
                        if (ret != DOCA_SUCCESS) {
                            DOCA_LOG_WARN("Multi-Rail: rdma_rma_rail1 connect failed: %s", doca_error_get_name(ret));
                        } else {
                            cw->host_conn_rail1 = cw->rdma_rma_rail1.connections[0];
                            cw->dual_rail = true;
                            DOCA_LOG_INFO("Multi-Rail: rail1 RMA connection established");

                            /* rail1 の接続記述子を通知に含める */
                            send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_rail1 =
                                (void *)cw->rdma_rma_rail1.local_conn_desc;
                            send_notify.ucp_connect_host_dpu.remote_ucp_worker_address_len_rail1 =
                                cw->rdma_rma_rail1.local_conn_desc_len;
                        }
                    }
                }
            }
        }
    }

    /* ---- Host doorbell remote mmap のインポート ---- */
    cw->doorbell_enabled = false;
    cw->doorbell_seq = 0;
    if (recv_cmd->ucp_connect_host_dpu.doorbell_export_desc_len > 0) {
        doca_error_t dbret = doca_remote_mem_create(&cw->host_doorbell_rmem, cw->rdma_dev,
            recv_cmd->ucp_connect_host_dpu.doorbell_export_desc,
            recv_cmd->ucp_connect_host_dpu.doorbell_export_desc_len);
        if (dbret == DOCA_SUCCESS) {
            cw->doorbell_enabled = true;
            DOCA_LOG_INFO("Doorbell remote mmap imported (addr=0x%lx, len=%zu)",
                          cw->host_doorbell_rmem.remote_addr, cw->host_doorbell_rmem.remote_len);
        } else {
            DOCA_LOG_WARN("Doorbell remote_mem create failed: %s", doca_error_get_name(dbret));
        }
    }

    /* ---- RDMA command slot 作成 (Host RDMA Write 受信用) ---- */
    cw->cmd_slot_enabled = false;
    cw->cmd_slot_buf = aligned_alloc(64, CMD_SLOT_SIZE);
    if (cw->cmd_slot_buf) {
        memset(cw->cmd_slot_buf, 0, CMD_SLOT_SIZE);
        cw->cmd_expected_seq = 1;  /* first expected seq */

        doca_error_t mret = doca_mmap_create(&cw->cmd_slot_mmap);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_add_dev(cw->cmd_slot_mmap, cw->rdma_dev);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_set_memrange(cw->cmd_slot_mmap, cw->cmd_slot_buf, CMD_SLOT_SIZE);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_set_permissions(cw->cmd_slot_mmap,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE);
        if (mret == DOCA_SUCCESS) mret = doca_mmap_start(cw->cmd_slot_mmap);
        if (mret == DOCA_SUCCESS) {
            mret = doca_mmap_export_rdma(cw->cmd_slot_mmap, cw->rdma_dev,
                                          &cw->cmd_slot_export_desc, &cw->cmd_slot_export_desc_len);
        }
        if (mret == DOCA_SUCCESS) {
            cw->cmd_slot_enabled = true;
            DOCA_LOG_INFO("RDMA command slot ready (%zu bytes export)", cw->cmd_slot_export_desc_len);
        } else {
            DOCA_LOG_WARN("RDMA command slot setup failed, using ComCh fallback");
        }
    }

    /* Include cmd_slot export desc in notify */
    if (cw->cmd_slot_enabled) {
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc = (void *)cw->cmd_slot_export_desc;
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc_len = cw->cmd_slot_export_desc_len;
    } else {
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc = NULL;
        send_notify.ucp_connect_host_dpu.cmd_slot_export_desc_len = 0;
    }

    /* 通知: DPU の接続記述子を Host に返送 */
    doca_error_t result = comch_send_control_notify(&send_notify, sample_objects);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send control notify");
    }

    DOCA_LOG_INFO("Host-DPU RDMA connection established (rank %d, dual_rail=%d, doorbell=%d, cmd_slot=%d)",
                  cw->mpi_rank, cw->dual_rail, cw->doorbell_enabled, cw->cmd_slot_enabled);
}

void execute_doca_create_ring_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    doca_error_t ret;
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t rank = sample_objects->rank;
    uint64_t world_size = sample_objects->world_size;

    /* per-ring job queues を初期化 (mutex/cv/stop flag) */
    if (ring_queues_init(cw) != 0) {
        DOCA_LOG_ERR("ring_queues_init failed");
        return;
    }

    /* ---- 1. バッファプール初期化 (working buffer は connect_host_dpu で確保済み) ---- */
    size_t slot_size = CW_SLOT_DEFAULT_BYTES;
    void *working_buf = cw->working_buf;
    size_t total_size = cw->working_buf_size;
    if (!working_buf) { DOCA_LOG_ERR("Working buffer not allocated (connect_host_dpu not called?)"); return; }

    cw_buffer_pool_init(&cw->send_pool, working_buf, slot_size);
    cw_buffer_pool_init(&cw->recv_pool, (char *)working_buf + slot_size * CW_BUFFER_POOL_SLOTS, slot_size);

    /* ---- 2. rdma_rma は connect_host_dpu で作成・接続済み ---- */

    /* ---- 3. Ring (DPU-DPU Send/Recv) RDMA コンテキスト作成 + 接続 ---- */
    for (size_t i = 0; i < world_size; i++) {
        if (rank != i && rank != ((i - 1 + world_size) % world_size))
            goto ring_barrier;

        if (rank == i) {
            /* rank i は recv 側: rdma_ring_recv を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_recv, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, DPU_GID_INDEX);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv init failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* num_recv_tasks=32 → 256 に拡大。
             * pmax=32 (最大 19 piece) + 4 step + dual-rail で 1 rail あたり 40 outstanding
             * となるため、32 を超えてプール枯渇 → DOCA_ERROR_NO_MEMORY を発生させていた。
             * 256 にすれば pmax=64 (32 piece) でも 32×4/2=64 outstanding で十分余裕。 */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv, 0, 0, 0, 256,
                                                     NULL, NULL, NULL, NULL,
                                                     NULL, NULL,
                                                     rdma_recv_comp_cb, rdma_recv_err_cb,
                                                     NULL);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv configure failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_recv, 0);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_recv export failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Multi-Rail Ring: rail1 recv コンテキスト作成 */
            bool ring_rail1_recv_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, DPU_GID_INDEX);
                if (ret == DOCA_SUCCESS) {
                    /* rail1 も同様に num_recv_tasks=256 */
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_rail1, 0, 0, 0, 256,
                                                             NULL, NULL, NULL, NULL,
                                                             NULL, NULL,
                                                             rdma_recv_comp_cb, rdma_recv_err_cb,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_rail1_recv_ok = true;
                        } else { DOCA_LOG_WARN("ring_recv_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("ring_recv_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("ring_recv_rail1 init failed"); }
            }

            /* MPI で接続記述子を交換 */
            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_recv.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_recv.local_conn_desc_len;
            if (ring_rail1_recv_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_recv_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_recv_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("MPI exchange failed"); free(recv_storage); MPI_Barrier(MPI_COMM_WORLD); continue; }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("ring_recv connect failed");
                cw->conn_from_prev = cw->rdma_ring_recv.connections[0];

                /* rail1 接続 */
                if (ring_rail1_recv_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("ring_recv_rail1 connect failed");
                    else cw->conn_from_prev_rail1 = cw->rdma_ring_recv_rail1.connections[0];
                }
            }
            free(recv_storage);
        } else {
            /* rank (i-1) は send 側: rdma_ring_send を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_send, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, DPU_GID_INDEX);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send init failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* num_write_tasks=4 (RDB barrier の RDMA Write 用)
             * num_send_tasks=32 → 256 (recv 側と対称) */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send, 0, 4, 256, 0,
                                                     NULL, NULL,
                                                     rdma_write_comp_cb, rdma_write_err_cb,
                                                     rdma_send_comp_cb, rdma_send_err_cb,
                                                     NULL, NULL,
                                                     NULL);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send configure failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_send, 0);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("ring_send export failed"); MPI_Barrier(MPI_COMM_WORLD); continue; }

            /* Multi-Rail Ring: rail1 send コンテキスト作成 */
            bool ring_rail1_send_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_send_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, DPU_GID_INDEX);
                if (ret == DOCA_SUCCESS) {
                    /* rail1 は RDB を使わないため Write タスクは 0 のまま。num_send_tasks=256 */
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_rail1, 0, 0, 256, 0,
                                                             NULL, NULL, NULL, NULL,
                                                             rdma_send_comp_cb, rdma_send_err_cb,
                                                             NULL, NULL,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_send_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_rail1_send_ok = true;
                        } else { DOCA_LOG_WARN("ring_send_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("ring_send_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("ring_send_rail1 init failed"); }
            }

            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_send.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_send.local_conn_desc_len;
            if (ring_rail1_send_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_send_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_send_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len);
            if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("MPI exchange failed"); free(recv_storage); MPI_Barrier(MPI_COMM_WORLD); continue; }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_send, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("ring_send connect failed");
                cw->conn_to_next = cw->rdma_ring_send.connections[0];

                /* rail1 接続 */
                if (ring_rail1_send_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("ring_send_rail1 connect failed");
                    else cw->conn_to_next_rail1 = cw->rdma_ring_send_rail1.connections[0];
                }
            }
            free(recv_storage);
        }
    ring_barrier:
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* ========================================================================
     * Ring 1 (parallel second ring)
     * Ring 0 と完全に同じパターンで 2 本目の Ring を構築する。
     * 重要:
     *   - MPI exchange tag を 0x501 に変更して Ring 0 と衝突しないようにする
     *   - 各 ctx は _r1 suffix を持つ別フィールド
     *   - conn/PE spinlock も独立
     *   - 失敗した場合は ring_r1_enabled=false で Ring 0 のみにフォールバック
     * ======================================================================== */
    cw->ring_r1_enabled = false;
    cw->ring_r1_rail1_enabled = false;
    pe_spin_init(&cw->ring_send_pe_spin_r1);
    pe_spin_init(&cw->ring_send_pe_spin_r1_rail1);
    pe_spin_init(&cw->ring_recv_pe_spin_r1);
    pe_spin_init(&cw->ring_recv_pe_spin_r1_rail1);

    bool ring_r1_setup_ok = true;
    for (size_t i = 0; i < world_size; i++) {
        if (rank != i && rank != ((i - 1 + world_size) % world_size))
            goto ring_r1_barrier;

        if (rank == i) {
            /* rank i は recv 側 (Ring 1): rdma_ring_recv_r1 を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_r1, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, DPU_GID_INDEX);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 init failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_r1, 0, 0, 0, 256,
                                                     NULL, NULL, NULL, NULL,
                                                     NULL, NULL,
                                                     rdma_recv_comp_cb, rdma_recv_err_cb,
                                                     NULL);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 configure failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_r1, 0);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_recv_r1 export failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Multi-Rail Ring 1: rail1 recv コンテキスト */
            bool ring_r1_rail1_recv_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_recv_r1_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, DPU_GID_INDEX);
                if (ret == DOCA_SUCCESS) {
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_recv_r1_rail1, 0, 0, 0, 256,
                                                             NULL, NULL, NULL, NULL,
                                                             NULL, NULL,
                                                             rdma_recv_comp_cb, rdma_recv_err_cb,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_recv_r1_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_r1_rail1_recv_ok = true;
                        } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 init failed"); }
            }

            /* MPI で接続記述子を交換 (tag 0x501) */
            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_recv_r1.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_recv_r1.local_conn_desc_len;
            if (ring_r1_rail1_recv_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_recv_r1_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_recv_r1_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd_tagged(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len,
                                            0x501);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] MPI exchange failed: %s", doca_error_get_name(ret));
                free(recv_storage); ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_r1, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("[R1] ring_recv_r1 connect failed: %s", doca_error_get_name(ret)); ring_r1_setup_ok = false; }
                else cw->conn_from_prev_r1 = cw->rdma_ring_recv_r1.connections[0];

                if (ring_r1_rail1_recv_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_recv_r1_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("[R1] ring_recv_r1_rail1 connect failed");
                    else cw->conn_from_prev_r1_rail1 = cw->rdma_ring_recv_r1_rail1.connections[0];
                }
            }
            free(recv_storage);
        } else {
            /* rank (i-1) は send 側 (Ring 1): rdma_ring_send_r1 を作成 */
            ret = doca_rdma_ctx_init(&cw->rdma_ring_send_r1, cw->ring_dev, NULL,
                                      working_buf, total_size,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                      64, 0, DPU_GID_INDEX);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 init failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Ring 1 は RDB を使わないので write tasks は 0 でよい。num_send_tasks=256 */
            ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_r1, 0, 0, 256, 0,
                                                     NULL, NULL, NULL, NULL,
                                                     rdma_send_comp_cb, rdma_send_err_cb,
                                                     NULL, NULL,
                                                     NULL);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 configure failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            ret = doca_rdma_ctx_export(&cw->rdma_ring_send_r1, 0);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] ring_send_r1 export failed: %s", doca_error_get_name(ret));
                ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            /* Multi-Rail Ring 1: rail1 send コンテキスト */
            bool ring_r1_rail1_send_ok = false;
            if (cw->ring_dev_rail1) {
                ret = doca_rdma_ctx_init(&cw->rdma_ring_send_r1_rail1, cw->ring_dev_rail1, NULL,
                                          working_buf, total_size,
                                          DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_WRITE,
                                          64, 0, DPU_GID_INDEX);
                if (ret == DOCA_SUCCESS) {
                    ret = doca_rdma_ctx_configure_and_start(&cw->rdma_ring_send_r1_rail1, 0, 0, 256, 0,
                                                             NULL, NULL, NULL, NULL,
                                                             rdma_send_comp_cb, rdma_send_err_cb,
                                                             NULL, NULL,
                                                             NULL);
                    if (ret == DOCA_SUCCESS) {
                        ret = doca_rdma_ctx_export(&cw->rdma_ring_send_r1_rail1, 0);
                        if (ret == DOCA_SUCCESS) {
                            ring_r1_rail1_send_ok = true;
                        } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 export failed"); }
                    } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 configure failed"); }
                } else { DOCA_LOG_WARN("[R1] ring_send_r1_rail1 init failed"); }
            }

            struct control_cmd send_cmd;
            memset(&send_cmd, 0, sizeof(send_cmd));
            send_cmd.type = CONTROL_CMD_UCP_CONNECT_DPU_DPU;
            send_cmd.ucp_connect_dpu_dpu.id = recv_cmd->ucp_create_ring.id;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address = (void *)cw->rdma_ring_send_r1.local_conn_desc;
            send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len = cw->rdma_ring_send_r1.local_conn_desc_len;
            if (ring_r1_rail1_send_ok) {
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1 = (void *)cw->rdma_ring_send_r1_rail1.local_conn_desc;
                send_cmd.ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 = cw->rdma_ring_send_r1_rail1.local_conn_desc_len;
            }

            struct control_cmd *rcmd = NULL;
            void *recv_storage = NULL;
            size_t recv_store_len = 0;
            ret = run_mpi_tag_exchange_cmd_tagged(rank, world_size, i,
                                            (i - 1 + world_size) % world_size,
                                            &send_cmd, &rcmd, &recv_storage, &recv_store_len,
                                            0x501);
            if (ret != DOCA_SUCCESS) {
                DOCA_LOG_ERR("[R1] MPI exchange failed: %s", doca_error_get_name(ret));
                free(recv_storage); ring_r1_setup_ok = false;
                MPI_Barrier(MPI_COMM_WORLD); continue;
            }

            if (rcmd && rcmd->type == CONTROL_CMD_UCP_CONNECT_DPU_DPU) {
                ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_r1, 0,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address,
                                             rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len);
                if (ret != DOCA_SUCCESS) { DOCA_LOG_ERR("[R1] ring_send_r1 connect failed: %s", doca_error_get_name(ret)); ring_r1_setup_ok = false; }
                else cw->conn_to_next_r1 = cw->rdma_ring_send_r1.connections[0];

                if (ring_r1_rail1_send_ok && rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1 > 0) {
                    ret = doca_rdma_ctx_connect(&cw->rdma_ring_send_r1_rail1, 0,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_rail1,
                                                 rcmd->ucp_connect_dpu_dpu.remote_ucp_worker_address_len_rail1);
                    if (ret != DOCA_SUCCESS) DOCA_LOG_WARN("[R1] ring_send_r1_rail1 connect failed");
                    else cw->conn_to_next_r1_rail1 = cw->rdma_ring_send_r1_rail1.connections[0];
                }
            }
            free(recv_storage);
        }
    ring_r1_barrier:
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Ring 1 の接続が両方向成功していれば有効化 */
    if (ring_r1_setup_ok && cw->conn_to_next_r1 && cw->conn_from_prev_r1) {
        cw->ring_r1_enabled = true;
        if (cw->conn_to_next_r1_rail1 && cw->conn_from_prev_r1_rail1) {
            cw->ring_r1_rail1_enabled = true;
        }
        if (rank == 0) {
            printf("[DPU] Ring 1 enabled (rail1=%s)\n",
                   cw->ring_r1_rail1_enabled ? "yes" : "no");
            fflush(stdout);
        }
    } else {
        if (rank == 0) {
            printf("[DPU] Ring 1 setup failed — falling back to Ring 0 only\n");
            fflush(stdout);
        }
    }

    /* ========================================================================
     * Per-ring MPI communicator の作成
     * Ring 0 / Ring 1 それぞれに MPI_Comm_dup で独立した通信子を割り当てる。
     * これにより MPI_Barrier フォールバック時に ring 間で barrier が混線しない。
     *
     * 重要: MPI_Comm_dup は collective 操作なので、全 rank が同じ順序で呼ぶ必要がある。
     * また Ring 1 が無効でも全 rank で Comm_dup を呼ぶ必要がある
     * (一部 rank のみ Comm_dup しないとハングする)。
     * ======================================================================== */
    for (int ri = 0; ri < N_RINGS; ri++) {
        cw->ring_comm[ri] = MPI_COMM_NULL;
        cw->ring_comm_valid[ri] = false;
    }
    for (int ri = 0; ri < N_RINGS; ri++) {
        int dup_rc = MPI_Comm_dup(MPI_COMM_WORLD, &cw->ring_comm[ri]);
        if (dup_rc == MPI_SUCCESS) {
            cw->ring_comm_valid[ri] = true;
        } else {
            DOCA_LOG_WARN("MPI_Comm_dup for Ring %d failed, using MPI_COMM_WORLD fallback", ri);
        }
    }
    if (rank == 0) {
        printf("[DPU] Per-ring MPI_Comm created (ring0=%s ring1=%s)\n",
               cw->ring_comm_valid[0] ? "ok" : "fallback",
               cw->ring_comm_valid[1] ? "ok" : "fallback");
        fflush(stdout);
    }

    /* piece config (env-tunable) を初回 1 度だけ初期化・出力 */
    ag_piece_config_init_once();

    /* ---- RDB barrier 初期化 (Ring 接続確立後) ----
     * MPI_Barrier (avg 325μs / max 281ms) を 2-pass Ring barrier (~40μs) に置換。
     * 失敗時は cw->rdb_enabled=false で MPI_Barrier フォールバック。 */
    cw->rdb_enabled = false;
    cw->rdb_my_seq = 0;
    cw->rdb_local_slot = NULL;
    cw->rdb_send_data = NULL;
    cw->rdb_next_remote_slot_addr = 0;
    {
        doca_error_t rdb_ret = rdb_init(cw, (int)rank, (int)world_size);
        if (rdb_ret != DOCA_SUCCESS) {
            DOCA_LOG_WARN("rdb_init returned %s, falling back to MPI_Barrier",
                          doca_error_get_name(rdb_ret));
        }
    }

    /* ---- 4. ワーカスレッドプール初期化 ---- */
    if (cw->worker_pool == NULL) {
        cw->worker_pool = (struct doca_worker_thread_pool_t *)calloc(1, sizeof(struct doca_worker_thread_pool_t));
        if (cw->worker_pool) {
            int pool_ret = doca_worker_thread_pool_init(cw->worker_pool, cw, (int)rank);
            if (pool_ret != 0) {
                DOCA_LOG_ERR("Failed to init DOCA worker thread pool");
                free(cw->worker_pool);
                cw->worker_pool = NULL;
            } else {
                printf("[Rank%lu] DOCA worker thread pool initialized\n", rank);
            }
        }
    }

    /* ---- 5. RS スレッドプール初期化 ---- */
    if (cw->rs_pool == NULL) {
        ret = rs_thread_pool_create(&cw->rs_pool, g_compute_cores);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create rs_thread_pool");
            cw->rs_pool = NULL;
        } else {
            printf("[Rank%lu] RS aggregation thread pool initialized\n", rank);
        }
    }

    /* ---- 6. 通知送信 ---- */
    struct control_notify send_notify;
    send_notify.type = CONTROL_NOTIFY_UCP_CREATE_RING;
    send_notify.ucp_create_ring.id = recv_cmd->ucp_create_ring.id;
    ret = comch_send_control_notify(&send_notify, sample_objects);
    if (ret != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to send ring notify");
    fflush(stdout);
}

/* =====================================================
 * GPU flag pool init handler
 *   ホストの GpuFlagPool 用 RDMA export を受け取り、DPU 側で remote_mem として保持する。
 *   1 度だけ呼ばれる (起動時の Python init)。
 *   AG 完了時に flag_pool_rmem を target に inline RDMA Write を発行する。
 * ===================================================== */
void execute_doca_init_flag_pool_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects)
{
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t base_addr = recv_cmd->ucp_init_flag_pool.base_addr;
    uint64_t length    = recv_cmd->ucp_init_flag_pool.length;
    void    *rkey_buf  = recv_cmd->ucp_init_flag_pool.rkey_buf;
    uint64_t rkey_len  = recv_cmd->ucp_init_flag_pool.rkey_buf_len;

    if (cw->flag_pool_enabled) {
        DOCA_LOG_WARN("flag pool already enabled, ignoring re-init");
        return;
    }
    if (base_addr == 0 || length == 0 || rkey_len == 0) {
        DOCA_LOG_ERR("invalid INIT_FLAG_POOL: addr=0x%lx len=%lu rkey_len=%lu",
                     base_addr, length, rkey_len);
        return;
    }

    /* rail0: DPU の RMA RDMA dev (= host_conn 経由で host GPU に RDMA Write) で remote_mem を作成 */
    doca_error_t ret = doca_remote_mem_create(&cw->flag_pool_rmem, cw->rdma_dev,
                                              rkey_buf, rkey_len);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG_ERR("flag pool remote_mem create rail0 failed: %s", doca_error_get_descr(ret));
        return;
    }

    /* rail1 (best effort) — host が rail1 export desc を送ってきていれば */
    if (cw->dual_rail && cw->rdma_dev_rail1 &&
        recv_cmd->ucp_init_flag_pool.rkey_buf_rail1 &&
        recv_cmd->ucp_init_flag_pool.rkey_buf_len_rail1 > 0) {
        doca_error_t r1 = doca_remote_mem_create(&cw->flag_pool_rmem_rail1, cw->rdma_dev_rail1,
                                                  recv_cmd->ucp_init_flag_pool.rkey_buf_rail1,
                                                  recv_cmd->ucp_init_flag_pool.rkey_buf_len_rail1);
        cw->flag_pool_rail1_enabled = (r1 == DOCA_SUCCESS);
        if (!cw->flag_pool_rail1_enabled) {
            DOCA_LOG_WARN("flag pool remote_mem create rail1 failed");
        }
    }

    /* ---- Cross-GVMI PCI import (optional) ---- */
    cw->flag_pool_pci_mmap = NULL;
    cw->flag_pool_pci_mmap_rail1 = NULL;
    cw->flag_pool_pci_enabled = false;
    if (recv_cmd->ucp_init_flag_pool.pci_export_buf &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_len > 0) {
        doca_error_t pret = doca_mmap_create_from_export(NULL,
            recv_cmd->ucp_init_flag_pool.pci_export_buf,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_len,
            cw->rdma_dev, &cw->flag_pool_pci_mmap);
        if (pret == DOCA_SUCCESS) {
            cw->flag_pool_pci_enabled = true;
            if (sample_objects->rank == 0) {
                printf("[DPU] flag pool PCI import rail0 SUCCESS (len=%lu)\n",
                       recv_cmd->ucp_init_flag_pool.pci_export_buf_len);
                fflush(stdout);
            }
        } else {
            cw->flag_pool_pci_mmap = NULL;
            DOCA_LOG_WARN("flag pool PCI import rail0 failed: %s (fallback to rdma remote_mem)",
                          doca_error_get_descr(pret));
        }
    }
    if (cw->flag_pool_pci_enabled &&
        cw->rdma_dev_rail1 &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_rail1 &&
        recv_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1 > 0) {
        doca_error_t pret = doca_mmap_create_from_export(NULL,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_rail1,
            recv_cmd->ucp_init_flag_pool.pci_export_buf_len_rail1,
            cw->rdma_dev_rail1, &cw->flag_pool_pci_mmap_rail1);
        if (pret != DOCA_SUCCESS) {
            cw->flag_pool_pci_mmap_rail1 = NULL;
            DOCA_LOG_WARN("flag pool PCI import rail1 failed: %s",
                          doca_error_get_descr(pret));
        }
    }

    /* DPU 側送信バッファ: 各 AG が 4 バイトの flag_value を inline でなく
     * バッファ経由で書き込む際に使用するリング状プール (256 エントリ)。
     * 256 個までの並行 AG が衝突せず使える。
     *
     * 重要: RDMA Write の local source は **DPU の登録済み mmap 内** に
     * なければならない。`aligned_alloc` で確保した plain heap は登録されておらず、
     * NIC が読めないため "Input/Output Operation Failed" となる。
     * → working_buf 末尾の RDB 領域 (192B) より手前 4KB を流用する。
     *   working_buf は rdma_rma の local mmap として登録済みなので NIC が読める。 */
    const size_t pool_entries = 256;
    const size_t flag_pool_bytes = pool_entries * 16;  /* 16 byte/entry (4B data + padding) = 4KB */
    if (cw->working_buf_size < (size_t)RDB_SLOT_REGION_SIZE + flag_pool_bytes) {
        DOCA_LOG_ERR("working_buf too small for flag pool: %zu < %zu",
                     cw->working_buf_size,
                     (size_t)RDB_SLOT_REGION_SIZE + flag_pool_bytes);
        doca_remote_mem_destroy(&cw->flag_pool_rmem);
        return;
    }
    /* Layout (working_buf 末尾):
     *   [working_buf - RDB_SLOT_REGION_SIZE - flag_pool_bytes ... -RDB_SLOT_REGION_SIZE)  ← flag pool
     *   [working_buf - RDB_SLOT_REGION_SIZE ... -1]                                       ← RDB
     */
    size_t flag_pool_offset = cw->working_buf_size - (size_t)RDB_SLOT_REGION_SIZE - flag_pool_bytes;
    cw->flag_send_buf_pool = (char *)cw->working_buf + flag_pool_offset;
    cw->flag_send_buf_pool_len = flag_pool_bytes;
    memset(cw->flag_send_buf_pool, 0, cw->flag_send_buf_pool_len);
    atomic_store(&cw->flag_send_buf_idx, 0);

    cw->flag_pool_enabled = true;
    /* Ring 0 / Ring 1 の両 thread が同時に flag write を発行する可能性が
     * あるため、rdma_rma ctx への書き込みを mutex で直列化する。 */
    pthread_mutex_init(&cw->flag_write_mtx, NULL);

    if (sample_objects->rank == 0) {
        printf("[DPU] flag pool initialized: addr=0x%lx len=%lu rail1=%s pool_entries=%zu pci_path=%s\n",
               base_addr, length,
               cw->flag_pool_rail1_enabled ? "yes" : "no", pool_entries,
               cw->flag_pool_pci_enabled ? "YES (Cross-GVMI)" : "no (legacy RDMA rkey)");
        fflush(stdout);
    }
}

/* AG 完了時に flag_value を flag_gpu_addr に書き込む。
 * RMA RDMA Write を発行 (ホスト GPU 上の int32 スロットを直接更新)。
 *
 * legacy: flag は pinned host memory 上にあり、DPU は rdma_rma ctx 経由で
 *   host memory に RDMA Write する。GPU は cuStreamWaitValue32 で pinned host を poll
 *   する (~1.8ms/AG のストール → fwd に蓄積)。
 *
 * PCI path: flag は GPU memory 上にあり、host 側 mmap は PCI_READ_WRITE permission
 *   付きで export されている (AG dst と同じパターン)。doca_remote_mem_create から
 *   得た flag_pool_rmem に対する RDMA Write が Cross-GVMI で GPU memory に直接着地する。
 *   GPU 側 cuStreamWaitValue32 は GPU-local な semaphore として即座に解放される。
 *
 * 順序保証: 直前の AG データ書き込みは PCIe 書き込みとして既に submit 済み。
 * doca_task_submit は同 ctx 内で submit 順序を保つ。
 * よって compute stream が flag を見た時には AG データが GPU 上に揃っている。
 */
static inline void phase14_write_flag(struct collective_worker_t *cw,
                                       uint64_t flag_gpu_addr, uint32_t flag_value)
{
    if (!cw->flag_pool_enabled || flag_gpu_addr == 0) return;

    /* 2 つの ring proc thread が同時に flag write するケースに備え、
     * rdma_rma ctx への submit を mutex で直列化する。critical section は μs 規模なので
     * throughput への影響は小さい。 */
    pthread_mutex_lock(&cw->flag_write_mtx);

    /* リング状送信バッファプールから次エントリを取得 (16 byte/entry, 256 エントリ) */
    unsigned idx = atomic_fetch_add_explicit(&cw->flag_send_buf_idx, 1, memory_order_relaxed);
    idx = idx & 0xFF;  /* 256 エントリ */
    uint32_t *send_slot = (uint32_t *)((char *)cw->flag_send_buf_pool + (size_t)idx * 16);
    *send_slot = flag_value;

    /* RDMA Write 発行 (rail0 を使用、4 バイト)。
     * flag_pool_rmem は host 側 mmap (PCI_READ_WRITE 付) から export された
     * rdma rkey で作成されている。GPU memory 上の flag でも Cross-GVMI で着地する。
     * AG dst の host_dst_rmem と同じパターン。 */
    struct doca_task_desc write_task;
    init_write_task(&write_task,
                    send_slot,
                    (void *)flag_gpu_addr,
                    sizeof(uint32_t),
                    &cw->flag_pool_rmem,
                    &cw->rdma_rma, cw->host_conn, 0);
    write_task.pe_spin = &cw->rma_pe_spin;
    submit_and_wait_doca(cw->worker_pool, &write_task);

    pthread_mutex_unlock(&cw->flag_write_mtx);
}

/* * collective コマンド実行 (dispatch) */

void execute_doca_collective_cmd(struct control_cmd *recv_cmd, struct comch_ctrl_path_server_objects *sample_objects, int ring_id)
{
    doca_error_t result;
    struct collective_worker_t *cw = &sample_objects->collective_worker;
    uint64_t id = recv_cmd->ucp_collective.id;
    /* ring_id は AG のみに影響。RS は常に Ring 0 を使う (collective_reduce_scatter は
     * ring_id を取らず内部で cw->rdma_ring_send/recv を直接参照する)。 */

    /* ---- Cross-GVMI PCI import with caching (dst + src, rail0 + rail1) ---- */
    struct doca_mmap *gpu_dst_local_mmap = NULL;
    struct doca_mmap *gpu_src_local_mmap = NULL;
    struct doca_mmap *gpu_dst_local_mmap_rail1 = NULL;
    struct doca_mmap *gpu_src_local_mmap_rail1 = NULL;

    /* dst PCI import (rail0 = ring_dev) */
    if (recv_cmd->ucp_collective.dst_pci_export_buf_len > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
        gpu_dst_local_mmap = pci_mmap_cache_find(cw->gpu_dst_pci_cache, addr, len);
        if (!gpu_dst_local_mmap) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.dst_pci_export_buf,
                recv_cmd->ucp_collective.dst_pci_export_buf_len,
                cw->ring_dev, &gpu_dst_local_mmap);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_dst_pci_cache, addr, len, gpu_dst_local_mmap);
            } else { gpu_dst_local_mmap = NULL; }
        }
    }

    /* src PCI import (rail0) */
    if (recv_cmd->ucp_collective.src_pci_export_buf_len > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_src_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_src_buffer_len;
        gpu_src_local_mmap = pci_mmap_cache_find(cw->gpu_src_pci_cache, addr, len);
        if (!gpu_src_local_mmap) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.src_pci_export_buf,
                recv_cmd->ucp_collective.src_pci_export_buf_len,
                cw->ring_dev, &gpu_src_local_mmap);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_src_pci_cache, addr, len, gpu_src_local_mmap);
            } else { gpu_src_local_mmap = NULL; }
        }
    }

    /* dst PCI import (rail1 = ring_dev_rail1) */
    if (cw->ring_dev_rail1 && recv_cmd->ucp_collective.dst_pci_export_buf_len_rail1 > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
        gpu_dst_local_mmap_rail1 = pci_mmap_cache_find(cw->gpu_dst_pci_cache_rail1, addr, len);
        if (!gpu_dst_local_mmap_rail1) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.dst_pci_export_buf_rail1,
                recv_cmd->ucp_collective.dst_pci_export_buf_len_rail1,
                cw->ring_dev_rail1, &gpu_dst_local_mmap_rail1);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_dst_pci_cache_rail1, addr, len, gpu_dst_local_mmap_rail1);
            } else {
                gpu_dst_local_mmap_rail1 = NULL;
                /* rail1 import 失敗の throttled ログ */
                static atomic_int g_dst_r1_fail;
                int n = atomic_fetch_add(&g_dst_r1_fail, 1) + 1;
                if (n <= 5 || n % 1000 == 0) {
                    DOCA_LOG_WARN("dst PCI import rail1 failed #%d: %s (addr=0x%lx len=%zu) — falling back to single-rail for this AG",
                                  n, doca_error_get_descr(pci_ret), addr, len);
                }
            }
        }
    }

    /* src PCI import (rail1) */
    if (cw->ring_dev_rail1 && recv_cmd->ucp_collective.src_pci_export_buf_len_rail1 > 0) {
        uint64_t addr = recv_cmd->ucp_collective.remote_src_buffer_address;
        size_t   len  = recv_cmd->ucp_collective.remote_src_buffer_len;
        gpu_src_local_mmap_rail1 = pci_mmap_cache_find(cw->gpu_src_pci_cache_rail1, addr, len);
        if (!gpu_src_local_mmap_rail1) {
            doca_error_t pci_ret = doca_mmap_create_from_export(NULL,
                recv_cmd->ucp_collective.src_pci_export_buf_rail1,
                recv_cmd->ucp_collective.src_pci_export_buf_len_rail1,
                cw->ring_dev_rail1, &gpu_src_local_mmap_rail1);
            if (pci_ret == DOCA_SUCCESS) {
                pci_mmap_cache_store(cw->gpu_src_pci_cache_rail1, addr, len, gpu_src_local_mmap_rail1);
            } else {
                gpu_src_local_mmap_rail1 = NULL;
                /* rail1 import 失敗の throttled ログ */
                static atomic_int g_src_r1_fail;
                int n = atomic_fetch_add(&g_src_r1_fail, 1) + 1;
                if (n <= 5 || n % 1000 == 0) {
                    DOCA_LOG_WARN("src PCI import rail1 failed #%d: %s (addr=0x%lx len=%zu) — falling back to single-rail for this AG",
                                  n, doca_error_get_descr(pci_ret), addr, len);
                }
            }
        }
    }

    /* GPU Direct AG かどうかを判定 (PCI import が両方成功 かつ AG) */
    bool is_ag = (recv_cmd->ucp_collective.collective_request.collective_op == COLLECTIVE_ALL_GATHER);
    /* CrossGVMI アブレーション: FORCE_STAGING=1 かつ AG なら PCI import 済み mmap を捨て、
     * gpu_direct を強制 off → host_dst_rmem/host_src_rmem 経由の RDMA staging 経路に落とす。
     * （RS は対象外。mmap キャッシュ自体は保持し、この op だけ staging にする。） */
    if (g_force_staging && is_ag) {
        gpu_dst_local_mmap = NULL;
        gpu_src_local_mmap = NULL;
        gpu_dst_local_mmap_rail1 = NULL;
        gpu_src_local_mmap_rail1 = NULL;
    }
    bool gpu_direct_ag = is_ag && gpu_dst_local_mmap && gpu_src_local_mmap;

    /* ---- 1. リモートメモリ作成 (キャッシュ付き) ---- */
    /* GPU Direct AG では RDMA Read/Write を使わないため rmem 作成をスキップ */
    if (!gpu_direct_ag) {
        /* src remote mmap: キャッシュ検索 */
        {
            uint64_t src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
            size_t   src_len  = recv_cmd->ucp_collective.remote_src_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->src_rmem_cache, src_addr, src_len);
            if (cached) {
                cw->host_src_rmem = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_src_rmem, cw->rdma_dev,
                                                recv_cmd->ucp_collective.src_rkey_buf,
                                                recv_cmd->ucp_collective.src_rkey_buf_len);
                if (result != DOCA_SUCCESS) { DOCA_LOG_ERR("src remote_mem create failed: %s", doca_error_get_name(result)); return; }
                rmem_cache_store(cw->src_rmem_cache, cw->host_src_rmem.remote_addr, cw->host_src_rmem.remote_len, &cw->host_src_rmem);
            }
        }

        /* dst remote mmap: キャッシュ検索 */
        {
            uint64_t dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
            size_t   dst_len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->dst_rmem_cache, dst_addr, dst_len);
            if (cached) {
                cw->host_dst_rmem = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_dst_rmem, cw->rdma_dev,
                                                recv_cmd->ucp_collective.dst_rkey_buf,
                                                recv_cmd->ucp_collective.dst_rkey_buf_len);
                if (result != DOCA_SUCCESS) { DOCA_LOG_ERR("dst remote_mem create failed: %s", doca_error_get_name(result)); return; }
                rmem_cache_store(cw->dst_rmem_cache, cw->host_dst_rmem.remote_addr, cw->host_dst_rmem.remote_len, &cw->host_dst_rmem);
            }
        }

        /* ---- Multi-Rail: rail1 用リモートメモリ作成 ---- */
        if (cw->dual_rail && recv_cmd->ucp_collective.src_rkey_buf_len_rail1 > 0) {
            uint64_t src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
            size_t   src_len  = recv_cmd->ucp_collective.remote_src_buffer_len;
            struct rmem_cache_entry *cached = rmem_cache_find(cw->src_rmem_cache_rail1, src_addr, src_len);
            if (cached) {
                cw->host_src_rmem_rail1 = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_src_rmem_rail1, cw->rdma_dev_rail1,
                                                recv_cmd->ucp_collective.src_rkey_buf_rail1,
                                                recv_cmd->ucp_collective.src_rkey_buf_len_rail1);
                if (result != DOCA_SUCCESS) { DOCA_LOG_WARN("rail1 src remote_mem create failed"); }
                else { rmem_cache_store(cw->src_rmem_cache_rail1, cw->host_src_rmem_rail1.remote_addr, cw->host_src_rmem_rail1.remote_len, &cw->host_src_rmem_rail1); }
            }

            uint64_t dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;
            size_t   dst_len  = recv_cmd->ucp_collective.remote_dst_buffer_len;
            cached = rmem_cache_find(cw->dst_rmem_cache_rail1, dst_addr, dst_len);
            if (cached) {
                cw->host_dst_rmem_rail1 = cached->rmem;
            } else {
                result = doca_remote_mem_create(&cw->host_dst_rmem_rail1, cw->rdma_dev_rail1,
                                                recv_cmd->ucp_collective.dst_rkey_buf_rail1,
                                                recv_cmd->ucp_collective.dst_rkey_buf_len_rail1);
                if (result != DOCA_SUCCESS) { DOCA_LOG_WARN("rail1 dst remote_mem create failed"); }
                else { rmem_cache_store(cw->dst_rmem_cache_rail1, cw->host_dst_rmem_rail1.remote_addr, cw->host_dst_rmem_rail1.remote_len, &cw->host_dst_rmem_rail1); }
            }
        }
    }

    /* ---- 2. バッファプールスロット取得 ---- */
    /* GPU Direct AG では DPU 中間バッファ不要 → スキップ */
    size_t working_buffer_len = (size_t)max_u64(recv_cmd->ucp_collective.remote_src_buffer_len,
                                                 recv_cmd->ucp_collective.remote_dst_buffer_len);
    size_t elem_size = sizeof(fp16_t);
    size_t num_elems = (working_buffer_len + elem_size - 1) / elem_size;
    working_buffer_len = elem_size * num_elems;

    void *send_buf = NULL;
    size_t send_cap = 0;
    int send_slot = -1;
    void *recv_buf = NULL;
    size_t recv_cap = 0;
    int recv_slot = -1;

    if (gpu_direct_ag) {
        /* GPU Direct AG: バッファプール不要 (GPU メモリに直接 Send/Recv) */
        recv_buf = NULL;
    } else {
        send_slot = cw_buffer_pool_acquire(&cw->send_pool, working_buffer_len, &send_buf, &send_cap);
        if (send_slot < 0) {
            DOCA_LOG_ERR("Failed to acquire send_pool slot");
            return;
        }

        if (is_ag) {
            recv_buf = send_buf;
        } else {
            recv_slot = cw_buffer_pool_acquire(&cw->recv_pool, working_buffer_len, &recv_buf, &recv_cap);
            if (recv_slot < 0) {
                DOCA_LOG_ERR("Failed to acquire recv_pool slot");
                cw_buffer_pool_release(&cw->send_pool, send_slot);
                return;
            }
        }
    }

    /* ---- 3. 集合通信の実行 ---- */
    uint64_t remote_src_addr = recv_cmd->ucp_collective.remote_src_buffer_address;
    uint64_t remote_dst_addr = recv_cmd->ucp_collective.remote_dst_buffer_address;

    switch (recv_cmd->ucp_collective.collective_request.collective_op) {
    case COLLECTIVE_REDUCE_SCATTER:
        result = collective_reduce_scatter(sample_objects->rank, sample_objects->world_size,
                                           cw, send_buf, recv_buf,
                                           working_buffer_len, remote_src_addr, remote_dst_addr, id);
        break;
    case COLLECTIVE_ALL_GATHER:
        result = collective_all_gather(sample_objects->rank, sample_objects->world_size,
                                       cw, send_buf,
                                       working_buffer_len, remote_src_addr, remote_dst_addr,
                                       gpu_src_local_mmap, id,
                                       gpu_dst_local_mmap,
                                       gpu_src_local_mmap_rail1,
                                       gpu_dst_local_mmap_rail1,
                                       ring_id);
        break;
    default:
        DOCA_LOG_ERR("未実装 collective op: %u", recv_cmd->ucp_collective.collective_request.collective_op);
        result = DOCA_ERROR_INVALID_VALUE;
        break;
    }

    /* AG 完了時に GPU flag を書き込む (doorbell より先)。
     * data → flag → doorbell の順序を厳守:
     *   PCIe writes within an endpoint are ordered, かつ doca_task_submit は同 ctx で
     *   submit 順序を保つので、Ring の最終 Recv → flag write → doorbell write の順で
     *   ホスト GPU/CPU 側に観測される。これにより compute stream は flag を見た時点で
     *   AG データが揃っていることを保証できる。 */
    if (recv_cmd->ucp_collective.flag_gpu_addr != 0 && cw->flag_pool_enabled) {
        phase14_write_flag(cw,
                           recv_cmd->ucp_collective.flag_gpu_addr,
                           recv_cmd->ucp_collective.flag_value);
    }

    /* ---- 4. 通知送信 (RDMA Doorbell or ComCh fallback) ---- */
    if (cw->doorbell_enabled) {
        /* send_buf が NULL (GPU Direct AG) の場合は working_buf の先頭を使う */
        volatile uint64_t *db_staging = (volatile uint64_t *)(send_buf ? send_buf : cw->working_buf);
        cw->doorbell_seq++;
        *db_staging = cw->doorbell_seq;  /* monotonic completion counter */
        struct doca_task_desc db_task;
        init_write_task(&db_task,
                        (void *)db_staging,
                        (void *)cw->host_doorbell_rmem.remote_addr,
                        sizeof(uint64_t),
                        &cw->host_doorbell_rmem,
                        &cw->rdma_rma, cw->host_conn, 0);
        db_task.pe_spin = &cw->rma_pe_spin;
        submit_and_wait_doca(cw->worker_pool, &db_task);
    } else {
        /* Fallback: ComCh notify */
        struct control_notify send_notify;
        send_notify.type = CONTROL_NOTIFY_UCP_COLLECTIVE;
        send_notify.ucp_collective.id = id;
        result = comch_send_control_notify(&send_notify, sample_objects);
        if (result != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to send collective notify");
    }

    /* ---- 5. local_cpu_buffer 処理 ---- */
    if (recv_cmd->ucp_collective.local_cpu_buffer_address != 0) {
        struct doca_remote_mem_t cpu_rmem;
        result = doca_remote_mem_create(&cpu_rmem, cw->rdma_dev,
                                         recv_cmd->ucp_collective.local_cpu_rkey_buf,
                                         recv_cmd->ucp_collective.local_cpu_rkey_buf_len);
        if (result == DOCA_SUCCESS) {
            struct doca_task_desc put_task;
            init_write_task(&put_task, send_buf,
                            (void *)recv_cmd->ucp_collective.local_cpu_buffer_address,
                            recv_cmd->ucp_collective.local_cpu_buffer_len,
                            &cpu_rmem, &cw->rdma_rma, cw->host_conn, 0);
            put_task.pe_spin = &cw->rma_pe_spin;
            submit_and_wait_doca(cw->worker_pool, &put_task);

            /* フラグ書き込み */
            if (recv_cmd->ucp_collective.local_cpu_flag_address != 0) {
                struct doca_remote_mem_t flag_rmem;
                result = doca_remote_mem_create(&flag_rmem, cw->rdma_dev,
                                                 recv_cmd->ucp_collective.local_cpu_flag_rkey_buf,
                                                 recv_cmd->ucp_collective.local_cpu_flag_rkey_buf_len);
                if (result == DOCA_SUCCESS) {
                    const int num_flags = recv_cmd->ucp_collective.local_cpu_flag_len / 4;
                    uint32_t *ones = (uint32_t *)send_buf;  /* 一時的に send_buf の先頭を使う */
                    for (int i = 0; i < num_flags; i++) ones[i] = 1;

                    struct doca_task_desc flag_task;
                    init_write_task(&flag_task, ones,
                                    (void *)recv_cmd->ucp_collective.local_cpu_flag_address,
                                    recv_cmd->ucp_collective.local_cpu_flag_len,
                                    &flag_rmem, &cw->rdma_rma, cw->host_conn, 0);
                    flag_task.pe_spin = &cw->rma_pe_spin;
                    submit_and_wait_doca(cw->worker_pool, &flag_task);
                    doca_remote_mem_destroy(&flag_rmem);
                }
            }
            doca_remote_mem_destroy(&cpu_rmem);
        }
    }

    /* ---- 6. クリーンアップ (キャッシュ済み rmem / PCI mmap は破棄しない) ---- */
    cw_buffer_pool_release(&cw->send_pool, send_slot);
    if (recv_slot >= 0) cw_buffer_pool_release(&cw->recv_pool, recv_slot);
}
