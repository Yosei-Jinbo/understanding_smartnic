//pybindで利用するためのインターフェースを作成する
#ifndef DOCA_COMCH_CLIENT_INTERFACE_H_
#define DOCA_COMCH_CLIENT_INTERFACE_H_

#include <stdint.h>
#include "../common/comch_mpi_common.h"
#ifdef __cplusplus
extern "C" {
#endif

void ucp_connect_host_dpu_request_c(uint64_t id);
void ucp_create_ring_request_c(uint64_t id);
void ucp_collective_request_c(uint64_t id,
                              uint64_t src_addr, uint64_t src_size,
                              uint64_t dst_addr, uint64_t dst_size,
                              struct CollectiveRequest collective_request,
                              uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size);


void doca_comch_client_init_c(const char *server_name, const char *pci_addr);
void doca_mpi_finalize_c(void);

/* 追加：完全非同期 API（handle は uint64_t） */
int ucp_collective_enqueue_c(uint64_t id,
                             uint64_t src_addr, uint64_t src_size,
                             uint64_t dst_addr, uint64_t dst_size,
                             struct CollectiveRequest collective_request,
                             uint64_t *out_handle,
                             uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size);

/* Phase 14: enqueue with GPU flag */
int ucp_collective_enqueue_with_flag_c(uint64_t id,
                             uint64_t src_addr, uint64_t src_size,
                             uint64_t dst_addr, uint64_t dst_size,
                             struct CollectiveRequest collective_request,
                             uint64_t *out_handle,
                             uint64_t local_addr, uint64_t local_size,
                             uint64_t local_flag_addr, uint64_t local_flag_size,
                             uint64_t flag_gpu_addr, uint32_t flag_value);

/* Phase 14: register the GPU flag pool (called once after CUDA tensor allocated) */
int comch_register_flag_pool_c(uint64_t base_addr, uint64_t total_len);

int comch_req_test_c(uint64_t handle, uint64_t *out_result);
int comch_req_wait_c(uint64_t handle, uint64_t *out_result);
void comch_req_release_c(uint64_t handle);

#ifdef __cplusplus
}
#endif

#endif