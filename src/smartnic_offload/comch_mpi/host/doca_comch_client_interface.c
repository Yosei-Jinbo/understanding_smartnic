#include "comch_client.h"
#include "doca_comch_client_interface.h"  // ★ 追加
#include "../common/comch_mpi_common.h"
#include <mpi.h>
#include <stdio.h>

void doca_comch_client_init_c(const char *server_name, const char *pci_addr)
{
    printf("in doca comch client init c, pci_addr=%s\n", pci_addr);
    comch_client_init(server_name, pci_addr);
}

void ucp_connect_host_dpu_request_c(uint64_t id)
{
    ucp_connect_host_dpu_request(id);
}

void ucp_create_ring_request_c(uint64_t id)
{
    ucp_create_ring_request(id);
}

void ucp_collective_request_c(uint64_t id,
                              uint64_t src_addr, uint64_t src_size,
                              uint64_t dst_addr, uint64_t dst_size,
                              struct CollectiveRequest collective_request,
                              uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size)
{
    ucp_collective_request(id, src_addr, src_size, dst_addr, dst_size, collective_request, local_addr, local_size, local_flag_addr, local_flag_size);
}

int ucp_collective_enqueue_c(uint64_t id,
                             uint64_t src_addr, uint64_t src_size,
                             uint64_t dst_addr, uint64_t dst_size,
                             struct CollectiveRequest collective_request,
                             uint64_t *out_handle,
                             uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size)
{
    return ucp_collective_enqueue_u64(id, src_addr, src_size, dst_addr, dst_size,
                                     collective_request, out_handle,
                                     local_addr, local_size, local_flag_addr, local_flag_size);
}

/* Phase 14: enqueue with GPU flag */
int ucp_collective_enqueue_with_flag_c(uint64_t id,
                             uint64_t src_addr, uint64_t src_size,
                             uint64_t dst_addr, uint64_t dst_size,
                             struct CollectiveRequest collective_request,
                             uint64_t *out_handle,
                             uint64_t local_addr, uint64_t local_size,
                             uint64_t local_flag_addr, uint64_t local_flag_size,
                             uint64_t flag_gpu_addr, uint32_t flag_value)
{
    return ucp_collective_enqueue_u64_with_flag(id, src_addr, src_size, dst_addr, dst_size,
                                     collective_request, out_handle,
                                     local_addr, local_size, local_flag_addr, local_flag_size,
                                     flag_gpu_addr, flag_value);
}

int comch_register_flag_pool_c(uint64_t base_addr, uint64_t total_len)
{
    doca_error_t r = comch_register_flag_pool(base_addr, total_len);
    return (r == DOCA_SUCCESS) ? 0 : (int)-r;
}

int comch_req_test_c(uint64_t handle, uint64_t *out_result)
{
    return comch_req_test_u64(handle, out_result);
}

int comch_req_wait_c(uint64_t handle, uint64_t *out_result)
{
    return comch_req_wait_u64(handle, out_result);
}

void comch_req_release_c(uint64_t handle)
{
    comch_req_release_u64(handle);
}

void doca_mpi_finalize_c(void)
{
    clean_comch_sample_objects();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}