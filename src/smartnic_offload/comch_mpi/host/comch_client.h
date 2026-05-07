#ifndef COMCH_CLIENT_H_
#define COMCH_CLIENT_H_

#include <stdint.h>
#include "../common/comch_mpi_common.h"

/* ===== 非同期 handle ===== */
/* =========================
 *  非同期 handle（C側で管理）
 *  - Python/pybind を跨ぐので uint64_t で受け渡し
 *  - 実体は内部で malloc した job 構造体のポインタを uintptr_t 経由でキャスト
 * ========================= */
typedef uint64_t comch_req_handle_u64_t;
/* enqueue only（即 return）
 * 戻り値: 0=成功, 負値=errno 互換
 * out_handle: 完了待ち/解放に使用する handle
 */
int ucp_collective_enqueue_u64(uint64_t id,
                               uint64_t src_addr, uint64_t src_size,
                               uint64_t dst_addr, uint64_t dst_size,
                               struct CollectiveRequest collective_request,
                               comch_req_handle_u64_t *out_handle,
                               uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size);

/* Phase 14: enqueue with GPU flag (DPU が AG 完了時に flag_value を flag_gpu_addr に書き込む) */
int ucp_collective_enqueue_u64_with_flag(uint64_t id,
                               uint64_t src_addr, uint64_t src_size,
                               uint64_t dst_addr, uint64_t dst_size,
                               struct CollectiveRequest collective_request,
                               comch_req_handle_u64_t *out_handle,
                               uint64_t local_addr, uint64_t local_size,
                               uint64_t local_flag_addr, uint64_t local_flag_size,
                               uint64_t flag_gpu_addr, uint32_t flag_value);

/* Phase 14: GPU flag pool 登録 (Python から 1 度だけ呼ぶ) */
doca_error_t comch_register_flag_pool(uint64_t base_addr, uint64_t total_len);

/* handle 操作 */
int  comch_req_test_u64(comch_req_handle_u64_t h, uint64_t *out_result_u64); /* done=1/0, 失敗<0 */
int  comch_req_wait_u64(comch_req_handle_u64_t h, uint64_t *out_result_u64); /* 0 or <0 */
void comch_req_release_u64(comch_req_handle_u64_t h);


int comch_client_init(const char *base_server_name, const char *dev_pci_addr);
void ucp_collective_request(uint64_t id, uint64_t src_addr, uint64_t src_size, uint64_t dst_addr, uint64_t dst_size, 
                            struct CollectiveRequest collective_request,
                            uint64_t local_addr, uint64_t local_size, uint64_t local_flag_addr, uint64_t local_flag_size);
void comch_client_shutdown(void);
void ucp_create_ring_request(uint64_t id);
void ucp_connect_host_dpu_request(uint64_t id);
void clean_comch_sample_objects(void);

#endif