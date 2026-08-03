#ifndef COMCH_CLIENT_INTERNAL_H_
#define COMCH_CLIENT_INTERNAL_H_

/* comch_client.c (制御路) と collective_queue.c (データ路) が共有する内部定義 */

/* atomic は doca_rdma_utils.h 経由で C/C++ 両対応 */
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <doca_comch.h>
#include "../common/comch_mpi_common.h"
#include "../common/doca_rdma_utils.h"
#include "uthash.h"

struct comch_ctrl_path_objects {
	struct doca_dev *hw_dev;	  /* Device used in the sample */
	struct doca_pe *pe;		  /* PE object used in the sample */
	struct doca_comch_client *client; /* Client object used in the sample */
	doca_error_t result;		  /* Holds result will be updated in callbacks */
	uint64_t rank;

    struct finish_flags_entry *finish_flags_map;
	/* DOCA RDMA: Host-DPU 接続用コンテキスト */
	struct doca_rdma_ctx_t host_rdma_ctx;

	/* Multi-Rail: 2 番目のポートの RDMA コンテキスト */
	struct doca_dev *hw_dev_rail1;         /* 2 番目の NIC デバイス */
	struct doca_rdma_ctx_t host_rdma_ctx_rail1;
	bool dual_rail;

	/* RDMA Doorbell (DPU→Host 完了通知) */
	volatile uint64_t doorbell __attribute__((aligned(64)));
	struct doca_mmap *doorbell_mmap;
	const void *doorbell_export_desc;
	size_t doorbell_export_desc_len;

	/* host RMA context 用ローカルバッファ (DOCA ctx_init が要求) */
	void *cmd_local_buf;
};

/* id ごとに持つ finish フラグ */
struct finish_flags_entry {
 	uint64_t id;  /* ハッシュキー */

    atomic_bool doca_send_task_finish;
	atomic_bool ucp_connect_host_dpu_finish; //UCPワーカ接続確立したかどうか
    atomic_bool ucp_create_ring_finish;
    atomic_bool ucp_collective_finish;

    UT_hash_handle hh; /* uthash 用ハンドル */
};

/* グローバル ComCh/RDMA 状態 (comch_client.c が定義) */
extern struct comch_ctrl_path_objects global_sample_objects;

/* finish flags (comch_client.c が定義)
 * id に対応するエントリを取得。存在しなければ必要に応じて作成 */
struct finish_flags_entry *get_finish_flags_entry(uint64_t id, bool create_if_missing);

/* ComCh 制御コマンド送信 (comch_client.c が定義) */
doca_error_t comch_send_control_cmd(struct control_cmd *send_cmd, uint64_t id);

#endif /* COMCH_CLIENT_INTERNAL_H_ */
