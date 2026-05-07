#ifndef UCP_UTILS_H
#define UCP_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif


#include "comch_mpi_common.h"
#include<doca_log.h>
#include<doca_error.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/spinlock.h>
#include <ucs/datastruct/mpool.h>

KHASH_MAP_INIT_INT64(ep, ucp_ep_h);

struct ucp_data_t {
    ucp_context_h *ucp_context;
    ucp_worker_h ucp_worker;
    ucp_address_t *worker_address;
    size_t ucp_addrlen;        //UCPワーカのアドレス長
    khash_t(ep) * eps;	       /* Worker endpoints map */

    ucs_list_link_t completed_reqs;
};

struct remote_buffer_info {
    uint64_t buffer_address;
    uint64_t buffer_len;
    ucp_rkey_h rkey;
};

struct local_buffer_info {
    uint64_t buffer_address;
    uint64_t buffer_len;
    ucp_mem_h memh;

    /* ★ 追加: メモリプールスロット番号 (-1: プール外) */
    int      pool_slot;
};

KHASH_MAP_INIT_INT64(remote_ucp_buffer_info, struct remote_buffer_info *);
KHASH_MAP_INIT_INT64(remote_rkey_cache, ucp_rkey_h);

struct remote_ucp_t {
    ucp_address_t *remote_worker_address;
    size_t remote_worker_address_len;

    ucp_ep_h ep_to_remote;

    khash_t(remote_ucp_buffer_info) *remote_ucp_buffer_infos; /* key -> remote_buffer_info* */
    ucs_spinlock_t remote_ucp_lock;

    khash_t(remote_rkey_cache) *rkey_cache;   /* (addr,len) -> ucp_rkey_h */
    ucs_spinlock_t rkey_cache_lock;
};

KHASH_MAP_INIT_INT64(local_ucp_put_get_data_buffer_info, struct local_buffer_info*);

//UCP put, getするための構造体を管理する
struct ucp_put_get_data_t {
    struct ucp_data_t ucp_data; //自分のUCPデータ情報

    khash_t(local_ucp_put_get_data_buffer_info) *local_ucp_put_get_data_buffer_infos;
    ucs_spinlock_t local_put_get_lock;

    //struct remote_ucp_t remote_ucp; //リモートの情報
    struct remote_ucp_t host_src_ucp;
    struct remote_ucp_t host_dst_ucp; //リモートの情報
    //ucp_ep_h ep_to_host;
};

KHASH_MAP_INIT_INT64(local_ucp_host_data_src_buffer_info, struct local_buffer_info*);
KHASH_MAP_INIT_INT64(local_ucp_host_data_dst_buffer_info, struct local_buffer_info*);

KHASH_MAP_INIT_INT64(local_ucp_host_data_local_buffer_info, struct local_buffer_info*);
KHASH_MAP_INIT_INT64(local_ucp_host_data_local_flag_buffer_info, struct local_buffer_info*);

//ホスト用のUCPに関する情報: リモートが勝手に読み出し、書き込みが行われるのでエンドポイント情報と一応メモリマップ情報を持てばいい
struct ucp_host_data_t {
    struct ucp_data_t ucp_data; //自分のUCPデータ情報

    khash_t(local_ucp_host_data_src_buffer_info) *local_ucp_host_data_src_buffer_infos;
    khash_t(local_ucp_host_data_dst_buffer_info) *local_ucp_host_data_dst_buffer_infos;
    ucs_spinlock_t host_src_lock;
    ucs_spinlock_t host_dst_lock;

    khash_t(local_ucp_host_data_local_buffer_info) *local_ucp_host_data_local_buffer_infos;
    ucs_spinlock_t host_local_lock;

    khash_t(local_ucp_host_data_local_flag_buffer_info) *local_ucp_host_data_local_flag_buffer_infos;
    ucs_spinlock_t host_local_flag_lock;

    struct remote_ucp_t remote_ucp;
};

KHASH_MAP_INIT_INT64(local_ucp_receive_data_buffer_info,     struct local_buffer_info*);
//receive用のUCPに関する情報: リモートからもらったものをreceiveするだけなので相手側のアドレスを知る必要なし
struct ucp_receive_data_t {
    struct ucp_data_t ucp_data; //自分のUCPデータ情報

    khash_t(local_ucp_receive_data_buffer_info) *local_ucp_receive_data_buffer_infos;
    ucs_spinlock_t local_receive_lock;

    struct remote_ucp_t prev_rank_send_ucp; /* このバッファアドレス情報は受信パスでは利用しない */
};

KHASH_MAP_INIT_INT64(local_ucp_send_data_buffer_info,        struct local_buffer_info*);
//send用のUCPに関する情報: sendおよびホスト側のsrc, dstから情報を受け取る必要がある
struct ucp_send_data_t {
    struct ucp_data_t ucp_data;

    khash_t(local_ucp_send_data_buffer_info) *local_ucp_send_data_buffer_infos; 
    ucs_spinlock_t local_send_lock;

    //struct remote_ucp_t host_src_ucp;
    //struct remote_ucp_t host_dst_ucp;

    struct remote_ucp_t next_rank_receive_ucp; //!!!!!! ここにあるバッファアドレス情報は使わないで
};

doca_error_t run_mpi_tag_exchange_cmd(uint64_t rank, uint64_t world_size,
                                     uint64_t src_rank, uint64_t dst_rank,
                                     struct control_cmd *send_cmd,
                                     struct control_cmd **recv_cmd,
                                     void **recv_storage,
                                     size_t *recv_storage_len);

doca_error_t create_ucp_worker(struct ucp_data_t *ucp_data);
doca_error_t register_ucp_worker_addresses(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size);
ucs_status_t ucx_wait(ucp_worker_h worker, void *request, const char *op_name);

//コマンドを送受信するための関数, これによりRDMAの記述子を交換するよ
doca_error_t run_ucp_tag_exchange_cmd(struct ucp_data_t *ucp_data, uint64_t rank, uint64_t world_size, uint64_t src_rank, uint64_t dst_rank,
                         struct control_cmd *send_cmd, struct control_cmd **recv_cmd, 
                         /*malloc した受信バッファ*/ void **recv_storage, /*そのサイズ (≒CONTROL_CMD_MAX_SIZE)*/ size_t *recv_storage_len);

doca_error_t remote_ucp_get_or_create_rkey(struct remote_ucp_t *remote,
                              ucp_ep_h ep,
                              uint64_t addr,
                              uint64_t len,
                              const void *rkey_buf,
                              size_t rkey_buf_len,
                              ucp_rkey_h *rkey_out);

/*--------------------キーハッシュ周りのユーティリティ関数のプロトタイプ宣言--------------------*/
/* ==== 初期化 / 破棄ヘルパ ==== */
/* remote_ucp_t: remote_ucp_buffer_infos マップ */
void remote_ucp_init(struct remote_ucp_t *remote);
void remote_ucp_destroy(struct remote_ucp_t *remote);

/* ucp_put_get_data_t: ローカルマップ＋remote_ucp */
void ucp_put_get_data_maps_init(struct ucp_put_get_data_t *d);
void ucp_put_get_data_maps_destroy(struct ucp_put_get_data_t *d);

/* ucp_host_data_t */
void ucp_host_data_maps_init(struct ucp_host_data_t *d);
void ucp_host_data_maps_destroy(struct ucp_host_data_t *d);

/* ucp_receive_data_t */
void ucp_receive_data_maps_init(struct ucp_receive_data_t *d);
void ucp_receive_data_maps_destroy(struct ucp_receive_data_t *d);

/* ucp_send_data_t */
void ucp_send_data_maps_init(struct ucp_send_data_t *d);
void ucp_send_data_maps_destroy(struct ucp_send_data_t *d);

/* ==== スピンロック付き put/get/del ==== */
/* remote_ucp_t.remote_ucp_buffer_infos (key: buffer_id 等, value: remote_buffer_info*) */
doca_error_t remote_ucp_buffer_info_put(struct remote_ucp_t *remote,
                                        uint64_t key,
                                        const struct remote_buffer_info *info_in);
doca_error_t remote_ucp_buffer_info_get(struct remote_ucp_t *remote,
                                        uint64_t key,
                                        struct remote_buffer_info **info_out);
doca_error_t remote_ucp_buffer_info_del(struct remote_ucp_t *remote,
                                        uint64_t key);

/* ucp_put_get_data_t.local_ucp_put_get_data_buffer_infos */
doca_error_t ucp_put_get_local_buffer_put(struct ucp_put_get_data_t *d,
                                          uint64_t key,
                                          const struct local_buffer_info *info_in);
doca_error_t ucp_put_get_local_buffer_get(struct ucp_put_get_data_t *d,
                                          uint64_t key,
                                          struct local_buffer_info **info_out);
doca_error_t ucp_put_get_local_buffer_del(struct ucp_put_get_data_t *d,
                                          uint64_t key);

/* ucp_host_data_t: src / dst */
doca_error_t ucp_host_src_buffer_put(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     const struct local_buffer_info *info_in);
doca_error_t ucp_host_src_buffer_get(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     struct local_buffer_info **info_out);
doca_error_t ucp_host_src_buffer_del(struct ucp_host_data_t *d,
                                     uint64_t key);

doca_error_t ucp_host_dst_buffer_put(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     const struct local_buffer_info *info_in);
doca_error_t ucp_host_dst_buffer_get(struct ucp_host_data_t *d,
                                     uint64_t key,
                                     struct local_buffer_info **info_out);
doca_error_t ucp_host_dst_buffer_del(struct ucp_host_data_t *d,
                                     uint64_t key);

doca_error_t ucp_host_local_buffer_put(struct ucp_host_data_t *d, uint64_t key, const struct local_buffer_info *info_in);
doca_error_t ucp_host_local_buffer_get(struct ucp_host_data_t *d, uint64_t key, struct local_buffer_info **info_out);
doca_error_t ucp_host_local_buffer_del(struct ucp_host_data_t *d, uint64_t key);

doca_error_t ucp_host_local_flag_buffer_put(struct ucp_host_data_t *d, uint64_t key, const struct local_buffer_info *info_in);
doca_error_t ucp_host_local_flag_buffer_get(struct ucp_host_data_t *d, uint64_t key, struct local_buffer_info **info_out);
doca_error_t ucp_host_local_flag_buffer_del(struct ucp_host_data_t *d, uint64_t key);

/* ucp_receive_data_t */
doca_error_t ucp_receive_local_buffer_put(struct ucp_receive_data_t *d,
                                          uint64_t key,
                                          const struct local_buffer_info *info_in);
doca_error_t ucp_receive_local_buffer_get(struct ucp_receive_data_t *d,
                                          uint64_t key,
                                          struct local_buffer_info **info_out);
doca_error_t ucp_receive_local_buffer_del(struct ucp_receive_data_t *d,
                                          uint64_t key);

/* ucp_send_data_t */
doca_error_t ucp_send_local_buffer_put(struct ucp_send_data_t *d,
                                       uint64_t key,
                                       const struct local_buffer_info *info_in);
doca_error_t ucp_send_local_buffer_get(struct ucp_send_data_t *d,
                                       uint64_t key,
                                       struct local_buffer_info **info_out);
doca_error_t ucp_send_local_buffer_del(struct ucp_send_data_t *d,
                                       uint64_t key);
/*--------------------キーハッシュ周りのユーティリティ関数ここまで--------------------*/

#ifdef __cplusplus
}
#endif

#endif