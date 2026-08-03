#ifndef HOST_CONFIG_H_
#define HOST_CONFIG_H_

/*
 * Host 側の実行環境設定。デプロイ先の構成に合わせてここだけ書き換える。
 */

/* RoCE GID インデックス (全 DOCA RDMA コンテキスト共通)。
 * RoCE v2 のエントリを指定し、DPU 側と RoCE バージョンを一致させる。 */
#define HOST_GID_INDEX 3

#endif /* HOST_CONFIG_H_ */
