#ifndef DPU_CONFIG_H_
#define DPU_CONFIG_H_

/*
 * DPU 側の実行環境設定。デプロイ先の構成に合わせてここだけ書き換える。
 */

/* Ring (DPU-DPU) 用ネットワークインターフェース名 (rail0 / rail1) */
static const char *const RING_IFACE[2] = { "enp3s0f0s0", "enp3s0f1s0" };

/* RoCE GID インデックス (全 DOCA RDMA コンテキスト共通)。
 * RoCE v2 のエントリを指定し、Host 側と RoCE バージョンを一致させる。 */
#define DPU_GID_INDEX 1

/* --- CPU コア割り当て (BF-3 Arm 16 コア、1 DPU = 2 ランク同居が前提) ---
 *   ワークロードの性質で配置方針を変える:
 *     - 通信 progress : busy-poll のため rank 分離（3 worker を C コアに packing）
 *     - 計算 RS 集約  : バースト的・メモリ律速のため rank 共有（動的バースト＋MLP）
 *   env で実行時変更可（再ビルド不要でスイープ）:
 *     COMM_CORES    : 通信 progress の rank あたりコア数 C (1..DOCA_WORKER_TYPE_COUNT, default 2)
 *     COMPUTE_CORES : 計算 RS 集約スレッド数 K（rank 共有, default 8）
 *   制約: 2*C + K + 4 <= 16（超過分は K を自動クランプ）
 *   レイアウト（自動計算）:
 *     comm[0..2C-1](rank別) | msg[2C,2C+1](rank別) | compute[2C+2 .. +K-1](共有) | main(rank別)
 *   既定 C=2, K=8 のとき: comm 0-3 | msg 4-5 | compute 6-13 | main 14-15
 *   (multicore/pipeline/full 構成の実測値。ablation sweep と運用手順が使う値に一致させている) */
#define COMM_CORES_DEFAULT     2
#define COMPUTE_CORES_DEFAULT  8

#endif /* DPU_CONFIG_H_ */
