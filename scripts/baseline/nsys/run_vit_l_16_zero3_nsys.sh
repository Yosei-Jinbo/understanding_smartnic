#!/bin/bash
# ViT-L/16 純粋 ZeRO-3 (--no-offload) prefetch + nsys プロファイル
#
# ZeRO-Offload 版 (run_vit_l_16_prefetch_nsys.sh) との差分は --no-offload のみ:
#   分割 fp16 / fp32 マスタ / Adam 状態を全て GPU 常駐にし、
#   Adam は in-process の torch.optim.Adam で実行する (CPU Adam worker なし)。
# バケット設定等は Offload 版と同一にして比較可能にする。
#
# 使用例:
#   bluefield01: ./run_vit_l_16_zero3_nsys.sh 0
#   bluefield02: ./run_vit_l_16_zero3_nsys.sh 1
#
# 出力: logs/nsys/<LABEL>_rank{0..3}.nsys-rep
set -u

LABEL=vit_l_16_zero3_prefetch
MODEL=vit_l_16
DATASET=cifar10
BATCH_SIZE=64
SEQ_LEN=
RB=1e8
PB=1e8
MLP=1.5e8
EXTRA_ARGS="--no-offload"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_prefetch_run.sh"
run_prefetch "$@"
