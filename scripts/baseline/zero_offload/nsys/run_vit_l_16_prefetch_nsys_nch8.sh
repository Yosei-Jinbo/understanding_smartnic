#!/bin/bash
# ViT-L/16 prefetch + nsys プロファイル、NCCL 8 チャネル版
#
# 目的: ViT は 2ch でも唯一 Tensor 実干渉が出るモデル (compute+comm で 60.6→45.2%)。
# 8ch (grid=8) にしたとき干渉がどこまで悪化するかを GPU メトリクスで直接観測する。
# end-to-end では nch8 が ViT を +20.6% 悪化させており、その機構的裏付けを取る。
# 比較対象は既存の vit_l_16_buf_prefetch_rank*.sqlite (2ch)。
#
# 使用例:
#   bluefield01: ./run_vit_l_16_prefetch_nsys_nch8.sh 0
#   bluefield02: ./run_vit_l_16_prefetch_nsys_nch8.sh 1
#
# 出力: logs/nsys/vit_l_16_buf_prefetch_nch8_rank{0..3}.nsys-rep
set -u

LABEL=vit_l_16_buf_prefetch_nch8
MODEL=vit_l_16
DATASET=cifar10
BATCH_SIZE=64
SEQ_LEN=
RB=1e8
PB=1e8
MLP=1.5e8

# torchrun は親シェルの環境を子プロセスへ継承する
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=8

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../_prefetch_run.sh"
run_prefetch "$@"
