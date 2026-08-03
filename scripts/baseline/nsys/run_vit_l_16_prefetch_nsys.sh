#!/bin/bash
# ViT-L/16 prefetch + nsys プロファイル
#
# 全ランク (4/4) を nsys で包む。起動部は _prefetch_run.sh に集約。
#
# 使用例:
#   bluefield01: ./run_vit_l_16_prefetch_nsys.sh 0
#   bluefield02: ./run_vit_l_16_prefetch_nsys.sh 1
#
# 出力: logs/nsys/<LABEL>_rank{0..3}.nsys-rep
set -u

LABEL=vit_l_16_buf_prefetch
MODEL=vit_l_16
DATASET=cifar10
BATCH_SIZE=64
SEQ_LEN=
RB=1e8
PB=1e8
MLP=1.5e8

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_prefetch_run.sh"
run_prefetch "$@"
