#!/bin/bash
# ViT-L/16 prefetch + CPU トレース (osrt/sampling)
#
# 全ランク (4/4) を nsys で包む。起動部は _prefetch_run.sh に集約。
#
# 使用例:
#   bluefield01: ./run_vit_l_16_prefetch_cputrace.sh 0
#   bluefield02: ./run_vit_l_16_prefetch_cputrace.sh 1
#
# 出力: logs/nsys/<LABEL>_rank{0..3}.nsys-rep
set -u

LABEL=vit_l_16_buf_prefetch_cputrace
MODEL=vit_l_16
DATASET=cifar10
BATCH_SIZE=64
SEQ_LEN=
RB=1e8
PB=1e8
MLP=1.5e8
MODE=cputrace

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_prefetch_run.sh"
run_prefetch "$@"
