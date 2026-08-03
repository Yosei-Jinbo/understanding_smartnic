#!/bin/bash
# OPT-1.3B 純粋 ZeRO-3 (--no-offload) seq512 + nsys プロファイル
# prod 比較 (run_opt_1.3b_zero3_seq512.sh) と同条件で CPU-GPU 同期待ちの分解を取る。
#
# 使用例:
#   bluefield01: ./run_opt_1.3b_zero3_nsys_seq512.sh 0
#   bluefield02: ./run_opt_1.3b_zero3_nsys_seq512.sh 1
set -u

LABEL=opt_zero3_seq512
MODEL=opt-1.3b
DATASET=wikitext-103
BATCH_SIZE=2
SEQ_LEN=512
RB=5e8
PB=5e8
MLP=5e8
EXTRA_ARGS="--no-offload"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_prefetch_run.sh"
run_prefetch "$@"
