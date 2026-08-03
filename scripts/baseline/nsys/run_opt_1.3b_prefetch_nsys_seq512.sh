#!/bin/bash
# OPT-1.3B ZeRO-Offload seq512 + nsys プロファイル
# prod 比較 (run_opt_1.3b_prefetch_seq512.sh) と同条件で CPU-GPU 同期待ちの分解を取る。
#
# 使用例:
#   bluefield01: ./run_opt_1.3b_prefetch_nsys_seq512.sh 0
#   bluefield02: ./run_opt_1.3b_prefetch_nsys_seq512.sh 1
set -u

LABEL=opt_buf_prefetch_seq512
MODEL=opt-1.3b
DATASET=wikitext-103
BATCH_SIZE=2
SEQ_LEN=512
RB=5e8
PB=5e8
MLP=5e8

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_prefetch_run.sh"
run_prefetch "$@"
