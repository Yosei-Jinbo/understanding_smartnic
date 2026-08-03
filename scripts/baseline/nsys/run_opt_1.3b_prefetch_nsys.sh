#!/bin/bash
# OPT-1.3B prefetch + nsys プロファイル
#
# 全ランク (4/4) を nsys で包む。起動部は _prefetch_run.sh に集約。
#
# 使用例:
#   bluefield01: ./run_opt_1.3b_prefetch_nsys.sh 0
#   bluefield02: ./run_opt_1.3b_prefetch_nsys.sh 1
#
# 出力: logs/nsys/<LABEL>_rank{0..3}.nsys-rep
set -u

LABEL=opt_buf_prefetch
MODEL=opt-1.3b
DATASET=wikitext-103
BATCH_SIZE=2
SEQ_LEN=1024
RB=5e8
PB=5e8
MLP=5e8

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_prefetch_run.sh"
run_prefetch "$@"
