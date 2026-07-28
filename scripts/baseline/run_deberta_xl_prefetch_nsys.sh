#!/bin/bash
# DeBERTa-XL prefetch + nsys プロファイル
#
# 全ランク (4/4) を nsys で包む。起動部は _prefetch_run.sh に集約。
#
# 使用例:
#   bluefield01: ./run_deberta_xl_prefetch_nsys.sh 0
#   bluefield02: ./run_deberta_xl_prefetch_nsys.sh 1
#
# 出力: logs/nsys/<LABEL>_rank{0..3}.nsys-rep
set -u

LABEL=deberta_xl_buf_prefetch
MODEL=deberta-xl
DATASET=wikitext-103
BATCH_SIZE=16
SEQ_LEN=128
RB=5e8
PB=3e8
MLP=3e8
MODE=nsys

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_prefetch_run.sh"
run_prefetch "$@"
