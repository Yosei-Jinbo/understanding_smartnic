#!/bin/bash
# OPT-1.3B prefetch + nsys プロファイル、NCCL 8 チャネル版
#
# 目的: チャネル数を 8 (grid=8, 通常の 2ch の 4 倍の SM 要求) にしたとき、
# SM 干渉 (compute+comm 区間での SMs Active / Tensor Active の低下、
# キュー遅延の増加) が起きるかを GPU メトリクスで検証する。
# 比較対象は既存の opt_buf_prefetch_rank*.sqlite (2ch)。
#
# 使用例:
#   bluefield01: ./run_opt_1.3b_prefetch_nsys_nch8.sh 0
#   bluefield02: ./run_opt_1.3b_prefetch_nsys_nch8.sh 1
#
# 出力: logs/nsys/opt_buf_prefetch_nch8_rank{0..3}.nsys-rep
set -u

LABEL=opt_buf_prefetch_nch8
MODEL=opt-1.3b
DATASET=wikitext-103
BATCH_SIZE=2
SEQ_LEN=1024
RB=5e8
PB=5e8
MLP=5e8

# torchrun は親シェルの環境を子プロセスへ継承する
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=8

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../_prefetch_run.sh"
run_prefetch "$@"
