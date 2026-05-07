#!/bin/bash
# OPT-1.3B prefetch + nsys プロファイル
# nsys は LOCAL_RANK=0 の process だけ包む (CUPTI 同居 segfault 回避、smartnic_offload 側と同方式)
#
# 使用例:
#   bluefield01: ./run_opt_1.3b_prefetch_nsys.sh 0
#   bluefield02: ./run_opt_1.3b_prefetch_nsys.sh 1
#
# 出力:
#   logs/nsys/opt_buf_prefetch_<host>_node<N>.nsys-rep  (LOCAL_RANK=0 のぶんだけ)

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

mkdir -p logs/nsys

export NSYS_OUTPUT_BASE="logs/nsys/opt_buf_prefetch"

ENABLE_FULL_PARAM_TRANSFER=0 \
USE_NVTX_RANGES=0 \
DISABLE_COMPLETION_POLLER=1 \
DISABLE_ADAM_FORK=1 \
NSYS_PROFILE_MEASURE_ITERS=25 \
NSYS_SYNC_RANGES=1 \
NCCL_SOCKET_IFNAME=enp207s0f0np0,enp207s0f1np1 \
NCCL_NET_GDR_LEVEL=SYS \
NCCL_P2P_LEVEL=SYS \
NCCL_CUMEM_ENABLE=0 \
torchrun \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --nproc_per_node=2 \
  --master_addr=172.16.0.1 \
  --master_port=29500 \
  /home/y-jinbo/understanding_smartnic/src/baseline/nsys_wrap_local_rank0.py \
  /home/y-jinbo/understanding_smartnic/src/baseline/run_zero.py \
    --model opt-1.3b \
    --dataset wikitext-103 \
    --batch-size 2 \
    --warmup-iters 5 \
    --measure-iters 30 \
    --seq-len 1024 \
    --reduce-bucket-size 5e8 \
    --prefetch-bucket-size 5e8 \
    --max-live-parameters 5e8 \
  2>&1 | tee "logs/nsys/opt_buf_prefetch_$(hostname)_node${NODE_RANK}.log"

echo ""
echo "[done] nsys file: ${NSYS_OUTPUT_BASE}_$(hostname)_node${NODE_RANK}.nsys-rep (only if LOCAL_RANK=0 ran on this host)"
