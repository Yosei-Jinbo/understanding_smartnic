#!/bin/bash
# OPT-1.3B prefetch + full param転送有効
# bluefield01: run_opt_1.3b_prefetch.sh 0 | tee logs/opt_1.3b_prefetch.log
# bluefield02: run_opt_1.3b_prefetch.sh 1

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

ENABLE_FULL_PARAM_TRANSFER=0 \
torchrun \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --nproc_per_node=2 \
  --master_addr=172.16.0.1 \
  --master_port=29500 \
  /home/y-jinbo/understanding_smartnic/src/baseline/run_zero.py \
    --model opt-1.3b \
    --dataset wikitext-103 \
    --batch-size 2 \
    --warmup-iters 10 \
    --measure-iters 300 \
    --seq-len 1024 \
    --reduce-bucket-size 5e8 \
    --prefetch-bucket-size 5e8 \
    --max-live-parameters 5e8
