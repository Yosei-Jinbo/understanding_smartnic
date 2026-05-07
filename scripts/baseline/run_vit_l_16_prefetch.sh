#!/bin/bash
# ViT-L/16 prefetch + full param転送有効
# bluefield01: run_vit_l_16_prefetch.sh 0 | tee logs/vit_l_16_prefetch.log
# bluefield02: run_vit_l_16_prefetch.sh 1

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

ENABLE_FULL_PARAM_TRANSFER=0 \
torchrun \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --nproc_per_node=2 \
  --master_addr=172.16.0.1 \
  --master_port=29500 \
  /home/y-jinbo/understanding_smartnic/src/baseline/run_zero.py \
    --model vit_l_16 \
    --dataset cifar10 \
    --batch-size 64 \
    --warmup-iters 25 \
    --measure-iters 1000 \
    --reduce-bucket-size 1e8 \
    --prefetch-bucket-size 1e8 \
    --max-live-parameters 1.5e8
