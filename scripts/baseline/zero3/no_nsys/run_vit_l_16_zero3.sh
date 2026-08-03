#!/bin/bash
# ViT-L/16 純粋 ZeRO-3 (全 GPU 常駐, in-process GPU Adam, --no-offload) prefetch [prod 非計装]
# ZeRO-Offload 版 (run_vit_l_16_prefetch.sh) との差分は --no-offload のみ。
# bluefield01: run_vit_l_16_zero3.sh 0 | tee logs/vit_l_16_zero3.log
# bluefield02: run_vit_l_16_zero3.sh 1

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

# NCCL は「真のデフォルト」で測る。対話シェルに残った export が torchrun に漏れるのを防ぐ。
unset NCCL_ALGO NCCL_PROTO

ENABLE_FULL_PARAM_TRANSFER=0 \
NCCL_SOCKET_IFNAME=enp207s0f0np0,enp207s0f1np1 \
NCCL_IB_DISABLE=0 \
NCCL_NET_GDR_LEVEL=SYS \
NCCL_P2P_LEVEL=SYS \
NCCL_CUMEM_ENABLE=0 \
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
    --warmup-iters 10 \
    --measure-iters 20 \
    --reduce-bucket-size 1e8 \
    --prefetch-bucket-size 1e8 \
    --max-live-parameters 1.5e8 \
    --no-offload
