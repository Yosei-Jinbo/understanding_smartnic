#!/bin/bash
# OPT-1.3B prefetch + NCCL user-buffer registration アブレーション (R3 リバッタル用)
#
# run_opt_1.3b_prefetch.sh との差分は 2 点のみ:
#   NCCL_CUMEM_ENABLE=1                              (user-buffer 登録は cuMem allocator が前提)
#   TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1  (caching allocator のセグメントを
#                                                     ncclCommRegister で NCCL に登録)
# これで NCCL >= 2.19 の user-buffer 経路 (内部ステージングコピーの除去) が有効になる。
# 登録が実際に効いているかは NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG の短ランで別途確認する。
#
# bluefield01: run_opt_1.3b_prefetch_ubreg.sh 0 | tee logs/opt_1.3b_prefetch_ubreg.log
# bluefield02: run_opt_1.3b_prefetch_ubreg.sh 1

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

# NCCL は「真のデフォルト」で測る。対話シェルに残った export が torchrun に漏れるのを防ぐ。
unset NCCL_ALGO NCCL_PROTO

ENABLE_FULL_PARAM_TRANSFER=0 \
NCCL_SOCKET_IFNAME=enp207s0f0np0,enp207s0f1np1 \
NCCL_IB_DISABLE=0 \
NCCL_NET_GDR_LEVEL=SYS \
NCCL_P2P_LEVEL=SYS \
NCCL_CUMEM_ENABLE=1 \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1 \
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
    --measure-iters 30 \
    --seq-len 1024 \
    --reduce-bucket-size 5e8 \
    --prefetch-bucket-size 5e8 \
    --max-live-parameters 5e8
