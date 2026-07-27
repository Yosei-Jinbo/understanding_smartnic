#!/bin/bash
# ViT-L/16 prefetch + nsys プロファイル (純 GPU 計算時間計測用)
# nsys は LOCAL_RANK=0 の process だけ包む (CUPTI 同居 segfault 回避、smartnic_offload 側と同方式)

NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

# NCCL は「真のデフォルト」で測る。対話シェルに残った export が torchrun に漏れるのを防ぐ。
unset NCCL_ALGO NCCL_PROTO

mkdir -p logs/nsys

export NSYS_OUTPUT_BASE="logs/nsys/vit_l_16_buf_prefetch"
# L2 cache hit rate 等を計測する場合 (GPU metrics collector を有効化)
export NSYS_EXTRA_ARGS="--gpu-metrics-devices=0 --gpu-metrics-frequency=10000 --gpu-metrics-set=ga10x"

ENABLE_FULL_PARAM_TRANSFER=0 \
USE_NVTX_RANGES=0 \
XFER_NVTX=1 \
DISABLE_COMPLETION_POLLER=1 \
DISABLE_ADAM_FORK=1 \
NSYS_PROFILE_MEASURE_ITERS=25 \
NSYS_SYNC_RANGES=1 \
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
  /home/y-jinbo/understanding_smartnic/src/baseline/nsys_wrap_local_rank0.py \
  /home/y-jinbo/understanding_smartnic/src/baseline/run_zero.py \
    --model vit_l_16 \
    --dataset cifar10 \
    --batch-size 64 \
    --warmup-iters 5 \
    --measure-iters 30 \
    --reduce-bucket-size 1e8 \
    --prefetch-bucket-size 1e8 \
    --max-live-parameters 1.5e8 \
  2>&1 | tee "logs/nsys/vit_l_16_buf_prefetch_$(hostname)_node${NODE_RANK}.log"

echo ""
echo "[done] nsys file: ${NSYS_OUTPUT_BASE}_$(hostname)_node${NODE_RANK}.nsys-rep (only if LOCAL_RANK=0 ran on this host)"
