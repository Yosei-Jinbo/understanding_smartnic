#!/bin/bash
# vit_l_16 を fp16 のまま cifar10 で複数エポック学習し、各エポック後の
# テストセット正答率 (accuracy_history) が上がっていくかを確認するテスト。
#
# 使い方 (各ノードで実行):
#   bluefield01: ./run_vit_accuracy.sh 0 [epochs]
#   bluefield02: ./run_vit_accuracy.sh 1 [epochs]
#   epochs 省略時は 5。
#
# 純粋 ZeRO-3 (--no-offload) で検証する場合は extra args に渡す:
#   ./run_vit_accuracy.sh 0 5 --no-offload
#
# 出力 (rank0 のログ):
#   [Epoch k/N] test_loss=... test_acc=...   ← 各エポック後
#   Accuracy history: [...]                  ← 最後にまとめて
#
# 注意:
#   - iteration ベース (--measure-iters) ではなく --epochs を使う (エポック単位で
#     評価を回すため)。--measure-iters を付けると途中打ち切りになり評価が走らない。
#   - ZeRO-3 では毎 forward で全 param を all-gather するため、cifar10 テスト全 1万枚の
#     評価は 1 エポックあたり数分かかることがある。速く回したいなら test をサブサンプル
#     するか batch-size を上げる。
NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02> [epochs] [extra run_zero.py args...]}
EPOCHS=${2:-5}
EXTRA_ARGS=("${@:3}")   # 例: --no-offload / --lr-decay 0.85

mkdir -p logs

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
    --epochs $EPOCHS \
    --eval-accuracy \
    --reduce-bucket-size 1e8 \
    --prefetch-bucket-size 1e8 \
    --max-live-parameters 1.5e8 \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/vit_accuracy_$(hostname)_node${NODE_RANK}.log"

echo ""
echo "[done] Accuracy history はログ末尾の 'Accuracy history:' 行を参照"
