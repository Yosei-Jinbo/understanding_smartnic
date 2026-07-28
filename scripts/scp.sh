#!/bin/bash
###############################################################################
# scp.sh — understanding_smartnic を各ノードへ rsync 差分同期する配布スクリプト
#
# 背景:
#   旧版は `scp -r smartnic_offload` だけで src/common/ を送らず、bluefield02 の
#   common/utils.py が古いまま（ThroughputMeter 無し）→ ImportError になっていた。
#   → リポジトリ全体を差分同期し、md・log・pycache・build・データセット等は除外。
#
# 同期対象:
#   - bluefield02 (host, ZeRO/Python 実行): understanding_smartnic 全体を丸ごと同期。
#   - dpu01 / dpu02 (DOCA C サーバ): smartnic_offload/comch_mpi/ のみ
#         → /home/ubuntu/doca_practice/comch_mpi/（DPU 側 build/ は残す＝再 build 前提）。
#
# 除外 (md / log / pycache / build / データセット / 生成物):
#   *.md, logs/, build/, __pycache__/, *.pyc, *.so, *.o, .git/, .claude/,
#   .venv/, data/, datasets/, *.arrow, *.bin, *.nsys-rep, *.qdrep, *.sqlite
#
# rsync: -a 保存 / -z 圧縮 / --info=progress2 進捗 / 差分のみ転送。
#   --delete は使わない（remote の成果物・ログ・生成バイナリを保護）。
#
# 使い方:
#   bash /home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host/scp.sh
###############################################################################
set -eu

REPO=/home/y-jinbo/understanding_smartnic
DPU_DST=/home/ubuntu/doca_practice

# 除外パターン（md・log・pycache・build・データセット・生成物）
EXCLUDES=(
  --exclude='*.md'
  --exclude='logs/'
  --exclude='build/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.so'
  --exclude='*.o'
  --exclude='.git/'
  --exclude='.claude/'
  --exclude='.venv/'
  --exclude='data/'
  --exclude='datasets/'
  --exclude='*.arrow'
  --exclude='*.bin'
  --exclude='*.nsys-rep'
  --exclude='*.qdrep'
  --exclude='*.sqlite'
)

RSYNC=(rsync -az --info=progress2 "${EXCLUDES[@]}")

# ---- host: bluefield02（Python/ZeRO） — リポジトリ全体を同期 ----------------
# 末尾スラッシュ: REPO の中身を同名 DST ディレクトリへ差分同期。
"${RSYNC[@]}" "$REPO/" y-jinbo@bluefield02:"$REPO/"

# ---- DPU: dpu01 / dpu02（DOCA C サーバ） — comch_mpi のみ --------------------
for dpu in dpu01 dpu02; do
  "${RSYNC[@]}" "$REPO/src/smartnic_offload/comch_mpi/" ubuntu@"$dpu":"$DPU_DST/comch_mpi/"
done

echo "[scp.sh] done: bluefield02 (repo 全体), dpu01/dpu02 (comch_mpi)"
