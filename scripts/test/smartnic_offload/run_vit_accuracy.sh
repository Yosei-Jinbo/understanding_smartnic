#!/bin/bash

# vit_l_16 を fp16 のまま cifar10 で複数エポック学習し、各エポック後の
# テストセット正答率 (accuracy_history) が上がっていくかを確認するテスト
# (SmartNIC オフロード版)。
#
# 事前準備: 両 DPU 側で先にサーバを起動しておくこと
# (host 側は接続待ちでブロックする):
#
#   dpu$ cd comch_mpi/dpu && mpirun --app dpu_appfile
#
# 使い方
# (どちらかのホストで1回だけ実行。mpirunが4 rankすべてを起動する):
#
#   ./run_vit_accuracy.sh [epochs] [method] \
#       [extra run_zero_mpi.py args...]
#
#   epochs:
#       省略時 5
#
#   method:
#       smartnic
#           RS + AGをDOCAで実行（デフォルト）
#
#       ag_smartnic
#           AGをDOCA、RSをNCCLで実行
#
# 出力（rank 0のログ）:
#
#   [Epoch k/N] test_loss=... test_acc=...
#   Accuracy history: [...]
#
# 注意:
#
#   - ベースappfileの--warmup-itersと--measure-itersは削除する。
#     iterationベースでは途中で打ち切られ、評価が走らないため。
#
#   - 代わりに--epochs Nと--eval-accuracyを追加する。
#
#   - ZeRO-3ではevalのforwardでも全parameterをall-gatherする。
#     CIFAR-10テストセット全1万枚の評価には、
#     1エポック当たり数分かかる場合がある。


EPOCHS=${1:-5}
METHOD=${2:-smartnic}
EXTRA_ARGS="${*:3}"


# ============================================================
# 使用するパス
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

APPFILE_DIR=/home/y-jinbo/understanding_smartnic/scripts/smartnic_offload/no_nsys

BASE_APPFILE="$APPFILE_DIR/host_appfile_vit_l_16_${METHOD}"

# 実際に実行するrun_zero_mpi.pyの絶対パス
RUN_ZERO_MPI=/home/y-jinbo/understanding_smartnic/src/smartnic_offload/run_zero_mpi.py


# ============================================================
# 入力とファイルの確認
# ============================================================

case "$METHOD" in
    smartnic|ag_smartnic)
        ;;
    *)
        echo "不正なmethodです: $METHOD" >&2
        echo "smartnic | ag_smartnic を指定してください" >&2
        exit 1
        ;;
esac

if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "epochsには1以上の整数を指定してください: $EPOCHS" >&2
    exit 1
fi

if [ ! -f "$BASE_APPFILE" ]; then
    echo "appfile not found: $BASE_APPFILE" >&2
    echo "method は smartnic | ag_smartnic を指定してください" >&2
    exit 1
fi

if [ ! -f "$RUN_ZERO_MPI" ]; then
    echo "run_zero_mpi.py not found: $RUN_ZERO_MPI" >&2
    exit 1
fi


# ============================================================
# 出力先
# ============================================================

mkdir -p "$SCRIPT_DIR/logs"

ACC_APPFILE="$SCRIPT_DIR/logs/host_appfile_vit_l_16_${METHOD}_accuracy"


# ============================================================
# appfileへ環境変数を追加
# ============================================================

# GRAD_CKPTなどが環境にあれば、
# appfileの各rankに-xで注入する。
#
# 例:
#
#   GRAD_CKPT=1 \
#       ./run_vit_accuracy.sh 5 ag_smartnic
#
#   USE_GPU_FLAG_WAIT=0 \
#       ./run_vit_accuracy.sh 2 smartnic

X_FLAGS=""

for VAR in \
    GRAD_CKPT \
    PYTORCH_CUDA_ALLOC_CONF \
    USE_GPU_FLAG_WAIT \
    GPU_FLAG_POOL_SIZE

do
    if [ -n "${!VAR}" ]; then
        X_FLAGS="$X_FLAGS -x $VAR=${!VAR}"
    fi
done


# ============================================================
# USE_GPU_FLAG_WAITの処理
# ============================================================

# 環境変数でUSE_GPU_FLAG_WAITを指定した場合、
# appfileに書き込まれている値を削除する。
#
# その後、X_FLAGSによって環境変数側の値を追加する。

STRIP_FLAG_EXPR=""

if [ -n "$USE_GPU_FLAG_WAIT" ]; then
    STRIP_FLAG_EXPR="s/ -x USE_GPU_FLAG_WAIT=[0-9]*//"
fi


# ============================================================
# 精度評価用appfileの生成
# ============================================================

# 以下の処理を行う:
#
# 1. appfile内のrun_zero_mpi.pyを正しい絶対パスへ置換
# 2. --warmup-itersを削除
# 3. --measure-itersを削除
# 4. 必要な環境変数を-xで追加
# 5. --epochsと--eval-accuracyを追加
#
# run_zero_mpi.pyの置換は、以下のいずれにも対応する:
#
#   run_zero_mpi.py
#   ../../run_zero_mpi.py
#   /home/y-jinbo/understanding_smartnic/scripts/run_zero_mpi.py

sed -E \
    -e "$STRIP_FLAG_EXPR" \
    -e 's/ --warmup-iters [0-9e.+-]*//' \
    -e 's/ --measure-iters [0-9e.+-]*//' \
    -e "s|([^[:space:]]*/)?run_zero_mpi\.py|$RUN_ZERO_MPI|g" \
    -e "s|^-n 1 |-n 1$X_FLAGS |" \
    -e "s|\$| --epochs $EPOCHS --eval-accuracy $EXTRA_ARGS|" \
    "$BASE_APPFILE" > "$ACC_APPFILE"

if [ "$?" -ne 0 ]; then
    echo "精度評価用appfileの生成に失敗しました" >&2
    exit 1
fi

if [ ! -s "$ACC_APPFILE" ]; then
    echo "生成したappfileが空です: $ACC_APPFILE" >&2
    exit 1
fi


# ============================================================
# 生成結果の確認
# ============================================================

echo "[run_vit_accuracy] generated appfile:"
echo "  $ACC_APPFILE"

echo "[run_vit_accuracy] run_zero_mpi.py command:"

grep -n 'run_zero_mpi.py' "$ACC_APPFILE"

if [ "$?" -ne 0 ]; then
    echo "生成したappfile内にrun_zero_mpi.pyがありません" >&2
    exit 1
fi


# ============================================================
# MPI実行
# ============================================================

LOG="$SCRIPT_DIR/logs/vit_accuracy_${METHOD}_$(date +%Y%m%d_%H%M%S).log"

echo "[run_vit_accuracy] log:"
echo "  $LOG"

echo ""
echo "[run_vit_accuracy] start"
echo ""

mpirun --app "$ACC_APPFILE" 2>&1 | tee "$LOG"

MPI_STATUS=${PIPESTATUS[0]}


# ============================================================
# 終了処理
# ============================================================

echo ""

if [ "$MPI_STATUS" -ne 0 ]; then
    echo "[error] mpirunが失敗しました" >&2
    echo "[error] exit status: $MPI_STATUS" >&2
    echo "[error] log: $LOG" >&2
    exit "$MPI_STATUS"
fi

echo "[done] Accuracy history はログ末尾の"
echo "       'Accuracy history:' 行を参照してください"
echo "[done] log: $LOG"