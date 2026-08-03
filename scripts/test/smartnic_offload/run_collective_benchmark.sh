#!/bin/bash
# 集合通信 (AllGather / ReduceScatter) 単体のベンチマーク兼 correctness テスト。
# DOCA (SmartNIC オフロード) 経路と NCCL 経路を同一サイズ系列で計測する。
#
# 事前準備:
#   1. 両 DPU 側で先にサーバを起動しておくこと (host 側は接続待ちでブロックする):
#        dpu$ cd src/smartnic_offload/comch_mpi/dpu
#        dpu$ mpirun --bind-to none --app dpu_appfile
#   2. NCCL 経路 (nccl_rs / nccl_ag / all) を使う場合は MASTER_ADDR / MASTER_PORT を
#      export しておくこと (appfile が -x で全 rank に配る)。
#
# 使い方 (どちらかのホストで 1 回だけ実行。mpirun が 4 rank 全てを起動する):
#   ./run_collective_benchmark.sh [run] [extra collective_benchmark.py args...]
#     run: all (default) | doca_rs | doca_ag | nccl_rs | nccl_ag
#
# 例:
#   ./run_collective_benchmark.sh doca_ag --num-iters 100
RUN=${1:-all}
EXTRA_ARGS="${*:2}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_APPFILE=$SCRIPT_DIR/host_appfile_collective_benchmark
if [ ! -f "$BASE_APPFILE" ]; then
    echo "appfile not found: $BASE_APPFILE" >&2
    exit 1
fi

case "$RUN" in
  all|nccl_rs|nccl_ag)
    if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ]; then
        echo "[warn] MASTER_ADDR / MASTER_PORT 未設定。NCCL 経路は init で失敗します。" >&2
        echo "       例: export MASTER_ADDR=172.16.0.1 MASTER_PORT=29500" >&2
    fi
    ;;
esac

mkdir -p "$SCRIPT_DIR/logs"

# ベース appfile に --run と追加引数を付与した appfile を生成する
RUN_APPFILE=$SCRIPT_DIR/logs/host_appfile_collective_benchmark_${RUN}
sed -e "s/\$/ --run $RUN $EXTRA_ARGS/" "$BASE_APPFILE" > "$RUN_APPFILE"

echo "[run_collective_benchmark] generated appfile: $RUN_APPFILE"
LOG="$SCRIPT_DIR/logs/collective_benchmark_${RUN}_$(date +%Y%m%d_%H%M%S).log"
mpirun --bind-to none --app "$RUN_APPFILE" 2>&1 | tee "$LOG"

echo ""
echo "[done] 結果はログを参照 ($LOG)"
