#!/bin/bash
###############################################################################
# nccl_proxy_cpu_check.sh — E2: NCCL proxy スレッドのコア割り当てと CPU 使用率を実測する
#
# 目的（査読 C7）:
#   「ベースラインが CPU 側で不当に制約されていないか」を確認する。示すのは
#   **「CPU を与えても速くならない ＝ CPU 側の制約を受けていない」** こと。
#
# やること:
#   2 構成を同条件で走らせ、その裏でスレッド別 CPU% と実行コアをサンプルする。
#     default : host_appfile_py        （proxy の配置は NCCL 任せ）
#     numa1   : host_appfile_py_numa1  （numactl で GPU/NIC と同じ NUMA1 に固定）
#   proxy のコアは 2 系統から取る:
#     (a) NCCL ログ    : `[Proxy Progress] Device N CPU core M`
#                        → sched_getcpu() のスナップショット。affinity 適用前の可能性あり
#     (b) /proc 実測   : task/<tid>/stat の processor フィールド（定常状態）
#   食い違ったら (b) が正しい。
#
# 重要: DPU は不要
#   BENCH_RUN=nccl のとき client.py は DOCA 初期化をスキップする。
#
# サンプル区間を確保するためサイズを多めに回す（既定 6 点、AG+RS で 1 構成あたり約 60〜90 秒）。
#
# 使い方:
#   bash scripts/test/nccl_proxy_cpu_check.sh
#   # サイズを変える:   BENCH_SIZES=1048576,8388608 bash ...
#   # 片方だけ:         ONLY=numa1 bash ...
###############################################################################
set -uo pipefail

REPO=/home/y-jinbo/understanding_smartnic
HOST_DIR=$REPO/src/smartnic_offload/comch_mpi/host
OUT_DIR=$REPO/logs/e2_proxy_cpu
SAMPLER=$REPO/scripts/test/sample_thread_cpu.sh

# サンプル区間を広げるため多めのサイズを回す（client.py は N>8388608 のとき 100 iters、他は 1000）
BENCH_SIZES_CSV=${BENCH_SIZES:-65536,262144,1048576,4194304,8388608,16777216}
ONLY=${ONLY:-}
SAMPLE_INT=${SAMPLE_INT:-0.2}
SAMPLE_N=${SAMPLE_N:-600}          # 0.2s × 600 = 最大 120 秒（ベンチ終了で自動的に打ち切り）

export MASTER_ADDR=${MASTER_ADDR:-172.16.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export UCX_TLS=${UCX_TLS:-rc,sm,self}
export BENCH_NO_PAUSE=1

mkdir -p "$OUT_DIR"

# 構成: "name|appfile|nccl_log_dir"
CONFIGS=(
  "default|host_appfile_py|$REPO/logs/nccl"
  "numa1|host_appfile_py_numa1|$REPO/logs/nccl_numa1"
)

for v in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS; do
  [[ -n "${!v:-}" ]] && { echo "!! シェルに $v=${!v} が残っています。unset してから実行してください。"; exit 1; }
done

run_one() {
  local name=$1 appfile=$2 nccl_dir=$3
  local bench_log="$OUT_DIR/${name}_bench.log"

  echo "============================================================"
  echo "[$name] appfile=$appfile  sizes=$BENCH_SIZES_CSV"

  mkdir -p "$nccl_dir"
  ssh -o ConnectTimeout=6 bluefield02 "mkdir -p $nccl_dir" 2>/dev/null
  rm -f "$nccl_dir"/coll.*.log
  ssh -o ConnectTimeout=6 bluefield02 "rm -f $nccl_dir/coll.*.log" 2>/dev/null

  # ベンチをバックグラウンドで起動
  ( cd "$HOST_DIR" && BENCH_RUN=nccl BENCH_SIZES="$BENCH_SIZES_CSV" \
      mpirun --app "$appfile" < /dev/null ) > "$bench_log" 2>&1 &
  local bench_pid=$!

  # このノード(bluefield01)のランクが立ち上がるのを待つ
  local pids=""
  for _ in $(seq 1 60); do
    sleep 1
    pids=$(pgrep -f "client.py collective_server" | tr '\n' ' ')
    [[ -n "$pids" ]] && break
  done
  if [[ -z "$pids" ]]; then
    echo "  !! client.py が起動しませんでした。$bench_log を確認してください"
    wait $bench_pid 2>/dev/null; return 1
  fi
  echo "  ローカルランク PID: $pids"

  # 各ランクを並行してサンプル（ベンチ終了時にプロセス消滅で自動停止）
  local sp=()
  for p in $pids; do
    bash "$SAMPLER" -p "$p" -i "$SAMPLE_INT" -n "$SAMPLE_N" -a \
      > "$OUT_DIR/${name}_cpu_pid${p}.txt" 2>&1 &
    sp+=($!)
  done

  wait $bench_pid 2>/dev/null
  echo "  ベンチ終了。サンプラの集計を待機..."
  for s in "${sp[@]}"; do wait "$s" 2>/dev/null; done

  # --- 結果表示 ---
  echo
  echo "  ---- (a) NCCL ログが報告する proxy コア ----"
  grep -hE "\[Proxy (Service|Progress)" "$nccl_dir"/coll.*.log 2>/dev/null \
    | sed 's/.*NCCL INFO /    /' | sort -u || echo "    (なし)"

  echo
  echo "  ---- (b) /proc 実測: NCCL 系スレッドの CPU% と実行コア ----"
  grep -hE "NCCL|nccl" "$OUT_DIR/${name}"_cpu_pid*.txt 2>/dev/null | sed 's/^/    /' \
    || echo "    (NCCL 名のスレッドが見つからない → 下の全スレッド一覧を参照)"

  echo
  echo "  ---- 参考: CPU% 上位スレッド ----"
  for f in "$OUT_DIR/${name}"_cpu_pid*.txt; do
    [[ -f "$f" ]] || continue
    echo "    [$(basename "$f")]"
    sed -n '/^thread /,/^$/p' "$f" | sed 's/^/      /' | head -12
  done

  echo
  echo "  ---- ベンチ結果 ----"
  grep -hE "^\[GPU (AG|RS) NCCL\]" "$bench_log" | sed 's/^/    /'
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name appfile nccl_dir <<< "$cfg"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue
  run_one "$name" "$appfile" "$nccl_dir"
done

echo
echo "============================================================"
echo "出力: $OUT_DIR"
echo "  <name>_bench.log        ベンチ標準出力"
echo "  <name>_cpu_pid<PID>.txt スレッド別 CPU% / 実行コア"
echo
echo "読み方:"
echo "  1. (b) の proxy スレッドの max% が 100 に遠ければ「1 コアを飽和させていない」"
echo "  2. cores が N1 (12-23) なら GPU/NIC と同一 NUMA"
echo "  3. default と numa1 でベンチ結果が誤差範囲なら「CPU を与えても速くならない」"
echo "     → E2 の地の文はこれで書ける。numa1 が速ければそちらを新ベースラインに採用する。"
