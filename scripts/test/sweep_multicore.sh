#!/bin/bash
###############################################################################
# sweep_multicore.sh — マルチコア設計（通信:計算コア配分）のアブレーション計測
#
# 目的（票2 / Ablation Plan）:
#   通信コア数 C=COMM_CORES と計算コア数 K=COMPUTE_CORES を振り、DOCA の
#   AllGather / ReduceScatter (flat) の avg / p50 / p99 を代表サイズで取得する。
#   「配分は計算・通信“両方”の性能に基づく」ことを示すためのデータ。
#
# 計測方針（重要）:
#   1) 対象は DOCA offload の RS/AG flat のみ（BENCH_RUN=doca で NCCL は省略＝高速化）。
#   2) 代表サイズ = per-rank メッセージ = chunk*2 bytes:
#        64KB → chunk 65536 / 1MB → chunk 1048576 / 16MB → chunk 16777216
#      これを BENCH_SIZES で client.py に渡す（client.py 側は _bench_sizes() が参照）。
#   3) 各構成ごとに:
#        (a) DPU 側で COMM_CORES/COMPUTE_CORES を設定して collective_server を再起動
#            （コア配分は DPU プロセス起動時に getenv で確定するため、構成ごとの再起動が必須）
#        (b) host 側で集合通信ベンチを 1 回流し、rank0 の出力をログへ保存
#        (c) ログから [DOCA RS/AG flat] 行の avg/p50/p99 を抽出し CSV へ
#   4) warmup/iters は client.py 固定（num_warmup=20 / num_iters=1000）。
#   5) 統計の読み方: RS は集約経路（rs_pool）のコールドスタートで最初の pool 使用サイズが
#      やや高く出ることがある（既知アーティファクト）。**中央値 p50 を主指標**にする。
#   6) 公平性: warmup/iters・dual_rail・コアピン・計算=rank共有/通信=rank分離 は全構成で同一。
#
# 前提:
#   - 最新ソースを DPU/host に配布し、DPU 側は ./build.sh 済み（env 対応バイナリ）。
#   - host 側 venv を activate 済み（torch/mpi）。
#   - DPU 再起動は既定で「手動（プロンプトで一時停止）」。SSH 自動化は下の restart_dpu を編集。
#
# 使い方:
#   cd /home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host
#   source /home/y-jinbo/.venv/bin/activate
#   bash /home/y-jinbo/understanding_smartnic/scripts/test/sweep_multicore.sh
###############################################################################
set -uo pipefail

# ---- 設定（必要ならここだけ編集） -------------------------------------------
HOST_DIR=/home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host
OUT_DIR=/home/y-jinbo/understanding_smartnic/logs/sweep_multicore
CSV="$OUT_DIR/results.csv"

# 振る構成: "C K"（通信コア数 計算コア数）
# 1 4は計算コア 1x4 + 4 + 4 = 12
CONFIGS=(
  "3 6"
  "2 6"
  "1 6"
  "1 8"
  "2 8"
)

# 代表サイズ（chunk = fp16 要素数）と表示ラベル（per-rank = chunk*2 bytes）
BENCH_SIZES_CSV="65536,1048576,16777216"
declare -A SIZE_LABEL=( [65536]=64KB [1048576]=1MB [16777216]=16MB )

# DPU 再起動方法: MANUAL=1 で手動（プロンプト）、0 で下の SSH 自動化を使用
MANUAL_DPU=1
DPU_SSH_LAUNCHER="ssh ubuntu@dpu01"    # dpu_appfile を叩くランチャ（dpu01→dpu01+dpu02 に展開）
DPU_DIR="/home/ubuntu/doca_practice/comch_mpi/dpu"
# -----------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
echo "comm_cores,compute_cores,coll,N,size_label,avg_ms,p50_ms,p99_ms,bw_gbps" > "$CSV"

# DPU 側で構成 C:K の collective_server を再起動する。
#   MANUAL_DPU=1: 手順を表示して Enter 待ち（ユーザが dpu01 で実行）。
#   MANUAL_DPU=0: SSH で kill→env→mpirun --bind-to noneをバックグラウンド起動（環境依存・要調整）。
restart_dpu() {
  local C=$1 K=$2
  if [[ "$MANUAL_DPU" == "1" ]]; then
    echo "============================================================"
    echo "[DPU] dpu01 で以下を実行し、両 DPU の collective_server を再起動してください:"
    echo "  cd $DPU_DIR"
    echo "  export COMM_CORES=$C COMPUTE_CORES=$K FORCE_STAGING=0 FORCE_SINGLE_RAIL=1 AG_PIECE_MAX=8"
    echo "  mpirun --bind-to none --app dpu_appfile   # 起動ログに 'COMM_CORES=$C COMPUTE_CORES=$K ... FORCE_SINGLE_RAIL=1' を確認"
    echo "------------------------------------------------------------"
    read -r -p "DPU が起動して待受状態になったら Enter: " _
  else
    # 環境依存。ssh・パス・pkill 対象はサイトに合わせて調整すること。
    # FORCE_STAGING=0 / FORCE_SINGLE_RAIL=1 / AG_PIECE_MAX=8 を明示（残留 export の漏れ防止）。
    $DPU_SSH_LAUNCHER "pkill -f doca_comch_server; sleep 2; \
      cd $DPU_DIR && COMM_CORES=$C COMPUTE_CORES=$K FORCE_STAGING=0 FORCE_SINGLE_RAIL=1 AG_PIECE_MAX=8 \
      nohup mpirun --bind-to none --app dpu_appfile > /tmp/dpu_${C}_${K}.log 2>&1 & sleep 5"
    sleep 5
  fi
}

# host 側ベンチを 1 回流してログを返す
run_host_bench() {
  local C=$1 K=$2 log=$3
  ( cd "$HOST_DIR" && \
    BENCH_RUN=doca BENCH_SIZES="$BENCH_SIZES_CSV" \
    mpirun --bind-to none --app host_appfile_py < /dev/null ) 2>&1 | tee "$log"
}

# ログから [DOCA RS/AG flat] 行の avg/p50/p99 を抽出 → "coll,N,avg,p50,p99"
parse_log() {
  awk '
    /\[DOCA (RS|AG) flat\]/ {
      coll = ($0 ~ /DOCA RS/) ? "RS" : "AG";
      n=$0;  sub(/.*N=[ ]*/,"",n); sub(/[^0-9].*/,"",n);
      a=$0;  sub(/.*avg=[ ]*/,"",a); sub(/[ ]*ms.*/,"",a);
      p=$0;  sub(/.*p50=[ ]*/,"",p); sub(/[ ]*ms.*/,"",p);
      q=$0;  sub(/.*p99=[ ]*/,"",q); sub(/[ ]*ms.*/,"",q);
      b=$0;  sub(/.*BW=[ ]*/,"",b); sub(/[ ]*GB.*/,"",b);
      if (n != "") print coll","n","a","p","q","b;
    }
  ' "$1"
}

echo "[sweep] configs=${#CONFIGS[@]}  sizes=$BENCH_SIZES_CSV  out=$OUT_DIR"
for cfg in "${CONFIGS[@]}"; do
  read -r C K <<< "$cfg"
  log="$OUT_DIR/doca_C${C}_K${K}.log"
  echo "==== [config] COMM_CORES=$C COMPUTE_CORES=$K ===="
  restart_dpu "$C" "$K"
  echo "[host] running DOCA AG/RS flat (sizes=$BENCH_SIZES_CSV) ..."
  run_host_bench "$C" "$K" "$log"

  # 抽出して CSV 追記
  while IFS=, read -r coll n avg p50 p99 bw; do
    [[ -z "$n" ]] && continue
    label="${SIZE_LABEL[$n]:-$n}"
    echo "$C,$K,$coll,$n,$label,$avg,$p50,$p99,$bw" >> "$CSV"
  done < <(parse_log "$log")
  echo "[host] done → $log"
done

echo
echo "================= SUMMARY (p50 ms) ================="
echo "CSV: $CSV"
# 見やすい表: 行=構成, 列=coll×size の p50
awk -F, 'NR>1 {
  key=$1":"$2; conf[key]=1;
  cell[key","$3","$5]=$7;   # p50
}
END{
  printf "%-10s", "C:K";
  split("AG:64KB AG:1MB AG:16MB RS:64KB RS:1MB RS:16MB", cols, " ");
  for(i=1;i<=6;i++){ split(cols[i],cc,":"); printf "%12s", cols[i]; }
  printf "\n";
  n=asorti(conf, sk);
  for(j=1;j<=n;j++){
    k=sk[j]; split(k,ck,":"); printf "%-10s", ck[1]":"ck[2];
    for(i=1;i<=6;i++){ split(cols[i],cc,":"); v=cell[k","cc[1]","cc[2]]; printf "%12s", (v==""?"-":v); }
    printf "\n";
  }
}' "$CSV" 2>/dev/null || echo "(summary は gawk が無い環境ではスキップ。CSV を参照)"
echo "===================================================="

# ピボット表（RS/AG × N を行、設定を列）を生成: pivot_<metric>.csv
if command -v python3 >/dev/null 2>&1; then
  for m in p50_ms avg_ms bw_gbps; do
    python3 "$(dirname "$0")/pivot.py" "$CSV" "$m"
  done
fi
