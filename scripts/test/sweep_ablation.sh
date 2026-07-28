#!/bin/bash
###############################################################################
# sweep_ablation.sh — 票1: 最適化アブレーション（積み上げ・依存順固定）
#
# 目的:
#   3 つの最適化を「依存順に積み上げ」て各段の効果を測る（OFAT ではない）。
#     依存順: マルチコア化 → チャンクパイプライン → CrossGVMI
#   単独 ON (OFAT) は不可（チャンクはマルチコア前提で単独では効かず誤指針になる）。
#
# 4 構成（各セルは DPU 側 env のみで切替。ホスト/client.py の変更は不要）:
#   +--------------------+------------+---------------+--------------+------------+---------------+
#   | 構成               | COMM_CORES | COMPUTE_CORES | AG_PIECE_MAX | RS_PREPOST | FORCE_STAGING |
#   +--------------------+------------+---------------+--------------+------------+---------------+
#   | baseline (all off) |     1      |       1       |      1       |     0      |   1 (staging) |
#   | + マルチコア化     |     2      |       8       |      1       |     0      |   1           |
#   | + パイプライン化   |     2      |       8       |      8       |     1      |   1           |
#   | + CrossGVMI (=full)|     2      |       8       |      8       |     1      |   0 (direct)  |
#   +--------------------+------------+---------------+--------------+------------+---------------+
#   マルチコア    : COMM_CORES/COMPUTE_CORES（off=1/1）。※sweep_multicore.sh と同じ env 方式。
#   パイプライン化: **AG_PIECE_MAX と RS_PREPOST を同時に on/off する**（1 段として扱う）。
#                   - AG_PIECE_MAX : チャンク分割（off=1 piece / on=8）
#                   - RS_PREPOST   : RS の Recv 先行 post（off=0 / on=1）
#                   コード上は独立した 2 つの env だが、どちらも「通信を先出しして重ねる」
#                   同種の最適化なので、票1 では 1 段にまとめて提示する。
#   CrossGVMI     : FORCE_STAGING（1=staging＝off / 0=GPU 直接＝on）。**AG のみ**に作用（RS は不変）。
#
# 注意（このサイズ域での実際の発動条件）:
#   ag_compute_num_pieces() は piece が AG_PIECE_MIN_SIZE(256KB) 未満なら 1 に落ち、
#   piece 数は ceil(chunk_size / AG_PIECE_TARGET=8MB) が上限。したがって:
#     - AG (chunk_size = N*2)   : 16MB 点 (=32MB) でのみ 4 piece に分割される
#     - RS (chunk_size = N/2)   : 全サイズで 1 piece（分割は発動しない）
#   → RS の「+パイプライン化」段は実質 **RS_PREPOST 単独の効果**である。
#
# 計測方針:
#   - 対象は DOCA offload の RS/AG flat のみ（BENCH_RUN=doca）。
#   - 代表サイズ 3 点（per-rank = chunk*2 bytes）: 64KB=65536 / 1MB=1048576 / 16MB=16777216。
#   - 取得統計: avg / p50 / p99。**主指標は p50**（RS 集約プールのコールドスタートで
#     最初の pool 使用サイズがやや高く出る既知アーティファクトの影響を受けにくい）。
#   - 読み筋: マルチコアは RS に効く／チャンク・CrossGVMI は大サイズ AG に効く／小サイズは効かない。
#   - RS の CrossGVMI 行は N/A（+チャンクと同値）— RS は reduction のため staging の概念が無い。
#   - 公平性: warmup/iters・dual_rail・コアピン・計算=rank共有/通信=rank分離 は全構成で同一。
#     構成間で変えるのは上表の 4 env のみ。
#
# 前提:
#   - 最新ソースを DPU/host に配布し、DPU 側は ./build.sh 済み（FORCE_STAGING 対応バイナリ）。
#   - host 側 venv activate 済み。dpu_appfile は AG_PIECE_MAX/COMM_CORES/COMPUTE_CORES/FORCE_STAGING
#     を -x で forward する版（bare）。
#
# 使い方:
#   cd /home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host
#   source /home/y-jinbo/.venv/bin/activate
#   bash /home/y-jinbo/understanding_smartnic/scripts/test/sweep_ablation.sh
###############################################################################
set -uo pipefail

# ---- 設定（必要ならここだけ編集） -------------------------------------------
HOST_DIR=/home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host
OUT_DIR=/home/y-jinbo/understanding_smartnic/logs/sweep_ablation
CSV="$OUT_DIR/results.csv"

BENCH_SIZES_CSV="65536,1048576,16777216"
declare -A SIZE_LABEL=( [65536]=64KB [1048576]=1MB [16777216]=16MB )

# 積み上げ構成: "name|COMM_CORES|COMPUTE_CORES|AG_PIECE_MAX|RS_PREPOST|FORCE_STAGING"
#   pipeline 段で AG_PIECE_MAX と RS_PREPOST を同時に切り替える
CONFIGS=(
  "baseline|1|1|1|0|1"
  "multicore|2|8|1|0|1"
  "pipeline|2|8|8|1|1"
  "full|2|8|8|1|0"
)

MANUAL_DPU=1
DPU_SSH_LAUNCHER="ssh ubuntu@dpu01"
DPU_DIR="/home/ubuntu/doca_practice/comch_mpi/dpu"
# -----------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
echo "config,comm_cores,compute_cores,ag_piece_max,rs_prepost,force_staging,coll,N,size_label,avg_ms,p50_ms,p99_ms,bw_gbps" > "$CSV"

restart_dpu() {
  local name=$1 C=$2 K=$3 P=$4 R=$5 S=$6
  if [[ "$MANUAL_DPU" == "1" ]]; then
    echo "============================================================"
    echo "[DPU] dpu01 で以下を実行して両 DPU の collective_server を再起動 ($name):"
    echo "  cd $DPU_DIR"
    echo "  export COMM_CORES=$C COMPUTE_CORES=$K AG_PIECE_MAX=$P RS_PREPOST=$R FORCE_STAGING=$S FORCE_SINGLE_RAIL=1"
    echo "  mpirun --bind-to none --app dpu_appfile"
    echo "  # 起動ログの core alloc 行を **4 ランクすべて** 確認すること:"
    echo "  #   COMM_CORES=$C COMPUTE_CORES=$K FORCE_STAGING=$S FORCE_SINGLE_RAIL=1 RS_PREPOST=$R"
    echo "  # 1 つでも欠けていたら dpu01/dpu02 のどちらかが再ビルドされていない (混在すると誤測定)"
    echo "------------------------------------------------------------"
    read -r -p "DPU が待受状態になったら Enter: " _
  else
    $DPU_SSH_LAUNCHER "pkill -f doca_comch_server; sleep 2; cd $DPU_DIR && \
      COMM_CORES=$C COMPUTE_CORES=$K AG_PIECE_MAX=$P RS_PREPOST=$R FORCE_STAGING=$S FORCE_SINGLE_RAIL=1 \
      nohup mpirun --bind-to none --app dpu_appfile > /tmp/dpu_${name}.log 2>&1 & sleep 5"
    sleep 5
  fi
}

run_host_bench() {
  local log=$1
  ( cd "$HOST_DIR" && BENCH_RUN=doca BENCH_SIZES="$BENCH_SIZES_CSV" \
    mpirun --bind-to none --app host_appfile_py < /dev/null ) 2>&1 | tee "$log"
}

# ログから [DOCA RS/AG flat] 行を抽出 → "coll,N,avg,p50,p99"
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

echo "[ablation] cumulative configs=${#CONFIGS[@]} sizes=$BENCH_SIZES_CSV out=$OUT_DIR"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name C K P R S <<< "$cfg"
  log="$OUT_DIR/${name}.log"
  echo "==== [config] $name (COMM=$C COMPUTE=$K AG_PIECE=$P RS_PREPOST=$R FORCE_STAGING=$S) ===="
  restart_dpu "$name" "$C" "$K" "$P" "$R" "$S"
  echo "[host] running DOCA AG/RS flat ..."
  run_host_bench "$log"
  while IFS=, read -r coll n avg p50 p99 bw; do
    [[ -z "$n" ]] && continue
    label="${SIZE_LABEL[$n]:-$n}"
    echo "$name,$C,$K,$P,$R,$S,$coll,$n,$label,$avg,$p50,$p99,$bw" >> "$CSV"
  done < <(parse_log "$log")
  echo "[host] done → $log"
done

echo
echo "========== SUMMARY: p50 (ms) と baseline 比 Δ% =========="
echo "CSV: $CSV"
# 行=構成（積み上げ順）, 列=coll×size。各セル: p50 (Δ% vs baseline)。
awk -F, '
NR>1 {
  order_seen[$1] || (order[++no]=$1, order_seen[$1]=1);
  key=$1","$7","$9;         # config,coll,size_label
  p50[key]=$11;
}
END{
  split("AG:64KB AG:1MB AG:16MB RS:64KB RS:1MB RS:16MB", cols, " ");
  printf "%-12s", "config";
  for(i=1;i<=6;i++) printf "%16s", cols[i];
  printf "\n";
  for(r=1;r<=no;r++){
    cfg=order[r]; printf "%-12s", cfg;
    for(i=1;i<=6;i++){
      split(cols[i],cc,":"); k=cfg","cc[1]","cc[2]; base="baseline","cc[1]","cc[2];
      v=p50[k]; b=p50[base];
      if(v==""){ printf "%16s","-"; }
      else if(cfg=="baseline" || b=="" || b+0==0){ printf "%16.3f", v; }
      else { d=(v-b)/b*100.0; printf "%10.3f(%+.0f%%)", v, d; }
    }
    printf "\n";
  }
  print "\n注) baseline 行は絶対値(ms)、以降は p50 と (baseline比 Δ%)。負=改善。";
  print "    RS の full 行は chunk と同値(CrossGVMI は RS 不作用)。";
}' "$CSV" 2>/dev/null || echo "(gawk 無し環境では summary スキップ。CSV を参照)"
echo "========================================================"

# ピボット表（RS/AG × N を行、設定を列）を生成: pivot_<metric>.csv
if command -v python3 >/dev/null 2>&1; then
  for m in p50_ms avg_ms bw_gbps; do
    python3 "$(dirname "$0")/pivot.py" "$CSV" "$m"
  done
fi
