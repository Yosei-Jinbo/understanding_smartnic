#!/bin/bash
###############################################################################
# sweep_ablation_c1k8.sh — 積み上げアブレーションの C1K8 再測定版
#
# sweep_ablation.sh との差分:
#   - multicore/pipeline/full のコア割当を C2K8 (2*2+8+4=16 = Arm 全コア飽和) から
#     C1K8 (2*1+8+4=14、2 コア残し) に変更。
#     sweep_multicore の実測で、全コア飽和構成のみ RS p99 が 1.7〜2.4 倍に膨れる
#     (72.8/94.3ms vs 非飽和 45.5〜49.9ms) ことが判明したため、
#     推奨運用点 C1K8 で表を作り直す。baseline (C1K1) は比較のため同条件で再取得。
#   - DPU 再起動を自動化 (MANUAL_DPU 廃止)。両 DPU の残プロセスを掃除してから
#     dpu01 の mpirun で再起動し、両 DPU の待受を確認して host bench を流す。
#   - 出力は logs/sweep_ablation_c1k8/ (既存の logs/sweep_ablation/ は保持)。
###############################################################################
set -uo pipefail

HOST_DIR=/home/y-jinbo/understanding_smartnic/src/smartnic_offload/comch_mpi/host
OUT_DIR=/home/y-jinbo/understanding_smartnic/logs/sweep_ablation_c1k8
CSV="$OUT_DIR/results.csv"
DPU_DIR="/home/ubuntu/doca_practice/comch_mpi/dpu"

BENCH_SIZES_CSV="65536,1048576,16777216"
declare -A SIZE_LABEL=( [65536]=64KB [1048576]=1MB [16777216]=16MB )

# "name|COMM_CORES|COMPUTE_CORES|AG_PIECE_MAX|RS_PREPOST|FORCE_STAGING"
CONFIGS=(
  "baseline|1|1|1|0|1"
  "multicore|1|8|1|0|1"
  "pipeline|1|8|8|1|1"
  "full|1|8|8|1|0"
)

mkdir -p "$OUT_DIR"
echo "config,comm_cores,compute_cores,ag_piece_max,rs_prepost,force_staging,coll,N,size_label,avg_ms,p50_ms,p99_ms,bw_gbps" > "$CSV"

# DPU 起動は手動 (ssh 自動化は mpirun が ssh セッションを掴んでハングするため廃止)。
# 各構成の開始時にコマンドを提示するので、dpu01 で実行して Enter を押す。
restart_dpu() {
  local name=$1 C=$2 K=$3 P=$4 R=$5 S=$6
  echo "============================================================"
  echo "[DPU] dpu01 で以下を実行して両 DPU の collective_server を再起動 ($name):"
  echo "  pkill -f doca_comch_server; sleep 2"
  echo "  cd $DPU_DIR"
  echo "  export COMM_CORES=$C COMPUTE_CORES=$K AG_PIECE_MAX=$P RS_PREPOST=$R FORCE_STAGING=$S FORCE_SINGLE_RAIL=1"
  echo "  mpirun --bind-to none --app dpu_appfile"
  echo "  # 起動ログの core alloc 行を 4 ランクすべて確認:"
  echo "  #   COMM_CORES=$C COMPUTE_CORES=$K FORCE_STAGING=$S RS_PREPOST=$R"
  echo "------------------------------------------------------------"
  read -r -p "DPU が待受状態になったら Enter: " _
}

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

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name C K P R S <<< "$cfg"
  log="$OUT_DIR/${name}.log"
  echo "==== [config] $name (COMM=$C COMPUTE=$K AG_PIECE=$P RS_PREPOST=$R FORCE_STAGING=$S) ===="
  restart_dpu "$name" "$C" "$K" "$P" "$R" "$S" || exit 1
  ( cd "$HOST_DIR" && BENCH_RUN=doca BENCH_SIZES="$BENCH_SIZES_CSV" BENCH_NO_PAUSE=1 \
    MASTER_ADDR=${MASTER_ADDR:-172.16.0.1} MASTER_PORT=${MASTER_PORT:-29500} UCX_TLS=${UCX_TLS:-rc,sm,self} \
    mpirun --bind-to none --app host_appfile_py < /dev/null ) > "$log" 2>&1
  while IFS=, read -r coll n avg p50 p99 bw; do
    [[ -z "$n" ]] && continue
    echo "$name,$C,$K,$P,$R,$S,$coll,$n,${SIZE_LABEL[$n]:-$n},$avg,$p50,$p99,$bw" >> "$CSV"
  done < <(parse_log "$log")
  echo "[done] $name"
done

echo "[完了] dpu01 の collective_server は手動で停止してください (pkill -f doca_comch_server)"
echo "CSV: $CSV"
column -s, -t "$CSV"
