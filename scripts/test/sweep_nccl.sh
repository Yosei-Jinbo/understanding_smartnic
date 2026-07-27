#!/bin/bash
###############################################################################
# sweep_nccl.sh — E1 段2: NCCL 設定の最小スイープ（既定が最良近傍かの確認）
#
# 目的（査読 C7）:
#   「ベースラインをどの程度注意深くチューニングしたか」に対し、
#   **既定設定を採用したうえで、主要パラメータを振っても大差ないこと**を数値で示す。
#   最良が見つかればそれを新ベースラインに採用する（どちらに転んでも成立する設計）。
#
# 重要: DPU は不要
#   client.py は BENCH_RUN が nccl 系のとき DOCA 初期化をスキップする（DPU 不要）。
#   したがって本スイープは **collective_server の起動なしで完全自動**に回せる。
#   （DOCA と比較したい場合は別途 BENCH_RUN=doca / all のランを行う）
#
# 構成: 既定を基準にした OFAT（1 つずつ変える）。全組合せは張らない。
#   default   : NCCL 既定（GDR=SYS のみ明示）           ← 採用候補
#   gdr_off   : NCCL_NET_GDR_LEVEL を外す               ← 旧測定の条件
#   algo_ring : NCCL_ALGO=Ring                          ← 旧測定の条件
#   proto_ll128 / proto_simple : NCCL_PROTO 指定
#   nch4 / nch8 : NCCL_MIN/MAX_NCHANNELS（既定は 2 チャネル）
#
# 前提:
#   - host 側 venv を activate 済み
#   - シェルに NCCL_ALGO / NCCL_PROTO を export していないこと
#
# 使い方:
#   bash /home/y-jinbo/understanding_smartnic/scripts/test/sweep_nccl.sh
#   # サイズを変える:  BENCH_SIZES=65536,8388608 bash ...
###############################################################################
set -uo pipefail

REPO=/home/y-jinbo/understanding_smartnic
HOST_DIR=$REPO/src/smartnic_offload/comch_mpi/host
APPFILE=$HOST_DIR/host_appfile_py
OUT_DIR=$REPO/logs/sweep_nccl
CSV="$OUT_DIR/results.csv"

BENCH_SIZES_CSV=${BENCH_SIZES:-65536,1048576,8388608,16777216}
declare -A SIZE_LABEL=( [65536]=64KB [1048576]=1MB [8388608]=8MB [16777216]=16MB )

# appfile は MASTER_ADDR / MASTER_PORT / UCX_TLS を起動シェルから継承する
export MASTER_ADDR=${MASTER_ADDR:-172.16.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export UCX_TLS=${UCX_TLS:-rc,sm,self}
export BENCH_NO_PAUSE=1          # 末尾の input() を抑止（連続実行のため）

# 構成: "name|GDR|ALGO|PROTO|NCHANNELS"   ('-' は未設定=既定)
CONFIGS=(
  "default|SYS|-|-|-"
  "gdr_off|-|-|-|-"
  "algo_ring|SYS|Ring|-|-"
  "proto_ll128|SYS|-|LL128|-"
  "proto_simple|SYS|-|Simple|-"
  "nch4|SYS|-|-|4"
  "nch8|SYS|-|-|8"
)

mkdir -p "$OUT_DIR"
echo "config,gdr,algo,proto,nchannels,coll,N,size_label,avg_ms,p50_ms,p99_ms,bw_gbps" > "$CSV"

# ---- 事前チェック -----------------------------------------------------------
for v in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS; do
  if [[ -n "${!v:-}" ]]; then
    echo "!! シェルに $v=${!v} が設定されています。スイープが汚染されるので unset してください。"
    exit 1
  fi
done

# ---- 一時 appfile 生成 (GDR は appfile に固定値があるので sed で外す) --------
make_appfile() {
  local name=$1 gdr=$2 tmp="$OUT_DIR/appfile.$name"
  if [[ "$gdr" == "-" ]]; then
    sed -e "s/-x NCCL_NET_GDR_LEVEL=SYS //g" \
        -e "s|-x NCCL_DEBUG_FILE=[^ ]*|-x NCCL_DEBUG_FILE=$OUT_DIR/nccl_${name}.%h.%p.log|g" \
        "$APPFILE" > "$tmp"
  else
    sed -e "s/-x NCCL_NET_GDR_LEVEL=[^ ]*/-x NCCL_NET_GDR_LEVEL=$gdr/g" \
        -e "s|-x NCCL_DEBUG_FILE=[^ ]*|-x NCCL_DEBUG_FILE=$OUT_DIR/nccl_${name}.%h.%p.log|g" \
        "$APPFILE" > "$tmp"
  fi
  echo "$tmp"
}

# ---- ログから [GPU RS/AG NCCL] を抽出 → "coll,N,avg,p50,p99,bw" -------------
parse_log() {
  awk '
    /^\[GPU (RS|AG) NCCL\]/ && /avg=/ {
      coll = ($0 ~ /GPU RS/) ? "RS" : "AG";
      n=$0;  sub(/.*N=[ ]*/,"",n); sub(/[^0-9].*/,"",n);
      a=$0;  sub(/.*avg=[ ]*/,"",a); sub(/[ ]*ms.*/,"",a);
      p=$0;  sub(/.*p50=[ ]*/,"",p); sub(/[ ]*ms.*/,"",p);
      q=$0;  sub(/.*p99=[ ]*/,"",q); sub(/[ ]*ms.*/,"",q);
      b=$0;  sub(/.*BW=[ ]*/,"",b); sub(/[ ]*GB.*/,"",b);
      if (n != "") print coll","n","a","p","q","b;
    }
  ' "$1"
}

echo "[sweep] configs=${#CONFIGS[@]} sizes=$BENCH_SIZES_CSV out=$OUT_DIR"
echo "[sweep] DPU 不要 (BENCH_RUN=nccl は DOCA 初期化をスキップ)"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name gdr algo proto nch <<< "$cfg"
  log="$OUT_DIR/${name}.log"
  echo "============================================================"
  echo "==== [config] $name (GDR=$gdr ALGO=$algo PROTO=$proto NCHANNELS=$nch) ===="
  tmp=$(make_appfile "$name" "$gdr")

  # bare -x で転送される変数は「設定しない＝既定」を表現できるので unset で制御
  unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
  [[ "$algo"  != "-" ]] && export NCCL_ALGO="$algo"
  [[ "$proto" != "-" ]] && export NCCL_PROTO="$proto"
  [[ "$nch"   != "-" ]] && { export NCCL_MIN_NCHANNELS="$nch"; export NCCL_MAX_NCHANNELS="$nch"; }

  ( cd "$HOST_DIR" && BENCH_RUN=nccl BENCH_SIZES="$BENCH_SIZES_CSV" \
      mpirun --app "$tmp" < /dev/null ) > "$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  !! 実行に失敗 (rc=$rc)。末尾を表示:"; tail -5 "$log"
  fi

  # 実際に解決された algo/proto をログから拾う（報告用）
  resolved=$(grep -hoE "AllGather: [0-9]+ Bytes -> Algo [A-Z]+ proto [A-Z0-9]+" \
              "$OUT_DIR/nccl_${name}."*.log 2>/dev/null | tail -1)
  [[ -n "$resolved" ]] && echo "  解決値: $resolved"

  cnt=0
  while IFS=, read -r coll n avg p50 p99 bw; do
    [[ -z "$n" ]] && continue
    label="${SIZE_LABEL[$n]:-$n}"
    echo "$name,$gdr,$algo,$proto,$nch,$coll,$n,$label,$avg,$p50,$p99,$bw" >> "$CSV"
    cnt=$((cnt+1))
  done < <(parse_log "$log")
  echo "  取得: ${cnt} 行 → $log"
done

echo
echo "========== SUMMARY: p50 (ms) と default 比 Δ% =========="
echo "CSV: $CSV"
awk -F, '
NR>1 {
  seen[$1] || (order[++no]=$1, seen[$1]=1);
  p50[$1","$6","$8]=$10;
  szseen[$8] || (szorder[++nsz]=$8, szseen[$8]=1);
}
END{
  split("AG RS", colls, " ");
  printf "%-14s", "config";
  for(c=1;c<=2;c++) for(s=1;s<=nsz;s++) printf "%16s", colls[c]":"szorder[s];
  printf "\n";
  for(r=1;r<=no;r++){
    cfg=order[r]; printf "%-14s", cfg;
    for(c=1;c<=2;c++) for(s=1;s<=nsz;s++){
      k=cfg","colls[c]","szorder[s]; base="default","colls[c]","szorder[s];
      v=p50[k]; b=p50[base];
      if(v==""){ printf "%16s","-"; }
      else if(cfg=="default" || b=="" || b+0==0){ printf "%16.3f", v; }
      else { printf "%10.3f(%+.0f%%)", v, (v-b)/b*100.0; }
    }
    printf "\n";
  }
  print "\n注) default 行は絶対値(ms)、以降は p50 と (default比 Δ%)。負=改善。";
  print "    どの行も default に大差なければ「既定が最良近傍」と主張できる。";
}' "$CSV" 2>/dev/null || echo "(gawk 無し環境では summary スキップ。CSV を参照)"
echo "========================================================"

if command -v python3 >/dev/null 2>&1; then
  for m in p50_ms avg_ms bw_gbps; do
    python3 "$(dirname "$0")/pivot.py" "$CSV" "$m" 2>/dev/null
  done
fi
