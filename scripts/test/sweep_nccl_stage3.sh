#!/bin/bash
###############################################################################
# sweep_nccl_stage3.sh — E1 段3: チャネル数の上限と GDR の是非を反復測定で確定する
#
# 背景（段2 = sweep_nccl.sh の結果、logs/sweep_nccl/）:
#   - チャネル数が支配的。既定の 2 チャネルは大サイズで明確に不利
#       nch8 は 8MB/16MB で p50 −27〜−49%、p99 −57〜−71%
#       2→4→8 が単調改善 → **16 も試す価値がある**
#   - GDR は行ごとに優劣が入れ替わり、一貫した優位が無い → **単発では決められない**
#   - プロトコルは既定で最適（小=LL / 大=SIMPLE を自動切替）→ 振らない
#   - NCCL_PROTO=LL128 は結果不正（順序保証を満たさない）→ 使わない
#
# したがって本スクリプトは **NCHANNELS × GDR の 2 軸だけ**を、**反復ありで**測る。
#
# 構成 (5) × 反復 (3) = 15 ラン:
#   default      : NCHANNELS 未設定 (=2ch) + GDR on   ← 参照点
#   nch8_gdron   / nch8_gdroff
#   nch16_gdron  / nch16_gdroff
#
# 実験計画上の注意:
#   反復は **rep を外側ループ**にする（構成ごとに 3 連続で回さない）。
#   熱・ネットワークのドリフトが特定構成に偏るのを避けるため。
#
# 重要: DPU は不要
#   BENCH_RUN=nccl のとき client.py は DOCA 初期化をスキップする。
#   collective_server の起動なしで完全自動に回せる。
#
# 前提:
#   - host 側 venv を activate 済み
#   - シェルに NCCL_ALGO / NCCL_PROTO / NCCL_*_NCHANNELS を export していないこと
#
# 使い方:
#   bash /home/y-jinbo/understanding_smartnic/scripts/test/sweep_nccl_stage3.sh
#   # 反復数を変える:  REPS=5 bash ...
#   # サイズを変える:  BENCH_SIZES=8388608,16777216 bash ...
###############################################################################
set -uo pipefail

REPO=/home/y-jinbo/understanding_smartnic
HOST_DIR=$REPO/src/smartnic_offload/comch_mpi/host
APPFILE=$HOST_DIR/host_appfile_py
OUT_DIR=$REPO/logs/sweep_nccl_stage3
CSV="$OUT_DIR/results.csv"          # 生データ (rep 列あり)
AGG="$OUT_DIR/summary.csv"          # 反復を集約した表

REPS=${REPS:-3}
BENCH_SIZES_CSV=${BENCH_SIZES:-65536,1048576,8388608,16777216}
declare -A SIZE_LABEL=( [65536]=64KB [1048576]=1MB [8388608]=8MB [16777216]=16MB )

# appfile は MASTER_ADDR / MASTER_PORT / UCX_TLS を起動シェルから継承する
export MASTER_ADDR=${MASTER_ADDR:-172.16.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export UCX_TLS=${UCX_TLS:-rc,sm,self}
export BENCH_NO_PAUSE=1              # 末尾の input() を抑止（連続実行のため）

# 構成: "name|GDR|NCHANNELS"   ('-' は未設定=既定)
CONFIGS=(
  "default|SYS|-"
  "nch8_gdron|SYS|8"
  "nch8_gdroff|-|8"
  "nch16_gdron|SYS|16"
  "nch16_gdroff|-|16"
)

mkdir -p "$OUT_DIR"
echo "config,gdr,nchannels,rep,coll,N,size_label,avg_ms,p50_ms,p99_ms,bw_gbps" > "$CSV"

# ---- 事前チェック -----------------------------------------------------------
for v in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS; do
  if [[ -n "${!v:-}" ]]; then
    echo "!! シェルに $v=${!v} が設定されています。スイープが汚染されるので unset してください。"
    exit 1
  fi
done

# ---- 一時 appfile 生成 (GDR は appfile に固定値があるので sed で差し替える) --
make_appfile() {
  local name=$1 gdr=$2 rep=$3 tmp="$OUT_DIR/appfile.$name"
  if [[ "$gdr" == "-" ]]; then
    sed -e "s/-x NCCL_NET_GDR_LEVEL=SYS //g" \
        -e "s|-x NCCL_DEBUG_FILE=[^ ]*|-x NCCL_DEBUG_FILE=$OUT_DIR/nccl_${name}_r${rep}.%h.%p.log|g" \
        "$APPFILE" > "$tmp"
  else
    sed -e "s/-x NCCL_NET_GDR_LEVEL=[^ ]*/-x NCCL_NET_GDR_LEVEL=$gdr/g" \
        -e "s|-x NCCL_DEBUG_FILE=[^ ]*|-x NCCL_DEBUG_FILE=$OUT_DIR/nccl_${name}_r${rep}.%h.%p.log|g" \
        "$APPFILE" > "$tmp"
  fi
  echo "$tmp"
}

# ---- ログから [GPU RS/AG NCCL] を抽出 → "coll,N,avg,p50,p99,bw" -------------
#      先頭一致 + avg= 必須にして [ERROR] 行を拾わない
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

echo "[stage3] configs=${#CONFIGS[@]} reps=$REPS sizes=$BENCH_SIZES_CSV out=$OUT_DIR"
echo "[stage3] DPU 不要 (BENCH_RUN=nccl は DOCA 初期化を省略)"
echo "[stage3] rep を外側ループにして構成間のドリフト偏りを避ける"

for rep in $(seq 1 "$REPS"); do
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name gdr nch <<< "$cfg"
    log="$OUT_DIR/${name}_r${rep}.log"
    echo "============================================================"
    echo "==== [rep $rep/$REPS] $name (GDR=$gdr NCHANNELS=$nch) ===="
    tmp=$(make_appfile "$name" "$gdr" "$rep")

    unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
    [[ "$nch" != "-" ]] && { export NCCL_MIN_NCHANNELS="$nch"; export NCCL_MAX_NCHANNELS="$nch"; }

    ( cd "$HOST_DIR" && BENCH_RUN=nccl BENCH_SIZES="$BENCH_SIZES_CSV" \
        mpirun --app "$tmp" < /dev/null ) > "$log" 2>&1
    rc=$?
    [[ $rc -ne 0 ]] && { echo "  !! rc=$rc。末尾:"; tail -5 "$log"; }

    # 設定が本当に効いたかログで検証（GDR と実チャネル数）
    gdr_act=$(grep -ho "use ring PXN [0-9] GDR [0-9]" "$OUT_DIR/nccl_${name}_r${rep}."*.log 2>/dev/null | tail -1)
    ch_act=$(grep -ho "[0-9]* coll channels" "$OUT_DIR/nccl_${name}_r${rep}."*.log 2>/dev/null | tail -1)
    echo "  実測設定: ${gdr_act:-GDR不明} / ${ch_act:-チャネル数不明}"
    if grep -q "mismatch" "$log"; then
      echo "  !! 正当性エラーあり (この構成の数値は採用しないこと)"
    fi

    cnt=0
    while IFS=, read -r coll n avg p50 p99 bw; do
      [[ -z "$n" ]] && continue
      echo "$name,$gdr,$nch,$rep,$coll,$n,${SIZE_LABEL[$n]:-$n},$avg,$p50,$p99,$bw" >> "$CSV"
      cnt=$((cnt+1))
    done < <(parse_log "$log")
    echo "  取得: ${cnt} 行 → $log"
  done
done

# ---- 反復を集約（中央値）+ ばらつき ----------------------------------------
echo
echo "========== 集約 (reps=$REPS の中央値) =========="
if command -v python3 >/dev/null 2>&1; then
python3 - "$CSV" "$AGG" <<'PYEOF'
import csv, io, sys, statistics as st
src, dst = sys.argv[1], sys.argv[2]
raw = [l for l in open(src) if "ERROR" not in l]   # 壊れ行(カンマ混入)を除去
rows = list(csv.DictReader(io.StringIO("".join(raw))))
if not rows:
    print("(データなし)"); sys.exit(0)
cfgs, keys = [], []
for r in rows:
    if r["config"] not in cfgs: cfgs.append(r["config"])
    k = (r["coll"], r["size_label"])
    if k not in keys: keys.append(k)
acc = {}
for r in rows:
    acc.setdefault((r["config"], r["coll"], r["size_label"]), []).append(
        (float(r["p50_ms"]), float(r["p99_ms"]), float(r["bw_gbps"])))
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config","coll","size_label","n_rep",
                "p50_med","p50_min","p50_max","p99_med","p99_min","p99_max","bw_med"])
    for c in cfgs:
        for coll, sz in keys:
            v = acc.get((c, coll, sz))
            if not v: continue
            p50s=[x[0] for x in v]; p99s=[x[1] for x in v]; bws=[x[2] for x in v]
            w.writerow([c,coll,sz,len(v),
                        f"{st.median(p50s):.3f}",f"{min(p50s):.3f}",f"{max(p50s):.3f}",
                        f"{st.median(p99s):.3f}",f"{min(p99s):.3f}",f"{max(p99s):.3f}",
                        f"{st.median(bws):.2f}"])
def show(metric, idx):
    print(f"\n=== {metric} 中央値 (ms) と default 比 Δ%  [min–max も併記] ===")
    print(f"{'coll/size':11s}" + "".join(f"{c:>26s}" for c in cfgs))
    for coll, sz in keys:
        base = acc.get(("default", coll, sz))
        b = st.median([x[idx] for x in base]) if base else None
        line = f"{coll+'/'+sz:11s}"
        for c in cfgs:
            v = acc.get((c, coll, sz))
            if not v: line += f"{'-':>26s}"; continue
            m = st.median([x[idx] for x in v]); lo=min(x[idx] for x in v); hi=max(x[idx] for x in v)
            d = "" if (c=="default" or not b) else f"({(m-b)/b*100:+.0f}%)"
            line += f"{m:9.3f}{d:>7s}[{lo:.2f}-{hi:.2f}]".rjust(26)
        print(line)
show("p50", 0); show("p99", 1)
print(f"\n[written] {dst}")
PYEOF
fi
echo "=================================================="
echo "生データ: $CSV"
echo "集約表  : $AGG"
