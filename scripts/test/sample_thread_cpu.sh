#!/bin/bash
###############################################################################
# sample_thread_cpu.sh — スレッド別の CPU 使用率と実行コアを /proc からサンプルする
#
# 目的（E2 / 査読 C7）:
#   NCCL proxy スレッドが「1 コアを飽和させていないか」「GPU/NIC と同じ NUMA にいるか」を
#   実測する。NCCL ログの `[Proxy Progress] ... CPU core N` は sched_getcpu() のスナップショットで
#   affinity 適用前の値の可能性があるため、定常状態を /proc から直接見る。
#
# 測り方の要点:
#   - /proc/<pid>/task/<tid>/stat の utime(14) / stime(15) / processor(39) を使う。
#     ただし comm フィールドは括弧内に空白を含み得るため、`)` までを削ってから数える
#     （削った後は index が 2 ずれる: utime=12, stime=13, processor=37）。
#   - proxy は集合通信が飛んでいる間だけ busy-poll する。全区間の平均では飽和が見えないので、
#     短い間隔でサンプルし **区間ごとの最大 CPU%** も出す。
#
# 使い方:
#   scripts/test/sample_thread_cpu.sh -m client.py            # プロセス名で検索
#   scripts/test/sample_thread_cpu.sh -p 12345                # PID 直指定
#   scripts/test/sample_thread_cpu.sh -m run_zero_mpi.py -i 0.2 -n 300 -o out.csv
#
# 出力:
#   標準出力 : スレッド別サマリ（平均 CPU% / 最大 CPU% / 使用コア / NUMA）
#   -o 指定時: 生サンプル CSV (ts,tid,comm,core,utime,stime)
###############################################################################
set -uo pipefail

PID=""; PAT=""; INT=0.2; N=150; OUT=""; SHOWALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PID=$2; shift 2;;
    -m) PAT=$2; shift 2;;
    -i) INT=$2; shift 2;;      # サンプル間隔(秒)
    -n) N=$2;   shift 2;;      # サンプル回数
    -o) OUT=$2; shift 2;;
    -a) SHOWALL=1; shift;;      # 低使用率スレッドも省略しない
    *) echo "unknown option: $1"; exit 1;;
  esac
done

if [[ -z "$PID" ]]; then
  [[ -z "$PAT" ]] && { echo "-p <pid> か -m <pattern> が必要"; exit 1; }
  PID=$(pgrep -f "$PAT" | head -1)
  [[ -z "$PID" ]] && { echo "プロセスが見つからない: $PAT"; exit 1; }
fi
[[ -d /proc/$PID ]] || { echo "PID $PID が存在しない"; exit 1; }

CLK=$(getconf CLK_TCK)
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Ctrl-C されても「そこまでのサンプル」で集計する（集計はループ後にあるため）
STOP=0
trap 'STOP=1; echo; echo "[sample] 中断 — ここまでのサンプルで集計します"' INT

echo "[sample] pid=$PID interval=${INT}s count=$N (最大 $(awk -v a="$INT" -v b="$N" 'BEGIN{printf "%.0f", a*b}')s)  CLK_TCK=$CLK"
echo "[sample] cmd: $(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | cut -c1-90)"
echo "[sample] 収集中... (Ctrl-C でいつでも打ち切って集計できます)"

got=0
for ((i=0; i<N; i++)); do
  [[ $STOP -eq 1 ]] && break
  [[ -d /proc/$PID ]] || { echo "[sample] プロセスが終了しました (i=$i)"; break; }
  ts=$(date +%s.%N)
  for t in /proc/$PID/task/*; do
    [[ -r "$t/stat" ]] || continue
    tid=${t##*/}
    name=$(cat "$t/comm" 2>/dev/null) || continue
    st=$(cat "$t/stat" 2>/dev/null) || continue
    rest=${st#*") "}                       # "pid (comm) " を除去 → 以降 index が 2 ずれる
    # shellcheck disable=SC2086
    set -- $rest
    echo "$ts,$tid,$name,${37},${12},${13}" >> "$TMP"
  done
  got=$((got+1))
  # 2 秒ごとに進捗（動いているか分かるように）
  if (( got % 10 == 0 )); then printf "\r[sample] %d/%d サンプル" "$got" "$N"; fi
  sleep "$INT"
done
echo
if (( got < 2 )); then
  echo "[sample] サンプルが $got 件しかありません。CPU% は 2 点以上必要です（もう少し長く回してください）。"
  exit 1
fi
echo "[sample] $got サンプル取得"

[[ -n "$OUT" ]] && { echo "ts,tid,comm,core,utime,stime" > "$OUT"; cat "$TMP" >> "$OUT"; echo "[written] $OUT"; }

echo
echo "================ スレッド別 CPU 使用率 ================"
awk -F, -v clk="$CLK" -v showall="$SHOWALL" '
{
  key=$2;
  if (!(key in first_ts)) { first_ts[key]=$1; first_c[key]=$5+$6; nm[key]=$3 }
  if (key in prev_ts) {
    dt = $1 - prev_ts[key];
    dc = ($5+$6) - prev_c[key];
    if (dt > 0) { pct = dc/clk/dt*100; if (pct > mx[key]) mx[key]=pct }
  }
  prev_ts[key]=$1; prev_c[key]=$5+$6;
  cores[key","$4]=1;
}
END{
  printf "%-18s %-8s %8s %8s   %s\n", "thread", "tid", "avg%", "max%", "cores(NUMA)";
  for (k in first_ts) {
    dt = prev_ts[k] - first_ts[k];
    dc = prev_c[k] - first_c[k];
    avg = (dt>0) ? dc/clk/dt*100 : 0;
    isnccl = (nm[k] ~ /NCCL|nccl/);
    if (!showall && !isnccl && avg < 0.5 && mx[k] < 5) continue;   # NCCL 系と -a 指定は常に表示
    cl="";
    for (ck in cores) { split(ck,a,","); if (a[1]==k) { n=(a[2]<12)?0:1; cl=cl a[2] "(N" n ") " } }
    printf "%-18s %-8s %8.1f %8.1f   %s\n", nm[k], k, avg, mx[k], cl;
  }
  print "";
  print "読み方: max% が 100 に近いスレッドは 1 コアを飽和させている（律速の疑い）。";
  print "        cores の N0/N1 は NUMA ノード（GPU/NIC は N1 = cores 12-23）。";
  print "        avg% が低くても max% が高ければ、通信区間だけ飽和している可能性がある。";
}' "$TMP"
