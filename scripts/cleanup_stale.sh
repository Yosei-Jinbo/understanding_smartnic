#!/bin/bash
###############################################################################
# cleanup_stale.sh — 実行前に残留プロセス／GPU メモリを掃除する
#
# なぜ必要か:
#   USE_GPU_FLAG_WAIT=1 のランは GPU ストリームが cuStreamWaitValue32 で
#   DPU のフラグを待つ。DPU 側の RDMA が壊れるとフラグが永遠に立たず、
#   **SIGINT/SIGTERM では止まらない**（GPU ストリーム待ちは割り込めない）。
#   さらに nsys ラッパ配下だと mpirun の kill が子プロセスまで届かない。
#   結果、python が GPU を 13 GiB 掴んだまま残り、次のランが
#     torch.OutOfMemoryError: ... Process <PID> has 12.99 GiB memory in use
#   で落ちる（2026-07-28 に実際に発生。3 世代分の残留が溜まった）。
#
#   nvidia-smi の compute-apps が空であることを確認せずに再実行すると、
#   同じ OOM を無限に繰り返すことになる。
#
# 使い方:
#   bash scripts/cleanup_stale.sh            # ホストのみ、確認あり
#   bash scripts/cleanup_stale.sh -y         # 確認なし（スクリプトから呼ぶ用）
#   bash scripts/cleanup_stale.sh -y --dpu   # DPU の collective_server も落とす
#   bash scripts/cleanup_stale.sh -n         # 何も殺さず現状確認だけ (dry-run)
#
# 終了コード:
#   0  = クリーン（GPU に compute プロセスなし）
#   1  = 掃除しても GPU を掴んだままのプロセスが残っている（手動対応が必要）
###############################################################################
set -u

HOSTS=${CLEANUP_HOSTS:-"bluefield01 bluefield02"}
DPUS=${CLEANUP_DPUS:-"dpu01 dpu02"}
DPU_USER=${CLEANUP_DPU_USER:-ubuntu}
SELF=$(hostname -s)

ASSUME_YES=0
DO_DPU=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    -y|--yes)  ASSUME_YES=1 ;;
    --dpu)     DO_DPU=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# 残留とみなすプロセス。ここに書いたものだけを殺す。
# 注意: パターンを引数に置くと ssh/bash 自身の argv に載って自分を巻き込むため、
#       payload は必ず **標準入力** から渡す（argv に一切出さない）。
read -r -d '' INSPECT_PAYLOAD <<'PAYLOAD'
set -u
echo "  [GPU]"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
  | sed 's/^/    /' || echo "    (nvidia-smi なし)"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
  && echo "    (compute プロセスなし)"
echo "  [残留プロセス]"
found=0
for pat in 'run_zero_mpi\.py' 'nsight-systems' 'nsys profile' 'prted' 'orted' 'host_appfile'; do
  pids=$(pgrep -f "$pat" 2>/dev/null)
  for p in $pids; do
    [ "$p" = "$$" ] && continue
    printf '    %-8s %-9s %s\n' "$p" "$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')" \
      "$(ps -o args= -p "$p" 2>/dev/null | cut -c1-72)"
    found=1
  done
done
[ "$found" = 0 ] && echo "    (なし)"
echo "  [/dev/shm]"
ls /dev/shm 2>/dev/null | grep -i -E 'nsys|nsight' | sed 's/^/    /' || true
ls /dev/shm 2>/dev/null | grep -qi -E 'nsys|nsight' || echo "    (なし)"
PAYLOAD

read -r -d '' KILL_PAYLOAD <<'PAYLOAD'
set -u
for pat in 'run_zero_mpi\.py' 'nsight-systems' 'nsys profile' 'prted' 'orted' 'host_appfile'; do
  pids=$(pgrep -f "$pat" 2>/dev/null)
  for p in $pids; do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null && echo "    killed $p"
  done
done
rm -f /dev/shm/sem.NSys-* /dev/shm/nsys* /dev/shm/NSys* 2>/dev/null
# GPU が解放されるまで待つ（プロセス消滅とドライバの解放にはラグがある）
for i in $(seq 1 20); do
  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$left" -eq 0 ] && break
  sleep 1
done
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$left" -eq 0 ]; then
  echo "    GPU: クリーン"
  exit 0
else
  echo "    GPU: ★まだ $left プロセスが占有中"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/      /'
  exit 1
fi
PAYLOAD

# $1=host, $2=payload  — payload は stdin 経由（argv に載せない）
run_on() {
  local host=$1 payload=$2
  if [ "$host" = "$SELF" ]; then
    printf '%s\n' "$payload" | bash -s
  else
    printf '%s\n' "$payload" | ssh -o ConnectTimeout=10 "$host" bash -s
  fi
}

echo "=============== 現状 ==============="
for h in $HOSTS; do
  echo "--- $h ---"
  run_on "$h" "$INSPECT_PAYLOAD"
done

if [ "$DO_DPU" = 1 ]; then
  for d in $DPUS; do
    printf -- "--- %s --- collective_server: " "$d"
    ssh -o ConnectTimeout=10 "$DPU_USER@$d" \
      'n=$(pgrep -cf doca_comch_server); [ "$n" -gt 0 ] && echo "$n 個 起動中" || echo なし' 2>/dev/null \
      || echo "(接続失敗)"
  done
fi

if [ "$DRY_RUN" = 1 ]; then
  echo; echo "(dry-run: 何も殺していません)"
  exit 0
fi

if [ "$ASSUME_YES" != 1 ]; then
  echo
  read -r -p "上記を kill -9 します。よろしいですか? [y/N] " ans
  case "$ans" in y|Y|yes) ;; *) echo "中止"; exit 2 ;; esac
fi

echo
echo "=============== 掃除 ==============="
rc=0
for h in $HOSTS; do
  echo "--- $h ---"
  run_on "$h" "$KILL_PAYLOAD" || rc=1
done

if [ "$DO_DPU" = 1 ]; then
  for d in $DPUS; do
    printf -- "--- %s --- " "$d"
    ssh -o ConnectTimeout=10 "$DPU_USER@$d" \
      'pkill -9 -f doca_comch_server 2>/dev/null; sleep 1; echo "collective_server 停止"' 2>/dev/null \
      || echo "(接続失敗)"
  done
  echo
  echo "※ DPU を落としました。再測定前に dpu_appfile で起動し直してください:"
  echo "   cd src/smartnic_offload/comch_mpi/dpu"
  echo "   export COMM_CORES=2 COMPUTE_CORES=8 AG_PIECE_MAX=8 RS_PREPOST=1 FORCE_STAGING=0 FORCE_SINGLE_RAIL=1"
  echo "   mpirun --bind-to none --app dpu_appfile 2>&1 | tee /tmp/dpu_<model>_<cfg>.log"
  echo "   # 起動ログの core alloc 行が **4 ランクすべて** 出ることを確認"
fi

echo
if [ "$rc" -eq 0 ]; then
  echo "=== 全ホストでクリーン。測定を開始できます ==="
else
  echo "=== ★GPU を掴んだままのプロセスが残っています ==="
  echo "    kill -9 で落ちない場合はカーネル側で D 状態の可能性があります。"
  echo "    sudo fuser -v /dev/nvidia* で保持者を確認し、最終手段としてノード再起動。"
fi
exit "$rc"
