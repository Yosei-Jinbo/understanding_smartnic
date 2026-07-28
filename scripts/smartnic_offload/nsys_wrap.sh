#!/bin/bash
###############################################################################
# nsys_wrap.sh — mpirun appfile から nsys を起動するためのラッパ
#
# なぜ必要か:
#   Open MPI の appfile は **1 行 1024 文字を超えると行が切り詰められ**、
#   `-host X:1` などの解釈が壊れて
#       "All nodes which are allocated for this job are already filled"
#   というスロットエラーになる（実測で確定: 1017 文字は成功 / 1095 文字は失敗）。
#   nsys のオプション列をインラインで書くと簡単に 1024 を超えるため、
#   ここへ追い出して appfile 側の行を短く保つ。
#
# 使い方（appfile 内）:
#   ... -x ... <このスクリプト> <profile 名> <実行するコマンド...>
#   例) .../nsys_wrap.sh opt_ag_smartnic_allrank /home/y-jinbo/.venv/bin/python run_zero_mpi.py ...
#
# ランク別の設定は環境変数から自動決定する（appfile を 4 行とも同一にできる）:
#   - 出力名          : <NSYS_OUT_DIR>/<profile 名>_rank<RANK>
#   - GPU metrics 対象 : local_rank = RANK % GPU数   （run_zero_mpi.py と同じ規則）
#
# 環境変数:
#   NSYS_OUT_DIR      出力先ディレクトリ (既定: <repo>/logs/nsys)
#   NSYS_GPU_METRICS  0 で GPU metrics を無効化 (既定: 1)
#   NSYS_TRACE        --trace の値 (既定: cuda,nvtx)
#   NSYS_SAMPLE       --sample の値 (既定: none)
#   NSYS_CPUCTXSW     --cpuctxsw の値 (既定: none)
#   NSYS_CAPTURE_END  --capture-range-end の値 (既定: stop / 理由は下の注記)
#   NSYS_BIN          nsys の場所 (既定: /usr/local/cuda/bin/nsys)
###############################################################################
set -u

PROFILE=${1:?usage: nsys_wrap.sh <profile-name> <command...>}
shift

NSYS_BIN=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
OUT_DIR=${NSYS_OUT_DIR:-/home/y-jinbo/understanding_smartnic/logs/nsys}
TRACE=${NSYS_TRACE:-cuda,nvtx}
SAMPLE=${NSYS_SAMPLE:-none}
CPUCTXSW=${NSYS_CPUCTXSW:-none}

# --capture-range-end は既定を 'stop' にする。
#
#   'stop-shutdown' は cudaProfilerStop() の瞬間に **対象アプリを強制終了する**
#   （nsys --help は 'stop' にだけ "Target app will continue running." と書いており、
#     この非対称は仕様。最小スクリプトで実測確認済み）。
#   全ランク profiling でこれを使うと:
#     1. 最初に stop に到達したランクの python が殺される
#     2. 残りのランクは集団通信で死んだピアを待って停止
#     3. mpirun が異常終了を検知してジョブ全体を SIGKILL
#     4. 書き出し途中の nsys が道連れになり、.qdstrm だけが /tmp に残る
#   実際 2026-07-28 の 4 ランク測定では rank1 の 1 本しかレポートが残らなかった。
#
#   さらに run_zero.py の NSYS_PROFILE_MEASURE_ITERS（「残り数 iter は nsys 抜きで
#   走らせて末尾の hang からレポートを守る」という設計）は **アプリが生き残る前提**
#   なので、'stop-shutdown' とは論理的に両立しない。
#
#   代償: 終了処理が hang すると nsys が待ち続けてレポートが出ない。その場合も
#   /tmp/nsys-report-*.qdstrm にデータは残るので QdstrmImporter で救出できる。
CAPTURE_END=${NSYS_CAPTURE_END:-stop}

# ランク: mpirun / torchrun のどちらでも拾えるように
RANK=${OMPI_COMM_WORLD_RANK:-${RANK:-0}}

mkdir -p "$OUT_DIR"

GPU_OPT=()
if [[ "${NSYS_GPU_METRICS:-1}" != "0" ]]; then
  # local_rank = RANK % GPU数（run_zero_mpi.py の local_rank 決定則と同一）
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [[ "$NGPU" -lt 1 ]] && NGPU=1
  DEV=$(( RANK % NGPU ))
  GPU_OPT=(--gpu-metrics-devices="$DEV"
           --gpu-metrics-frequency=10000
           --gpu-metrics-set=ga10x)
fi

exec "$NSYS_BIN" profile \
  --trace="$TRACE" \
  --sample="$SAMPLE" \
  --cpuctxsw="$CPUCTXSW" \
  "${GPU_OPT[@]}" \
  --capture-range=cudaProfilerApi \
  --capture-range-end="$CAPTURE_END" \
  --force-overwrite=true \
  -o "$OUT_DIR/${PROFILE}_rank${RANK}" \
  "$@"
