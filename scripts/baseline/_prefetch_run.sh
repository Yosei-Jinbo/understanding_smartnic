###############################################################################
# _prefetch_run.sh — baseline (prefetch) の nsys ラン共通処理
#
# なぜ共通化したか:
#   run_*_nsys.sh の各スクリプトが torchrun 起動部をコピペで持っており、
#   nsys 設定の修正が一部にしか入らない事故が実際に起きた
#   （--capture-range-end=stop-shutdown、--gpu-metrics-devices=0 固定など）。
#   起動部はここ 1 箇所に集約し、各スクリプトはモデル固有値だけを持つ。
#
# nsys の設定は smartnic_offload 側と同じ scripts/smartnic_offload/nsys_wrap.sh に
# 委譲する。torchrun は RANK を環境変数に設定するので、ラッパの
#   RANK=${OMPI_COMM_WORLD_RANK:-${RANK:-0}}
# がそのまま global rank を拾い、GPU も RANK % GPU数 で torchrun の local_rank と一致する。
# これで mpirun 側と torchrun 側の nsys 設定が二度と食い違わない。
#
# 呼び出し側で設定する変数:
#   LABEL       出力プロファイル名 (例: opt_buf_prefetch)
#   MODEL       --model
#   DATASET     --dataset
#   BATCH_SIZE  --batch-size
#   SEQ_LEN     --seq-len (空なら渡さない: ViT など)
#   RB / PB / MLP
#               --reduce-bucket-size / --prefetch-bucket-size / --max-live-parameters
#   EXTRA_ARGS  run_zero.py に追加で渡す引数 (例: "--no-offload" で純粋 ZeRO-3)
#
# 使い方 (2 ノードで別々に起動する):
#   bluefield01: ./run_opt_1.3b_prefetch_nsys.sh 0
#   bluefield02: ./run_opt_1.3b_prefetch_nsys.sh 1
###############################################################################

REPO=/home/y-jinbo/understanding_smartnic
PYTHON=/home/y-jinbo/.venv/bin/python

run_prefetch() {
  local NODE_RANK=${1:?Usage: $0 <node_rank: 0=bluefield01, 1=bluefield02>}

  # NCCL は「真のデフォルト」で測る。対話シェルに残った export が torchrun に漏れるのを防ぐ。
  unset NCCL_ALGO NCCL_PROTO

  # 出力先は絶対パスにする。以前は cwd 相対の logs/nsys だったため、
  # 起動ディレクトリ次第で src/smartnic_offload/comch_mpi/host/logs/nsys などに散らばった。
  local OUT_DIR="$REPO/logs/nsys"
  mkdir -p "$OUT_DIR"
  export NSYS_OUT_DIR="$OUT_DIR"

  local SEQ_ARGS=()
  [[ -n "${SEQ_LEN:-}" ]] && SEQ_ARGS=(--seq-len "$SEQ_LEN")

  ENABLE_FULL_PARAM_TRANSFER=0 \
  USE_NVTX_RANGES=0 \
  STEP_NVTX=1 \
  DISABLE_COMPLETION_POLLER=1 \
  DISABLE_ADAM_FORK=1 \
  NSYS_PROFILE_MEASURE_ITERS=25 \
  NSYS_SYNC_RANGES=1 \
  NCCL_SOCKET_IFNAME=enp207s0f0np0,enp207s0f1np1 \
  NCCL_IB_DISABLE=0 \
  NCCL_NET_GDR_LEVEL=SYS \
  NCCL_P2P_LEVEL=SYS \
  NCCL_CUMEM_ENABLE=0 \
  torchrun \
    --nnodes=2 \
    --node_rank="$NODE_RANK" \
    --nproc_per_node=2 \
    --master_addr=172.16.0.1 \
    --master_port=29500 \
    --no-python \
    "$REPO/scripts/smartnic_offload/nsys_wrap.sh" "$LABEL" \
    "$PYTHON" "$REPO/src/baseline/run_zero.py" \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --batch-size "$BATCH_SIZE" \
      --warmup-iters 5 \
      --measure-iters 30 \
      "${SEQ_ARGS[@]}" \
      --reduce-bucket-size "$RB" \
      --prefetch-bucket-size "$PB" \
      --max-live-parameters "$MLP" \
      ${EXTRA_ARGS:-} \
    2>&1 | tee "$OUT_DIR/${LABEL}_node${NODE_RANK}.log"

  echo
  echo "[done] $(hostname -s) は rank $((NODE_RANK * 2)), $((NODE_RANK * 2 + 1)) を担当"
  ls -l "$OUT_DIR/${LABEL}"_rank*.nsys-rep 2>/dev/null \
    || echo "  ★このノードのレポートが出ていません"
  echo "  ※ 4 ランク揃ったかは両ノードで確認すること"
}
