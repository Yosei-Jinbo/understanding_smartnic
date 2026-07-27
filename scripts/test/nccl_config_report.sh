#!/bin/bash
###############################################################################
# nccl_config_report.sh — E1 段1: NCCL の「解決された設定」を取得して報告する
#
# 目的（査読 C7）:
#   「関連する NCCL 設定、(可能であれば) プロトコル/チャネルの挙動、CPU プロキシ
#     スレッドの挙動、CPU 使用率を報告してほしい」
#   → 追加のチューニングをせず、**既定設定が何に解決されたか**を開示する。
#
# やること:
#   1. host_appfile_py から一時 appfile を生成（NCCL_DEBUG_SUBSYS=ALL + 専用ログ dir）
#      ※ 元の appfile は書き換えない
#   2. マイクロベンチを 1 回実行（NCCL arm を含む）
#   3. 両ノードの NCCL ログを回収し、algo / proto / チャネル数 / GDR / プロキシ
#      スレッド / NIC を抽出して表示
#
# 前提:
#   - host 側 venv を activate 済み
#   - NCCL_ALGO / NCCL_PROTO をシェルで export していないこと（下で検出して警告する）
#
# DPU について（重要）:
#   client.py は NCCL arm のみでも起動時に DOCA comch を初期化するため、
#   **DPU 側 collective_server が「新鮮な状態で」起動している必要がある**。
#   サーバは切断イベントで finish=true になり以後接続を受けないので、
#   ラン毎に再起動が要る。本スクリプトは既定で「手動起動」(プロンプトで待つ)。
#
# 使い方:
#   bash /home/y-jinbo/understanding_smartnic/scripts/test/nccl_config_report.sh
#   # サイズを変える場合:  BENCH_SIZES=8388608 bash ...
#   # DOCA も一緒に測る場合: BENCH_RUN=all bash ...   (既定は nccl_ag → nccl_rs の 2 回)
#   # DPU 自動起動を試す場合(非推奨): MANUAL_DPU=0 RESTART_DPU=1 bash ...
###############################################################################
set -uo pipefail

REPO=/home/y-jinbo/understanding_smartnic
HOST_DIR=$REPO/src/smartnic_offload/comch_mpi/host
APPFILE=$HOST_DIR/host_appfile_py
OUT_DIR=$REPO/logs/nccl_config
TMP_APPFILE=$OUT_DIR/host_appfile_py.subsys_all
PEER_HOST=bluefield02

BENCH_SIZES=${BENCH_SIZES:-8388608}     # 表A の代表点 (8MB)
BENCH_RUN=${BENCH_RUN:-}                # 空なら nccl_ag → nccl_rs を順に実行

# host_appfile_py は MASTER_ADDR / MASTER_PORT / UCX_TLS を bare `-x` で
# 「起動シェルから継承」する形になっている。スクリプト単体で完結させるため既定値を与える。
# (値は scripts/baseline/run_*_prefetch.sh の torchrun 引数と同じ)
export MASTER_ADDR=${MASTER_ADDR:-172.16.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export UCX_TLS=${UCX_TLS:-rc,sm,self}
# DPU 起動方法:
#   1 = 手動（既定。プロンプトで停止し、dpu01 で起動してもらう）
#       ※ sweep_ablation.sh / sweep_multicore.sh と同じ MANUAL_DPU=1 の流儀。
#   0 = ssh 自動起動。**現状うまく動かない**（下記 KNOWN ISSUE）ので推奨しない。
MANUAL_DPU=${MANUAL_DPU:-1}
#
# KNOWN ISSUE (2026-07-27):
#   `ssh -f ubuntu@dpu01 "bash -lc 'nohup mpirun --app dpu_appfile &'"` で自動起動すると、
#   DPU サーバの初期化までは完了する（worker pool / RS pool までログに出る）が、
#   ホスト側が `execute ucp create ring notify` で無限に待つ。
#   ring 生成は DPU 側 4 ランクの集団処理を伴うため、ssh 切断で mpirun のジョブ制御が
#   壊れて集団処理が完了しないものと考えられる。手動起動（端末に mpirun を残す）では発生しない。
RESTART_DPU=${RESTART_DPU:-0}
DPU_HOST=dpu01
DPU_DIR=/home/ubuntu/doca_practice/comch_mpi/dpu
DPU_BIN=/tmp/build/doca_comch_server

mkdir -p "$OUT_DIR"

# DPU 側 collective_server を再起動して待受状態にする。
#   dpu_appfile が dpu01 から dpu01/dpu02 の 2 ランクずつを起動する。
#   env は採用済みの既定値を **明示** する (シェル残留による誤測定を防ぐ)。
restart_dpu() {
  echo "  DPU 停止中..."
  ssh -o ConnectTimeout=8 ubuntu@dpu01 "pkill -f '$DPU_BIN'; pkill -f 'mpirun --app dpu_appfile'" 2>/dev/null
  ssh -o ConnectTimeout=8 ubuntu@dpu02 "pkill -f '$DPU_BIN'" 2>/dev/null
  sleep 3
  echo "  DPU 起動中 (COMM_CORES=3 COMPUTE_CORES=6 AG_PIECE_MAX=8 FORCE_STAGING=0 FORCE_SINGLE_RAIL=1)..."
  # 非対話 ssh では mpirun が PATH に無いのでログインシェル経由で起動する
  ssh -o ConnectTimeout=8 -f ubuntu@"$DPU_HOST" \
    "bash -lc 'cd $DPU_DIR && COMM_CORES=3 COMPUTE_CORES=6 AG_PIECE_MAX=8 FORCE_STAGING=0 FORCE_SINGLE_RAIL=1 \
     nohup mpirun --app dpu_appfile > /tmp/dpu_nccl_config.log 2>&1 &'" 2>/dev/null
  # 待受になるまで待つ (両ノードで 2 プロセスずつ)
  for i in $(seq 1 30); do
    sleep 2
    n1=$(ssh -o ConnectTimeout=5 ubuntu@dpu01 "pgrep -c -f '$DPU_BIN' 2>/dev/null || echo 0" 2>/dev/null)
    n2=$(ssh -o ConnectTimeout=5 ubuntu@dpu02 "pgrep -c -f '$DPU_BIN' 2>/dev/null || echo 0" 2>/dev/null)
    if [[ "${n1:-0}" -ge 2 && "${n2:-0}" -ge 2 ]]; then
      echo "  DPU 待受 OK (dpu01=$n1, dpu02=$n2 プロセス)"; sleep 3; return 0
    fi
  done
  echo "  !! DPU が待受状態になりませんでした (dpu01=${n1:-?}, dpu02=${n2:-?})"
  echo "     dpu01 で手動起動してください: cd $DPU_DIR && mpirun --app dpu_appfile"
  return 1
}

# ---- 0. 事前チェック -------------------------------------------------------
echo "============================================================"
echo "[0] 事前チェック"
leak=0
for v in NCCL_ALGO NCCL_PROTO; do
  if [[ -n "${!v:-}" ]]; then
    echo "  !! シェルに $v=${!v} が設定されています。"
    echo "     E1 は『真のデフォルト』を測るので unset してから実行してください:"
    echo "       unset $v"
    leak=1
  fi
done
[[ $leak -eq 0 ]] && echo "  OK: NCCL_ALGO / NCCL_PROTO はシェルに設定されていない"
if [[ $leak -ne 0 ]]; then exit 1; fi
echo "  ログ出力先: $OUT_DIR"

# ---- 1. 一時 appfile の生成 (元ファイルは書き換えない) ----------------------
echo "[1] SUBSYS=ALL の一時 appfile を生成: $TMP_APPFILE"
sed -e "s|-x NCCL_DEBUG_SUBSYS=[^ ]*|-x NCCL_DEBUG_SUBSYS=ALL|g" \
    -e "s|-x NCCL_DEBUG_FILE=[^ ]*|-x NCCL_DEBUG_FILE=$OUT_DIR/coll.%h.%p.log|g" \
    "$APPFILE" > "$TMP_APPFILE"
echo -n "  変更確認: "; tr ' ' '\n' < "$TMP_APPFILE" | grep -c "NCCL_DEBUG_SUBSYS=ALL" | xargs echo "SUBSYS=ALL のランク数 ="

# 相手ノードにも出力先を作る (home は共有 FS ではない)
ssh -o ConnectTimeout=8 "$PEER_HOST" "mkdir -p $OUT_DIR" 2>/dev/null \
  && echo "  $PEER_HOST 側の出力先を作成" || echo "  !! $PEER_HOST に ssh できず (後でログ回収を手動で)"

# 古いログを退避
rm -f "$OUT_DIR"/coll.*.log
ssh -o ConnectTimeout=8 "$PEER_HOST" "rm -f $OUT_DIR/coll.*.log" 2>/dev/null

# ---- 2. マイクロベンチ実行 --------------------------------------------------
run_bench() {
  local sel=$1
  echo "------------------------------------------------------------"
  echo "[2] マイクロベンチ実行: BENCH_RUN=$sel BENCH_SIZES=$BENCH_SIZES"
  # collective_server は切断で finish=true になるため、**ラン毎に再起動が必要**
  if [[ "$MANUAL_DPU" == "1" ]]; then
    echo "============================================================"
    echo "[DPU] dpu01 で以下を実行して collective_server を起動してください:"
    echo "  cd $DPU_DIR"
    echo "  export COMM_CORES=3 COMPUTE_CORES=6 AG_PIECE_MAX=8 FORCE_STAGING=0 FORCE_SINGLE_RAIL=1"
    echo "  mpirun --app dpu_appfile"
    echo "  # ※ 前のランで起動したサーバは終了しているので、毎回起動し直すこと"
    echo "------------------------------------------------------------"
    read -r -p "DPU が待受状態になったら Enter: " _
  elif [[ "$RESTART_DPU" == "1" ]]; then
    restart_dpu || return 1
  fi
  ( cd "$HOST_DIR" && BENCH_RUN="$sel" BENCH_SIZES="$BENCH_SIZES" \
      mpirun --app "$TMP_APPFILE" < /dev/null ) 2>&1 | tee "$OUT_DIR/bench_${sel}.log"
  # 接続失敗を早期に検出して明示する (segfault の生ログだけだと原因が分かりにくい)
  if grep -q "Failed to comch send control cmd" "$OUT_DIR/bench_${sel}.log"; then
    echo "  !! DPU への接続に失敗しました (collective_server が待受状態でない)。"
    echo "     dpu01 で再起動してから RESTART_DPU=0 で再実行してください。"
    return 1
  fi
}
if [[ -n "$BENCH_RUN" ]]; then
  run_bench "$BENCH_RUN"
else
  run_bench nccl_ag
  run_bench nccl_rs
fi

# ---- 3. 相手ノードのログを回収 ---------------------------------------------
echo "------------------------------------------------------------"
echo "[3] $PEER_HOST の NCCL ログを回収"
scp -q "$PEER_HOST:$OUT_DIR/coll.*.log" "$OUT_DIR/" 2>/dev/null \
  && echo "  回収完了" || echo "  !! 回収失敗 (bluefield02 のログは現地に残っています)"
echo -n "  収集できたログ数: "; ls -1 "$OUT_DIR"/coll.*.log 2>/dev/null | wc -l

# ---- 4. 解決値の抽出 --------------------------------------------------------
LOG=$(ls -1 "$OUT_DIR"/coll.*.log 2>/dev/null | head -1)
if [[ -z "$LOG" ]]; then
  echo "!! NCCL ログが見つかりません。NCCL_DEBUG_FILE の書き込み権限を確認してください。"
  exit 1
fi

sec() { echo; echo "── $1 ──"; }
echo
echo "############################################################"
echo "# 表A 段1: NCCL の解決された設定  (代表ログ: $(basename "$LOG"))"
echo "############################################################"

sec "バージョン"
grep -hoE "NCCL version [^ ]+.*" "$LOG" | sort -u | head -2

sec "環境変数から設定された値 (=これ以外はすべて既定)"
grep -h "set by environment" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u

sec "アルゴリズム / プロトコル"
grep -hiE "NCCL INFO.*(Algo|Proto|PAT|Ring [0-9]|Trees? \[|CollNet|NVLS)" "$OUT_DIR"/coll.*.log \
  | sed 's/.*NCCL INFO //' | sort -u | head -20
echo "  (上に proto が出ない場合、そのサイズでは既定チューナの選択がログに現れていない)"

sec "チャネル数"
grep -h "coll channels" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u
grep -hE "Pattern [0-9]+, crossNic" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u

sec "GPUDirect RDMA (実行時の判断)"
grep -hE "GPU Direct RDMA|use ring PXN" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u
echo "  → 'GDR 1' なら有効、'GDR 0' なら無効"

sec "NIC / トランスポート"
grep -hE "NET/IB : Using|Made virtual device|NET/IB: \[" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u | head -8

sec "CPU プロキシスレッド (スレッド → コア)"
grep -hE "\[Proxy (Service|Progress)" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u

sec "バッファサイズ / しきい値"
grep -hE "Chunksize|threadThresholds|Buffsize|BUFFSIZE" "$OUT_DIR"/coll.*.log | sed 's/.*NCCL INFO //' | sort -u | head -6

sec "ベンチマーク結果 (表A の数値)"
grep -hE "\[GPU (AG|RS) NCCL\]|\[DOCA (AG|RS) flat\]" "$OUT_DIR"/bench_*.log | sort -u

echo
echo "############################################################"
echo "ログ一式: $OUT_DIR"
echo "  NCCL 詳細ログ : coll.<host>.<pid>.log"
echo "  ベンチ標準出力 : bench_<run>.log"
echo "############################################################"
