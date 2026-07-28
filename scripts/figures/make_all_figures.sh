#!/bin/bash
###############################################################################
# make_all_figures.sh — 全モデルの idle 分解図を一括生成する
#
# 処理の流れ:
#   1. 各ノードで idle_decomp_allrank.py を走らせ、そのノードにあるランクを解析
#   2. JSON を bluefield01 に集約
#   3. plot_idle_decomp.py で 3 種類の図を生成
#
# ランクのノード配置は手法によって違う（prefetch は torchrun、ag_smartnic は
# mpirun appfile で rank→host の割当が逆）。ハードコードすると片方が壊れるので、
# 各ノードで <prefix>_rank*.sqlite を glob して「在るものを解析する」方式にしてある。
#
# 前提:
#   logs/nsys/<prefix>_rank<N>.sqlite が両ノードに存在すること
#   （nsys export --type sqlite で事前に変換しておく）
#
# 使い方:
#   bash scripts/figures/make_all_figures.sh                # 全モデル
#   bash scripts/figures/make_all_figures.sh opt_1_3b       # 指定モデルのみ
###############################################################################
set -u

REPO=/home/y-jinbo/understanding_smartnic
PY=/home/y-jinbo/.venv/bin/python
NSYS_DIR=$REPO/logs/nsys
JSON_DIR=$REPO/logs/idle_decomp
FIG_DIR=$REPO/scripts/figures
REMOTE=bluefield02
SELF=$(hostname -s)

mkdir -p "$JSON_DIR"

# model_key | prefetch prefix | offload prefix | label | ytick(busy) | ytick(idle) | ytick(other)
MODELS=(
  "opt_1_3b|opt_buf_prefetch|opt_ag_smartnic_allrank|OPT-1.3B|500|500|250"
  "deberta_xl|deberta_xl_buf_prefetch|deberta_xl_ag_smartnic|DeBERTa-XL|500|500|250"
  "vit_l_16|vit_l_16_buf_prefetch|vit_l_16_ag_smartnic|ViT-L/16|250|250|100"
)

# $1=prefix $2=出力 JSON 名の接頭辞
analyze() {
  local prefix=$1 tag=$2 out=()
  for host in "$SELF" "$REMOTE"; do
    local dst="$JSON_DIR/${tag}_${host}.json"
    if [ "$host" = "$SELF" ]; then
      local files
      files=$(ls "$NSYS_DIR/${prefix}"_rank*.sqlite 2>/dev/null)
      [ -z "$files" ] && continue
      $PY "$REPO/scripts/idle_decomp_allrank.py" --label "$tag" --json "$dst" $files \
        > /dev/null 2>&1 || { echo "  ★$host $prefix の解析失敗" >&2; continue; }
    else
      ssh "$host" "ls $NSYS_DIR/${prefix}_rank*.sqlite >/dev/null 2>&1 && \
        $PY $REPO/scripts/idle_decomp_allrank.py --label $tag \
          --json $NSYS_DIR/${tag}_remote.json \$(ls $NSYS_DIR/${prefix}_rank*.sqlite) \
          > /dev/null 2>&1" || continue
      scp -q "$host:$NSYS_DIR/${tag}_remote.json" "$dst" 2>/dev/null || continue
    fi
    out+=("$dst")
  done
  echo "${out[@]}"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r key pf_prefix ag_prefix label yb yi yo <<< "$spec"
  if [ $# -gt 0 ]; then
    case " $* " in *" $key "*) ;; *) continue ;; esac
  fi

  echo "################ $label ################"
  pf_json=$(analyze "$pf_prefix" "${key}_pf")
  ag_json=$(analyze "$ag_prefix" "${key}_ag")

  if [ -z "$pf_json" ] || [ -z "$ag_json" ]; then
    echo "  ★スキップ: JSON が揃わなかった (pf='$pf_json' ag='$ag_json')"
    continue
  fi

  # ylabel は LaTeX subfloat の左端パネルにだけ付ける（論文側の運用に合わせる）。
  # 既定の左端は ViT。YLABEL_MODEL で変更できる。
  ylabel_opt=""
  [ "$key" = "${YLABEL_MODEL:-vit_l_16}" ] && ylabel_opt="--with-ylabel"

  # 凡例はモデル共通なので最初に処理したモデルで 1 回だけ出す。
  # 特定モデルに紐付けると、そのモデルだけ指定して実行したときに凡例が出ない。
  legend_opt="--no-legend"
  if [ -z "${legend_done:-}" ]; then
    legend_opt=""
    legend_done=1
  fi

  $PY "$FIG_DIR/plot_idle_decomp.py" \
    --model-key "$key" --model-label "$label" $ylabel_opt $legend_opt \
    --prefetch $pf_json --offload $ag_json \
    --ytick-busy "$yb" --ytick-idle "$yi" --ytick-other "$yo" \
    --out-dir "$FIG_DIR" 2>&1 | grep -v findfont
  echo
done

echo "=== 生成物 ==="
ls -1 "$FIG_DIR/figure_png/" 2>/dev/null | sed 's/^/  /'
