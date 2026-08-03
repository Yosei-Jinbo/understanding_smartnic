#!/bin/bash
###############################################################################
# make_all_figures.sh — 全モデルの idle 分解図を一括生成する
#
# 処理の流れ:
#   1. idle_decomp_allrank.py でこのノード (bluefield01) にあるランクを解析
#   2. plot_idle_decomp.py で 3 種類の図を生成
#
# 前提:
#   bluefield01 に .nsys-rep が存在すること:
#     baseline : logs/baseline/nsys-report/<prefix>_rank<N>.nsys-rep
#     smartnic : logs/smartnic_offload/nsys-report/<prefix>_rank<N>.nsys-rep
#   sqlite が無い/古い場合は自動で nsys export する。
#
# 使い方:
#   bash scripts/figures/make_all_figures.sh                # 全モデル
#   bash scripts/figures/make_all_figures.sh opt_1_3b       # 指定モデルのみ
###############################################################################
set -u

REPO=/home/y-jinbo/understanding_smartnic
PY=/home/y-jinbo/.venv/bin/python
NSYS_BIN=/usr/local/cuda/bin/nsys
PF_DIR=$REPO/logs/baseline/nsys-report
AG_DIR=$REPO/logs/smartnic_offload/nsys-report
JSON_DIR=$REPO/logs/idle_decomp
FIG_DIR=$REPO/scripts/figures

mkdir -p "$JSON_DIR"

# model_key | prefetch prefix | offload prefix | label | ytick(busy) | ytick(idle) | ytick(other)
MODELS=(
  "opt_1_3b|opt_buf_prefetch|opt_ag_smartnic|OPT-1.3B|500|500|250"
  "deberta_xl|deberta_xl_buf_prefetch|deberta_xl_ag_smartnic|DeBERTa-XL|500|500|250"
  "vit_l_16|vit_l_16_buf_prefetch|vit_l_16_ag_smartnic|ViT-L/16|250|250|100"
)

# $1=nsys-report ディレクトリ $2=prefix $3=出力 JSON 名の接頭辞
analyze() {
  local dir=$1 prefix=$2 tag=$3
  local dst="$JSON_DIR/${tag}.json"
  local rep sq files
  for rep in "$dir/${prefix}"_rank*.nsys-rep; do
    [ -e "$rep" ] || continue
    sq="${rep%.nsys-rep}.sqlite"
    [ "$sq" -nt "$rep" ] || "$NSYS_BIN" export --type sqlite --force-overwrite true \
      -o "$sq" "$rep" > /dev/null 2>&1
  done
  files=$(ls "$dir/${prefix}"_rank*.sqlite 2>/dev/null)
  [ -z "$files" ] && return
  $PY "$REPO/scripts/idle_decomp_allrank.py" --label "$tag" --json "$dst" $files \
    > /dev/null 2>&1 || { echo "  ★$prefix の解析失敗" >&2; return; }
  echo "$dst"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r key pf_prefix ag_prefix label yb yi yo <<< "$spec"
  if [ $# -gt 0 ]; then
    case " $* " in *" $key "*) ;; *) continue ;; esac
  fi

  echo "################ $label ################"
  pf_json=$(analyze "$PF_DIR" "$pf_prefix" "${key}_pf")
  ag_json=$(analyze "$AG_DIR" "$ag_prefix" "${key}_ag")

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
