#!/usr/bin/env python3
"""idle_decomp_allrank.py の JSON から、論文と同じ体裁の図を生成する。

生成する図（モデルごと）:
  1. resource_busy_idle_<key>      Compute Kernel / Idle の 2 段積み
  2. resource_idle_decomp_<key>    Idle を AllGather / ReduceScatter / Memcpy / Other に分解
  3. other_breakdown_<key>         ★ "Other" をさらに細粒度分解（本スクリプトの主目的）

1・2 は performance_evaluation/figures/plot_resource_liberation_split.py と
同じ配色・フォント・サイズにしてあり、LaTeX の subfloat に混ぜても違和感が出ない。
凡例は本体から切り離して *_legend として別ファイルに出す（論文側と同じ運用）。

■ "Other" の細粒度分解について
  図 2 の "Other" は「Idle から AllGather / ReduceScatter / Memcpy を引いた残り」で、
  査読で「これは何なのか」「起動オーバーヘッドではないのか」と問われている部分。
  図 3 はこれを次の粒度に割る:
      - CUDA API (起動)          host_api。cudaLaunchKernel 等。純粋な起動コスト
      - 他ストリームのカーネル      other_stream_kernel。torch.cat 等、実は GPU が働いている
      - Param AllGather 待ち      GPU 完全停止のうち、ホストが AG 関連関数にいた分
      - 勾配処理待ち              同、勾配 D2H / partition_grads 等
      - パラメータ管理            同、free_param / partition / swap 等
      - フック/フレーム            同、autograd フックや forward フレーム自体
      - その他
  上 2 つは「GPU が動いている or ホストが CUDA を呼んでいる」時間、
  下 5 つは「GPU 完全停止（true_idle）」時間で、意味が違うので色系統を分けている。

使い方:
    python3 scripts/figures/plot_idle_decomp.py \
        --model-key opt_1_3b --model-label "OPT-1.3B" \
        --prefetch a.json b.json --offload c.json d.json \
        [--with-ylabel] [--include-step-phases] [--out-dir DIR]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
# graph_config は論文用に Linux Libertine / Times 等を並べるが、この環境には
# DejaVu Serif しか無い。フォールバックは正常動作なので警告だけ黙らせる。
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_config import common_rcParams, tol_bright

# ============================================================================
# Style（論文側 plot_resource_liberation_split.py と一致させる）
# ============================================================================
DPI = 300
BAR_WIDTH = 0.55
EDGE_COLOR = "black"
EDGE_WIDTH = 0.8
ANNOT_FONTSIZE = 13
TICK_FONTSIZE = 16
YTICK_FONTSIZE = 14
YLABEL_FONTSIZE = 12

BUSY_COLOR = tol_bright[0]    # blue
IDLE_COLOR = tol_bright[1]    # red
AG_COLOR = tol_bright[1]      # red
RS_COLOR = tol_bright[3]      # yellow
MEMCPY_COLOR = tol_bright[2]  # green
OTHER_COLOR = tol_bright[5]   # purple

CONFIG_LABELS = ["ZO", "SO(AG)"]
FIG_SIZE = (6.0, 3.2)

BUSY_LABEL = "Compute Kernel"
IDLE_LABEL = "Idle"
DECOMP_LABELS = ["AllGather", "ReduceScatter", "Memcpy", "Other"]
Y_LABEL_BUSY_IDLE = "Stream breakdown (ms)"
Y_LABEL_IDLE_DECOMP = "Idle breakdown (ms)"
Y_LABEL_OTHER = "Other breakdown (ms)"

# "Other" の細粒度分解。(凡例ラベル, JSON 上のキー, 色)
# 積む順は「意味の近いものを隣に」: 上 2 つ = GPU が動いている/CUDA を呼んでいる、
# 残り 5 つ = GPU 完全停止 (true_idle) をホストの居場所で割ったもの。
#
# 色は Tol bright を 0..6 の順にそのまま使う。パレット内で相互に識別できるよう
# 設計されているため。tol_muted と混ぜると CUDA API と Param Mgmt がほぼ同色になり、
# 凡例で区別できなかった（実際に発生）。
OTHER_STACK = [
    ("CUDA API",            "host_api",                     tol_bright[0]),
    ("Other-stream Kernel", "other_stream",                 tol_bright[2]),
    ("Param AllGather",     "cat:AG関連(パラメータ取得)",       tol_bright[1]),
    ("Gradient",            "cat:勾配処理",                   tol_bright[3]),
    ("Param Mgmt",          "cat:パラメータ管理(解放/分割)",     tol_bright[4]),
    ("Hook/Frame",          "cat:フック/フレーム",             tol_bright[5]),
    ("Misc",                "cat:その他",                     tol_bright[6]),
]

# idle_decomp_allrank.py が出すクラス名（日本語）との対応
CLS_AG = "ag(AllGatherカーネル)"
CLS_RS = "rs(ReduceScatterカーネル)"
CLS_MEM = "mem(memcpy)"
CLS_DPU = "dpu_ag_wait"
CLS_OTHER_STREAM = "other_stream_kernel"
CLS_HOST_API = "host_api(起動オーバーヘッド)"
CLS_NCCL_OTHER = "nccl_other"


# ============================================================================
# データ読み込み
# ============================================================================
def load(files, phases):
    """複数ランクの JSON を読み、指定フェーズを合算してランク平均を返す。

    ランクをまたぐ平均を取るのは、2026-07-28 の実測で step:barrier に
    最大 741ms のスキューがあり、単一ランクでは代表性が無いと判明したため。
    """
    per_rank = []
    for path in files:
        blob = json.load(open(path))
        for res in blob["results"]:
            agg = {"wall": 0.0, "compute": 0.0, "idle": 0.0, "true_idle": 0.0,
                   "ag": 0.0, "rs": 0.0, "mem": 0.0, "dpu": 0.0,
                   "other_stream": 0.0, "host_api": 0.0, "nccl_other": 0.0,
                   "cats": {}}
            for pname in phases:
                p = res["phases"].get(pname)
                if not p:
                    continue
                agg["wall"] += p["wall"]
                agg["compute"] += p["compute"]
                agg["idle"] += p["idle"]
                agg["true_idle"] += p["true_idle"]
                cl = p["classes"]
                agg["ag"] += cl.get(CLS_AG, 0.0)
                agg["rs"] += cl.get(CLS_RS, 0.0)
                agg["mem"] += cl.get(CLS_MEM, 0.0)
                agg["dpu"] += cl.get(CLS_DPU, 0.0)
                agg["other_stream"] += cl.get(CLS_OTHER_STREAM, 0.0)
                agg["host_api"] += cl.get(CLS_HOST_API, 0.0)
                agg["nccl_other"] += cl.get(CLS_NCCL_OTHER, 0.0)
                for k, v in p["true_idle_cat"].items():
                    agg["cats"][k] = agg["cats"].get(k, 0.0) + v
            per_rank.append(agg)

    if not per_rank:
        raise SystemExit("JSON からデータが読めなかった")

    n = len(per_rank)
    out = {k: sum(r[k] for r in per_rank) / n
           for k in per_rank[0] if k != "cats"}
    cats = {}
    for r in per_rank:
        for k, v in r["cats"].items():
            cats[k] = cats.get(k, 0.0) + v / n
    for k, v in cats.items():
        out["cat:" + k] = v

    # 図 2 の "Other" は積み上げが idle と一致する必要があるので、残差で定義する。
    # （nccl_other 等の小クラスを取りこぼして棒がずれるのを防ぐ）
    out["host_overhead"] = out["idle"] - out["ag"] - out["dpu"] - out["rs"] - out["mem"]
    out["n_ranks"] = n
    return out


# ============================================================================
# 描画
# ============================================================================
def save_fig(fig, name, out_dir):
    for fmt in ("pdf", "png"):
        d = os.path.join(out_dir, f"figure_{fmt}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def _annotate(ax, xi, bottom, v, ymax, state=None):
    """棒の中央に値を書く。

    state を渡すと直前に書いたラベルとの距離を見て、近すぎる場合は省略する。
    細粒度分解は 7 セグメントあり、小さい区画が隣接すると数字が重なって
    読めなくなるため（実測で 44 と 46 が衝突した）。
    """
    if v < ymax * 0.04:
        return
    center = bottom + v / 2
    if state is not None:
        min_gap = ymax * 0.055   # フォント高さぶん
        if state["last"] is not None and abs(center - state["last"]) < min_gap:
            return
        state["last"] = center
    ax.text(xi, center, f"{int(round(v))}",
            ha="center", va="center",
            fontsize=ANNOT_FONTSIZE, color="black", weight="bold")


def _finish(ax, ymax, step, ylabel, with_ylabel):
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(CONFIG_LABELS, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls="--", alpha=0.3)
    if with_ylabel:
        ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)


def plot_busy_idle(data, with_ylabel, step):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(2)
    busy = np.array([d["compute"] for d in data])
    idle = np.array([d["idle"] for d in data])
    ax.bar(x, busy, BAR_WIDTH, color=BUSY_COLOR,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.bar(x, idle, BAR_WIDTH, bottom=busy, color=IDLE_COLOR,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ymax = max(busy + idle) * 1.15
    for xi, b, i in zip(x, busy, idle):
        _annotate(ax, xi, 0, b, ymax)
        _annotate(ax, xi, b, i, ymax)
    _finish(ax, ymax, step, Y_LABEL_BUSY_IDLE, with_ylabel)
    fig.tight_layout()
    return fig


def plot_idle_decomp(data, with_ylabel, step):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    stack = [(["ag", "dpu"], AG_COLOR), (["rs"], RS_COLOR),
             (["mem"], MEMCPY_COLOR), (["host_overhead"], OTHER_COLOR)]
    ymax = max(d["idle"] for d in data) * 1.15
    for ci, d in enumerate(data):
        bottom = 0.0
        for keys, color in stack:
            v = sum(d[k] for k in keys)
            ax.bar(ci, v, BAR_WIDTH, bottom=bottom, color=color,
                   edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
            _annotate(ax, ci, bottom, v, ymax)
            bottom += v
    _finish(ax, ymax, step, Y_LABEL_IDLE_DECOMP, with_ylabel)
    fig.tight_layout()
    return fig


def plot_other_breakdown(data, with_ylabel, step):
    """図 2 の "Other" をさらに割る。査読 C3 への直接の回答図。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    totals = [sum(d.get(k, 0.0) for _, k, _ in OTHER_STACK) for d in data]
    ymax = max(totals) * 1.15
    for ci, d in enumerate(data):
        bottom = 0.0
        state = {"last": None}
        for _, key, color in OTHER_STACK:
            v = d.get(key, 0.0)
            if v <= 0:
                continue
            ax.bar(ci, v, BAR_WIDTH, bottom=bottom, color=color,
                   edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
            _annotate(ax, ci, bottom, v, ymax, state)
            bottom += v
    _finish(ax, ymax, step, Y_LABEL_OTHER, with_ylabel)
    fig.tight_layout()
    return fig


def plot_legend_only(pairs, name, ncol, out_dir):
    handles = [Patch(facecolor=c, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
               for _, c in pairs]
    labels = [l for l, _ in pairs]
    fig = plt.figure(figsize=(0.1, 0.1))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol,
                     fontsize=ANNOT_FONTSIZE, frameon=False,
                     handlelength=1.2, handleheight=0.8,
                     borderpad=0.0, borderaxespad=0.0,
                     columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)
    for fmt in ("pdf", "png"):
        d = os.path.join(out_dir, f"figure_{fmt}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.{fmt}")
        fig.savefig(path, dpi=DPI,
                    bbox_inches=leg.get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted()),
                    pad_inches=0.18)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", nargs="+", required=True,
                    help="ZO (baseline) の idle_decomp_allrank JSON")
    ap.add_argument("--offload", nargs="+", required=True,
                    help="SO(AG) の idle_decomp_allrank JSON")
    ap.add_argument("--model-key", required=True, help="出力ファイル名に使う (例: opt_1_3b)")
    ap.add_argument("--model-label", default="", help="ログ表示用")
    ap.add_argument("--with-ylabel", action="store_true",
                    help="LaTeX subfloat の左端パネルにだけ付ける")
    ap.add_argument("--include-step-phases", action="store_true",
                    help="forward/backward に加えて step:* も合算する "
                         "(update_new_params 等が丸ごと Other なので影響が大きい)")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--ytick-busy", type=float, default=500)
    ap.add_argument("--ytick-idle", type=float, default=500)
    ap.add_argument("--ytick-other", type=float, default=250)
    ap.add_argument("--no-legend", action="store_true")
    a = ap.parse_args()

    phases = ["forward", "backward"]
    if a.include_step_phases:
        phases += ["step:opt_dpu", "step:opt", "step:update_new_params", "step:barrier"]

    common_rcParams()
    data = [load(a.prefetch, phases), load(a.offload, phases)]

    print(f"--- {a.model_label or a.model_key}  phases={phases} "
          f"ranks={[d['n_ranks'] for d in data]} ---")
    for name, d in zip(CONFIG_LABELS, data):
        other = sum(d.get(k, 0.0) for _, k, _ in OTHER_STACK)
        print(f"  {name:7s} wall={d['wall']:8.1f} compute={d['compute']:7.1f} "
              f"idle={d['idle']:8.1f} other={d['host_overhead']:7.1f} "
              f"(細粒度合計={other:7.1f})")

    save_fig(plot_busy_idle(data, a.with_ylabel, a.ytick_busy),
             f"resource_busy_idle_{a.model_key}", a.out_dir)
    save_fig(plot_idle_decomp(data, a.with_ylabel, a.ytick_idle),
             f"resource_idle_decomp_{a.model_key}", a.out_dir)
    save_fig(plot_other_breakdown(data, a.with_ylabel, a.ytick_other),
             f"other_breakdown_{a.model_key}", a.out_dir)

    if not a.no_legend:
        plot_legend_only([(BUSY_LABEL, BUSY_COLOR), (IDLE_LABEL, IDLE_COLOR)],
                         "resource_busy_idle_legend", 2, a.out_dir)
        plot_legend_only(list(zip(DECOMP_LABELS,
                                  [AG_COLOR, RS_COLOR, MEMCPY_COLOR, OTHER_COLOR])),
                         "resource_idle_decomp_legend", 4, a.out_dir)
        plot_legend_only([(l, c) for l, _, c in OTHER_STACK],
                         "other_breakdown_legend", 4, a.out_dir)


if __name__ == "__main__":
    main()
