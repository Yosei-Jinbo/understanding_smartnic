#!/usr/bin/env python

import matplotlib.pyplot as plt
from cycler import cycler

# Tol's Color Palettes
# [Tol 2021] Color-Blind Safe Palettes:
# https://sronpersonalpages.nl/~pault/data/colourschemes.pdf

tol_bright = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        # Reordered from its original order so that similar colors don't sit next to each other.
        (68,119,170),  # blue
        (238,102,119), # red
        (34,136,51),   # green
        (204,187,68),  # yellow
        (102,204,238), # cyan
        (170,51,119),  # purple
        (187,187,187)  # grey
    ]
]

tol_high_contrast = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        (0,68,136), (221,170,51), (187,85,102)
    ]
]

tol_vibrant = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        (0,119,187), (51,187,238), (0,153,136),
        (238,119,51), (204,51,17), (238,51,119), (187,187,187)
    ]
]

tol_muted = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        (51,34,136), (136,204,238), (68,170,153),
        (17,119,51), (153,153,51), (221,204,119),
        (204,102,119), (136,34,85), (170,68,153)
    ]
]

tol_medium_contrast = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        (102,51,153), (136,204,238), (68,170,153),
        (17,119,51), (153,153,51), (204,102,119)
    ]
]

tol_light = [
    (r/255.0, g/255.0, b/255.0) for r, g, b in
    [
        (119,170,221), (153,221,255), (68,187,153),
        (187,204,51), (238,221,136), (238,136,102), (255,170,187)
    ]
]

_LABEL_FONTSIZE = 22  # 軸ラベル
_TICK_FONTSIZE = 20   # 目盛り（x/y）
_LEGEND_FONTSIZE = 18 # 凡例

def common_rcParams():
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=tol_bright),
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.family": [
                "DejaVu Serif",       # available on this system
                "Linux Libertine O",  # for ACM paper
                "Times",              # for IEEE paper
                "Helvetica", "Arial", # General Sans-Serif fonts
            ],
            "font.size": _TICK_FONTSIZE,
            "axes.labelsize": _LABEL_FONTSIZE,
            "xtick.labelsize": _TICK_FONTSIZE,
            "ytick.labelsize": _TICK_FONTSIZE,
            "legend.fontsize": _LEGEND_FONTSIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
