# -*- coding: utf-8 -*-
"""
Grayscale figure utilities.
Used by: evaluate.py. Per user preference, results are reported as grayscale
figures rather than tables. A Korean-capable font is registered automatically so
jamo/syllable labels render; axes/titles otherwise stay ASCII-friendly.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _set_korean_font():
    """Register a Korean-capable font so jamo/syllable labels don't break.
    Uses Noto Sans CJK KR / Nanum / Malgun if available, else warns."""
    from matplotlib import font_manager as fm
    prefer = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
              "Malgun Gothic", "AppleGothic", "Noto Sans CJK JP"]
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in prefer:
        if cand in names:
            plt.rcParams["font.family"] = cand
            return cand
    # try registering by file path when the name is not registered (e.g. .ttc)
    import glob
    for pat in ["/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
                "*NanumGothic*", "*malgun*"]:
        for fp in glob.glob(pat):
            try:
                fm.fontManager.addfont(fp)
                name = fm.FontProperties(fname=fp).get_name()
                plt.rcParams["font.family"] = name
                return name
            except Exception:
                pass
    import warnings
    warnings.warn("No Korean font found; Hangul labels in figures may not render "
                  "(e.g. apt-get install fonts-nanum).")
    return None


plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 9, "axes.grid": False,
    "image.cmap": "gray",
    "axes.unicode_minus": False,
})

_set_korean_font()


def confusion_matrix_fig(cm, labels, path, title="Confusion matrix"):
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.28),) * 2)
    ax.imshow(1 - cmn, cmap="gray", vmin=0, vmax=1)          # darker = larger value
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def bar_fig(names, values, path, ylabel, title, baseline=None):
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 0.9), 3.4))
    x = np.arange(len(names))
    ax.bar(x, values, color="0.25", edgecolor="black", width=0.6)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, color="black", ls="--", lw=1)
        ax.text(len(names) - 0.5, baseline, f" chance={baseline:.3f}",
                va="bottom", ha="right", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def grouped_bar_fig(groups, series, path, ylabel, title):
    """series: dict[name] = list(len(groups))."""
    fig, ax = plt.subplots(figsize=(max(4, len(groups) * 1.2), 3.6))
    x = np.arange(len(groups))
    n = len(series); w = 0.8 / n
    shades = np.linspace(0.15, 0.7, n)
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + (k - (n - 1) / 2) * w, vals, width=w, label=name,
               color=str(shades[k]), edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def curve_fig(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(ep, [h["train_acc"] for h in history], color="0.5", label="train")
    ax.plot(ep, [h["val_acc"] for h in history], color="0.0", label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_title("Training curve")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
