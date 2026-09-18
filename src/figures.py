# -*- coding: utf-8 -*-
"""
흑백(그레이스케일) figure 유틸.
사용 위치: evaluate.py. 사용자 선호에 따라 결과는 표가 아니라 흑백 figure로 낸다.
한글 폰트가 없을 수 있으므로 축/제목은 영문·기호 위주로 쓴다.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _set_korean_font():
    """한글 자모/음절 라벨이 깨지지 않도록 한국어 지원 폰트를 자동 등록.
    Noto Sans CJK KR / Nanum / Malgun 등이 있으면 사용, 없으면 경고만."""
    from matplotlib import font_manager as fm
    prefer = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
              "Malgun Gothic", "AppleGothic", "Noto Sans CJK JP"]
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in prefer:
        if cand in names:
            plt.rcParams["font.family"] = cand
            return cand
    # ttc 등 이름 미등록 시 파일 경로로 직접 등록 시도
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
    warnings.warn("한국어 폰트를 찾지 못했습니다. figure의 한글 라벨이 깨질 수 있습니다. "
                  "(예: apt-get install fonts-nanum)")
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
    ax.imshow(1 - cmn, cmap="gray", vmin=0, vmax=1)          # 진할수록 값 큼
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
