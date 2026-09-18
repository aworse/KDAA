# -*- coding: utf-8 -*-
"""
Reorganize tool B: dataset integrity check + coverage report.
Usage:
  python -m scripts.validate_dataset --config config.yaml
Checks:
  * metadata.csv required columns, file paths exist, audio loads / sample rate
  * jamo labels are within the allowed 33 base jamo (i.e. compound vowels/finals
    were not accidentally left in the labels)
  * per-jamo / per-scenario / per-participant keystroke counts (class imbalance,
    missing cells)
  * whether a session-wise split is possible (>=2 sessions per scenario)
Outputs:
  docs/validate_jamo_counts.png, docs/validate_scenario_counts.png (grayscale)
  a PASS/FAIL summary to stdout
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.config import load_config, parse_overrides
from src.hangul import LABELS
from src.audio_io import load_audio

REQUIRED = ["clip_id", "filepath", "jamo", "scenario", "participant", "session"]
ALLOWED = set(LABELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check-audio", type=int, default=30,
                    help="number of clips to actually load-check (0=skip)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    root = cfg.data.root
    warns, errs = [], []

    df = pd.read_csv(cfg.data.metadata)
    df["jamo"] = df["jamo"].astype(str)

    # 1) columns
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        errs.append(f"metadata missing required columns: {miss}")

    # 2) label validity
    bad = sorted(set(df["jamo"]) - ALLOWED)
    if bad:
        errs.append(f"disallowed jamo labels (compound vowel/final not split?): {bad}")

    # 3) files exist + (sampled) load/sample-rate
    if "filepath" in df.columns:
        missing_files = 0
        for fp in df["filepath"]:
            p = fp if os.path.isabs(fp) else os.path.join(root, fp)
            if not os.path.exists(p):
                missing_files += 1
        if missing_files:
            errs.append(f"{missing_files} audio files do not exist")
        if a.check_audio:
            sample = df.sample(min(a.check_audio, len(df)), random_state=0)
            srs = set()
            for fp in sample["filepath"]:
                p = fp if os.path.isabs(fp) else os.path.join(root, fp)
                try:
                    wav, sr = load_audio(p, None)
                    srs.add(sr)
                    if wav.size == 0:
                        warns.append(f"empty audio: {fp}")
                except Exception as e:
                    errs.append(f"failed to load {fp}: {e}")
            if len(srs) > 1:
                warns.append(f"mixed sample rates in sample: {sorted(srs)} (unify recommended)")

    # 4) jamo / scenario / participant counts
    jc = Counter(df["jamo"])
    low = [j for j in LABELS if jc[j] < 25]
    if low:
        warns.append(f"{len(low)} jamo below 25 keystrokes: {low}")
    missing_labels = [j for j in LABELS if jc[j] == 0]
    if missing_labels:
        warns.append(f"jamo entirely absent: {missing_labels}")

    # 5) session-split feasibility
    for scen, g in df.groupby("scenario"):
        n = g["session"].nunique()
        if n < 2:
            warns.append(f"scenario '{scen}' has {n} session(s) -> session-wise split "
                         f"not possible (>=2 recommended)")

    # --- figures ---
    os.makedirs("docs", exist_ok=True)
    try:
        from src import figures as FIG  # sets Korean font
        import matplotlib.pyplot as plt
        vals = [jc[j] for j in LABELS]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.bar(range(len(LABELS)), vals, color="0.3", edgecolor="black")
        ax.axhline(25, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylabel("count"); ax.set_title("Per-jamo keystroke count (dataset)")
        fig.tight_layout(); fig.savefig("docs/validate_jamo_counts.png", dpi=130)
        plt.close(fig)

        scen = sorted(df["scenario"].unique())
        parts = sorted(df["participant"].unique())
        M = np.array([[len(df[(df.scenario==s)&(df.participant==p)]) for s in scen] for p in parts])
        fig, ax = plt.subplots(figsize=(1.6*len(scen)+2, 0.7*len(parts)+2))
        ax.imshow(M, cmap="gray_r")
        ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen)
        ax.set_yticks(range(len(parts))); ax.set_yticklabels(parts)
        for i in range(len(parts)):
            for j in range(len(scen)):
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        color="white" if M[i,j] > M.max()/2 else "black", fontsize=9)
        ax.set_title("Keystrokes: participant x scenario")
        fig.tight_layout(); fig.savefig("docs/validate_scenario_counts.png", dpi=130)
        plt.close(fig)
        print("[figures] docs/validate_jamo_counts.png, docs/validate_scenario_counts.png")
    except Exception as e:
        print("[figure] skipped:", e)

    # --- summary ---
    print(f"\ntotal keystrokes {len(df)}, sessions {df['session'].nunique()}, "
          f"jamo types {df['jamo'].nunique()}/{len(LABELS)}")
    for w in warns:
        print("  [warn]", w)
    for e in errs:
        print("  [error]", e)
    print("\nresult:", "FAIL (has errors)" if errs else ("PASS (warnings only)" if warns else "PASS"))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
