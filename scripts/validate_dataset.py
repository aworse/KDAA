# -*- coding: utf-8 -*-
"""
재편 도구 B: 데이터셋 정합성 점검 + 커버리지 리포트.
사용 위치 / 실행:
  python -m scripts.validate_dataset --config config.yaml
점검 항목:
  * metadata.csv 필수 컬럼 존재, 파일 경로 실재, 오디오 로딩 가능/샘플레이트
  * jamo 라벨이 허용된 33개 base 자모 집합에 속하는지(복합모음/겹받침이 라벨에
    잘못 들어갔는지)
  * 자모별/시나리오별/참가자별 타건 수 (클래스 불균형, 셀 결측)
  * 세션 분할 가능 여부(시나리오마다 세션 2개 이상인지)
산출:
  docs/validate_jamo_counts.png, docs/validate_scenario_counts.png (흑백)
  콘솔에 PASS/경고 요약
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
                    help="오디오 실제 로딩 점검 표본 수(0=생략)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    root = cfg.data.root
    warns, errs = [], []

    df = pd.read_csv(cfg.data.metadata)
    df["jamo"] = df["jamo"].astype(str)

    # 1) 컬럼
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        errs.append(f"metadata 필수 컬럼 누락: {miss}")

    # 2) 라벨 유효성
    bad = sorted(set(df["jamo"]) - ALLOWED)
    if bad:
        errs.append(f"허용되지 않은 jamo 라벨(복합모음/겹받침을 쪼개지 않았을 수 있음): {bad}")

    # 3) 파일 실재 + (표본) 로딩/샘플레이트
    if "filepath" in df.columns:
        missing_files = 0
        for fp in df["filepath"]:
            p = fp if os.path.isabs(fp) else os.path.join(root, fp)
            if not os.path.exists(p):
                missing_files += 1
        if missing_files:
            errs.append(f"존재하지 않는 오디오 파일 {missing_files}개")
        if a.check_audio:
            sample = df.sample(min(a.check_audio, len(df)), random_state=0)
            srs = set()
            for fp in sample["filepath"]:
                p = fp if os.path.isabs(fp) else os.path.join(root, fp)
                try:
                    wav, sr = load_audio(p, None)
                    srs.add(sr)
                    if wav.size == 0:
                        warns.append(f"빈 오디오: {fp}")
                except Exception as e:
                    errs.append(f"로딩 실패 {fp}: {e}")
            if len(srs) > 1:
                warns.append(f"표본 샘플레이트가 섞여 있음: {sorted(srs)} (통일 권장)")

    # 4) 자모/시나리오/참가자 카운트
    jc = Counter(df["jamo"])
    low = [j for j in LABELS if jc[j] < 25]
    if low:
        warns.append(f"25타건 미만 자모 {len(low)}개: {low}")
    missing_labels = [j for j in LABELS if jc[j] == 0]
    if missing_labels:
        warns.append(f"전혀 없는 자모: {missing_labels}")

    # 5) 세션 분할 가능성
    for scen, g in df.groupby("scenario"):
        n = g["session"].nunique()
        if n < 2:
            warns.append(f"시나리오 '{scen}' 세션 {n}개 -> session-wise 분할 불가(>=2 권장)")

    # --- figure ---
    os.makedirs("docs", exist_ok=True)
    try:
        from src import figures as FIG  # 한글폰트 자동설정
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
        im = ax.imshow(M, cmap="gray_r")
        ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen)
        ax.set_yticks(range(len(parts))); ax.set_yticklabels(parts)
        for i in range(len(parts)):
            for j in range(len(scen)):
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        color="white" if M[i,j] > M.max()/2 else "black", fontsize=9)
        ax.set_title("Keystrokes: participant × scenario")
        fig.tight_layout(); fig.savefig("docs/validate_scenario_counts.png", dpi=130)
        plt.close(fig)
        print("[figures] docs/validate_jamo_counts.png, docs/validate_scenario_counts.png")
    except Exception as e:
        print("[figure] skipped:", e)

    # --- 요약 ---
    print(f"\n총 타건 {len(df)}, 세션 {df['session'].nunique()}, "
          f"자모종류 {df['jamo'].nunique()}/{len(LABELS)}")
    for w in warns:
        print("  [경고]", w)
    for e in errs:
        print("  [오류]", e)
    print("\n결과:", "FAIL (오류 있음)" if errs else ("PASS (경고만)" if warns else "PASS"))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
