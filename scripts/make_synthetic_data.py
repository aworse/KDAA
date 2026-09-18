# -*- coding: utf-8 -*-
"""
Synthetic data generator (smoke test / pipeline validation).
Usage:
  python -m scripts.make_synthetic_data --config config.yaml
Produces: data/sessions/<sid>.wav + <sid>.csv  (KDAA 'session' format)
  -> then `python -m src.segment` to convert into clips/ + metadata.csv

Note: this is NOT real keyboard audio. Each jamo gets a distinct spectral
      signature (synthetic clicks) so we can check the pipeline runs end to end
      and whether the automaton correction helps. In a real experiment, replace
      this with actual recordings in data/sessions/ or data/clips/.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, parse_overrides
from src.audio_io import save_wav
from src.utils import ensure_dir, set_seed
from src.hangul import LABELS, decompose_text, is_consonant

SENTENCES = [
    "안녕하세요 반갑습니다", "오늘 날씨가 좋네요", "과학전람회 준비 잘 되고 있어요",
    "키보드 소리로 무엇을 입력했는지 추정합니다", "값을 계산하고 결과를 저장했다",
    "깎아서 만든 조각 작품", "빨리 뛰어가는 강아지", "우리 학교 도서관은 넓다",
    "음향 사이드채널 공격 연구", "두벌식 자판 실험 데이터", "복합모음 과 왜 위 의",
    "받침이 있는 글자 값 닭 삶", "된소리 까 따 빠 싸 짜", "청소년 컴퓨터 사이언스 학회",
    "마이크 거리에 따른 정확도 변화", "배경 소음이 많은 환경에서 녹음",
]


def jamo_signature(jamo, sr, n):
    """Per-jamo spectral signature = sum of decaying sinusoids at specific formants."""
    idx = LABELS.index(jamo) if jamo in LABELS else 0
    rng = np.random.RandomState(1000 + idx)
    base = 800 + idx * 320                      # distinct center frequency per jamo
    freqs = base + rng.uniform(-120, 120, size=3) + np.array([0, 1500, 4200])
    t = np.arange(n) / sr
    decay = np.exp(-t / (0.006 + 0.004 * (idx % 3)))   # short transient
    sig = np.zeros(n, dtype=np.float32)
    for f, a in zip(freqs, [1.0, 0.5, 0.3]):
        sig += a * np.sin(2 * np.pi * f * t)
    return (sig * decay).astype(np.float32)


def make_keystroke(jamo, sr, scenario):
    """Double-peak click: press peak + (weaker) release peak."""
    press = jamo_signature(jamo, sr, int(0.03 * sr))
    rel = 0.35 * jamo_signature(jamo, sr, int(0.02 * sr))
    gap = int(np.random.uniform(0.04, 0.07) * sr)      # press-release gap
    ks = np.zeros(gap + len(rel), dtype=np.float32)
    ks[:len(press)] += press[:len(ks)]
    ks[gap:gap + len(rel)] += rel
    return ks


def reverb(x, sr, scenario):
    """Per-scenario channel: near=none, far=reverb tail, noise=handled via SNR below."""
    if scenario == "near":
        return x
    if scenario == "far":
        ir_len = int(0.05 * sr)
        ir = np.exp(-np.arange(ir_len) / (0.012 * sr)).astype(np.float32)
        ir[0] = 1.0
        y = np.convolve(x, ir)[:len(x) + ir_len]
        return y.astype(np.float32)
    return x  # 'noise' handled by SNR below


def add_noise(x, scenario):
    snr = {"near": 30, "far": 18, "noise": 8}[scenario]
    p = np.mean(x ** 2) + 1e-9
    noise = np.random.randn(len(x)).astype(np.float32)
    k = np.sqrt(p / ((np.mean(noise ** 2) + 1e-9) * 10 ** (snr / 10)))
    return x + k * noise


def build_session(sentences, sr, scenario, participant, seed):
    rng = np.random.RandomState(seed)
    rows = []
    buf = []
    total_len = 0
    for sent in sentences:
        for j in decompose_text(sent):
            if j == "<sp>":
                gap = int(rng.uniform(0.15, 0.25) * sr)   # longer gap between words
                buf.append(np.zeros(gap, dtype=np.float32))
                total_len += gap
                continue
            ks = make_keystroke(j, sr, scenario)
            onset = total_len / sr
            buf.append(ks)
            total_len += len(ks)
            gap = int(rng.uniform(0.06, 0.14) * sr)       # inter-keystroke gap
            buf.append(np.zeros(gap, dtype=np.float32))
            total_len += gap
            rows.append(dict(onset_s=onset, key="", jamo=j,
                             shift=1 if j in "ㄲㄸㅃㅆㅉㅒㅖ" else 0,
                             scenario=scenario, participant=participant))
    wav = np.concatenate(buf) if buf else np.zeros(sr, dtype=np.float32)
    wav = reverb(wav, sr, scenario)[:len(wav)]
    wav = add_noise(wav, scenario)
    wav = wav / (np.max(np.abs(wav)) + 1e-9) * 0.9
    return wav.astype(np.float32), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sessions-per-cell", type=int, default=2,
                    help="sessions per (participant x scenario) cell")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    set_seed(cfg.seed)
    sr = cfg.data.sample_rate
    sdir = ensure_dir(os.path.join(cfg.data.root, "sessions"))

    participants = ["p1", "p2", "p3"]
    scenarios = ["near", "far", "noise"]
    seed = cfg.seed
    for p in participants:
        for sc in scenarios:
            for k in range(a.sessions_per_cell):
                seed += 1
                sents = list(SENTENCES)
                np.random.RandomState(seed).shuffle(sents)
                wav, labels = build_session(sents, sr, sc, p, seed)
                sid = f"{p}_{sc}_s{k}"
                save_wav(os.path.join(sdir, sid + ".wav"), wav, sr)
                labels.to_csv(os.path.join(sdir, sid + ".csv"), index=False)
                print(f"[gen] {sid}: {len(labels)} keystrokes, {len(wav)/sr:.1f}s")
    print(f"[ok] sessions -> {sdir}")


if __name__ == "__main__":
    main()
