# -*- coding: utf-8 -*-
"""
합성 데이터 생성기 (스모크 테스트 / 파이프라인 검증용).
사용 위치 / 실행:
  python -m scripts.make_synthetic_data --config config.yaml
생성물: data/sessions/<sid>.wav + <sid>.csv   (KDAA '세션' 포맷)
  -> 이후 `python -m src.segment` 로 clips/ + metadata.csv 변환

주의: 이것은 '진짜 키보드 소리'가 아니라, 각 자모에 고유한 스펙트럼 서명을 부여한
      합성 클릭음이다. 코드 파이프라인이 끝까지 동작하는지, 오토마타 보정이
      효과가 있는지 확인하는 용도. 실제 실험에서는 이 스크립트 대신
      진짜 녹음을 data/sessions/ 또는 data/clips/ 에 넣으면 된다.
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
    """자모별 고유 스펙트럼 서명 = 특정 formant 주파수들의 감쇠 정현파 합."""
    idx = LABELS.index(jamo) if jamo in LABELS else 0
    rng = np.random.RandomState(1000 + idx)
    base = 800 + idx * 320                      # 자모마다 다른 중심 주파수
    freqs = base + rng.uniform(-120, 120, size=3) + np.array([0, 1500, 4200])
    t = np.arange(n) / sr
    decay = np.exp(-t / (0.006 + 0.004 * (idx % 3)))   # 짧은 transient
    sig = np.zeros(n, dtype=np.float32)
    for f, a in zip(freqs, [1.0, 0.5, 0.3]):
        sig += a * np.sin(2 * np.pi * f * t)
    return (sig * decay).astype(np.float32)


def make_keystroke(jamo, sr, scenario):
    """press peak + release peak(약하게) 의 이중 피크 클릭 생성."""
    press = jamo_signature(jamo, sr, int(0.03 * sr))
    rel = 0.35 * jamo_signature(jamo, sr, int(0.02 * sr))
    gap = int(np.random.uniform(0.04, 0.07) * sr)      # press-release 간격
    ks = np.zeros(gap + len(rel), dtype=np.float32)
    ks[:len(press)] += press[:len(ks)]
    ks[gap:gap + len(rel)] += rel
    # Shift(된소리)는 동시 타건 느낌 -> 살짝 겹친 저역 성분 추가
    return ks


def reverb(x, sr, scenario):
    """시나리오별 채널 효과: 근접(near)=거의 없음, 원거리(far)=잔향, 소음(noise)=SNR↓."""
    if scenario == "near":
        return x
    if scenario == "far":
        ir_len = int(0.05 * sr)
        ir = np.exp(-np.arange(ir_len) / (0.012 * sr)).astype(np.float32)
        ir[0] = 1.0
        y = np.convolve(x, ir)[:len(x) + ir_len]
        return y.astype(np.float32)
    return x  # noise 는 아래에서 SNR 로 처리


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
                gap = int(rng.uniform(0.15, 0.25) * sr)   # 단어 사이 긴 공백
                buf.append(np.zeros(gap, dtype=np.float32))
                total_len += gap
                continue
            ks = make_keystroke(j, sr, scenario)
            onset = total_len / sr
            buf.append(ks)
            total_len += len(ks)
            gap = int(rng.uniform(0.06, 0.14) * sr)       # 타건 간 간격
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
                    help="(참가자 x 시나리오) 셀당 세션 수")
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
