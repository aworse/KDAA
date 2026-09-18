# -*- coding: utf-8 -*-
"""
데이터 증강.
사용 위치: dataset.py (학습 split 에만 적용).
설계: 배경소음 조건은 '독립변인'이므로 증강으로 대체하지 않고 실제 녹음을 쓴다.
      여기서의 노이즈 믹싱은 일반화 향상을 위한 보조 증강일 뿐이다.
"""
from __future__ import annotations
import numpy as np


def time_shift(wav, sr, max_ms):
    if max_ms <= 0:
        return wav
    n = int(np.random.uniform(-max_ms, max_ms) / 1000 * sr)
    return np.roll(wav, n)


def mix_noise(wav, snr_db_range, prob):
    """가우시안 노이즈를 지정 SNR 범위로 랜덤 믹싱."""
    if np.random.rand() > prob:
        return wav
    snr = np.random.uniform(*snr_db_range)
    sig_p = np.mean(wav ** 2) + 1e-12
    noise = np.random.randn(*wav.shape).astype(np.float32)
    noise_p = np.mean(noise ** 2) + 1e-12
    k = np.sqrt(sig_p / (noise_p * (10 ** (snr / 10))))
    return (wav + k * noise).astype(np.float32)


def spec_augment(logmel, max_t, max_f):
    """로그멜 (1, M, T) 에 시간/주파수 마스크. 입력을 복사하지 않고 in-place."""
    _, M, T = logmel.shape
    if max_t > 0 and T > max_t:
        w = np.random.randint(0, max_t + 1)
        if w > 0:
            s = np.random.randint(0, T - w)
            logmel[:, :, s:s + w] = 0.0
    if max_f > 0 and M > max_f:
        w = np.random.randint(0, max_f + 1)
        if w > 0:
            s = np.random.randint(0, M - w)
            logmel[:, s:s + w, :] = 0.0
    return logmel
