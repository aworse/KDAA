# -*- coding: utf-8 -*-
"""
Data augmentation.
Used by: dataset.py (training split only).
Design: the background-noise condition is an *independent variable*, so it is NOT
        replaced by augmentation — it must be recorded for real. The noise mixing
        here is only an auxiliary augmentation to improve generalization.
"""
from __future__ import annotations
import numpy as np


def time_shift(wav, sr, max_ms):
    if max_ms <= 0:
        return wav
    n = int(np.random.uniform(-max_ms, max_ms) / 1000 * sr)
    return np.roll(wav, n)


def mix_noise(wav, snr_db_range, prob):
    """Mix Gaussian noise at a random SNR within the given range."""
    if np.random.rand() > prob:
        return wav
    snr = np.random.uniform(*snr_db_range)
    sig_p = np.mean(wav ** 2) + 1e-12
    noise = np.random.randn(*wav.shape).astype(np.float32)
    noise_p = np.mean(noise ** 2) + 1e-12
    k = np.sqrt(sig_p / (noise_p * (10 ** (snr / 10))))
    return (wav + k * noise).astype(np.float32)


def spec_augment(logmel, max_t, max_f):
    """Time/frequency masking on log-mel (1, M, T), in place (no copy)."""
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
