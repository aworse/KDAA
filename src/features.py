# -*- coding: utf-8 -*-
"""
mel-spectrogram feature extraction.
Used by: dataset.py (clip -> log-mel image), the model input for train/eval.
Design: Harrison et al. (2023) style — treat the log mel-spectrogram as an image
        fed to a CNN. Uses torchaudio if available, otherwise scipy STFT + a mel
        filterbank producing the same result.
"""
from __future__ import annotations
import numpy as np


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """(n_mels, n_fft//2+1) mel filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    m_min, m_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        left = (fft_freqs - lo) / max(ce - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ce, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


class MelExtractor:
    """Reusable extractor built from config.feature."""

    def __init__(self, fcfg, sample_rate):
        self.sr = sample_rate
        self.n_fft = int(fcfg.n_fft)
        self.win = max(4, int(fcfg.win_ms * sample_rate / 1000))
        self.hop = max(1, int(fcfg.hop_ms * sample_rate / 1000))
        self.n_mels = int(fcfg.n_mels)
        self.fmin = float(fcfg.fmin)
        self.fmax = min(float(fcfg.fmax), sample_rate / 2)
        self.log_offset = float(fcfg.log_offset)
        self.target_frames = int(fcfg.target_frames)
        self.fb = mel_filterbank(sample_rate, self.n_fft, self.n_mels, self.fmin, self.fmax)
        self.window = np.hanning(self.win).astype(np.float32)

    def _stft_power(self, wav):
        from scipy.signal import stft
        _, _, Z = stft(
            wav, fs=self.sr, window=self.window, nperseg=self.win,
            noverlap=self.win - self.hop, nfft=self.n_fft,
            boundary=None, padded=False,
        )
        return (np.abs(Z) ** 2).astype(np.float32)      # (n_freqs, frames)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """Return log-mel (1, n_mels, target_frames) float32, per-clip normalized."""
        wav = wav.astype(np.float32)
        if wav.size < self.win:
            wav = np.pad(wav, (0, self.win - wav.size))
        power = self._stft_power(wav)                   # (F, Tframes)
        mel = self.fb @ power                            # (n_mels, Tframes)
        logmel = np.log(mel + self.log_offset)
        logmel = self._fix_frames(logmel)
        # per-clip normalization (mitigates recording-device domain shift)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        return logmel[None, :, :].astype(np.float32)

    def _fix_frames(self, x):
        T = x.shape[1]
        if T == self.target_frames:
            return x
        if T > self.target_frames:                       # center crop
            s = (T - self.target_frames) // 2
            return x[:, s:s + self.target_frames]
        pad = self.target_frames - T                     # right pad
        return np.pad(x, ((0, 0), (0, pad)), mode="edge")
