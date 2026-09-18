# -*- coding: utf-8 -*-
"""
오디오 로딩/저장 (외부 무거운 의존성 없이 scipy 사용).
사용 위치: segment.py, features.py, dataset.py, make_synthetic_data.py
지원: .wav (scipy), .npy (numpy 배열, mono float32). soundfile 있으면 우선 사용.
"""
from __future__ import annotations
import numpy as np


def load_audio(path: str, target_sr: int | None = None):
    """반환: (waveform float32 mono [-1,1], sr)."""
    if path.endswith(".npy"):
        wav = np.load(path).astype(np.float32)
        sr = target_sr or 48000
    else:
        try:
            import soundfile as sf
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            wav = _to_float(data)
    if wav.ndim > 1:                      # 스테레오 -> 모노
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if target_sr and target_sr != sr:
        wav = _resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_wav(path: str, wav: np.ndarray, sr: int):
    from scipy.io import wavfile
    x = np.clip(wav, -1.0, 1.0)
    wavfile.write(path, sr, (x * 32767.0).astype(np.int16))


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """폴리페이즈 리샘플(scipy). 정수비 아니어도 동작."""
    from math import gcd
    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    from scipy.signal import resample_poly
    return resample_poly(wav, up, down).astype(np.float32)
