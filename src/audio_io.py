"""
Audio loading/saving (scipy-based, no heavy dependencies).
Used by: segment.py, features.py, dataset.py, make_synthetic_data.py
Supports: .wav (scipy), .npy (numpy array, mono float32). Uses soundfile if present.
"""
from __future__ import annotations
import numpy as np


def load_audio(path: str, target_sr: int | None = None):
    """Return (waveform float32 mono in [-1,1], sr)."""
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
    if wav.ndim > 1:                      # stereo -> mono
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
    """Polyphase resample (scipy). Works for non-integer ratios."""
    from math import gcd
    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    from scipy.signal import resample_poly
    return resample_poly(wav, up, down).astype(np.float32)
