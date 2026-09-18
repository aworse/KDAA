# -*- coding: utf-8 -*-
"""
Segmentation: continuous session recording -> individual keystroke clips.
Used by:
  - converting 'sessions/' format (continuous recording + onset-label CSV) into clips/
  - CLI:  python -m src.segment --config config.yaml
Pipeline stage 1. If this fails everything downstream breaks, so we detect
push-peak onsets with an energy bandpass + adaptive threshold + minimum-gap
suppression.

Note: if the label CSV provides onset_s, we trust those times and cut windows
      there (supervised labels). If onset_s is absent, we align auto-detected
      onsets to the label order (matched by count).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from .config import load_config, parse_overrides
from .audio_io import load_audio, save_wav
from .utils import ensure_dir


def bandpass(wav, sr, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    ny = sr / 2
    sos = butter(4, [max(lo, 1) / ny, min(hi, ny - 1) / ny], btype="band", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def detect_onsets(wav, sr, scfg):
    """Return keystroke onset times (seconds) via an adaptive energy threshold."""
    x = bandpass(wav, sr, scfg.band_hz[0], scfg.band_hz[1])
    frame = max(1, int(scfg.frame_ms * sr / 1000))
    hop = max(1, int(scfg.hop_ms * sr / 1000))
    # frame energy
    n = 1 + (len(x) - frame) // hop if len(x) >= frame else 0
    energy = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop:i * hop + frame]
        energy[i] = np.sum(seg * seg)
    if n == 0:
        return []
    thr = np.percentile(energy, scfg.onset_percentile)
    min_gap = int(scfg.min_gap_ms / 1000 * sr / hop)
    onsets = []
    last = -10 ** 9
    for i in range(1, n - 1):
        if energy[i] > thr and energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1]:
            if i - last >= min_gap:
                onsets.append(i * hop / sr)
                last = i
    return onsets


def cut_clip(wav, sr, onset_s, scfg):
    pre = int(scfg.pre_ms / 1000 * sr)
    post = int(scfg.post_ms / 1000 * sr)
    c = int(onset_s * sr)
    s, e = max(0, c - pre), min(len(wav), c + post)
    clip = wav[s:e]
    need = (pre + post) - len(clip)
    if need > 0:
        clip = np.pad(clip, (0, need))
    return clip.astype(np.float32)


def segment_sessions(cfg):
    """
    Read data/sessions/<sid>.wav + data/sessions/<sid>.csv and produce
    data/clips/*.wav + data/metadata.csv.
    Session CSV columns: onset_s(optional), key, jamo, shift, scenario, participant
    """
    root = cfg.data.root
    sdir = os.path.join(root, "sessions")
    cdir = ensure_dir(os.path.join(root, "clips"))
    rows = []
    sess_files = [f for f in os.listdir(sdir) if f.endswith(".wav")]
    for sf_ in sorted(sess_files):
        sid = sf_[:-4]
        wav, sr = load_audio(os.path.join(sdir, sf_), cfg.data.sample_rate)
        labels = pd.read_csv(os.path.join(sdir, sid + ".csv"))
        if "onset_s" in labels.columns and labels["onset_s"].notna().all():
            onsets = labels["onset_s"].tolist()
        else:
            det = detect_onsets(wav, sr, cfg.segment)
            if len(det) != len(labels):
                print(f"[warn] {sid}: detected {len(det)} != labels {len(labels)}, "
                      f"aligning from the start")
            onsets = det[:len(labels)] + [None] * max(0, len(labels) - len(det))
        for i, (onset, (_, lab)) in enumerate(zip(onsets, labels.iterrows())):
            if onset is None:
                continue
            clip = cut_clip(wav, sr, onset, cfg.segment)
            cid = f"{sid}_{i:04d}"
            fp = os.path.join(cdir, cid + ".wav")
            save_wav(fp, clip, sr)
            rows.append(dict(
                clip_id=cid, filepath=os.path.relpath(fp, root),
                jamo=lab["jamo"], key=lab.get("key", ""),
                shift=int(lab.get("shift", 0)),
                scenario=lab["scenario"], participant=lab["participant"],
                session=sid, sample_rate=sr,
            ))
    out = pd.DataFrame(rows)
    mpath = cfg.data.metadata
    out.to_csv(mpath, index=False)
    print(f"[ok] {len(out)} clips -> {mpath}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    segment_sessions(cfg)


if __name__ == "__main__":
    main()
