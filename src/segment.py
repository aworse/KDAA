# -*- coding: utf-8 -*-
"""
세그멘테이션: 연속 세션 녹음 -> 개별 타건 클립.
사용 위치:
  - 데이터가 'sessions/' 형식(연속 녹음 + 온셋 라벨 CSV)일 때 clips/ 로 변환
  - CLI:  python -m src.segment --config config.yaml
논문 파이프라인 1단계. 여기서 실패하면 이후가 모두 망가지므로,
에너지 밴드패스 + 적응형 임계 + 최소간격 억제로 push-peak 온셋을 검출한다.

주의: 라벨 CSV의 onset_s가 주어지면 그 시점을 신뢰해 창을 자른다(지도 라벨).
      onset_s가 없으면 자동 검출 온셋과 라벨 순서를 정렬(라벨 개수 기준).
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
    """적응형 임계 기반 타건 온셋(초 단위) 리스트 반환."""
    x = bandpass(wav, sr, scfg.band_hz[0], scfg.band_hz[1])
    frame = max(1, int(scfg.frame_ms * sr / 1000))
    hop = max(1, int(scfg.hop_ms * sr / 1000))
    # 프레임 에너지
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
    data/sessions/<sid>.wav + data/sessions/<sid>.csv 를 읽어
    data/clips/*.wav + data/metadata.csv 를 생성.
    세션 CSV 컬럼: onset_s(선택), key, jamo, shift, scenario, participant
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
                print(f"[warn] {sid}: 검출 {len(det)} != 라벨 {len(labels)}, "
                      f"앞에서부터 정렬")
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
    print(f"[ok] {len(out)} 클립 -> {mpath}")
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
