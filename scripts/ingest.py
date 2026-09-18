# -*- coding: utf-8 -*-
"""
Reorganize tool A: arbitrary recording + ground-truth text -> KDAA 'sessions/' format.
Usage:
  # when each recording types a KNOWN prompt sentence (collected without a keylogger)
  python -m scripts.ingest --mode known-text \
      --audio raw/p1_near_s0.wav --text "안녕하세요 반갑습니다" \
      --participant p1 --scenario near --sid p1_near_s0

known-text mode:
  Decompose the ground-truth text into a Dubeolsik jamo sequence (decompose_text)
  and emit that many labels. onset_s is left blank; `python -m src.segment` then
  auto-detects onsets and aligns them to the label order (counts must match).
  => the labeling path for when a keylogger cannot be used.

Batch several files with a manifest CSV:
  columns: audio, text, participant, scenario, sid
  python -m scripts.ingest --mode known-text --manifest raw/manifest.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hangul import decompose_text
from src.audio_io import load_audio, save_wav


def write_session(audio, text, participant, scenario, sid, root, sr):
    wav, _ = load_audio(audio, sr)
    outdir = os.path.join(root, "sessions")
    os.makedirs(outdir, exist_ok=True)
    save_wav(os.path.join(outdir, sid + ".wav"), wav, sr)
    jamos = [j for j in decompose_text(text) if j != "<sp>"]
    cpath = os.path.join(outdir, sid + ".csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_s", "key", "jamo", "shift", "scenario", "participant"])
        for j in jamos:
            shift = 1 if j in "ㄲㄸㅃㅆㅉㅒㅖ" else 0
            w.writerow(["", "", j, shift, scenario, participant])
    print(f"[ok] {sid}: {len(jamos)} labels (onset unset -> aligned in segment), {len(wav)/sr:.1f}s")
    return len(jamos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="known-text", choices=["known-text"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--audio"); ap.add_argument("--text")
    ap.add_argument("--participant"); ap.add_argument("--scenario")
    ap.add_argument("--sid")
    ap.add_argument("--root", default="data")
    ap.add_argument("--sample-rate", type=int, default=48000)
    a = ap.parse_args()

    if a.manifest:
        import pandas as pd
        rows = pd.read_csv(a.manifest)
        for _, r in rows.iterrows():
            write_session(r["audio"], r["text"], r["participant"],
                          r["scenario"], r["sid"], a.root, a.sample_rate)
    else:
        assert a.audio and a.text and a.sid, "--audio --text --sid required"
        write_session(a.audio, a.text, a.participant, a.scenario,
                      a.sid, a.root, a.sample_rate)
    print("next: python -m src.segment  to build clips/ + metadata.csv")


if __name__ == "__main__":
    main()
