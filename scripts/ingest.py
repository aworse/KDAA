# -*- coding: utf-8 -*-
"""
재편 도구 A: 임의의 녹음 + 정답 텍스트 -> KDAA 'sessions/' 포맷으로 변환.
사용 위치 / 실행:
  # 각 녹음이 '알려진 프롬프트 문장'을 친 것일 때(키로거 없이 수집한 경우)
  python -m scripts.ingest --mode known-text \
      --audio raw/p1_near_s0.wav --text "안녕하세요 반갑습니다" \
      --participant p1 --scenario near --sid p1_near_s0

동작(known-text 모드):
  정답 텍스트를 두벌식 자모열로 분해(decompose_text) -> 그 개수만큼의 라벨 생성.
  onset_s는 비워두고 CSV를 쓴다. 이후 `python -m src.segment` 가 자동 온셋 검출로
  타건을 찾아 라벨과 순서 정렬(개수가 맞아야 함).
  => 키로거를 못 쓴 경우의 라벨링 경로.

여러 파일을 한 번에 처리하려면 --manifest CSV 사용:
  컬럼: audio, text, participant, scenario, sid
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
    print(f"[ok] {sid}: {len(jamos)} 라벨 (onset 미지정 -> segment에서 자동정렬), {len(wav)/sr:.1f}s")
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
        assert a.audio and a.text and a.sid, "--audio --text --sid 필요"
        write_session(a.audio, a.text, a.participant, a.scenario,
                      a.sid, a.root, a.sample_rate)
    print("다음: python -m src.segment  로 clips/ + metadata.csv 생성")


if __name__ == "__main__":
    main()
