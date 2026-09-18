# -*- coding: utf-8 -*-
"""
녹음 도구 (데이터 '측정').
마이크 오디오를 녹음하면서 물리 키 입력을 타임스탬프와 함께 기록해,
KDAA 'sessions/' 포맷(<sid>.wav + <sid>.csv)을 자동 생성한다.

핵심: 한글 IME를 파싱하지 않는다. 물리 QWERTY 키다운을 잡아
      두벌식 키맵(src.hangul.KEYMAP_BASE/SHIFT)으로 자모로 변환하므로
      onset_s(정확한 타건 시각)와 jamo 라벨을 '자동으로' 얻는다.
      -> 수작업 라벨링이 필요 없다.

필요 패키지(참가자 PC에 설치):
    pip install sounddevice pynput scipy numpy
    # Linux는 PortAudio 필요: sudo apt-get install libportaudio2
    # macOS는 '손쉬운 사용> 입력 모니터링'에서 터미널/파이썬 권한 허용 필요

실행 예:
    python -m scripts.record_session --participant p1 --scenario near --sid p1_near_s0
    # 엔터로 녹음 시작 -> 프롬프트(scripts/prompts_ko.txt) 보고 타이핑 -> ESC로 종료

주의:
  * '실제 비밀번호/개인정보는 절대 입력하지 말 것' (프롬프트 문장만 타이핑).
  * 참가자에게 이 스크립트가 물리 키를 기록함을 사전 고지하고 동의받을 것.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hangul import KEYMAP_BASE, KEYMAP_SHIFT


def _load_prompts(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l for l in f.read().splitlines() if l.strip()]
    return ["(프롬프트 파일이 없습니다. scripts.make_corpus 로 생성하세요)"]


def record(args):
    import numpy as np
    import sounddevice as sd
    from pynput import keyboard
    from scipy.io import wavfile

    sr = args.sample_rate
    prompts = _load_prompts(args.prompts)

    events = []          # (t_rel, jamo, shift)
    shift_down = {"v": False}
    audio_chunks = []
    t0 = {"v": None}
    stop = threading.Event()

    def audio_cb(indata, frames, time_info, status):
        if t0["v"] is None:
            t0["v"] = time.monotonic()
        audio_chunks.append(indata.copy())

    def on_press(key):
        # Shift 상태
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            shift_down["v"] = True
            return
        try:
            ch = key.char
        except AttributeError:
            if key == keyboard.Key.esc:
                stop.set()
                return False
            return
        if ch is None or t0["v"] is None:
            return
        base = ch.lower()
        if shift_down["v"] and base.upper() in KEYMAP_SHIFT:
            jamo, shift = KEYMAP_SHIFT[base.upper()], 1
        elif base in KEYMAP_BASE:
            jamo, shift = KEYMAP_BASE[base], 0
        else:
            return                       # 두벌식 자모가 아닌 키는 무시
        t_rel = time.monotonic() - t0["v"]
        events.append((t_rel, jamo, shift))

    def on_release(key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            shift_down["v"] = False

    print("=" * 60)
    print(f" 세션 {args.sid}  |  참가자 {args.participant}  |  시나리오 {args.scenario}")
    print("=" * 60)
    print(" 아래 프롬프트를 순서대로 타이핑하세요. 끝나면 ESC.")
    print(" (한/영 상태 무관 — 물리 키를 기록합니다. 실제 비밀번호 입력 금지)\n")
    for i, p in enumerate(prompts, 1):
        print(f"  {i:2d}. {p}")
    print("\n [Enter] 를 누르면 녹음이 시작됩니다...")
    input()

    stream = sd.InputStream(samplerate=sr, channels=1, callback=audio_cb)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    with stream:
        listener.start()
        print(" ● 녹음 중... (종료: ESC)")
        while not stop.is_set():
            time.sleep(0.05)
        listener.stop()

    # 저장
    wav = np.concatenate(audio_chunks, axis=0).reshape(-1).astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-9) * 0.9
    outdir = os.path.join(args.root, "sessions")
    os.makedirs(outdir, exist_ok=True)
    wpath = os.path.join(outdir, args.sid + ".wav")
    wavfile.write(wpath, sr, (np.clip(wav, -1, 1) * 32767).astype("int16"))

    import csv
    cpath = os.path.join(outdir, args.sid + ".csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_s", "key", "jamo", "shift", "scenario", "participant"])
        for t_rel, jamo, shift in events:
            w.writerow([f"{t_rel:.4f}", "", jamo, shift, args.scenario, args.participant])

    print(f"\n [ok] {len(events)} 타건, {len(wav)/sr:.1f}s")
    print(f"      {wpath}\n      {cpath}")
    print(" 다음: python -m src.segment  (clips/ + metadata.csv 로 변환)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="p1/p2/p3")
    ap.add_argument("--scenario", required=True, choices=["near", "far", "noise"])
    ap.add_argument("--sid", required=True, help="세션 ID 예: p1_near_s0")
    ap.add_argument("--root", default="data")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--prompts", default="scripts/prompts_ko.txt")
    args = ap.parse_args()
    try:
        record(args)
    except ImportError as e:
        print("[의존성 오류] sounddevice / pynput 설치 필요:")
        print("  pip install sounddevice pynput scipy numpy")
        print("  (Linux: sudo apt-get install libportaudio2)")
        print("원인:", e)


if __name__ == "__main__":
    main()
