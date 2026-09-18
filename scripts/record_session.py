# -*- coding: utf-8 -*-
"""
Recording tool (data 'measurement').
Records mic audio while logging physical key presses with timestamps, producing
the KDAA 'sessions/' format (<sid>.wav + <sid>.csv) automatically.

Key idea: we do NOT parse the Hangul IME. We capture physical QWERTY key-down
          events and map them to jamo via the Dubeolsik key map
          (src.hangul.KEYMAP_BASE/SHIFT), so we get exact onset_s and jamo labels
          automatically -> no manual labeling.

Dependencies (install on the participant's PC):
    pip install sounddevice pynput scipy numpy
    # Linux needs PortAudio: sudo apt-get install libportaudio2
    # macOS: allow the terminal/python under Privacy > Input Monitoring

Example:
    python -m scripts.record_session --participant p1 --scenario near --sid p1_near_s0
    # Enter to start -> type the prompts (scripts/prompts_ko.txt) -> ESC to stop

Warnings:
  * NEVER type real passwords/personal info (type only the prompt sentences).
  * Tell participants this script logs physical keys and obtain consent first.
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
    return ["(no prompt file found; generate one with scripts.make_corpus)"]


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
        # Shift state
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
            return                       # ignore non-Dubeolsik keys
        t_rel = time.monotonic() - t0["v"]
        events.append((t_rel, jamo, shift))

    def on_release(key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            shift_down["v"] = False

    print("=" * 60)
    print(f" session {args.sid}  |  participant {args.participant}  |  scenario {args.scenario}")
    print("=" * 60)
    print(" Type the prompts below in order. Press ESC when done.")
    print(" (Hangul IME state doesn't matter -- physical keys are logged."
          " Do NOT type real passwords.)\n")
    for i, p in enumerate(prompts, 1):
        print(f"  {i:2d}. {p}")
    print("\n Press [Enter] to start recording...")
    input()

    stream = sd.InputStream(samplerate=sr, channels=1, callback=audio_cb)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    with stream:
        listener.start()
        print(" ● recording... (stop: ESC)")
        while not stop.is_set():
            time.sleep(0.05)
        listener.stop()

    # save
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

    print(f"\n [ok] {len(events)} keystrokes, {len(wav)/sr:.1f}s")
    print(f"      {wpath}\n      {cpath}")
    print(" next: python -m src.segment  (convert to clips/ + metadata.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="p1/p2/p3")
    ap.add_argument("--scenario", required=True, choices=["near", "far", "noise"])
    ap.add_argument("--sid", required=True, help="session id, e.g. p1_near_s0")
    ap.add_argument("--root", default="data")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--prompts", default="scripts/prompts_ko.txt")
    args = ap.parse_args()
    try:
        record(args)
    except ImportError as e:
        print("[dependency error] sounddevice / pynput required:")
        print("  pip install sounddevice pynput scipy numpy")
        print("  (Linux: sudo apt-get install libportaudio2)")
        print("cause:", e)


if __name__ == "__main__":
    main()
