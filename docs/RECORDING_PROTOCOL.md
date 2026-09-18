# KDAA Data Measurement & Reorganization Protocol

The complete data-collection procedure for the experiment that recovers Dubeolsik
Hangul input from keystroke sounds. This single document covers "what to record, how,
and how to organize it into the format the code expects."

Overall flow:
```
① prepare (hardware/software)  →  ② record sessions (record_session.py)
      →  ③ reorganize & validate (validate_dataset.py)  →  ④ segment (src.segment)
      →  ⑤ train & evaluate (src.train / src.evaluate)
```

---

## 0. What and how much to collect (targets)

- The final unit is **one keystroke clip = one jamo**. The classifier predicts one of
  33 base jamo.
- **>= 25 keystrokes per jamo** (Harrison et al.). Typing one full pass of
  `scripts/prompts_ko.txt` automatically yields >= 25/jamo per participant
  (~1,840 keystrokes total).
- **Independent variable = 3 attack scenarios**: `near` / `far` / `noise`.
- **3 participants** (including yourself). Each participant records **>= 2 sessions** in
  each scenario.
  - Why >= 2 sessions: train/eval must be separated **by session** (session-wise) or
    accuracy is overestimated. With only one session per scenario, separation is
    impossible.

Recommended minimum design: **3 participants × 3 scenarios × 2 sessions = 18 sessions**.
One prompt pass (~1,840 keystrokes) per session → ~30k keystrokes total. If time is short,
split the prompt set into halves and type them as sessions A/B (two sessions cover one pass).

---

## 1. Equipment (budget allocation)

Core principle: **use one keyboard from start to finish** (control variable). Only the
microphone changes per scenario.

- **Near mic**: one USB condenser or lavalier mic, 20–30 cm from the keyboard → `near`.
- **Far mic**: a shotgun/directional mic, or the same condenser placed 1.5–2 m away →
  `far`. One mic moved between positions is fine (gain consistency matters more).
- **One keyboard** (clicky mechanical leaks the most, but a common laptop/membrane board
  is also meaningful for the study — whatever you use, keep the **same model for all
  sessions**).
- **Tripod/mic stand**, a USB audio interface (optional), an ordinary lab room (no need
  for a treated room).
- Spend leftover budget on a spare mic/cable and an external SSD (raw backup).

> Whether you buy two mics or move one is up to budget, but **keeping the input gain fixed
> across a session** affects results more than the mic type.

---

## 2. Software (recording PC)

```bash
pip install sounddevice pynput scipy numpy
# Linux: sudo apt-get install libportaudio2
# macOS: System Settings > Privacy & Security > Input Monitoring -> allow terminal/python
# Windows: run the terminal as Administrator
```

The analysis PC only needs the project `requirements.txt` (torch etc.). The recording and
analysis PCs can be the same or different.

---

## 3. Scenario definitions & mic placement (independent variable)

The three conditions must differ **only in mic/environment** — keyboard, typing, and
corpus stay identical.

- **`near`**: mic **20–30 cm** from the keyboard. Quiet room. High SNR.
  - Attack interpretation: a phone/laptop mic on the same desk.
- **`far`**: same room, mic at **1.5–2 m**. Keep it quiet.
  - The main cause of degradation here is **reverberation**, not SNR — the transient
    smears and location cues vanish. Change only distance, not the room, to isolate this.
- **`noise`**: back at close distance (same 20–30 cm as `near`), but **play real
  background noise** — café noise / AC / conversation. An intentionally low-SNR condition.
  - Important: **actually play and record the noise.** Training may add synthetic noise
    augmentation, but that is auxiliary — **do not replace an independent-variable
    condition with augmentation.**

**Control-variable checklist (identical across all sessions)**:
- Same keyboard, same desk/chair, same seated posture and hand position
- Fixed mic gain, fixed sample rate 48 kHz, mono
- Same corpus (`prompts_ko.txt`), same typing habit (don't mix fast/slow)
- Recording all three scenarios back-to-back on the same day reduces equipment drift

---

## 4. Session recording procedure (recommended: keylogger method)

This method **automates labeling.** It logs physical keys with timestamps, so you get
which jamo was pressed at which second (onset_s + jamo) with no manual work. It doesn't
matter whether the Hangul IME is on — it captures **physical QWERTY keys** and maps them
via the Dubeolsik keymap.

Record one session:
```bash
python -m scripts.record_session --participant p1 --scenario near --sid p1_near_s0
# prompts appear -> [Enter] to start -> type the prompts -> [ESC] to stop
```

- `--sid` convention: **`<participant>_<scenario>_s<number>`** (e.g. `p2_far_s1`). This
  name is the session id and the basis for ordering clips after segmentation.
- On finish, `data/sessions/p1_near_s0.wav` + `.csv` are generated automatically.
- Repeat 18 times, changing only `--sid` (participant × scenario × number combos).

**Rules while recording**:
- Type only the prompt sentences. **Never type real passwords / personal info / real names.**
- On a typo, don't backspace — just move to the next line (backspace is ignored as a
  label, but a deleted char misaligns sound vs. label). On a big mistake, discard that
  session and re-record.
- Record each session **in one continuous take.** If you must pause, split into `s0`, `s1`.

**30-second pre-check**: verify the gain isn't clipping (waveform flattened at the top)
nor too small (waveform stuck at the floor). For `near`, a peak around -6 dB is ideal.

---

## 5. Session recording procedure (fallback: known-text method)

If a keylogger can't be used (permissions etc.), record a **known** sentence and label it
afterward.

1. With any recorder, capture the sound of typing one sentence (or one pass) of
   `prompts_ko.txt` as `raw/*.wav`. Record exactly what was typed as text.
2. Convert:
   ```bash
   python -m scripts.ingest --mode known-text \
       --audio raw/p1_near_s0.wav --text "안녕하세요 반갑습니다 ..." \
       --participant p1 --scenario near --sid p1_near_s0
   ```
   For many files, batch with a manifest CSV:
   ```bash
   # raw/manifest.csv columns: audio,text,participant,scenario,sid
   python -m scripts.ingest --mode known-text --manifest raw/manifest.csv
   ```
3. Here `onset_s` is empty, so segmentation **auto-detects onsets and aligns them to the
   label order.** The **detected count must equal the label count** for correct alignment.
   - Many run-together keystrokes → fewer detections than labels → later labels shift.
   - So the known-text method works best when you type **slowly, with wide gaps.**
   - If `src.segment` prints a `detected N != labels M` warning, discard that session or
     tune `segment.onset_percentile` / `min_gap_ms` in `config.yaml` and re-detect.

> The keylogger method is superior in both accuracy and effort. known-text is a last resort.

---

## 6. Reorganize: build the folder structure and validate

After recording, `data/sessions/` holds `<sid>.wav` + `<sid>.csv` pairs — the KDAA
"session format." Two steps from here:

**(1) Integrity check** — run before segmentation to catch problems early:
```bash
python -m scripts.validate_dataset --config config.yaml
```
Checks: required columns, file existence, sample-rate consistency, **disallowed jamo
labels** (e.g. a compound vowel ㅘ or compound final ㄳ not split), per-jamo /
per-scenario / per-participant counts, session counts. Coverage is shown in
`docs/validate_jamo_counts.png` and `docs/validate_scenario_counts.png` (grayscale).
If the result is `FAIL`, fix the errors first.

**(2) Segmentation** — cut sessions into individual keystroke clips:
```bash
python -m src.segment --config config.yaml
# data/sessions/*.wav  ->  data/clips/*.wav  +  data/metadata.csv
```
`metadata.csv` is the training input. Layout at this point:
```
data/
  sessions/        # raw session recordings + labels (keep)
  clips/           # cut keystroke clips (segmentation output)
  metadata.csv     # per-clip label table (training input)
```

> If you segmented clips another way, you can provide `data/clips/` + `metadata.csv`
> directly. See the 'Format B' section of `docs/DEVELOPMENT.md` for the required columns.

---

## 7. Quality bar (must pass before training)

- `validate_dataset` result has **no errors (not FAIL)**.
- All 33 jamo present, each **>= 25 keystrokes** (above the dashed line in the figure).
- **>= 2 sessions per scenario** (session-wise split possible).
- Audio is **48 kHz mono, no clipping**, no mixed sample rates.
- The three scenarios are not badly skewed in keystroke count (check the
  participant × scenario grid).

---

## 8. Common failures & fixes

- **Clipping**: lower the gain and re-record that session. Clipped data is unusable.
- **Count mismatch (known-text)**: type more slowly, or raise `min_gap_ms` / lower
  `onset_percentile`. Switching to the keylogger method fixes this at the root.
- **Disallowed jamo warning**: a compound vowel/final not split into two keystrokes. The
  keylogger method logs physical keys so it splits automatically; known-text uses
  `decompose_text`.
- **Accuracy collapses in far**: expected (reverberation). That *is* the result (the IV
  effect). Don't change the room.
- **Per-participant accuracy variance**: hand-position/typing-habit differences. Report
  cross-user generalization separately with `train.split=participant` — it becomes a
  discussion point rather than a problem.

---

## 9. Ethics & consent (required in the report)

- Inform all 3 participants that **physical key input is logged** and obtain written
  consent beforehand.
- **No real passwords/personal info** beyond the prompt sentences. Discard any incidental
  speech captured in the recording.
- State the **raw-recording deletion date** in the consent form (e.g. 30 days after the
  competition).
- Do not release the full attack code/raw data indiscriminately (include defenses in the
  discussion).

---

## 10. Final run order

```bash
# (prepare) generate prompts — verify jamo coverage
python -m scripts.make_corpus --per-jamo 25

# (measure) repeat per session: 3 participants × 3 scenarios × 2 sessions = 18
python -m scripts.record_session --participant p1 --scenario near --sid p1_near_s0
#  ... p1_near_s1, p1_far_s0, ... through p3_noise_s1

# (reorganize & validate)
python -m scripts.validate_dataset --config config.yaml
python -m src.segment --config config.yaml
python -m scripts.validate_dataset --config config.yaml   # re-check on clips (optional)

# (train & evaluate)
python -m src.train    --config config.yaml
python -m src.evaluate --config config.yaml
```
