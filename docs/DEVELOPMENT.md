# KDAA — Developer Notes & Data Format

Detailed data-format spec and design notes. For the public overview see the root
`README.md`; for the recording protocol see `RECORDING_PROTOCOL.md`.

---

## 0. Install

```bash
pip install -r requirements.txt   # torch, numpy, scipy, scikit-learn, matplotlib, pandas, pyyaml
# if Hangul labels in figures break, install a Korean font: (e.g.) apt-get install fonts-nanum
```

Runs on CPU (just slower). Verified on Python 3.11.

---

## 1. Data format — what you provide

Use one of two formats.

### Format A) Continuous session recording + labels (recommended)

```
data/
  sessions/
    p1_near_s0.wav        # one session = one participant typing continuously in one scenario
    p1_near_s0.csv        # keystroke labels for that session (columns below)
    p1_far_s0.wav
    p1_far_s0.csv
    ...
```

Session CSV columns:

- `onset_s` (recommended) — start time (seconds) of each keystroke. If present, clips
  are cut at these times (supervised labels). If absent, auto-detected onsets are
  aligned to the label order.
- `jamo` (required) — the single **base jamo** produced by that keystroke
  (e.g. `ㄱ`, `ㅏ`, `ㄲ`). This is the classification label.
- `key` (optional) — physical QWERTY key. May be blank.
- `shift` (optional) — 1 if Shift was held (tense consonants / ㅒㅖ), else 0.
- `scenario` (required) — **independent variable.** one of `near` / `far` / `noise`.
- `participant` (required) — `p1` / `p2` / `p3`.

> `jamo` must be a single keystroke's letter. A compound vowel `ㅘ` (=ㅗ+ㅏ) or
> compound final `ㄳ` (=ㄱ+ㅅ) is two keystrokes, so write two rows (`ㅗ`,`ㅏ` /
> `ㄱ`,`ㅅ`); the automaton merges them. If you have the ground-truth sentence,
> `src.hangul.decompose_text("...")` generates this jamo sequence automatically.

The 33 base jamo usable as labels:
```
consonants: ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ   tense: ㄲㄸㅃㅆㅉ
vowels:     ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ        shift vowels: ㅒㅖ
```

Convert session recordings into clips:
```bash
python -m src.segment --config config.yaml     # sessions/ -> clips/ + metadata.csv
```

### Format B) Pre-segmented clips + metadata.csv

If you segmented the audio yourself, provide this form directly:

```
data/
  clips/
    p1_near_s0_0000.wav    # one file = one keystroke (press+release, ~100ms)
    p1_near_s0_0001.wav
    ...
  metadata.csv
```

`metadata.csv` columns:

- `clip_id` — unique id. **Prefer `<session>_<integer_index>`** — used to restore
  per-session keystroke order at evaluation.
- `filepath` — path relative to `data/` (e.g. `clips/p1_near_s0_0000.wav`).
- `jamo` — base jamo label.
- `shift` — 0/1.
- `scenario` — `near`/`far`/`noise`.
- `participant` — `p1`/`p2`/`p3`.
- `session` — session id (**the unit of session-wise split** — important).
- `sample_rate` — (optional) original sample rate.

### Audio spec
- **Mono, 48 kHz recommended** (transient resolution). Other rates are auto-resampled.
- `.wav` (recommended) or `.npy` (mono float32). With `soundfile` installed, other
  formats load too.
- Clip length is unified by `config.yaml` `segment.pre_ms + post_ms` (default 100ms).

### Recommended volume (experiment design)
- Harrison et al.: **>= 25 keystrokes per key.** 3 participants × 33 jamo × 25 ≈ 2,475
  keystrokes is the lower bound.
- The `noise` condition must be **recorded for real, not replaced by augmentation**
  (it is an independent variable).
- **Never collect real passwords.** State the deletion date in the consent form.

---

## 2. Run

```bash
bash scripts/run_all.sh          # synthetic data -> segment -> train -> evaluate (validation)
```

Individual steps:
```bash
python -m src.segment  --config config.yaml                 # 1. segmentation
python -m src.train    --config config.yaml                 # 2. train
python -m src.evaluate --config config.yaml --run runs/exp1 # 3. evaluate (figures)
python -m src.decode   --config config.yaml --run runs/exp1 --wav <rec.wav>  # 4. attack demo
```

Override config on the CLI:
```bash
python -m src.train --set train.epochs=60 model.arch=coatnet_lite train.split=participant
```

---

## 3. File layout

```
config.yaml                all hyperparameters / paths
src/
  config.py                config loading + CLI override
  audio_io.py              wav/npy load/save (scipy-based, no heavy deps)
  hangul.py    ★core       Dubeolsik keymap, jamo<->syllable compose/decompose, constrained beam
  segment.py    [stage 1]  continuous recording -> onset detection -> clip cutting
  features.py   [stage 2]  log mel-spectrogram (works without torchaudio)
  augment.py               SpecAugment / time-shift / noise mixing (train split only)
  dataset.py               PyTorch Dataset + session/participant/random split
  model.py      [stage 3]  SmallCNN / CoAtNetLite
  train.py                 training loop -> runs/<exp>/best.pt
  evaluate.py   [stage 4]  top-1/5, per-scenario, confusion matrix, syllable recovery
  decode.py     [attack]   new recording -> recovered text (whole pipeline)
  figures.py               grayscale figure utils (auto-registers Korean font)
scripts/
  record_session.py        [measure] mic recording + keylogger -> sessions/ (auto labels)
  ingest.py                [reorganize] arbitrary recording + text -> sessions/ (known-text)
  validate_dataset.py      [reorganize] file/sample-rate/coverage checks + grayscale figures
  make_corpus.py           generate + verify the recording prompt corpus
  make_synthetic_data.py   synthetic data (smoke test)
  run_all.sh               run everything
docs/
  RECORDING_PROTOCOL.md    full measurement/reorganization protocol (printable)
tests/
  test_hangul.py           automaton round-trip / jong migration / beam correction
```

---

## 4. Evaluation outputs (grayscale figures, not tables)

`python -m src.evaluate` -> `runs/exp1/figures/`:

- `topk_accuracy.png` — jamo classification top-1/top-5 (chance = 1/num_classes shown)
- `per_scenario_accuracy.png` — near/far/noise accuracy (independent-variable effect)
- `confusion_matrix.png` — jamo confusion matrix
- `syllable_recovery.png` — syllable recovery **before (greedy) vs after (constrained beam)**
- `training_curve.png`, `metrics.json`

---

## 5. Methodology notes (for review)

- **Split**: default is `session` (per-session separation). Random splits overestimate
  accuracy. Report cross-user generalization separately with `participant`.
- **Baseline**: chance = 1/num_classes (dashed line in figures).
- **Automaton contribution**: when the classifier emits an invalid jamo sequence (many
  orphan jamo), the constrained beam recovers a valid syllable from a lower-ranked
  candidate. `tests/test_hangul.py::test_beam_fixes_invalid` verifies `ㄱㄱ -> 가`.
- Syllable-boundary rhythm, tense-consonant Shift chords, and the Han/Eng toggle key are
  future feature extensions (the current code focuses on base-jamo classification + the
  automaton).

---

## 6. Ethics

Experiment only on consenting teammates; never collect real passwords/personal info.
Discuss defenses (masking noise, autofill, etc.) in the report. Consider not releasing
the full attack code/data indiscriminately.
