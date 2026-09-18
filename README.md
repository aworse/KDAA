# KDAA — Korean Dubeolsik Acoustic Attack

OFFICIAL : https://regx64.github.io/kdaa

Recover **Dubeolsik Hangul input from keyboard sound** — an acoustic side-channel attack
(ASCA) study. Keystroke audio → jamo classification (CNN) → **Dubeolsik
automaton-constrained decoding** → Hangul text.

> **Abstract.** Acoustic side-channel attacks on keyboards have been studied almost
> exclusively on English QWERTY, where a spell-checker cleans up the classifier's
> mistakes. Korean is different: the Hangul composition rules (cho/jung/jong) act as an
> almost *deterministic* filter that discards impossible jamo sequences outright. KDAA is,
> to our knowledge, the first pipeline aimed at the **Dubeolsik** layout, and it measures
> how much that automaton constraint recovers over a raw acoustic classifier.

---

## ⚠️ Responsible-research notice

This repository is for **defensive research and education**. Recovering someone else's
input by eavesdropping is illegal and harmful.

- Experiments use data from **consenting participants** only.
- **Never collect real passwords or personal information** (type only predefined prompts).
- Raw data / model weights needed to reproduce the attack are not distributed.
- The goal is to establish whether Korean keyboards are vulnerable to acoustic attacks —
  and to discuss **defenses**.

---

## Core idea

The physical keys are identical to QWERTY, so the **acoustic classification stage itself
is language-agnostic**. The Korean difference lives around it:

- An English spell-checker corrects probabilistically, but the **Dubeolsik jamo
  composition rules are an almost deterministic filter** — they discard impossible jamo
  sequences (runs of orphan letters) wholesale.
- As a result, a **constrained beam search** recovers valid syllables from lower-ranked
  candidates when the classifier errs.
  (`tests/test_hangul.py` verifies `ㄱㄱ → 가` recovery concretely.)

Quantifying that "automaton-filter contribution" is the point of the project.

---

## Pipeline

```
recording(.wav)  ─►  segmentation  ─►  log mel-spectrogram  ─►  CNN jamo classifier
   keylogger          (onsets)          (features)               (33-class)
     └─ onset+jamo labels auto                                        │
                                                                      ▼
              Hangul text  ◄─  Dubeolsik automaton-constrained beam  ◄─ jamo candidate probs
```

---

## Install

```bash
pip install -r requirements.txt
#   torch · numpy · scipy · scikit-learn · matplotlib · pandas · pyyaml
# if Hangul labels in figures break, install a Korean font: (e.g.) apt-get install fonts-nanum
```

Runs on CPU (just slower). Verified on Python 3.11.

---

## Quick start (check the pipeline on synthetic data)

You can verify the whole pipeline runs without any real recordings:

```bash
bash scripts/run_all.sh
# synthetic data → segmentation → train → evaluate → figures in runs/exp1/figures/
```

> `scripts/make_synthetic_data.py` makes synthetic clicks with a distinct spectrum per
> jamo. It is **not real keyboard audio** — it is for code validation. Replace it with real
> recordings for the experiment.

---

## Run on real data

How to record and organize the data — hardware, mic placement, labeling, validation — is
covered end to end in **[`docs/RECORDING_PROTOCOL.md`](docs/RECORDING_PROTOCOL.md)**. In short:

```bash
# 1) generate recording prompts (auto-balances jamo coverage)
python -m scripts.make_corpus --per-jamo 25

# 2) record sessions — mic + keylogger auto-labels (repeat per participant/scenario/session)
python -m scripts.record_session --participant p1 --scenario near --sid p1_near_s0

# 3) validate → segment
python -m scripts.validate_dataset --config config.yaml
python -m src.segment --config config.yaml

# 4) train → evaluate
python -m src.train    --config config.yaml
python -m src.evaluate --config config.yaml
```

The **independent variable** is the 3 attack scenarios — `near` / `far` (reverberant) /
`noise` (background noise). Training/evaluation separates data **by session
(session-wise)** by default to avoid overestimating accuracy.

Data-format details (session format / clip format) are in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

---

## Outputs

`python -m src.evaluate` → `runs/<exp>/figures/` (reported as grayscale figures, not tables):

- `topk_accuracy.png` — jamo classification top-1/top-5 (chance baseline shown)
- `per_scenario_accuracy.png` — near/far/noise accuracy (independent-variable effect)
- `confusion_matrix.png` — jamo confusion matrix
- `syllable_recovery.png` — syllable recovery **before (greedy) vs after (constrained beam)**
- `training_curve.png`, `metrics.json`

---

## Project structure

```
config.yaml                all hyperparameters / paths
src/
  hangul.py    ★           Dubeolsik keymap, jamo<->syllable compose/decompose, constrained beam
  segment.py               continuous recording -> onset detection -> clip cutting
  features.py              log mel-spectrogram (no torchaudio needed)
  dataset.py               PyTorch Dataset + session/participant/random split
  model.py                 SmallCNN / CoAtNetLite
  train.py / evaluate.py / decode.py
  augment.py / figures.py / audio_io.py / config.py / utils.py
scripts/
  record_session.py        [measure]    mic recording + keylogger → auto labels
  ingest.py                [reorganize] arbitrary recording + ground-truth text → session format
  validate_dataset.py      [reorganize] integrity/coverage checks + grayscale figures
  make_corpus.py           generate/verify the recording prompt corpus
  make_synthetic_data.py   synthetic data (smoke test)
docs/
  RECORDING_PROTOCOL.md    full measurement/reorganization protocol (printable)
  DEVELOPMENT.md           data-format details / design notes
tests/
  test_hangul.py           automaton round-trip / jong migration / beam correction
```

---

## Design highlights

- **Session-wise split by default.** Randomly splitting the same recording session
  overestimates accuracy. `train.split=participant` also measures cross-user generalization.
- **Automated labeling.** Physical QWERTY keys are logged with timestamps to produce
  `onset_s`+`jamo` automatically. No Hangul-IME parsing, so it's accurate and low-effort.
- **Minimal dependencies.** Audio loading and feature extraction are implemented with
  scipy, so it works without torchaudio/librosa.

---

## Related work

- Asonov & Agrawal, *Keyboard Acoustic Emanations*, IEEE S&P 2004.
- Zhuang, Zhou, Tygar, *Keyboard Acoustic Emanations Revisited*, ACM CCS 2005.
- Berger, Wool, Yeredor, *Dictionary Attacks Using Keyboard Acoustic Emanations*, CCS 2006.
- Compagno et al., *Don't Skype & Type!*, ASIA CCS 2017.
- Harrison, Toreini, Mehrnezhad, *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*, IEEE EuroS&PW 2023.
- Taheritajar & Rahaeimehr, *A Survey on Acoustic Side Channel Attacks on Keyboards*, 2023–24.

No public ASCA study on Dubeolsik jamo input was found — that gap is this project's
starting point.

---

## Ethics & disclaimer

- Experiment only with consenting participants; do not collect real passwords/personal info.
- Include defenses (masking noise, autofill, sound homogenization, etc.) in the discussion.
- The authors are not responsible for any use of this code to eavesdrop on or surveil
  others without authorization.

## License

TBD (research/education). Contact the author before redistribution or reuse.
