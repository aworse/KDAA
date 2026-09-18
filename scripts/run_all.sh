#!/usr/bin/env bash
# Run the whole pipeline at once.
# For a real experiment: delete the make_synthetic_data line, put real recordings
# in data/sessions/ or data/clips/, then run the rest as-is.
set -e
cd "$(dirname "$0")/.."

# 0) (validation only) generate synthetic data -- delete for real experiments
python -m scripts.make_synthetic_data --config config.yaml --sessions-per-cell 2

# 1) segmentation: sessions/ -> clips/ + metadata.csv
#    (skip if you already provide data in clips/ + metadata.csv form)
python -m src.segment --config config.yaml

# 2) train
python -m src.train --config config.yaml

# 3) evaluate (produces grayscale figures + metrics.json)
python -m src.evaluate --config config.yaml

# 4) attack demo (one continuous recording -> text)
#    python -m src.decode --config config.yaml --wav data/sessions/p1_near_s0.wav
echo "== done: see runs/exp1/figures =="
