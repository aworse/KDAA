#!/usr/bin/env bash
# 전체 파이프라인 한 번에 실행.
# 실제 실험: make_synthetic_data 줄을 지우고 data/sessions/ 또는 data/clips/ 에
#            진짜 녹음을 넣은 뒤 아래를 그대로 실행하면 된다.
set -e
cd "$(dirname "$0")/.."

# 0) (검증용) 합성 데이터 생성 — 실제 실험에선 삭제
python -m scripts.make_synthetic_data --config config.yaml --sessions-per-cell 2

# 1) 세그멘테이션: sessions/ -> clips/ + metadata.csv
#    (이미 clips/ + metadata.csv 형식으로 데이터를 넣었다면 이 단계 생략)
python -m src.segment --config config.yaml

# 2) 학습
python -m src.train --config config.yaml

# 3) 평가 (흑백 figure + metrics.json 생성)
python -m src.evaluate --config config.yaml

# 4) 공격 데모 (연속 녹음 한 파일 -> 텍스트)
#    python -m src.decode --config config.yaml --wav data/sessions/p1_near_s0.wav
echo "== done: runs/exp1/figures 확인 =="
