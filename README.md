# KDAA — Korean Dubeolsik Acoustic Attack

두벌식 한글 키보드를 대상으로 한 음향 사이드채널 공격(ASCA) 연구용 코드베이스.
키보드 타건 소리 → 자모 분류(CNN) → **두벌식 오토마타 제약 디코딩** → 한글 텍스트 복원.

영어권 ASCA 연구가 스펠체크(확률적 보정)에 기대는 것과 달리, 이 프로젝트는
초성·중성·종성 결합 규칙이라는 **거의 결정적인 필터**로 음향 분류기의 오류를
잘라낸다. 그 효과 측정이 핵심 기여점이다.

---

## 0. 설치

```bash
pip install -r requirements.txt      # torch, numpy, scipy, scikit-learn, matplotlib, pandas, pyyaml
# figure 한글 라벨이 깨지면 한국어 폰트 설치:  (예) apt-get install fonts-nanum
```

GPU 없어도 CPU로 동작한다(느릴 뿐). 검증 환경: Python 3.11.

---

## 1. 데이터 형식 — **네가 채워 넣어야 할 부분**

두 가지 형식 중 하나로 데이터를 넣으면 된다.

### 형식 A) 연속 세션 녹음 + 라벨 (권장)

```
data/
  sessions/
    p1_near_s0.wav        # 한 세션 = 한 참가자가 한 시나리오에서 쭉 타이핑한 녹음
    p1_near_s0.csv        # 그 세션의 타건 라벨 (아래 컬럼)
    p1_far_s0.wav
    p1_far_s0.csv
    ...
```

세션 CSV 컬럼:

| 컬럼         | 필수 | 설명 |
|--------------|------|------|
| `onset_s`    | 권장 | 각 타건의 시작 시각(초). 있으면 이 시점으로 클립을 자른다(지도 라벨). 없으면 자동 온셋 검출 결과와 순서 정렬. |
| `jamo`       | ✅   | 그 타건으로 입력된 **base 자모** 1개 (예: `ㄱ`, `ㅏ`, `ㄲ`). 분류 라벨. |
| `key`        | –    | (선택) 물리 QWERTY 키. 비워도 됨. |
| `shift`      | –    | 된소리/ㅒㅖ 등 Shift 동반이면 1, 아니면 0. |
| `scenario`   | ✅   | **독립변인.** `near` / `far` / `noise` 중 하나. |
| `participant`| ✅   | `p1` / `p2` / `p3`. |

> `jamo`는 **한 타에 나오는 낱자**여야 한다. 복합모음 `ㅘ`(=ㅗ+ㅏ)나 겹받침 `ㄳ`(=ㄱ+ㅅ)은
> 두 타건이므로 각각 `ㅗ`,`ㅏ` / `ㄱ`,`ㅅ` 두 행으로 나눠 적는다. 오토마타가 알아서 합친다.
> 정답 텍스트 문장이 있으면 `src.hangul.decompose_text("문장")`으로 이 자모열을 자동 생성할 수 있다.

라벨로 쓸 수 있는 33개 base 자모:
```
자음:   ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ   된소리: ㄲㄸㅃㅆㅉ
모음:   ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ         Shift모음: ㅒㅖ
```

세션 녹음 → 개별 클립 변환:
```bash
python -m src.segment --config config.yaml     # sessions/ -> clips/ + metadata.csv
```

### 형식 B) 이미 잘린 클립 + metadata.csv

세그멘테이션을 직접 했다면 이 형식으로 바로 넣는다:

```
data/
  clips/
    p1_near_s0_0000.wav    # 한 파일 = 한 타건(press+release 포함, 약 100ms)
    p1_near_s0_0001.wav
    ...
  metadata.csv
```

`metadata.csv` 컬럼:

| 컬럼 | 설명 |
|------|------|
| `clip_id` | 고유 ID. **`<session>_<정수순번>` 형식 권장** — 평가 시 세션 내 타건 순서 복원에 쓴다. |
| `filepath` | `data/` 기준 상대경로 (예: `clips/p1_near_s0_0000.wav`). |
| `jamo` | base 자모 라벨. |
| `shift` | 0/1. |
| `scenario` | `near`/`far`/`noise`. |
| `participant` | `p1`/`p2`/`p3`. |
| `session` | 세션 ID (**session-wise 분할의 단위** — 중요). |
| `sample_rate` | (선택) 원본 샘플레이트. |

> **실제로 어떻게 녹음하고 정리하는지**는 `docs/RECORDING_PROTOCOL.md`에 전 과정이 있다.

### 오디오 규격
- **모노, 48 kHz 권장** (transient 해상도 확보). 다른 샘플레이트면 자동 리샘플.
- `.wav`(권장) 또는 `.npy`(mono float32). `soundfile` 설치 시 다른 포맷도 로딩.
- 클립 길이는 `config.yaml`의 `segment.pre_ms + post_ms`(기본 100ms)로 통일된다.

### 권장 수집량 (실험 설계)
- Harrison et al. 기준 **키당 최소 25회**. 참가자 3명 × 33 자모 × 25회 ≈ 2,475 타건이 하한.
- 배경소음(`noise`) 조건은 **증강으로 대체하지 말고 실제 녹음**할 것(독립변인이므로).
- **실제 비밀번호는 절대 수집 금지.** 동의서에 녹음 폐기 시점 명시.

---

## 2. 실행

```bash
bash scripts/run_all.sh          # 합성데이터 생성→세그→학습→평가 전체 (검증용)
```

개별 단계:
```bash
python -m src.segment  --config config.yaml                 # 1. 세그멘테이션
python -m src.train    --config config.yaml                 # 2. 학습
python -m src.evaluate --config config.yaml --run runs/exp1 # 3. 평가(figure 생성)
python -m src.decode   --config config.yaml --run runs/exp1 --wav <녹음.wav>  # 4. 공격 데모
```

설정은 `config.yaml`에서 바꾸거나 CLI로 덮어쓴다:
```bash
python -m src.train --set train.epochs=60 model.arch=coatnet_lite train.split=participant
```

---

## 3. 파일 구조와 역할

```
config.yaml                  모든 하이퍼파라미터/경로 (한 곳에서 관리)
src/
  config.py                  config 로딩 + CLI override
  audio_io.py                wav/npy 로딩·저장 (scipy 기반, 무거운 의존성 없음)
  hangul.py    ★핵심         두벌식 키맵, 자모↔음절 조합/분해, 오토마타 제약 빔서치
  segment.py    [파이프 1]   연속 녹음 → 타건 온셋 검출 → 클립 절단
  features.py   [파이프 2]   로그 mel-spectrogram 추출 (torchaudio 없이 동작)
  augment.py                 SpecAugment / 시간이동 / 노이즈믹싱 (학습 split만)
  dataset.py                 PyTorch Dataset + **session/participant/random 분할**
  model.py      [파이프 3]   SmallCNN / CoAtNetLite
  train.py                   학습 루프 → runs/<exp>/best.pt
  evaluate.py   [파이프 4]   top-1/5, 시나리오별, 혼동행렬, 음절복원(오토마타 전후)
  decode.py     [공격 데모]  새 녹음 → 텍스트 복원 (전 단계 통합)
  figures.py                 흑백 figure 유틸 (한글 폰트 자동 등록)
scripts/
  record_session.py          [측정] 마이크녹음+키로거 -> sessions/ 자동생성(라벨 자동)
  ingest.py                  [재편] 임의 녹음+정답텍스트 -> sessions/ (known-text 방식)
  validate_dataset.py        [재편] 파일/샘플레이트/자모커버리지 점검 + 흑백 figure
  make_corpus.py             녹음 프롬프트 코퍼스 생성 + 커버리지 검증
  make_synthetic_data.py     포맷에 맞는 합성 데이터 생성 (스모크 테스트용)
  run_all.sh                 전체 실행
docs/
  RECORDING_PROTOCOL.md      ★ 데이터 측정·재편 전 과정 프로토콜 (인쇄용)
tests/
  test_hangul.py             오토마타 왕복/종성이동/빔서치 보정 검증
```

---

## 4. 평가 산출물 (표 대신 흑백 figure)

`python -m src.evaluate` → `runs/exp1/figures/` :

- `topk_accuracy.png` — 자모 분류 top-1/top-5 (chance = 1/클래스수 기준선 표시)
- `per_scenario_accuracy.png` — **근접/원거리/배경소음별 정확도** (독립변인 효과)
- `confusion_matrix.png` — 자모 혼동행렬
- `syllable_recovery.png` — **오토마타 필터 적용 전(greedy) vs 후(제약 빔서치)** 음절 복원율
- `training_curve.png`, `metrics.json`

---

## 5. 방법론 메모 (심사 대비)

- **분할**: 기본이 `session`(세션 단위 분리). 같은 세션을 무작위로 나누면 정확도가
  과대평가된다. `participant` 분할로 **사용자 간 일반화**를 별도 보고할 것.
- **베이스라인**: 무작위 = 1/클래스수 (figure에 점선으로 표시).
- **오토마타 기여**: 분류기가 유효하지 않은 자모열(고아 낱자 다발)을 낼 때, 제약 빔서치가
  2순위 후보로 유효 음절을 복원한다. `tests/test_hangul.py`의 `test_beam_fixes_invalid`가
  `ㄱㄱ → 가` 복원을 구체적으로 검증한다.
- 음절 경계 리듬, 된소리 Shift 동시타건, 한/영 전환키 등은 향후 특징으로 확장 가능
  (현재 코드는 base 자모 분류 + 오토마타 결합에 집중).

---

## 6. 윤리

동의한 팀원 데이터로만 실험하고, 실제 비밀번호·개인정보는 수집하지 않는다.
방어 기법(마스킹 노이즈, 자동입력 등) 논의는 보고서 고찰에 포함할 것.
공격 재현에 필요한 전체 코드/데이터를 무분별하게 공개하지 않는 것도 고려한다.
