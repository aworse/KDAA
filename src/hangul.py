# -*- coding: utf-8 -*-
"""
두벌식 한글 오토마타.
사용 위치:
  - decode.py 에서 자모 시퀀스 -> 음절 조합 및 제약 빔서치 디코딩에 사용
  - evaluate.py 에서 '오토마타 필터 적용 전/후 음절 복원율' 계산에 사용
  - dataset.py 에서 라벨(자모) 집합 검증에 사용

핵심 아이디어(KDAA의 차별점):
  영어 스펠체크는 확률적 보정이지만, 두벌식 초성/중성/종성 결합 규칙은
  거의 '결정적' 필터다. 유효하지 않은 자모 시퀀스를 대량으로 잘라내므로
  음향 분류기의 약점을 언어권 연구보다 강하게 보정한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------
# 1) 두벌식 키맵 (QWERTY 물리키 -> 자모)
#    분류기가 예측하는 라벨은 '한 타에 나오는 base 자모'다.
#    복합모음(ㅘ 등)/겹받침(ㄳ 등)은 단일 키가 아니라 연타로 만들어지므로
#    분류 라벨에는 들어가지 않고, 오토마타가 조합한다.
# ---------------------------------------------------------------
KEYMAP_BASE = {
    'q': 'ㅂ', 'w': 'ㅈ', 'e': 'ㄷ', 'r': 'ㄱ', 't': 'ㅅ',
    'y': 'ㅛ', 'u': 'ㅕ', 'i': 'ㅑ', 'o': 'ㅐ', 'p': 'ㅔ',
    'a': 'ㅁ', 's': 'ㄴ', 'd': 'ㅇ', 'f': 'ㄹ', 'g': 'ㅎ',
    'h': 'ㅗ', 'j': 'ㅓ', 'k': 'ㅏ', 'l': 'ㅣ',
    'z': 'ㅋ', 'x': 'ㅌ', 'c': 'ㅊ', 'v': 'ㅍ',
    'b': 'ㅠ', 'n': 'ㅜ', 'm': 'ㅡ',
}
KEYMAP_SHIFT = {  # 된소리/ㅒㅖ (Shift 동반)
    'Q': 'ㅃ', 'W': 'ㅉ', 'E': 'ㄸ', 'R': 'ㄲ', 'T': 'ㅆ',
    'O': 'ㅒ', 'P': 'ㅖ',
}

# 물리키 위치(QWERTY 소문자) -> 클래스 기준으로 삼고 싶을 때를 위해 역맵도 제공
JAMO_TO_KEY = {v: k for k, v in KEYMAP_BASE.items()}
JAMO_TO_KEY.update({v: k for k, v in KEYMAP_SHIFT.items()})

# 분류기 라벨로 쓰는 33개 base 자모 + 특수토큰
CONSONANTS_BASE = list('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ')
CONSONANTS_TENSE = list('ㄲㄸㅃㅆㅉ')            # 된소리
VOWELS_BASE = list('ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ')
VOWELS_SHIFT = list('ㅒㅖ')
SPECIAL = ['<sp>']                              # 스페이스(선택)

LABELS = CONSONANTS_BASE + CONSONANTS_TENSE + VOWELS_BASE + VOWELS_SHIFT
ALL_CONSONANTS = set(CONSONANTS_BASE + CONSONANTS_TENSE)
ALL_VOWELS = set(VOWELS_BASE + VOWELS_SHIFT)


def is_consonant(j: str) -> bool:
    return j in ALL_CONSONANTS


def is_vowel(j: str) -> bool:
    return j in ALL_VOWELS


# ---------------------------------------------------------------
# 2) 유니코드 조합용 인덱스 테이블 (표준 한글 음절 = 0xAC00 + ...)
# ---------------------------------------------------------------
CHO = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')          # 19
JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')     # 21
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')  # 28

CHO_IDX = {c: i for i, c in enumerate(CHO)}
JUNG_IDX = {v: i for i, v in enumerate(JUNG)}
JONG_IDX = {t: i for i, t in enumerate(JONG)}

# 복합모음: (첫 모음, 둘째 모음) -> 복합모음
VOWEL_COMBINE = {
    ('ㅗ', 'ㅏ'): 'ㅘ', ('ㅗ', 'ㅐ'): 'ㅙ', ('ㅗ', 'ㅣ'): 'ㅚ',
    ('ㅜ', 'ㅓ'): 'ㅝ', ('ㅜ', 'ㅔ'): 'ㅞ', ('ㅜ', 'ㅣ'): 'ㅟ',
    ('ㅡ', 'ㅣ'): 'ㅢ',
}
# 겹받침: (첫 자음, 둘째 자음) -> 겹받침
JONG_COMBINE = {
    ('ㄱ', 'ㅅ'): 'ㄳ', ('ㄴ', 'ㅈ'): 'ㄵ', ('ㄴ', 'ㅎ'): 'ㄶ',
    ('ㄹ', 'ㄱ'): 'ㄺ', ('ㄹ', 'ㅁ'): 'ㄻ', ('ㄹ', 'ㅂ'): 'ㄼ',
    ('ㄹ', 'ㅅ'): 'ㄽ', ('ㄹ', 'ㅌ'): 'ㄾ', ('ㄹ', 'ㅍ'): 'ㄿ',
    ('ㄹ', 'ㅎ'): 'ㅀ', ('ㅂ', 'ㅅ'): 'ㅄ',
}
# 겹받침 -> (앞, 뒤) 분해 (모음이 뒤따라오면 뒤 자음이 다음 초성으로 이동)
JONG_SPLIT = {v: k for k, v in JONG_COMBINE.items()}


def compose_syllable(cho: str | None, jung: str | None, jong: str) -> str:
    """초/중/종 -> 완성형 음절 1글자. 불완전하면 자모를 그대로 이어붙인다."""
    if cho is not None and jung is not None and cho in CHO_IDX and jung in JUNG_IDX:
        code = 0xAC00 + (CHO_IDX[cho] * 21 + JUNG_IDX[jung]) * 28 + JONG_IDX.get(jong, 0)
        return chr(code)
    # 불완전: 낱자 나열
    return (cho or '') + (jung or '') + (jong or '')


# ---------------------------------------------------------------
# 3) 두벌식 오토마타 (실제 IME 동작 재현)
#    feed(jamo) 로 자모를 한 타씩 흘려넣으면 음절을 조합/방출한다.
#    orphan(고아 낱자: 초성만/중성만으로 끝난 것)을 표시해 언어 우도로 쓴다.
# ---------------------------------------------------------------
@dataclass
class _Unit:
    text: str
    orphan: bool          # 완성 음절이 아니면 True (초성단독/중성단독 등)


@dataclass
class Dubeolsik:
    cho: str | None = None
    jung: str | None = None
    jong: str = ''
    out: List[_Unit] = field(default_factory=list)

    # --- 내부 상태 방출 ---
    def _emit(self):
        if self.cho is None and self.jung is None and not self.jong:
            return
        has_full = self.cho is not None and self.jung is not None
        text = compose_syllable(self.cho, self.jung, self.jong)
        self.out.append(_Unit(text=text, orphan=not has_full))
        self.cho, self.jung, self.jong = None, None, ''

    def feed(self, j: str):
        # 스페이스/기타 토큰 -> 음절 경계
        if j == '<sp>':
            self._emit()
            self.out.append(_Unit(text=' ', orphan=False))
            return
        if is_consonant(j):
            self._feed_cons(j)
        elif is_vowel(j):
            self._feed_vowel(j)
        else:
            # 알 수 없는 토큰: 경계 처리
            self._emit()
            self.out.append(_Unit(text=j, orphan=True))

    def _feed_cons(self, c: str):
        if self.cho is None and self.jung is None:
            self.cho = c                                   # 초성 시작
        elif self.cho is not None and self.jung is None:
            # 초성만 있는데 또 자음 -> 앞 초성은 고아 낱자로 방출
            self._emit()
            self.cho = c
        elif self.jung is not None and not self.jong:
            # CV 상태에서 자음 -> 종성 시도
            if c in JONG_IDX and c != '':
                self.jong = c
            else:
                self._emit()
                self.cho = c
        else:
            # 이미 종성 있음 -> 겹받침 시도
            if (self.jong, c) in JONG_COMBINE:
                self.jong = JONG_COMBINE[(self.jong, c)]
            else:
                self._emit()
                self.cho = c

    def _feed_vowel(self, v: str):
        if self.cho is None and self.jung is None:
            self.jung = v                                  # 중성 단독(→ 고아 낱자)
        elif self.cho is not None and self.jung is None:
            self.jung = v                                  # CV 완성
        elif self.jung is not None and not self.jong:
            # 복합모음 시도
            if (self.jung, v) in VOWEL_COMBINE:
                self.jung = VOWEL_COMBINE[(self.jung, v)]
            else:
                self._emit()
                self.jung = v
        else:
            # 종성 있는 상태에서 모음 -> 종성(마지막 자음)이 다음 초성으로 이동
            moved, keep = self._steal_jong()
            self.jong = keep
            self._emit()
            self.cho = moved
            self.jung = v

    def _steal_jong(self):
        """모음이 왔을 때 종성에서 다음 초성으로 넘길 자음과 남길 종성 반환."""
        if self.jong in JONG_SPLIT:            # 겹받침이면 뒤 자음만 이동
            first, second = JONG_SPLIT[self.jong]
            return second, first
        return self.jong, ''                    # 단일 종성이면 통째로 이동

    def flush(self) -> None:
        self._emit()

    def units(self) -> List[_Unit]:
        self.flush()
        return self.out


def compose(jamo_seq: List[str]) -> str:
    """자모 시퀀스 -> 조합된 한글 문자열."""
    a = Dubeolsik()
    for j in jamo_seq:
        a.feed(j)
    return ''.join(u.text for u in a.units())


def compose_units(jamo_seq: List[str]) -> List[_Unit]:
    a = Dubeolsik()
    for j in jamo_seq:
        a.feed(j)
    return a.units()


def count_orphans(jamo_seq: List[str]) -> int:
    """조합 결과에서 완성되지 못한 고아 낱자 수(언어 우도 페널티에 사용)."""
    return sum(1 for u in compose_units(jamo_seq) if u.orphan)


# ---------------------------------------------------------------
# 4) 오토마타 제약 빔서치 디코딩
#    입력: 타건별 자모 후보의 로그확률 (T, C)
#    출력: 오토마타 우도(고아 낱자 페널티)로 보정한 최적 자모 시퀀스/텍스트
# ---------------------------------------------------------------
def constrained_beam_decode(
    logprobs,                      # np.ndarray (T, C)
    idx2label: List[str],          # 클래스 idx -> 자모 라벨
    beam_width: int = 8,
    orphan_penalty: float = 2.0,   # 고아 낱자 1개당 감점(log 스케일)
    cand_per_step: int = 5,        # 각 타건에서 고려할 상위 후보 수
):
    """
    반환: (best_jamo_seq: List[str], best_text: str, best_score: float)
    점수 = Σ 분류기 logprob  -  orphan_penalty * (고아 낱자 수)
    """
    import numpy as np
    T, C = logprobs.shape
    # 각 스텝 상위 후보 idx
    topc = np.argsort(-logprobs, axis=1)[:, :cand_per_step]

    # 빔: (jamo_seq(list), cum_logprob)
    beams = [([], 0.0)]
    for t in range(T):
        new = []
        for seq, lp in beams:
            for ci in topc[t]:
                j = idx2label[ci]
                new.append((seq + [j], lp + float(logprobs[t, ci])))
        # 오토마타 우도로 재랭킹 후 상위 beam_width 유지
        scored = []
        for seq, lp in new:
            score = lp - orphan_penalty * count_orphans(seq)
            scored.append((seq, lp, score))
        scored.sort(key=lambda x: -x[2])
        beams = [(s, lp) for s, lp, _ in scored[:beam_width]]

    best_seq, best_lp = beams[0]
    best_score = best_lp - orphan_penalty * count_orphans(best_seq)
    return best_seq, compose(best_seq), best_score


def greedy_decode(logprobs, idx2label: List[str]):
    """오토마타 미적용(비교 기준선): argmax 자모를 그대로 조합."""
    import numpy as np
    idxs = np.argmax(logprobs, axis=1)
    seq = [idx2label[i] for i in idxs]
    return seq, compose(seq)


# ---------------------------------------------------------------
# 5) 역변환: 한글 텍스트 -> 두벌식 base 자모 타건 시퀀스
#    사용 위치: make_synthetic_data.py (정답 텍스트 -> 타건열 생성),
#    데이터 검증(조합-분해 왕복 테스트).
# ---------------------------------------------------------------
VOWEL_SPLIT = {v: k for k, v in VOWEL_COMBINE.items()}   # 복합모음 -> (기본, 기본)


def decompose_text(text: str) -> List[str]:
    """한글 문자열 -> 실제로 눌러야 하는 base 자모 키 시퀀스."""
    seq: List[str] = []
    for ch in text:
        if ch == ' ':
            seq.append('<sp>')
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:                     # 완성형 음절
            s = code - 0xAC00
            cho = CHO[s // (21 * 28)]
            jung = JUNG[(s % (21 * 28)) // 28]
            jong = JONG[s % 28]
            seq.append(cho)
            seq.extend(VOWEL_SPLIT.get(jung, (jung,)))
            if jong:
                seq.extend(JONG_SPLIT.get(jong, (jong,)))
        elif ch in ALL_CONSONANTS or ch in ALL_VOWELS:   # 낱자
            seq.append(ch)
        # 그 외 문자는 무시(데이터 생성용)
    return seq
