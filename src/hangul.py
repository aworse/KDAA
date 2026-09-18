# -*- coding: utf-8 -*-
"""
Dubeolsik Hangul automaton.
Used by:
  - decode.py: compose jamo sequences into syllables and run constrained beam search
  - evaluate.py: compute syllable-recovery rate with/without the automaton filter
  - dataset.py: validate the label (jamo) set

Core idea (KDAA's differentiator):
  An English spell-checker corrects probabilistically, but the Dubeolsik
  cho/jung/jong composition rules act as an almost *deterministic* filter.
  They discard impossible jamo sequences outright, so they constrain the acoustic
  classifier more strongly than the language models used in English ASCA work.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------
# 1) Dubeolsik key map (physical QWERTY key -> jamo)
#    The classifier's label is the *base jamo produced by one keystroke*.
#    Compound vowels (ㅘ etc.) and compound finals (ㄳ etc.) are NOT single keys;
#    they come from two keystrokes and are assembled by the automaton, so they
#    are not classifier labels.
# ---------------------------------------------------------------
KEYMAP_BASE = {
    'q': 'ㅂ', 'w': 'ㅈ', 'e': 'ㄷ', 'r': 'ㄱ', 't': 'ㅅ',
    'y': 'ㅛ', 'u': 'ㅕ', 'i': 'ㅑ', 'o': 'ㅐ', 'p': 'ㅔ',
    'a': 'ㅁ', 's': 'ㄴ', 'd': 'ㅇ', 'f': 'ㄹ', 'g': 'ㅎ',
    'h': 'ㅗ', 'j': 'ㅓ', 'k': 'ㅏ', 'l': 'ㅣ',
    'z': 'ㅋ', 'x': 'ㅌ', 'c': 'ㅊ', 'v': 'ㅍ',
    'b': 'ㅠ', 'n': 'ㅜ', 'm': 'ㅡ',
}
KEYMAP_SHIFT = {  # tense consonants / ㅒㅖ (require Shift)
    'Q': 'ㅃ', 'W': 'ㅉ', 'E': 'ㄸ', 'R': 'ㄲ', 'T': 'ㅆ',
    'O': 'ㅒ', 'P': 'ㅖ',
}

# Reverse map (jamo -> physical key), in case class labels are keyed by position.
JAMO_TO_KEY = {v: k for k, v in KEYMAP_BASE.items()}
JAMO_TO_KEY.update({v: k for k, v in KEYMAP_SHIFT.items()})

# The 33 base jamo used as classifier labels + special tokens.
CONSONANTS_BASE = list('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ')
CONSONANTS_TENSE = list('ㄲㄸㅃㅆㅉ')            # tense consonants
VOWELS_BASE = list('ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ')
VOWELS_SHIFT = list('ㅒㅖ')
SPECIAL = ['<sp>']                              # space (optional)

LABELS = CONSONANTS_BASE + CONSONANTS_TENSE + VOWELS_BASE + VOWELS_SHIFT
ALL_CONSONANTS = set(CONSONANTS_BASE + CONSONANTS_TENSE)
ALL_VOWELS = set(VOWELS_BASE + VOWELS_SHIFT)


def is_consonant(j: str) -> bool:
    return j in ALL_CONSONANTS


def is_vowel(j: str) -> bool:
    return j in ALL_VOWELS


# ---------------------------------------------------------------
# 2) Unicode composition tables (standard syllable = 0xAC00 + ...)
# ---------------------------------------------------------------
CHO = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')          # 19 leading consonants
JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')     # 21 medial vowels
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')  # 28 finals

CHO_IDX = {c: i for i, c in enumerate(CHO)}
JUNG_IDX = {v: i for i, v in enumerate(JUNG)}
JONG_IDX = {t: i for i, t in enumerate(JONG)}

# Compound vowels: (first vowel, second vowel) -> compound vowel
VOWEL_COMBINE = {
    ('ㅗ', 'ㅏ'): 'ㅘ', ('ㅗ', 'ㅐ'): 'ㅙ', ('ㅗ', 'ㅣ'): 'ㅚ',
    ('ㅜ', 'ㅓ'): 'ㅝ', ('ㅜ', 'ㅔ'): 'ㅞ', ('ㅜ', 'ㅣ'): 'ㅟ',
    ('ㅡ', 'ㅣ'): 'ㅢ',
}
# Compound finals: (first consonant, second consonant) -> compound final
JONG_COMBINE = {
    ('ㄱ', 'ㅅ'): 'ㄳ', ('ㄴ', 'ㅈ'): 'ㄵ', ('ㄴ', 'ㅎ'): 'ㄶ',
    ('ㄹ', 'ㄱ'): 'ㄺ', ('ㄹ', 'ㅁ'): 'ㄻ', ('ㄹ', 'ㅂ'): 'ㄼ',
    ('ㄹ', 'ㅅ'): 'ㄽ', ('ㄹ', 'ㅌ'): 'ㄾ', ('ㄹ', 'ㅍ'): 'ㄿ',
    ('ㄹ', 'ㅎ'): 'ㅀ', ('ㅂ', 'ㅅ'): 'ㅄ',
}
# compound final -> (first, second); when a vowel follows, the 2nd moves to next cho
JONG_SPLIT = {v: k for k, v in JONG_COMBINE.items()}


def compose_syllable(cho: str | None, jung: str | None, jong: str) -> str:
    """Compose cho/jung/jong into one syllable char; concatenate jamo if incomplete."""
    if cho is not None and jung is not None and cho in CHO_IDX and jung in JUNG_IDX:
        code = 0xAC00 + (CHO_IDX[cho] * 21 + JUNG_IDX[jung]) * 28 + JONG_IDX.get(jong, 0)
        return chr(code)
    # incomplete: list the jamo as-is
    return (cho or '') + (jung or '') + (jong or '')


# ---------------------------------------------------------------
# 3) Dubeolsik automaton (reproduces real IME behavior).
#    feed(jamo) streams one keystroke at a time and composes/emits syllables.
#    'orphan' jamo (a lone leading consonant / lone vowel) are flagged so they
#    can be used as a language prior.
# ---------------------------------------------------------------
@dataclass
class _Unit:
    text: str
    orphan: bool          # True if not a complete syllable (lone cho / lone jung, etc.)


@dataclass
class Dubeolsik:
    cho: str | None = None
    jung: str | None = None
    jong: str = ''
    out: List[_Unit] = field(default_factory=list)

    # --- emit the current buffer ---
    def _emit(self):
        if self.cho is None and self.jung is None and not self.jong:
            return
        has_full = self.cho is not None and self.jung is not None
        text = compose_syllable(self.cho, self.jung, self.jong)
        self.out.append(_Unit(text=text, orphan=not has_full))
        self.cho, self.jung, self.jong = None, None, ''

    def feed(self, j: str):
        # space / other token -> syllable boundary
        if j == '<sp>':
            self._emit()
            self.out.append(_Unit(text=' ', orphan=False))
            return
        if is_consonant(j):
            self._feed_cons(j)
        elif is_vowel(j):
            self._feed_vowel(j)
        else:
            # unknown token: treat as boundary
            self._emit()
            self.out.append(_Unit(text=j, orphan=True))

    def _feed_cons(self, c: str):
        if self.cho is None and self.jung is None:
            self.cho = c                                   # start leading consonant
        elif self.cho is not None and self.jung is None:
            # leading consonant then another consonant -> emit the first as an orphan
            self._emit()
            self.cho = c
        elif self.jung is not None and not self.jong:
            # CV state, consonant arrives -> try as final
            if c in JONG_IDX and c != '':
                self.jong = c
            else:
                self._emit()
                self.cho = c
        else:
            # already have a final -> try to form a compound final
            if (self.jong, c) in JONG_COMBINE:
                self.jong = JONG_COMBINE[(self.jong, c)]
            else:
                self._emit()
                self.cho = c

    def _feed_vowel(self, v: str):
        if self.cho is None and self.jung is None:
            self.jung = v                                  # lone vowel (-> orphan)
        elif self.cho is not None and self.jung is None:
            self.jung = v                                  # complete CV
        elif self.jung is not None and not self.jong:
            # try a compound vowel
            if (self.jung, v) in VOWEL_COMBINE:
                self.jung = VOWEL_COMBINE[(self.jung, v)]
            else:
                self._emit()
                self.jung = v
        else:
            # vowel after a final -> the final (last consonant) migrates to next cho
            moved, keep = self._steal_jong()
            self.jong = keep
            self._emit()
            self.cho = moved
            self.jung = v

    def _steal_jong(self):
        """When a vowel arrives, return (consonant moved to next cho, final kept)."""
        if self.jong in JONG_SPLIT:            # compound final: only 2nd moves
            first, second = JONG_SPLIT[self.jong]
            return second, first
        return self.jong, ''                    # single final: the whole thing moves

    def flush(self) -> None:
        self._emit()

    def units(self) -> List[_Unit]:
        self.flush()
        return self.out


def compose(jamo_seq: List[str]) -> str:
    """jamo sequence -> composed Hangul string."""
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
    """Number of incomplete 'orphan' units (used as a language-prior penalty)."""
    return sum(1 for u in compose_units(jamo_seq) if u.orphan)


# ---------------------------------------------------------------
# 4) Automaton-constrained beam search decoding.
#    Input: per-keystroke jamo-candidate log-probabilities (T, C)
#    Output: the best jamo sequence / text after re-ranking by the automaton
#            prior (orphan penalty).
# ---------------------------------------------------------------
def constrained_beam_decode(
    logprobs,                      # np.ndarray (T, C)
    idx2label: List[str],          # class idx -> jamo label
    beam_width: int = 8,
    orphan_penalty: float = 2.0,   # penalty per orphan jamo (log scale)
    cand_per_step: int = 5,        # top candidates considered per keystroke
):
    """
    Returns: (best_jamo_seq: List[str], best_text: str, best_score: float)
    score = sum(classifier logprob) - orphan_penalty * (number of orphan jamo)
    """
    import numpy as np
    T, C = logprobs.shape
    # top candidates per step
    topc = np.argsort(-logprobs, axis=1)[:, :cand_per_step]

    # beam entries: (jamo_seq(list), cum_logprob)
    beams = [([], 0.0)]
    for t in range(T):
        new = []
        for seq, lp in beams:
            for ci in topc[t]:
                j = idx2label[ci]
                new.append((seq + [j], lp + float(logprobs[t, ci])))
        # re-rank by the automaton prior, keep top beam_width
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
    """No automaton (baseline): compose the argmax jamo directly."""
    import numpy as np
    idxs = np.argmax(logprobs, axis=1)
    seq = [idx2label[i] for i in idxs]
    return seq, compose(seq)


# ---------------------------------------------------------------
# 5) Inverse transform: Hangul text -> Dubeolsik base-jamo keystroke sequence.
#    Used by: make_synthetic_data.py (ground-truth text -> keystrokes),
#    dataset validation (compose/decompose round-trip).
# ---------------------------------------------------------------
VOWEL_SPLIT = {v: k for k, v in VOWEL_COMBINE.items()}   # compound vowel -> (base, base)


def decompose_text(text: str) -> List[str]:
    """Hangul string -> the sequence of base jamo keys actually pressed."""
    seq: List[str] = []
    for ch in text:
        if ch == ' ':
            seq.append('<sp>')
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:                     # composed syllable
            s = code - 0xAC00
            cho = CHO[s // (21 * 28)]
            jung = JUNG[(s % (21 * 28)) // 28]
            jong = JONG[s % 28]
            seq.append(cho)
            seq.extend(VOWEL_SPLIT.get(jung, (jung,)))
            if jong:
                seq.extend(JONG_SPLIT.get(jong, (jong,)))
        elif ch in ALL_CONSONANTS or ch in ALL_VOWELS:   # bare jamo
            seq.append(ch)
        # anything else is ignored (for data generation)
    return seq
