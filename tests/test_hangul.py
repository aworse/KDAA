# -*- coding: utf-8 -*-
"""
Unit tests for the automaton / decoder.
Run: python -m tests.test_hangul   (works without pytest)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import hangul as H


def test_roundtrip():
    """decompose->compose round-trip equals the original (compound final/vowel, tense)."""
    for t in ["안녕하세요", "값을 계산했다", "과학전람회", "깎아 만든 닭",
              "복합모음 과 왜 위 의", "받침 삶 넓다"]:
        seq = H.decompose_text(t)
        assert H.compose(seq) == t, f"{t} -> {H.compose(seq)}"
    print("[ok] roundtrip")


def test_jong_migration():
    """The rule where a final migrates to the next syllable's leading consonant."""
    assert H.compose(list("ㅁㅓㄱㅇㅓ")) == "먹어"
    assert H.compose(list("ㅁㅓㄱㅓ")) == "머거"
    print("[ok] jong migration")


def test_orphan_count():
    """Valid sequence has 0 orphans; a run of consonants has many."""
    assert H.count_orphans(list("ㅇㅏㄴ")) == 0            # 안
    assert H.count_orphans(list("ㄱㄴㄷ")) == 3            # 3 lone jamo
    print("[ok] orphan count")


def test_beam_fixes_invalid():
    """
    Core check: argmax (greedy) yields an INVALID orphan sequence, but the
    automaton-constrained beam recovers a valid syllable from the 2nd candidate.
    Scenario: target '가' = ㄱ ㅏ.
      step0: ㄱ certain
      step1: ㅏ (correct, completes '가') vs ㄱ (wrong, 'ㄱㄱ' = 2 orphans),
             with ㄱ marginally higher.
    """
    labels = ["ㄱ", "ㅏ"]
    lp = np.log(np.array([
        [0.90, 0.10],     # step0 -> ㄱ
        [0.55, 0.45],     # step1 -> greedy picks ㄱ (wrong), beam picks ㅏ (correct)
    ]))
    g_seq, g_text = H.greedy_decode(lp, {0: "ㄱ", 1: "ㅏ"})
    b_seq, b_text, _ = H.constrained_beam_decode(lp, labels, beam_width=4,
                                                 orphan_penalty=2.0)
    assert g_text != "가", f"test is moot if greedy is already correct: {g_text}"
    assert b_text == "가", f"beam failed to recover: {b_text}"
    print(f"[ok] beam fixes invalid: greedy='{g_text}' -> beam='{b_text}'")


def test_label_space_size():
    """Number of classifier labels (base jamo)."""
    assert len(H.LABELS) == 33, len(H.LABELS)
    print(f"[ok] {len(H.LABELS)} base jamo labels")


if __name__ == "__main__":
    test_roundtrip()
    test_jong_migration()
    test_orphan_count()
    test_beam_fixes_invalid()
    test_label_space_size()
    print("\nALL TESTS PASSED")
