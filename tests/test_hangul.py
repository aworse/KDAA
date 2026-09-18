# -*- coding: utf-8 -*-
"""
오토마타/디코더 단위 테스트.
실행: python -m tests.test_hangul   (pytest 없이도 동작)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import hangul as H


def test_roundtrip():
    """분해->조합 왕복이 원문과 일치(겹받침/복합모음/된소리 포함)."""
    for t in ["안녕하세요", "값을 계산했다", "과학전람회", "깎아 만든 닭",
              "복합모음 과 왜 위 의", "받침 삶 넓다"]:
        seq = H.decompose_text(t)
        assert H.compose(seq) == t, f"{t} -> {H.compose(seq)}"
    print("[ok] roundtrip")


def test_jong_migration():
    """종성이 뒤 모음으로 이동하는 규칙."""
    assert H.compose(list("ㅁㅓㄱㅇㅓ")) == "먹어"
    assert H.compose(list("ㅁㅓㄱㅓ")) == "머거"
    print("[ok] jong migration")


def test_orphan_count():
    """유효 시퀀스는 고아 0, 자음 나열은 고아 다수."""
    assert H.count_orphans(list("ㅇㅏㄴ")) == 0            # 안
    assert H.count_orphans(list("ㄱㄴㄷ")) == 3            # 낱자 3
    print("[ok] orphan count")


def test_beam_fixes_invalid():
    """
    핵심 검증: argmax(greedy)가 '유효하지 않은 고아 낱자' 시퀀스를 내지만,
    오토마타 제약 빔서치는 2순위 후보로 유효 음절을 복원한다.
    시나리오: 목표 '가' = ㄱ ㅏ.
      step0: ㄱ 확실
      step1: ㅏ(정답, '가' 완성) vs ㄱ(오답, 'ㄱㄱ' 고아2) 이 근소하게 ㄱ 우세
    """
    labels = ["ㄱ", "ㅏ"]
    lp = np.log(np.array([
        [0.90, 0.10],     # step0 -> ㄱ
        [0.55, 0.45],     # step1 -> greedy는 ㄱ(오답), beam은 ㅏ(정답)
    ]))
    g_seq, g_text = H.greedy_decode(lp, {0: "ㄱ", 1: "ㅏ"})
    b_seq, b_text, _ = H.constrained_beam_decode(lp, labels, beam_width=4,
                                                 orphan_penalty=2.0)
    assert g_text != "가", f"greedy가 이미 정답이면 테스트 무의미: {g_text}"
    assert b_text == "가", f"beam 복원 실패: {b_text}"
    print(f"[ok] beam fixes invalid: greedy='{g_text}' -> beam='{b_text}'")


def test_label_space_size():
    """분류 라벨(base 자모) 개수 확인."""
    assert len(H.LABELS) == 33, len(H.LABELS)
    print(f"[ok] {len(H.LABELS)} base jamo labels")


if __name__ == "__main__":
    test_roundtrip()
    test_jong_migration()
    test_orphan_count()
    test_beam_fixes_invalid()
    test_label_space_size()
    print("\nALL TESTS PASSED")
