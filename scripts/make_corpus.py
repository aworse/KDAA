# -*- coding: utf-8 -*-
"""
녹음용 프롬프트 코퍼스 생성 + 자모 커버리지 검증.
사용 위치 / 실행:
  python -m scripts.make_corpus --config config.yaml --per-jamo 25
동작:
  1) 자연문장 + 희귀자모(된소리/ㅋㅌㅍ/ㅒㅖ/ㅛㅠ/겹받침/복합모음) 드릴 문장을 합쳐
     모든 33개 base 자모가 --per-jamo 회 이상 등장하도록 프롬프트 세트를 구성
  2) scripts/prompts_ko.txt 저장 (한 줄 = 한 프롬프트, 참가자가 이 순서대로 타이핑)
  3) 자모별 등장 횟수를 흑백 figure(docs/corpus_coverage.png)로 출력
목표: 참가자 1명이 이 코퍼스를 1회 타이핑하면 자모당 >=25타건 확보(Harrison 기준).
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hangul import decompose_text, LABELS

# --- 자연문장(일상/연구 맥락) ---
NATURAL = [
    "안녕하세요 오늘도 좋은 하루 보내세요",
    "우리 학교 도서관은 넓고 조용해서 공부하기 좋다",
    "음향 사이드채널 공격은 키보드 소리로 입력을 추정한다",
    "마이크 거리에 따라 복원 정확도가 크게 달라진다",
    "배경 소음이 많으면 신호 대 잡음비가 낮아진다",
    "두벌식 자판은 초성 중성 종성으로 글자를 만든다",
    "실험 참가자는 동의서를 작성한 뒤 녹음을 시작한다",
    "결과를 표로 정리하고 그림으로 시각화했다",
    "컴퓨터 공학과 전자공학을 함께 공부하고 싶다",
    "회로 이론과 미적분을 병행해서 배우는 중이다",
    "청소년 과학 탐구 대회에 출품할 주제를 정했다",
    "데이터를 모으고 모델을 학습시켜 평가한다",
    "손가락 위치와 타건 세기가 소리를 바꾼다",
    "잔향이 심한 방에서는 타건음이 뭉개진다",
    "정답 문장을 알면 자모 순서를 미리 계산할 수 있다",
    "예산은 오십만 원이며 마이크 두 종을 구매한다",
]

# --- 희귀/특수 자모 드릴 문장 ---
DRILL = [
    # 된소리 ㄲㄸㅃㅆㅉ
    "까치 딸기 빵 쌀 짜장 꿀 떡 뿌리 씨앗 찌개",
    "깨끗한 땅에 빨간 꽃이 피었고 짝꿍이 웃었다",
    "꼬마가 뚜껑을 따고 빼빼로를 씹으며 찡그렸다",
    # ㅋㅌㅍ (거센소리 자음)
    "커피 코드 카메라 타조 통계 트럭 파도 편지 표범 프린터",
    "콘크리트 텐트 파이프 키트 토치 펜치 캠프 파크",
    # ㅒㅖ (Shift 모음)
    "얘기 예의 걔네 계획 예술 얘들아 예년 계좌 폐교",
    # ㅛㅠ (드문 기본 모음)
    "교육 유리 요리 규칙 표류 뉴스 쇼핑 튜브 육교 향유",
    # 겹받침 / 받침 다양
    "값 닭 삶 넓다 읽다 앉다 많다 훑다 밟다 옳다 없다 굶다",
    "흙 위에 앉아 삶은 닭을 나눠 먹으니 값이 아깝지 않다",
    # 복합모음 과 왜 외 워 웨 위 의
    "과학 왜냐하면 외국 원인 웨딩 위치 의사 화요일 궤도 취미",
    "괴물 뒷과 훼손 귀가 희망 의의 왕관 궂은 쥐",
    # ㅐㅔ 구분
    "개미 게시판 배게 세배 재미 제비 해변 헤엄",
]


def coverage(lines):
    c = Counter()
    for ln in lines:
        for j in decompose_text(ln):
            if j != "<sp>":
                c[j] += 1
    return c


def topup(lines, per_jamo):
    """부족한 자모를 채우는 반복 드릴 라인 자동 추가."""
    # 각 자모를 확실히 포함하는 최소 단어(초성/중성/종성 배치)
    filler_word = {
        'ㄱ': '가각', 'ㄴ': '나난', 'ㄷ': '다닫', 'ㄹ': '라랄', 'ㅁ': '마맘',
        'ㅂ': '바밥', 'ㅅ': '사삿', 'ㅇ': '아앙', 'ㅈ': '자잦', 'ㅊ': '차찾',
        'ㅋ': '카칵', 'ㅌ': '타탓', 'ㅍ': '파팦', 'ㅎ': '하핳',
        'ㄲ': '까깎', 'ㄸ': '따', 'ㅃ': '빠', 'ㅆ': '싸쌌', 'ㅉ': '짜',
        'ㅏ': '아', 'ㅐ': '애', 'ㅑ': '야', 'ㅓ': '어', 'ㅔ': '에',
        'ㅕ': '여', 'ㅗ': '오', 'ㅛ': '요', 'ㅜ': '우', 'ㅠ': '유',
        'ㅡ': '으', 'ㅣ': '이', 'ㅒ': '얘', 'ㅖ': '예',
    }
    extra = []
    for _ in range(200):
        c = coverage(lines + extra)
        deficits = [j for j in LABELS if c[j] < per_jamo]
        if not deficits:
            break
        # 부족 자모들을 한 줄에 모아 드릴
        line = ' '.join(filler_word.get(j, '') * 2 for j in deficits[:12])
        extra.append(line)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-jamo", type=int, default=25, help="자모당 목표 최소 등장수")
    ap.add_argument("--out", default="scripts/prompts_ko.txt")
    a = ap.parse_args()

    lines = NATURAL + DRILL
    extra = topup(lines, a.per_jamo)
    lines = lines + extra
    c = coverage(lines)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total = sum(c.values())
    mn = min(c[j] for j in LABELS)
    print(f"[corpus] 프롬프트 {len(lines)}줄, 총 타건 {total}, 자모 최소 등장 {mn}")
    missing = [j for j in LABELS if c[j] == 0]
    print(f"[coverage] 미커버 자모: {missing or '없음'}  (드릴 자동추가 {len(extra)}줄)")

    # 흑백 figure
    try:
        from src import figures as FIG
        import matplotlib.pyplot as plt
        import numpy as np
        vals = [c[j] for j in LABELS]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.bar(range(len(LABELS)), vals, color="0.3", edgecolor="black")
        ax.axhline(a.per_jamo, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylabel("count / participant")
        ax.set_title(f"Jamo coverage (target ≥ {a.per_jamo})")
        fig.tight_layout()
        os.makedirs("docs", exist_ok=True)
        fig.savefig("docs/corpus_coverage.png", dpi=130)
        print("[figure] docs/corpus_coverage.png")
    except Exception as e:
        print("[figure] skipped:", e)


if __name__ == "__main__":
    main()
