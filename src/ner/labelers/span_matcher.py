"""LLM 엔티티 span을 문자 오프셋 span으로 변환한다.

LLM은 위치 정보 없이 {"text": "東京", "type": "地名"} 형태로 엔티티를 반환한다.
이 모듈은 원문 텍스트에서 문자 오프셋을 찾는다.

알고리즘 (부분 문자열 안전):
1. 원문에서 각 엔티티 텍스트의 모든 출현 위치를 찾는다
2. 텍스트 길이 내림차순으로 엔티티를 정렬한다 (긴 것 우선)
3. 각 엔티티를 가장 왼쪽의 미소비 위치에 할당한다
4. 소비된 문자 범위를 추적하여 중복을 방지한다

현재 particle/suffix strip 폴백은 일본어 지향이며 VI/KO 텍스트엔 무해한 no-op이다.
"""
import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def match_spans(original_text: str, llm_spans: List[dict]) -> List[dict]:
    """LLM 엔티티 span을 원문 텍스트의 문자 오프셋에 매핑한다.

    Args:
        original_text: 원문 문장 텍스트.
        llm_spans: LLM 출력 span, 각각 {"text": str, "type": str} 형태.

    Returns:
        오프셋이 포함된 매칭 span: [{"text": str, "type": str, "start": int, "end": int}]
    """
    if not original_text or not llm_spans:
        return []

    # 가장 긴 엔티티부터 매칭하기 위해 텍스트 길이 내림차순으로 정렬한다
    indexed_spans = sorted(
        enumerate(llm_spans),
        key=lambda x: len(x[1].get("text", "")),
        reverse=True,
    )

    consumed: Set[int] = set()  # 소비된 문자 인덱스
    results: List[Tuple[int, dict]] = []  # (원래 인덱스, 매칭된 span)

    for orig_idx, span in indexed_spans:
        text = span.get("text", "").strip()
        etype = span.get("type", "").strip()
        if not text or not etype:
            continue

        # 1. 정확한 매칭
        candidates = _find_all_occurrences(original_text, text)

        # 2. 폴백: 내부 공백 제거 (LLM이 때때로 공백을 추가함)
        if not candidates:
            collapsed = re.sub(r"\s+", "", text)
            if collapsed != text:
                candidates = _find_all_occurrences(original_text, collapsed)
                if candidates:
                    text = collapsed

        # 3. 폴백: 일본어 조사/접미사를 제거하고 재시도
        if not candidates:
            stripped = _strip_particles(text)
            if stripped != text:
                candidates = _find_all_occurrences(original_text, stripped)
                if candidates:
                    text = stripped

        # 완전히 미소비된 범위의 가장 왼쪽 후보를 선택한다
        matched = False
        for start, end in candidates:
            span_range = set(range(start, end))
            if not span_range & consumed:  # 소비된 범위와 겹치지 않는 경우
                consumed |= span_range
                results.append((orig_idx, {
                    "text": text,
                    "type": etype,
                    "start": start,
                    "end": end,
                }))
                matched = True
                break

        if not matched:
            logger.warning("Span '%s' (%s) not found in text: %s", text, etype, original_text[:80])

    # 결과를 원래 순서로 정렬한다
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# LLM이 때때로 포함하는 일본어 조사와 접미사
_PARTICLES = ("は", "が", "を", "に", "で", "と", "の", "へ", "から", "まで", "も", "や", "より")
_SUFFIXES = ("氏", "さん", "君", "ちゃん", "様")


def _strip_particles(text: str) -> str:
    """Strip trailing Japanese particles/suffixes from entity text."""
    for p in sorted(_PARTICLES, key=len, reverse=True):
        if text.endswith(p) and len(text) > len(p):
            text = text[:-len(p)]
            break
    for s in _SUFFIXES:
        if text.endswith(s) and len(text) > len(s):
            text = text[:-len(s)]
            break
    return text


def _find_all_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
    """Find all (start, end) positions of substring in text."""
    positions = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(substring)))
        start = idx + 1
    return positions
