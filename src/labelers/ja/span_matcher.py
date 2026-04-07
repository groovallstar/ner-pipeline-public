"""Convert LLM entity spans to character offset spans.

The LLM returns entities as {"text": "東京", "type": "地名"} without position info.
This module finds the character offsets in the original text.

Algorithm (substring-safe):
1. Find all occurrences of each entity text in the original text
2. Sort entities by text length DESCENDING (longest first)
3. Assign each entity to the leftmost unconsumed position
4. Track consumed character ranges to prevent overlap
"""
import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def match_spans(original_text: str, llm_spans: List[dict]) -> List[dict]:
    """Match LLM entity spans to character offsets in original text.

    Args:
        original_text: The original sentence text.
        llm_spans: LLM output spans, each {"text": str, "type": str}.

    Returns:
        Matched spans with offsets: [{"text": str, "type": str, "start": int, "end": int}]
    """
    if not original_text or not llm_spans:
        return []

    # Sort by text length descending to match longest entities first
    indexed_spans = sorted(
        enumerate(llm_spans),
        key=lambda x: len(x[1].get("text", "")),
        reverse=True,
    )

    consumed: Set[int] = set()  # consumed character indices
    results: List[Tuple[int, dict]] = []  # (original_index, matched_span)

    for orig_idx, span in indexed_spans:
        text = span.get("text", "").strip()
        etype = span.get("type", "").strip()
        if not text or not etype:
            continue

        # 1. Exact match
        candidates = _find_all_occurrences(original_text, text)

        # 2. Fallback: remove internal whitespace (LLM sometimes adds spaces)
        if not candidates:
            collapsed = re.sub(r"\s+", "", text)
            if collapsed != text:
                candidates = _find_all_occurrences(original_text, collapsed)
                if candidates:
                    text = collapsed

        # 3. Fallback: strip Japanese particles/suffixes and retry
        if not candidates:
            stripped = _strip_particles(text)
            if stripped != text:
                candidates = _find_all_occurrences(original_text, stripped)
                if candidates:
                    text = stripped

        # Pick the leftmost candidate whose range is fully unconsumed
        matched = False
        for start, end in candidates:
            span_range = set(range(start, end))
            if not span_range & consumed:  # no overlap with consumed
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

    # Sort results back to original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# Japanese particles and suffixes that LLMs sometimes include
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
