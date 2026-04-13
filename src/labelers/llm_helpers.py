"""Language-neutral helpers shared by NER labeler backends.

Lifted verbatim (logic-preserving) from ko/ja *_ner_labeler.py duplicates:
- `split_sentences(text, lang)` — sentence splitter, parameterized punctuation.
- `parse_spans(raw)` — JSON span list extractor with <think> and fallback handling.
- `spans_to_bio(tokens, spans)` — whitespace-token BIO tagging with exact+substring match.
"""
import logging
import re
from typing import List

from labelers.labeler_base import parse_json_response

logger = logging.getLogger(__name__)


def split_sentences(text: str, lang: str = "ko") -> List[str]:
    """Split text into sentences on punctuation or newlines.

    Args:
        text: Input text to split.
        lang: Language hint. "ja" adds Japanese punctuation (。！？) to the split set.
    """
    if lang == "ja":
        pattern = r'(?<=[.!?。！？])\s*|\n+'
    else:
        pattern = r'(?<=[.!?])\s+|\n+'
    parts = re.split(pattern, text.strip())
    sentences: List[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= 10:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences if sentences else [text.strip()]


def parse_spans(raw: str) -> List[dict]:
    """Extract JSON span list from LLM output. Returns [] on failure.

    Delegates to `labeler_base.parse_json_response` for the core parse, then
    applies a `[...]` regex fallback if the primary parse returns [].
    """
    # parse_json_response already handles <think>, bare arrays, wrapped dicts.
    result = parse_json_response(raw)
    if result:
        return result

    # Fallback: extract first [..] substring
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            import json
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except Exception:  # noqa: BLE001
            pass
    return []


def spans_to_bio(tokens: List[str], spans: List[dict]) -> List[str]:
    """Convert entity spans to token-level BIO tags (whitespace-split tokens).

    Matching strategy:
    1. Exact multi-token match.
    2. Substring match per token (handles attached particles / 助詞).
    """
    tags = ["O"] * len(tokens)
    for span in spans:
        entity_text = span.get("text", "").strip()
        entity_type = span.get("type", "").strip()
        if not entity_text or not entity_type:
            continue
        span_tokens = entity_text.split()
        n = len(span_tokens)
        if n == 0:
            continue
        matched = False

        # 1. Exact match
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == span_tokens:
                tags[i] = f"B-{entity_type}"
                for j in range(1, n):
                    tags[i + j] = f"I-{entity_type}"
                matched = True
                break

        # 2. Substring match (handles particles attached to root)
        if not matched:
            for i in range(len(tokens) - n + 1):
                if all(
                    sp in tokens[i + j] or tokens[i + j] in sp
                    for j, sp in enumerate(span_tokens)
                ):
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, n):
                        tags[i + j] = f"I-{entity_type}"
                    matched = True
                    break

        if not matched:
            logger.warning("Span '%s' not found in tokens: %s", entity_text, tokens)
    return tags
