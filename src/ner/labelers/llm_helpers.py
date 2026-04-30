"""NER 라벨러 백엔드가 공유하는 언어 중립 헬퍼.

ko/ja *_ner_labeler.py 중복 코드에서 로직을 보존하여 추출한 함수들:
- `split_sentences(text, lang)` — 문장 분리기, 구두점 매개변수화.
- `parse_spans(raw)` — <think> 및 폴백 처리를 포함한 JSON span 리스트 추출기.
- `spans_to_bio(tokens, spans)` — 정확/부분 매칭을 사용한 공백 분리 토큰 BIO 태깅.
"""
import logging
import re
from typing import List

from labelers.labeler_base import parse_json_response

logger = logging.getLogger(__name__)


def split_sentences(text: str, lang: str = "ko") -> List[str]:
    """구두점이나 줄바꿈을 기준으로 텍스트를 문장으로 분리한다.

    Args:
        text: 분리할 입력 텍스트.
        lang: 언어 힌트. "ja"이면 일본어 구두점(。！？)을 분리 집합에 추가한다.
    """
    if lang == "ja":
        # 일본어 구두점(。！？)은 공백 없이 분리, ASCII .!?는 공백이 뒤따를 때만 분리
        # (예: "gmail.com"처럼 도메인·URL 내부의 점을 보호)
        pattern = r'(?<=[。！？])\s*|(?<=[.!?])\s+|\n+'
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
    """LLM 출력에서 JSON span 리스트를 추출한다. 실패 시 []를 반환한다.

    `labeler_base.parse_json_response`의 얇은 별칭으로, <think> 블록,
    단독 배열, `{"entities": [...]}` 래퍼를 이미 처리한다.
    """
    return parse_json_response(raw)


def spans_to_bio(tokens: List[str], spans: List[dict]) -> List[str]:
    """엔티티 span을 토큰 수준 BIO 태그로 변환한다 (공백 분리 토큰 기준).

    매칭 전략:
    1. 정확한 멀티 토큰 매칭.
    2. 토큰별 부분 문자열 매칭 (조사/助詞 부착 처리).
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

        # 1. 정확한 매칭
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == span_tokens:
                tags[i] = f"B-{entity_type}"
                for j in range(1, n):
                    tags[i + j] = f"I-{entity_type}"
                matched = True
                break

        # 2. 부분 문자열 매칭 (어근에 붙은 조사 처리)
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
