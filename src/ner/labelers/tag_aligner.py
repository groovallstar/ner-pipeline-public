"""gold 태그와 예측 태그 정렬 유틸리티.

사전 토큰화된 gold 데이터와 재구성 텍스트에서 공백 분리된
LLM 라벨러 출력 간의 토큰화 불일치를 처리한다.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 언어별 태그 정규화 맵
# 한국어: canonical 명칭(PER/LOC/ORG/DAT)으로 표준화. KLUE 약어(PS/LC/OG/DT)
# 입력도 canonical로 수렴시킨다. TI/QT는 매핑하지 않고 통과(후속 gold 단계 드롭).
_TAG_NORMALIZE_MAP_KO = {
    "PS": "PER",
    "LC": "LOC",
    "OG": "ORG",
    "DT": "DAT",
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "DATE": "DAT",
}

# 일본어 (KLUE 약어 표준): PER→PS, LOC→LC 등으로 정규화
_TAG_NORMALIZE_MAP_JA = {
    "PER": "PS",
    "LOC": "LC",
    "ORG": "OG",
    "DAT": "DT",
    "TIM": "TI",
    "NUM": "QT",
    "QUANTITY": "QT",
    "DATE": "DT",
    "TIME": "TI",
    "PERSON": "PS",
    "LOCATION": "LC",
    "ORGANIZATION": "OG",
}

# 베트남어: PER, LOC, ORG로 표준화 (국제 표준 태그 유지)
_TAG_NORMALIZE_MAP_VI = {
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "MISCELLANEOUS": "MISC",
}

_TAG_NORMALIZE_MAPS = {
    "ko": _TAG_NORMALIZE_MAP_KO,
    "ja": _TAG_NORMALIZE_MAP_JA,
    "vi": _TAG_NORMALIZE_MAP_VI,
}


def normalize_tag(tag: str, lang: str = "ko") -> str:
    """BIO 태그를 언어별 표준으로 정규화한다.

    한국어: B-PS → B-PER, B-LC → B-LOC (canonical 표준)
    일본어: B-PER → B-PS, B-LOC → B-LC (KLUE 약어 표준)
    베트남어: B-PERSON → B-PER, B-LOCATION → B-LOC (국제 표준)
    """
    if tag == "O" or not tag:
        return "O"
    parts = tag.split("-", 1)
    if len(parts) != 2:
        return tag
    prefix, entity = parts
    entity_upper = entity.upper()
    tag_map = _TAG_NORMALIZE_MAPS.get(lang, _TAG_NORMALIZE_MAP_KO)
    normalized = tag_map.get(entity_upper, entity)
    return f"{prefix}-{normalized}"


def normalize_tags(tags: List[str], lang: str = "ko") -> List[str]:
    return [normalize_tag(t, lang=lang) for t in tags]


def _find_ignore_spaces(text: str, pattern: str, start: int = 0) -> tuple:
    """양쪽의 공백을 무시하고 텍스트에서 패턴을 찾는다.

    `text` 내의 (start_offset, end_offset)을 반환하며, 없으면 (-1, -1)을 반환한다.
    예: _find_ignore_spaces("지난 19일", "지난19일") → (0, 5)
    """
    pattern_nospace = pattern.replace(" ", "")
    if not pattern_nospace:
        return (-1, -1)
    p_len = len(pattern_nospace)
    p_idx = 0
    match_start = -1
    for i in range(start, len(text)):
        if text[i] == " ":
            continue
        if text[i] == pattern_nospace[p_idx]:
            if p_idx == 0:
                match_start = i
            p_idx += 1
            if p_idx == p_len:
                return (match_start, i + 1)
        else:
            if match_start != -1:
                p_idx = 0
                match_start = -1
    return (-1, -1)


def extract_spans_from_bio(
    tokens: List[str], tags: List[str], lang: str = "ko"
) -> List[dict]:
    """BIO 태그에서 엔티티 span을 추출한다.

    음절 수준 토큰(KLUE 방식, 공백 토큰 포함)과
    단어 수준 토큰(WikiANN 방식, 공백 토큰 없음)을 모두 처리한다.
    [{"text": "경찰", "type": "OG"}, ...] 형태로 반환한다.
    """
    tags = normalize_tags(tags, lang=lang)
    # 토큰 수준 감지: 순수 공백 토큰이 있으면 음절 수준이다
    has_space_tokens = any(t.strip() == "" for t in tokens)
    joiner = "" if has_space_tokens else " "

    spans: List[dict] = []
    current_chars: List[str] = []
    current_type: str = ""

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            current_chars = [tok] if tok.strip() else []
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type and tag[2:] == current_type:
            if tok.strip():
                current_chars.append(tok)
        else:
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            current_chars = []
            current_type = ""

    if current_chars and current_type:
        spans.append({"text": joiner.join(current_chars), "type": current_type})

    return spans


class TagAligner:
    """예측 BIO 태그를 gold 토큰 경계에 정렬한다."""

    @staticmethod
    def reconstruct_text(tokens: List[str]) -> str:
        """음절 토큰에서 원문 텍스트를 재구성한다.

        KLUE NER 토큰은 명시적 공백 토큰(' ' 또는 '')을 포함한다.
        음절 토큰은 직접 연결하며, 공백 토큰만 출력에서 공백을 생성한다.
        """
        has_space_tokens = any(t.strip() == "" for t in tokens)
        if has_space_tokens:
            # 명시적 공백 마커가 있는 KLUE 방식 음절 토큰
            return "".join(t if t.strip() != "" else " " for t in tokens)
        # 단어 수준 토큰 (예: kor_ner) — 이전과 같이 공백으로 연결
        return " ".join(tokens)

    @staticmethod
    def align(
        gold_tokens: List[str],
        gold_tags: List[str],
        pred_tokens: List[str],
        pred_tags: List[str],
        lang: str = "ko",
    ) -> Tuple[List[str], List[str]]:
        """예측 태그를 gold 토큰 그리드에 정렬한다.

        동일 길이의 (gold_tags, aligned_pred_tags)를 반환한다.
        """
        gold_tags = normalize_tags(gold_tags, lang=lang)
        pred_tags = normalize_tags(pred_tags, lang=lang)

        # 빠른 경로: 토큰화가 정확히 일치하는 경우
        if gold_tokens == pred_tokens:
            return gold_tags, pred_tags

        # 문자 오프셋 기반 정렬
        aligned_pred = TagAligner._char_offset_align(
            gold_tokens, pred_tokens, pred_tags
        )
        return gold_tags, aligned_pred

    @staticmethod
    def align_syllable_to_word(
        syllable_tokens: List[str],
        syllable_tags: List[str],
        word_tokens: List[str],
        word_tags: List[str],
        lang: str = "ko",
    ) -> Tuple[List[str], List[str]]:
        """단어 수준 예측 태그를 음절 수준 gold 토큰 그리드에 정렬한다.

        KLUE NER은 단어 사이에 공백 토큰을 포함한 음절(음절) 토큰을 사용한다.
        모델은 단어 수준 예측을 생성한다. 이 함수는 단어 예측을 음절 그리드로
        다시 매핑한다.
        """
        syllable_tags = normalize_tags(syllable_tags, lang=lang)
        word_tags = normalize_tags(word_tags, lang=lang)

        # 어느 음절 토큰이 어느 단어에 속하는지 재구성한다
        # 공백 토큰(' ' 또는 '')은 구분자 역할을 한다
        aligned_pred = []
        word_idx = 0
        in_word_pos = 0  # 현재 단어 내의 문자 위치

        for syl_tok in syllable_tokens:
            if syl_tok.strip() == "":
                # Space token -> 'O'
                aligned_pred.append("O")
                if word_idx < len(word_tokens):
                    word_idx += 1
                    in_word_pos = 0
                continue

            if word_idx < len(word_tags):
                tag = word_tags[word_idx]
                # 단어의 첫 음절은 태그를 그대로 사용한다
                # 이후 음절: B-X → I-X
                if in_word_pos > 0 and tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                aligned_pred.append(tag)
            else:
                aligned_pred.append("O")
            in_word_pos += 1

        return syllable_tags, aligned_pred

    @staticmethod
    def spans_to_syllable_bio(
        text: str,
        syllable_tokens: List[str],
        spans: List[dict],
        lang: str = "ko",
    ) -> List[str]:
        """원시 엔티티 span을 문자 오프셋을 통해 직접 음절 토큰에 매핑한다.

        손실이 있는 단어 수준 BIO 중간 단계를 피할 수 있다.
        span 형식: [{"text": "경찰", "type": "OG"}, ...].
        """
        # 원문 텍스트에서 문자 → 음절 인덱스 맵을 구성한다
        char_to_syl = {}
        text_pos = 0
        for syl_idx, tok in enumerate(syllable_tokens):
            if tok.strip() == "":
                # 공백 토큰
                if text_pos < len(text) and text[text_pos] == " ":
                    text_pos += 1
                continue
            found = text.find(tok, text_pos)
            if found == -1:
                # Single char fallback
                if len(tok) == 1 and text_pos < len(text) and text[text_pos] == tok:
                    char_to_syl[text_pos] = syl_idx
                    text_pos += 1
                continue
            for c in range(found, found + len(tok)):
                char_to_syl[c] = syl_idx
            text_pos = found + len(tok)

        tags = ["O"] * len(syllable_tokens)
        # 중복 처리를 위해 엔티티 텍스트별 소비 위치를 추적한다.
        # 키: entity_text, 값: 다음 출현 탐색의 최소 시작 위치.
        _next_search: dict[str, int] = {}

        for span in spans:
            entity_text = span.get("text", "").strip()
            entity_type = span.get("type", "").strip()
            if not entity_text or not entity_type:
                continue

            # 이전 매칭을 지나쳐 원문에서 엔티티 텍스트를 찾는다
            search_start = _next_search.get(entity_text, 0)
            exact_found = text.find(entity_text, search_start)
            if exact_found != -1:
                match_start, match_end = exact_found, exact_found + len(entity_text)
            else:
                match_start, match_end = _find_ignore_spaces(text, entity_text, search_start)
            if match_start == -1:
                # 처음부터 재시도: span이 순서 없이 도착한 경우 search_start 이전에
                # 엔티티가 나타날 수 있다
                if search_start > 0:
                    exact_found = text.find(entity_text, 0)
                    if exact_found != -1:
                        match_start, match_end = exact_found, exact_found + len(entity_text)
                    else:
                        match_start, match_end = _find_ignore_spaces(text, entity_text, 0)
                if match_start == -1:
                    continue
            _next_search[entity_text] = match_end

            # 엔티티 문자를 음절 인덱스로 매핑한다
            matched_syls = []
            for c in range(match_start, match_end):
                syl_idx = char_to_syl.get(c)
                if syl_idx is not None and syl_idx not in matched_syls:
                    matched_syls.append(syl_idx)

            # 매칭된 음절에만 B/I 태그를 부여한다
            for j, syl_idx in enumerate(matched_syls):
                if tags[syl_idx] == "O":  # 기존 태그를 덮어쓰지 않는다
                    tags[syl_idx] = f"B-{entity_type}" if j == 0 else f"I-{entity_type}"

        return normalize_tags(tags, lang=lang)

    @staticmethod
    def _char_offset_align(
        gold_tokens: List[str],
        pred_tokens: List[str],
        pred_tags: List[str],
    ) -> List[str]:
        """문자 오프셋을 사용하여 예측 태그를 gold 토큰에 매핑한다.

        pred 토큰에서 텍스트를 재구성(공백 결합)하고, 각
        gold 음절 토큰을 동일 문자를 커버하는 pred 토큰에 매핑한다.
        gold 공백 토큰(빈 문자열/공백)은 항상 'O'로 태깅된다.
        """
        # pred 텍스트를 재구성하고 문자 → pred 토큰 인덱스 맵을 구성한다
        pred_text = " ".join(pred_tokens)
        pred_char_map = {}  # 문자 위치 → pred 토큰 인덱스
        pos = 0
        for idx, tok in enumerate(pred_tokens):
            for c in range(len(tok)):
                pred_char_map[pos + c] = idx
            pos += len(tok) + 1  # +1 for space separator

        # gold 비공백 토큰을 pred 텍스트에 순차적으로 매칭한다
        aligned = []
        pred_pos = 0
        for gold_tok in gold_tokens:
            if gold_tok.strip() == "":
                aligned.append("O")
                continue

            # Find this gold token in pred_text starting from pred_pos
            found = pred_text.find(gold_tok, pred_pos)
            if found != -1:
                mid = found + len(gold_tok) // 2
                pred_idx = pred_char_map.get(mid)
                if pred_idx is not None and pred_idx < len(pred_tags):
                    aligned.append(pred_tags[pred_idx])
                else:
                    aligned.append("O")
                pred_pos = found + len(gold_tok)
            else:
                # 폴백: 음절에 대해 단일 문자 매칭을 시도한다
                found_char = pred_text.find(gold_tok[0], pred_pos)
                if found_char != -1:
                    pred_idx = pred_char_map.get(found_char)
                    if pred_idx is not None and pred_idx < len(pred_tags):
                        aligned.append(pred_tags[pred_idx])
                    else:
                        aligned.append("O")
                    pred_pos = found_char + 1
                else:
                    aligned.append("O")

        return aligned
