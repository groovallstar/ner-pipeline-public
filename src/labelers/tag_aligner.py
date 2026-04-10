"""Gold-vs-predicted tag alignment utilities.

Handles tokenization mismatches between gold data (pre-tokenized)
and LLM labeler output (whitespace-split from reconstructed text).
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Language-specific tag normalization maps
# Korean (KLUE): standardize to PS, LC, OG, DT, TI, QT
_TAG_NORMALIZE_MAP_KO = {
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

# Vietnamese: standardize to PER, LOC, ORG (keep international standard tags)
_TAG_NORMALIZE_MAP_VI = {
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "MISCELLANEOUS": "MISC",
}

_TAG_NORMALIZE_MAPS = {
    "ko": _TAG_NORMALIZE_MAP_KO,
    "ja": _TAG_NORMALIZE_MAP_KO,  # Japanese uses same mapping as Korean
    "vi": _TAG_NORMALIZE_MAP_VI,
}


def normalize_tag(tag: str, lang: str = "ko") -> str:
    """Normalize a BIO tag to language-specific standard.

    Korean/Japanese: B-PER → B-PS, B-LOC → B-LC (KLUE standard)
    Vietnamese: B-PERSON → B-PER, B-LOCATION → B-LOC (international standard)
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
    """Find pattern in text ignoring spaces in both sides.

    Returns (start_offset, end_offset) in `text`, or (-1, -1).
    E.g. _find_ignore_spaces("지난 19일", "지난19일") → (0, 5)
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
    """Extract entity spans from BIO tags.

    Handles both syllable-level tokens (KLUE-style, with space tokens)
    and word-level tokens (WikiANN-style, no space tokens).
    Returns [{"text": "경찰", "type": "OG"}, ...].
    """
    tags = normalize_tags(tags, lang=lang)
    # Detect token level: if any token is pure whitespace, it's syllable-level
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
    """Aligns predicted BIO tags to gold token boundaries."""

    @staticmethod
    def reconstruct_text(tokens: List[str]) -> str:
        """Reconstruct original text from syllable tokens.

        KLUE NER tokens include explicit space tokens (' ' or '').
        Syllable tokens should be concatenated directly; only space tokens
        produce whitespace in the output.
        """
        has_space_tokens = any(t.strip() == "" for t in tokens)
        if has_space_tokens:
            # KLUE-style syllable tokens with explicit space markers
            return "".join(t if t.strip() != "" else " " for t in tokens)
        # Word-level tokens (e.g. kor_ner) — space-join as before
        return " ".join(tokens)

    @staticmethod
    def align(
        gold_tokens: List[str],
        gold_tags: List[str],
        pred_tokens: List[str],
        pred_tags: List[str],
        lang: str = "ko",
    ) -> Tuple[List[str], List[str]]:
        """Align predicted tags to gold token grid.

        Returns (gold_tags, aligned_pred_tags) of equal length.
        """
        gold_tags = normalize_tags(gold_tags, lang=lang)
        pred_tags = normalize_tags(pred_tags, lang=lang)

        # Fast path: tokenizations match exactly
        if gold_tokens == pred_tokens:
            return gold_tags, pred_tags

        # Character-offset based alignment
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
        """Align word-level predicted tags to syllable-level gold token grid.

        KLUE NER uses syllable tokens (음절) with space tokens between words.
        Models produce word-level predictions. This maps word predictions back
        to the syllable grid.
        """
        syllable_tags = normalize_tags(syllable_tags, lang=lang)
        word_tags = normalize_tags(word_tags, lang=lang)

        # Reconstruct which syllable tokens belong to which word
        # Space tokens (' ' or '') are separators
        aligned_pred = []
        word_idx = 0
        in_word_pos = 0  # position within current word's characters

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
                # First syllable of word gets the tag as-is
                # Subsequent syllables: B-X -> I-X
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
        """Map raw entity spans directly to syllable tokens via character offsets.

        This avoids the lossy word-level BIO intermediate step.
        Spans are [{"text": "경찰", "type": "OG"}, ...].
        """
        # Build char→syllable_idx map from original text
        char_to_syl = {}
        text_pos = 0
        for syl_idx, tok in enumerate(syllable_tokens):
            if tok.strip() == "":
                # Space token
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
        # Track how far we've consumed per entity text to handle duplicates.
        # Key: entity_text, Value: minimum search_start for next occurrence.
        _next_search: dict[str, int] = {}

        for span in spans:
            entity_text = span.get("text", "").strip()
            entity_type = span.get("type", "").strip()
            if not entity_text or not entity_type:
                continue

            # Find entity text in the original text, advancing past previous matches
            search_start = _next_search.get(entity_text, 0)
            exact_found = text.find(entity_text, search_start)
            if exact_found != -1:
                match_start, match_end = exact_found, exact_found + len(entity_text)
            else:
                match_start, match_end = _find_ignore_spaces(text, entity_text, search_start)
            if match_start == -1:
                # Wrap around: retry from beginning (entity may appear before search_start
                # if spans arrive out of order)
                if search_start > 0:
                    exact_found = text.find(entity_text, 0)
                    if exact_found != -1:
                        match_start, match_end = exact_found, exact_found + len(entity_text)
                    else:
                        match_start, match_end = _find_ignore_spaces(text, entity_text, 0)
                if match_start == -1:
                    continue
            _next_search[entity_text] = match_end

            # Map entity chars to syllable indices
            matched_syls = []
            for c in range(match_start, match_end):
                syl_idx = char_to_syl.get(c)
                if syl_idx is not None and syl_idx not in matched_syls:
                    matched_syls.append(syl_idx)

            # Assign B/I tags only to matched syllables
            for j, syl_idx in enumerate(matched_syls):
                if tags[syl_idx] == "O":  # don't overwrite existing tags
                    tags[syl_idx] = f"B-{entity_type}" if j == 0 else f"I-{entity_type}"

        return normalize_tags(tags, lang=lang)

    @staticmethod
    def _char_offset_align(
        gold_tokens: List[str],
        pred_tokens: List[str],
        pred_tags: List[str],
    ) -> List[str]:
        """Map predicted tags to gold tokens using character offsets.

        Reconstructs text from pred tokens (space-joined) and maps each
        gold syllable token to the pred token that covers the same characters.
        Gold space tokens (empty/whitespace) are always tagged 'O'.
        """
        # Reconstruct pred text and build char→pred_token_idx map
        pred_text = " ".join(pred_tokens)
        pred_char_map = {}  # char_pos -> pred token index
        pos = 0
        for idx, tok in enumerate(pred_tokens):
            for c in range(len(tok)):
                pred_char_map[pos + c] = idx
            pos += len(tok) + 1  # +1 for space separator

        # Match gold non-space tokens to pred text sequentially
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
                # Fallback: try single-char match for syllables
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
