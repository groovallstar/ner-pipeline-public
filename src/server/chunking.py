"""긴 입력 분할 — 모델 max_length 초과 텍스트를 문장 단위로 쪼갠다.

각 청크를 `(substring, base_offset)` 로 반환한다. substring 은 원문의 연속
슬라이스이고 base_offset 은 원문에서의 시작 char 위치라, 청크 내 span 의
start/end 에 base_offset 을 더하면 원문 char offset 으로 정확히 복원된다.

토큰 한도 안에 드는 텍스트는 통째로 한 청크 `(text, 0)` 로 둬, 정상 입력의
추론 동작이 분할 없이 그대로 유지되도록 한다.

**엔티티가 경계를 가로지르지 않는 것은 문장 단위로 자를 때만 보장된다.**
한 문장이 혼자 한도를 넘으면 단어 경계로 강제 분할하므로 그 안의 엔티티는
잘릴 수 있다. 한도보다 긴 문장을 모델에 넣을 방법이 없어 남는 한계다.
"""

import re
from typing import List, Tuple

from ner.classifier.data_utils import _is_phobert, _word_spans_vi

# 문장 경계: 마침표류·물음표·느낌표(전각 포함)와 개행. 경계 문자를 문장
# 끝에 포함해 연속(tile)되게 매칭한다 → 청크 결합 시 원문 슬라이스 보존.
_SENT_RE = re.compile(r'[^。．！？!?\n]*(?:[。．！？!?\n]+|$)')
_WORD_RE = re.compile(r'\S+\s*')

# 경계 문자 뒤에 이것이 오면 문장이 안 끝난 것으로 본다. 작품 제목 같은
# 표면이 `Do You Know ? ( The Ping Pong Song )` 처럼 물음표를 품는데, 그
# 자리를 문장 끝으로 보면 엔티티 한가운데에 청크 경계가 생겨 회수에 실패한다.
_NOT_SENTENCE_END_NEXT = '([{（［｛'


def _token_count(text: str, tokenizer) -> int:
    """tokenizer 로 special token 제외 토큰 수를 센다."""
    if getattr(tokenizer, 'is_fast', False):
        enc = tokenizer(text, add_special_tokens=False)
        return len(enc['input_ids'])
    if _is_phobert(tokenizer):
        return sum(len(tokenizer.tokenize(surface))
                   for surface, _, _ in _word_spans_vi(text))
    return len(tokenizer.tokenize(text))


def _ends_sentence(text: str, end: int) -> bool:
    """`end` 위치의 경계가 실제 문장 끝인지 — 뒤따르는 글자로 판단한다."""
    nxt = text[end:].lstrip()[:1]
    if not nxt:
        return True
    return not (nxt in _NOT_SENTENCE_END_NEXT or nxt.islower())


def _sentences(text: str) -> List[Tuple[str, int]]:
    """(문장, 원문 시작 offset) 리스트 — 공백뿐인 조각은 제외.

    경계 문자가 나와도 뒤따르는 글자가 문장 시작으로 안 보이면 다음 조각과
    이어 붙인다. 이어 붙인 조각도 원문의 연속 슬라이스라 offset 은 그대로다.
    """
    out: List[Tuple[str, int]] = []
    pending = ''
    pending_start = 0
    for m in _SENT_RE.finditer(text):
        seg = m.group()
        if not seg or not seg.strip():
            continue
        start = m.start()
        if pending:
            seg = text[pending_start:m.end()]
            start = pending_start
            pending = ''
        if not _ends_sentence(text, m.end()):
            pending, pending_start = seg, start
            continue
        out.append((seg, start))
    if pending:
        out.append((pending, pending_start))
    return out


def _split_word(text: str, base: int, tokenizer,
                budget: int) -> List[Tuple[str, int]]:
    """단어 자체가 한도를 넘으면 문자 중간에서 나누고 양쪽 길이를 재검사한다."""
    if _token_count(text, tokenizer) <= budget:
        return [(text, base)]
    if len(text) <= 1:
        raise ValueError('token budget is too small to encode one character')
    mid = len(text) // 2
    return (_split_word(text[:mid], base, tokenizer, budget)
            + _split_word(text[mid:], base + mid, tokenizer, budget))


def _char_windows(sent: str, base: int, tokenizer,
                  budget: int) -> List[Tuple[str, int]]:
    """단일 문장이 budget 초과 시 단어 경계로 강제 분할(offset 보존)."""
    out: List[Tuple[str, int]] = []
    cur = ''
    cur_start = base
    for m in _WORD_RE.finditer(sent):
        word = m.group()
        word_start = base + m.start()
        if _token_count(word, tokenizer) > budget:
            if cur:
                out.append((cur, cur_start))
                cur = ''
            out.extend(_split_word(word, word_start, tokenizer, budget))
            continue
        cand = cur + word if cur else word
        if cur and _token_count(cand, tokenizer) > budget:
            out.append((cur, cur_start))
            cur, cur_start = word, word_start
        else:
            if not cur:
                cur_start = word_start
            cur = cand
    if cur:
        out.append((cur, cur_start))
    return out


def split_for_length(text: str, tokenizer,
                     max_length: int) -> List[Tuple[str, int]]:
    """텍스트를 토큰 한도에 맞는 `(substring, base_offset)` 청크로 분할.

    한도 안이면 `[(text, 0)]`. 초과하면 문장 단위로 묶되, 한 문장이 단독으로
    한도를 넘으면 단어 경계로 더 쪼개고, 단어도 넘으면 문자 단위로 나눈다.
    """
    budget = max(1, max_length - 2)  # CLS/SEP 여유
    if _token_count(text, tokenizer) <= budget:
        return [(text, 0)]

    chunks: List[Tuple[str, int]] = []
    cur = ''
    cur_start = 0
    for sent, start in _sentences(text):
        if _token_count(sent, tokenizer) > budget:
            if cur:
                chunks.append((cur, cur_start))
                cur = ''
            chunks.extend(_char_windows(sent, start, tokenizer, budget))
            continue
        # cur 누적은 원문 슬라이스로 — 문장 사이 스킵된 공백/개행을 보존해
        # offset 불변식(text[base:base+len]==sub)을 깨지 않는다.
        cand = text[cur_start:start + len(sent)] if cur else sent
        if cur and _token_count(cand, tokenizer) > budget:
            chunks.append((cur, cur_start))
            cur, cur_start = sent, start
        else:
            if not cur:
                cur_start = start
            cur = cand
    if cur:
        chunks.append((cur, cur_start))
    return chunks
