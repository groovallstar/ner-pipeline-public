"""긴 입력 분할 — 모델 max_length 초과 텍스트를 문장 단위로 쪼갠다.

각 청크를 `(substring, base_offset)` 로 반환한다. substring 은 원문의 연속
슬라이스이고 base_offset 은 원문에서의 시작 char 위치라, 청크 내 span 의
start/end 에 base_offset 을 더하면 원문 char offset 으로 정확히 복원된다.

토큰 한도 안에 드는 텍스트는 통째로 한 청크 `(text, 0)` 로 둬, 정상 입력의
추론 동작이 분할 없이 그대로 유지되도록 한다.
"""

import re
from typing import List, Tuple

# 문장 경계: 마침표류·물음표·느낌표(전각 포함)와 개행. 경계 문자를 문장
# 끝에 포함해 연속(tile)되게 매칭한다 → 청크 결합 시 원문 슬라이스 보존.
_SENT_RE = re.compile(r'[^。．！？!?\n]*(?:[。．！？!?\n]+|$)')
_WORD_RE = re.compile(r'\S+\s*')


def _token_count(text: str, tokenizer) -> int:
    """tokenizer 로 special token 제외 토큰 수를 센다."""
    if getattr(tokenizer, 'is_fast', False):
        enc = tokenizer(text, add_special_tokens=False)
        return len(enc['input_ids'])
    return len(tokenizer.tokenize(text))


def _sentences(text: str) -> List[Tuple[str, int]]:
    """(문장, 원문 시작 offset) 리스트 — 공백뿐인 조각은 제외."""
    out: List[Tuple[str, int]] = []
    for m in _SENT_RE.finditer(text):
        seg = m.group()
        if not seg or not seg.strip():
            continue
        out.append((seg, m.start()))
    return out


def _char_windows(sent: str, base: int, tokenizer,
                  budget: int) -> List[Tuple[str, int]]:
    """단일 문장이 budget 초과 시 단어 경계로 강제 분할(offset 보존)."""
    out: List[Tuple[str, int]] = []
    cur = ''
    cur_start = base
    for m in _WORD_RE.finditer(sent):
        word = m.group()
        word_start = base + m.start()
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
    한도를 넘으면 단어 경계로 더 쪼갠다.
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
