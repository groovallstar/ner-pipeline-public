"""긴 입력 분할 단위 테스트 — 가짜 토크나이저로 offset 보존 검증.

핵심 불변식: 각 청크 `(sub, base)` 에 대해 `text[base:base+len(sub)] == sub`.
이게 보장돼야 청크 내 span offset 에 base 를 더해 원문 offset 으로 복원된다.
"""

from ner.server.chunking import split_for_length


class FakeTok:
    """공백 단어 수를 토큰 수로 세는 가짜 fast 토크나이저."""

    is_fast = True

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


def _assert_contiguous(text, chunks):
    for sub, base in chunks:
        assert text[base:base + len(sub)] == sub


def test_short_text_single_chunk():
    """한도 안 텍스트는 통째로 (text, 0)."""
    tok = FakeTok()
    text = 'aa bb cc.'
    assert split_for_length(text, tok, max_length=10) == [(text, 0)]


def test_long_text_splits_by_sentence_offset_preserved():
    """문장 단위 분할 — 모든 청크가 원문 슬라이스(offset 보존)."""
    tok = FakeTok()
    text = 'aa bb cc. dd ee ff. gg hh ii. jj kk ll.'
    chunks = split_for_length(text, tok, max_length=5)  # budget=3 words
    assert len(chunks) > 1
    _assert_contiguous(text, chunks)


def test_overlong_single_sentence_char_window():
    """단일 문장이 한도 초과 시 단어 경계로 강제 분할(offset 보존)."""
    tok = FakeTok()
    text = 'w1 w2 w3 w4 w5 w6 w7 w8 w9'  # 문장부호 없는 한 문장, 9 words
    chunks = split_for_length(text, tok, max_length=5)  # budget=3 words
    assert len(chunks) > 1
    _assert_contiguous(text, chunks)


def test_chunks_cover_all_entity_chars():
    """청크들이 원문을 빠짐없이 덮어 어느 위치 span 도 복원 가능."""
    tok = FakeTok()
    text = 'aa bb cc. dd ee ff. gg hh ii.'
    chunks = split_for_length(text, tok, max_length=5)
    # 청크는 연속 타일이라 이어붙이면 원문과 동일
    assert ''.join(sub for sub, _ in chunks) == text
