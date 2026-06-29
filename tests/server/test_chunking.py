"""긴 입력 분할 단위 테스트 — 가짜 토크나이저로 offset 보존 검증.

핵심 불변식: 각 청크 `(sub, base)` 에 대해 `text[base:base+len(sub)] == sub`.
이게 보장돼야 청크 내 span offset 에 base 를 더해 원문 offset 으로 복원된다.
"""

from server.chunking import split_for_length


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


def test_sentence_merge_preserves_offset_across_skipped_gap():
    """문장 경계(。) 뒤 공백+개행이 낀 장문도 offset 보존.

    `。 \\n` 처럼 문장부호 뒤 공백-개행은 독립 조각이 되어 _sentences 의
    공백 스킵에 걸린다. 인접 문장을 이어붙일 때 그 스킵 길이만큼 원문과
    어긋나, 청크 내 span offset 이 밀려 엔티티 위치가 깨진다.
    """
    tok = FakeTok()
    text = 'aa bb。 \ncc dd。 \nee ff。'
    chunks = split_for_length(text, tok, max_length=5)  # budget 3 tokens
    assert len(chunks) > 1  # 병합 분할 확인 — 테스트가 공허하지 않음
    _assert_contiguous(text, chunks)


def test_sentence_entities_stay_within_chunks():
    """문장부호로 나뉜 텍스트의 엔티티는 한 청크에 온전히 들어간다.

    청크 경계가 문장 경계(。)에 맞아 문장 내부 엔티티는 잘리지 않는다 —
    chunk recall 보존의 핵심 불변식(엔티티가 경계를 가로지르면 회수 실패).
    """
    tok = FakeTok()
    text = 'aa BB cc。dd EE ff。gg HH ii。jj KK ll。'
    chunks = split_for_length(text, tok, max_length=6)  # budget 4 tokens
    assert len(chunks) > 1  # 강제 분할 확인 — 테스트가 공허하지 않음
    spans = [(base, base + len(sub)) for sub, base in chunks]
    for surface in ('BB', 'EE', 'HH', 'KK'):
        s = text.index(surface)
        e = s + len(surface)
        assert any(cs <= s and e <= ce for cs, ce in spans), \
            f'entity {surface!r} straddles a chunk boundary'
