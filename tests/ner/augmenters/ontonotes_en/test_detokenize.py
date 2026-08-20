"""자연문 복원 — 정규화 표와 공백 규칙을 고정한다.

엔티티↔토큰 대조 검사는 정규화 표 **자체**는 검증하지 못한다(양쪽에 같은
정규화를 걸기 때문). 그 표를 고정하는 것이 이 파일의 몫이다.
"""
from ner.augmenters.ontonotes_en.detokenize import detokenize, normalize_token


def test_ptb_bracket_escapes_are_unescaped():
    """실제 영문에 없는 표기라 남으면 그대로 학습 데이터가 된다."""
    assert normalize_token('-LRB-') == '('
    assert normalize_token('-RRB-') == ')'
    assert normalize_token('-LSB-') == '['
    assert normalize_token('-RSB-') == ']'
    assert normalize_token('-LCB-') == '{'
    assert normalize_token('-RCB-') == '}'


def test_directional_quotes_become_symmetric():
    assert normalize_token('``') == '"'
    assert normalize_token('`') == '"'
    assert normalize_token("''") == '"'


def test_speech_and_disfluency_markers():
    assert normalize_token('/.') == '.'
    assert normalize_token('/?') == '?'
    assert normalize_token('/-') == '-'
    assert normalize_token('%um') == 'um'
    assert normalize_token('%uh') == 'uh'


def test_plain_tokens_pass_through():
    for token in ('Boston', 'U.S.', 'AT&T', '10,000', "'s", 'Inc.'):
        assert normalize_token(token) == token


def test_punctuation_attaches_left():
    text, _ = detokenize(['He', 'left', ',', 'then', 'returned', '.'])
    assert text == 'He left, then returned.'


def test_possessive_and_contraction_attach_left():
    text, _ = detokenize(["Yugoslavia", "'s", "envoy", "did", "n't", "go"])
    assert text == "Yugoslavia's envoy didn't go"


def test_hyphen_joins_two_words():
    text, _ = detokenize(['time', '-', 'consuming', 'work'])
    assert text == 'time-consuming work'


def test_slash_joins_two_words():
    text, _ = detokenize(['Princeton', '/', 'Newport'])
    assert text == 'Princeton/Newport'


def test_ampersand_keeps_spaces():
    """`Fleet & Leasing` 은 붙이면 실제 영문에 없는 표기가 된다."""
    text, _ = detokenize(['Fleet', '&', 'Leasing'])
    assert text == 'Fleet & Leasing'


def test_quotes_hug_their_content():
    text, _ = detokenize(['``', 'It', 'works', "''", ',', 'he', 'said'])
    assert text == '"It works", he said'


def test_brackets_hug_their_content():
    text, _ = detokenize(['Wallop', '-LRB-', 'R.', '-RRB-', 'spoke'])
    assert text == 'Wallop (R.) spoke'


def test_offsets_point_at_normalized_surfaces():
    tokens = ['Wallop', '-LRB-', 'R.', '-RRB-', 'spoke']
    text, offsets = detokenize(tokens)
    for token, (start, end) in zip(tokens, offsets):
        assert text[start:end] == normalize_token(token)


def test_offsets_are_monotonic_and_within_text():
    tokens = ['A', 'test', ',', 'with', '-', 'marks', '.']
    text, offsets = detokenize(tokens)
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    for (_, prev_end), (start, _) in zip(offsets, offsets[1:]):
        assert start >= prev_end
