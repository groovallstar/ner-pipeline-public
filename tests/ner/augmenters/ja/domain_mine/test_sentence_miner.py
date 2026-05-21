"""sentence_miner 순수 함수 단위 테스트 (네트워크 호출 없음)."""

from ner.augmenters.ja.domain_mine import sentence_miner as sm


def test_split_sentences_basic():
    text = '東京は都市だ。京都も都市だ！大阪は？'
    assert sm.split_sentences(text) == [
        '東京は都市だ。', '京都も都市だ！', '大阪は？'
    ]


def test_split_sentences_handles_newlines_and_blanks():
    text = 'A。\n\nB。  \nC'
    assert sm.split_sentences(text) == ['A。', 'B。', 'C']


def test_split_sentences_empty():
    assert sm.split_sentences('') == []


def test_extract_sentences_with_name_offset_correct():
    text = 'これは前置き。「トイレの神様」は植村花菜の楽曲である。'
    hits = sm.extract_sentences_with_name(text, 'トイレの神様')
    assert len(hits) == 1
    sent = hits[0]['text']
    (s, e) = hits[0]['anchors'][0]
    assert sent[s:e] == 'トイレの神様'


def test_extract_sentences_with_name_length_filter():
    # min_len=8 보다 짧은 문장은 제외 (anchor 가 있어도)
    text = 'ICOCA。'
    assert sm.extract_sentences_with_name(text, 'ICOCA', min_len=8) == []


def test_extract_sentences_with_name_no_match():
    text = 'まったく関係ない文章である。'
    assert sm.extract_sentences_with_name(text, 'Suica') == []


def test_extract_sentences_with_name_multi_occurrence():
    text = 'SuicaとSuicaを比べる長めの説明文である。'
    hits = sm.extract_sentences_with_name(text, 'Suica')
    assert len(hits) == 1
    assert len(hits[0]['anchors']) == 2
