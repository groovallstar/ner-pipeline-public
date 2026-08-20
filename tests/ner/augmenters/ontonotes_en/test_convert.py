"""BIO 디코드와 레코드 변환 — 합성 입력으로 경계를 고정한다."""
import pytest

from ner.augmenters.ontonotes_en.convert import (
    convert_record,
    check_entity_token_match,
    decode_bio,
    source_types,
    strip_ws,
)
from ner.augmenters.ontonotes_en.mapping import UndeclaredTypeError

ID2LABEL = {
    0: 'O',
    1: 'B-PERSON', 2: 'I-PERSON',
    3: 'B-ORG', 4: 'I-ORG',
    5: 'B-CARDINAL',
    6: 'B-GPE',
    7: 'B-FAC',
}


def test_source_types_ignores_outside_tag():
    assert source_types(ID2LABEL) == {
        'PERSON', 'ORG', 'CARDINAL', 'GPE', 'FAC',
    }


def test_decode_bio_reads_contiguous_span():
    # Phil Rosen / works / at / Fleet Leasing
    tags = [1, 2, 0, 0, 3, 4]
    assert decode_bio(tags, ID2LABEL) == [
        (0, 1, 'PERSON'), (4, 5, 'ORG'),
    ]


def test_decode_bio_splits_adjacent_same_type_on_b_prefix():
    """`B-` 는 앞 span 을 끊는다 — 두 사람을 한 명으로 합치면 안 된다."""
    tags = [1, 1]
    assert decode_bio(tags, ID2LABEL) == [
        (0, 0, 'PERSON'), (1, 1, 'PERSON'),
    ]


def test_decode_bio_recovers_from_leading_i_tag():
    """`I-` 로 시작하는 비정형 열도 버리지 않는다 — 버리면 gold 가 준다."""
    tags = [0, 2, 2, 0]
    assert decode_bio(tags, ID2LABEL) == [(1, 2, 'PERSON')]


def test_decode_bio_closes_span_at_end_of_sentence():
    tags = [0, 1, 2]
    assert decode_bio(tags, ID2LABEL) == [(1, 2, 'PERSON')]


def test_convert_record_maps_and_drops():
    """FAC 는 ORG 로 흡수되고 CARDINAL 은 드롭된다."""
    tokens = ['Narita', 'Airport', 'saw', 'three', 'flights']
    tags = [7, 0, 0, 5, 0]
    record = convert_record(tokens, tags, ID2LABEL, 'en-test-000000', 'test')
    assert [e['label'] for e in record['entities']] == ['ORG']
    assert record['entities'][0]['text'] == 'Narita'


def test_convert_record_span_is_self_consistent():
    tokens = ['Phil', 'Rosen', 'left', 'Boston', '.']
    tags = [1, 2, 0, 6, 0]
    record = convert_record(tokens, tags, ID2LABEL, 'en-test-000001', 'test')
    for entity in record['entities']:
        sliced = record['text'][entity['start_char']:entity['end_char']]
        assert sliced == entity['text']


def test_convert_record_carries_split_and_group_key():
    record = convert_record(['Hi'], [0], ID2LABEL, 'en-valid-000007', 'valid')
    assert record['split'] == 'valid'
    assert record['id'] == 'en-valid-000007'
    assert record['orig'] == 'en-valid-000007'


def test_convert_record_rejects_undeclared_type():
    id2label = dict(ID2LABEL) | {8: 'B-NEW_TYPE'}
    with pytest.raises(UndeclaredTypeError):
        convert_record(['x'], [8], id2label, 'en-test-000002', 'test')


def test_entity_token_check_passes_on_clean_conversion():
    tokens = ['Fleet', '&', 'Leasing', 'Management', 'Inc.', 'grew']
    tags = [3, 4, 4, 4, 4, 0]
    record = convert_record(tokens, tags, ID2LABEL, 'en-test-000003', 'test')
    assert check_entity_token_match(record, tokens, tags, ID2LABEL) == []


def test_entity_token_check_catches_drifted_offsets():
    """offset 이 밀리면 잡힌다 — 이 검사가 존재하는 이유다."""
    tokens = ['Phil', 'Rosen', 'left']
    tags = [1, 2, 0]
    record = convert_record(tokens, tags, ID2LABEL, 'en-test-000004', 'test')
    record['entities'][0]['text'] = record['text'][1:6]
    assert check_entity_token_match(record, tokens, tags, ID2LABEL) != []


def test_entity_token_check_catches_dropped_span():
    tokens = ['Phil', 'Rosen', 'left']
    tags = [1, 2, 0]
    record = convert_record(tokens, tags, ID2LABEL, 'en-test-000005', 'test')
    record['entities'].clear()
    assert check_entity_token_match(record, tokens, tags, ID2LABEL) != []


def test_strip_ws_is_whitespace_only_normalization():
    assert strip_ws('Fleet & Leasing') == 'Fleet&Leasing'
    assert strip_ws('  a\tb\nc ') == 'abc'


def test_assign_text_groups_shares_orig_across_identical_texts():
    """같은 문장을 가진 행은 한 unit 이 된다 — 아니면 그룹 보호가 no-op 이다."""
    from ner.augmenters.ontonotes_en.convert import assign_text_groups

    records = [
        {'text': 'yeah', 'id': 'en-train-000000'},
        {'text': 'Boston grew', 'id': 'en-train-000001'},
        {'text': 'yeah', 'id': 'en-train-000002'},
    ]
    assign_text_groups(records)
    assert records[0]['orig'] == records[2]['orig'] == 'en-train-000000'
    assert records[1]['orig'] == 'en-train-000001'


def test_assign_text_groups_is_stable_on_reruns():
    from ner.augmenters.ontonotes_en.convert import assign_text_groups

    records = [
        {'text': 'a', 'id': 'en-test-000000'},
        {'text': 'a', 'id': 'en-test-000001'},
    ]
    assign_text_groups(records)
    first = [r['orig'] for r in records]
    assign_text_groups(records)
    assert [r['orig'] for r in records] == first
