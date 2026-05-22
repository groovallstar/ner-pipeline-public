"""domain_mine.schema 순수 함수 단위 테스트 (LLM 호출 없음)."""

from ner.augmenters.ja.domain_mine import schema


def test_span_to_entity_renames_fields():
    span = {'type': 'PROD', 'start': 2, 'end': 5, 'text': 'ABC'}
    ent = schema.span_to_entity(span)
    assert ent == {
        'label': 'PROD', 'start_char': 2, 'end_char': 5, 'text': 'ABC'
    }


def test_spans_to_entities_maps_all():
    spans = [
        {'type': 'LOC', 'start': 0, 'end': 2, 'text': '東京'},
        {'type': 'DAT', 'start': 3, 'end': 5, 'text': '昨日'},
    ]
    ents = schema.spans_to_entities(spans)
    assert [e['label'] for e in ents] == ['LOC', 'DAT']
    assert ents[1]['start_char'] == 3


def test_make_record_shapes_contract():
    rec = schema.make_record(7, 'abc', [])
    assert rec == {'id': '7', 'text': 'abc', 'entities': []}


def test_find_occurrences_multiple_non_overlapping():
    text = 'ICOCAとICOCAの話'
    assert schema.find_occurrences(text, 'ICOCA') == [(0, 5), (6, 11)]


def test_find_occurrences_absent_and_empty():
    assert schema.find_occurrences('abc', 'xyz') == []
    assert schema.find_occurrences('abc', '') == []


def test_make_anchor_entity_uses_text_slice():
    text = '新法ICOCA法案を可決'
    ent = schema.make_anchor_entity(text, 'ICOCA法案', 2)
    assert ent['label'] == 'PROD'
    assert ent['start_char'] == 2 and ent['end_char'] == 9
    assert ent['text'] == text[2:9] == 'ICOCA法案'


def test_validate_labels_flags_unknown():
    ents = [{'label': 'PROD'}, {'label': 'FOO'}, {'label': 'LOC'}]
    errors = schema.validate_labels(ents)
    assert len(errors) == 1
    assert 'FOO' in errors[0]


def test_validate_offsets_passes_clean_record():
    text = '東京で会う'
    ents = [{'label': 'LOC', 'start_char': 0,
             'end_char': 2, 'text': '東京'}]
    assert schema.validate_offsets(text, ents) == []


def test_validate_offsets_catches_text_mismatch():
    # Phase 3 의 offset shift 손상 재현: 좌표가 한 칸 밀린 경우.
    text = '東京で会う'
    ents = [{'label': 'LOC', 'start_char': 1,
             'end_char': 3, 'text': '東京'}]
    errors = schema.validate_offsets(text, ents)
    assert len(errors) == 1 and 'text mismatch' in errors[0]


def test_validate_offsets_catches_out_of_range():
    text = 'abc'
    ents = [{'label': 'LOC', 'start_char': 2,
             'end_char': 9, 'text': 'c'}]
    errors = schema.validate_offsets(text, ents)
    assert len(errors) == 1 and 'out of range' in errors[0]


def test_find_overlaps_detects_overlap():
    ents = [
        {'start_char': 0, 'end_char': 4},
        {'start_char': 2, 'end_char': 6},
        {'start_char': 6, 'end_char': 8},
    ]
    assert schema.find_overlaps(ents) == [(0, 1)]


def test_find_overlaps_none_when_disjoint():
    ents = [
        {'start_char': 0, 'end_char': 2},
        {'start_char': 2, 'end_char': 4},
    ]
    assert schema.find_overlaps(ents) == []


def test_validate_record_accepts_valid():
    rec = schema.make_record(
        '1', '東京で会う',
        [{'label': 'LOC', 'start_char': 0,
          'end_char': 2, 'text': '東京'}],
    )
    assert schema.validate_record(rec) == []


def test_validate_record_reports_corrupted_offset():
    rec = schema.make_record(
        '1', '東京で会う',
        [{'label': 'LOC', 'start_char': 1,
          'end_char': 3, 'text': '東京'}],
    )
    errors = schema.validate_record(rec)
    assert any('text mismatch' in e for e in errors)


def test_validate_record_reports_missing_key():
    errors = schema.validate_record({'text': 'x', 'entities': []})
    assert errors == ['missing key: id']
