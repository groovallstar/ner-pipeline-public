"""주입 산출물의 `orig`·`split` 복원.

`augmenters.pii` 의 `Record` 는 `text`·`entities`·`id` 만 담아, 주입을 지나면
변환 단계가 심어둔 `orig`(형제 묶음 키)와 `split` 이 사라진다. JA 는
`--group-key id` 라 안 겪는 문제지만 EN 은 `orig` 를 쓴다 — OntoNotes 가 같은
문장을 여러 번 담기 때문이다.
"""
import json

import pytest

from ner.augmenters.ontonotes_en.restore_groups import (
    CARRIED_FIELDS,
    build_index,
    restore,
)


def _write(path, rows):
    path.write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
        encoding='utf-8',
    )
    return path


@pytest.fixture
def source(tmp_path):
    return _write(tmp_path / 'src.jsonl', [
        {'text': 'yeah', 'entities': [], 'id': 'en-train-000000',
         'orig': 'en-train-000000', 'split': 'train'},
        {'text': 'Boston grew', 'entities': [], 'id': 'en-train-000001',
         'orig': 'en-train-000001', 'split': 'train'},
        {'text': 'yeah', 'entities': [], 'id': 'en-train-000002',
         'orig': 'en-train-000000', 'split': 'train'},
    ])


def test_index_carries_only_the_lost_fields(source):
    index = build_index(source)
    assert set(index) == {
        'en-train-000000', 'en-train-000001', 'en-train-000002',
    }
    assert set(index['en-train-000000']) == set(CARRIED_FIELDS)


def test_restore_puts_orig_and_split_back(tmp_path, source):
    injected = _write(tmp_path / 'inj.jsonl', [
        {'text': 'yeah, call 555-0101', 'entities': [],
         'id': 'en-train-000000'},
        {'text': 'yeah again at 555-0102', 'entities': [],
         'id': 'en-train-000002'},
    ])
    stats = restore(injected, build_index(source), injected)
    assert stats['written'] == 2
    rows = [json.loads(line) for line in injected.read_text().splitlines()]
    # 주입이 문장을 다시 써서 text 는 서로 다르지만 원문은 같다.
    assert rows[0]['text'] != rows[1]['text']
    assert rows[0]['orig'] == rows[1]['orig'] == 'en-train-000000'
    assert all(r['split'] == 'train' for r in rows)


def test_in_place_restore_does_not_truncate(tmp_path, source):
    """읽는 파일을 그대로 쓰기 모드로 열면 읽기 전에 비워진다."""
    injected = _write(tmp_path / 'inj.jsonl', [
        {'text': 'x', 'entities': [], 'id': 'en-train-000001'},
    ])
    restore(injected, build_index(source), injected)
    assert injected.read_text().strip()
    assert not list(tmp_path.glob('*.tmp'))


def test_unknown_id_stops_instead_of_skipping(tmp_path, source):
    """조용히 건너뛰면 그 행만 그룹 키 없이 학습에 들어간다."""
    injected = _write(tmp_path / 'inj.jsonl', [
        {'text': 'x', 'entities': [], 'id': 'en-train-999999'},
    ])
    with pytest.raises(KeyError):
        restore(injected, build_index(source), injected)


def test_injected_entities_are_untouched(tmp_path, source):
    injected = _write(tmp_path / 'inj.jsonl', [
        {'text': 'Boston grew, call 555-0101', 'id': 'en-train-000001',
         'entities': [
             {'label': 'LOC', 'start_char': 0, 'end_char': 6,
              'text': 'Boston'},
             {'label': 'PHONE', 'start_char': 18, 'end_char': 26,
              'text': '555-0101'},
         ]},
    ])
    restore(injected, build_index(source), injected)
    row = json.loads(injected.read_text().splitlines()[0])
    assert [e['label'] for e in row['entities']] == ['LOC', 'PHONE']


def test_source_missing_the_carried_fields_stops_the_build(tmp_path):
    """옮길 필드가 원본에 없으면 표를 만들다 멈춘다.

    건너뛰면 표에 빈 칸이 실리고 `restore` 는 아무것도 안 하면서 그 행을
    복원했다고 센다 — `orig` 없는 행이 성공 로그와 함께 학습에 들어간다.
    """
    source = _write(tmp_path / 'src.jsonl', [
        {'text': 'yeah', 'entities': [], 'id': 'en-train-000000',
         'split': 'train'},  # orig 가 없다
    ])
    with pytest.raises(KeyError):
        build_index(source)
