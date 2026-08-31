"""커밋된 소형 픽스처로 수락 기준을 **무조건** 돌린다.

전량 불변식 테스트(`test_corpus_invariants.py`)는 `data/` 가 gitignore 라
클론 직후 통째로 skip 된다. skip 은 초록으로 보고돼 "통과했다" 와 "못 돌았다"
가 구별되지 않는다 — 하네스가 게이트의 fail-open 에 대해 경계하는 것과 같은
문제다. 그래서 원본 18 종을 모두 덮는 21 문장을 저장소에 넣고, 같은 검사를
그 위에서 조건 없이 돌린다.

픽스처는 규모가 작아 회귀 탐지력이 전량만 못하다. 둘은 서로를 대체하지
않는다 — 이 파일은 **검사 기제가 살아 있는지**를, 전량 테스트는 **수치가
맞는지**를 본다.

## 이 파일이 덮지 **않는** 것

- **매핑표 각 행이 옳은가**는 여기서 안 본다. 그것은
  `test_mapping.py::test_kept_types_map_as_expected` 가 18 행을 독립 리터럴로
  박아 무조건 층에서 본다.
"""
import collections
import json
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.convert import (
    check_entity_token_match,
    convert_record,
    count_source_spans,
    load_id2label,
    source_types,
)
from ner.augmenters.ontonotes_en.mapping import assert_exhaustive

# 픽스처의 원본 타입별 span 수 — 매핑을 거치지 않은 값이라 BIO 디코드
# 단계만 본다. canonical 쪽 수와 함께 두면 실패 단계가 갈린다.
FIXTURE_SOURCE_SPANS = {
    'CARDINAL': 9, 'DATE': 16, 'EVENT': 1, 'FAC': 1, 'GPE': 7,
    'LANGUAGE': 1, 'LAW': 1, 'LOC': 2, 'MONEY': 2, 'NORP': 2,
    'ORDINAL': 1, 'ORG': 22, 'PERCENT': 6, 'PERSON': 4, 'PRODUCT': 1,
    'QUANTITY': 1, 'TIME': 1, 'WORK_OF_ART': 3,
}

# 변환 뒤 canonical 타입별 span 수 — **손으로 박은 리터럴**이다.
# `FIXTURE_SOURCE_SPANS` 에 `resolve()` 를 먹여 만들면 매핑축이 등식 양변에
# 들어가 매핑을 뒤집어도 통과한다(2·3 라운드에 실제로 그렇게 실패했다).
FIXTURE_CANONICAL_SPANS = {
    'DAT': 16, 'EVT': 1, 'LOC': 10, 'ORG': 22, 'PER': 4, 'PROD': 4,
}

# 픽스처의 단 하나뿐인 `FAC` span 과 표가 그것에 준 판정. 경로형(`Avenue`)이라
# `LOC` 이며, 이 한 줄이 **분할이 무조건 층에서 실제로 일어나는지**를 본다 —
# `FIXTURE_CANONICAL_SPANS` 만으로는 `LOC` 하나가 어디서 왔는지 안 갈린다.
FIXTURE_FAC_SURFACE = 'Arthur Avenue'
FIXTURE_FAC_VERDICT = 'LOC'

FIXTURE = Path(__file__).resolve().parents[2] / 'golden' / 'ontonotes_en'


@pytest.fixture(scope='module')
def converted():
    id2label = load_id2label(FIXTURE / 'label.json')
    records, problems = [], []
    source_counts: collections.Counter = collections.Counter()
    with (FIXTURE / 'sample.json').open(encoding='utf-8') as fh:
        for index, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            source_counts.update(count_source_spans(row['tags'], id2label))
            record = convert_record(
                row['tokens'], row['tags'], id2label,
                record_id=f'en-fixture-{index:06d}', split='fixture',
            )
            problems += check_entity_token_match(
                record, row['tokens'], row['tags'], id2label,
            )
            records.append((record, row))
    return {'records': records, 'problems': problems,
            'source_counts': source_counts, 'id2label': id2label}


def test_fixture_covers_every_source_type(converted):
    """픽스처가 18 종을 다 덮지 않으면 이 파일의 보증이 좁아진다."""
    assert set(converted['source_counts']) == set(
        source_types(converted['id2label'])
    )
    assert len(converted['source_counts']) == 18


def test_entity_token_match_holds(converted):
    assert converted['problems'] == []


def test_slice_equality_is_tautological_not_a_guard(converted):
    """`text[start:end] == entity.text` 는 변환 산출물에서 항진명제다.

    `convert_record` 가 `entity['text']` 를 그 슬라이스로 대입하므로 늘 참이다.
    한때 수락 기준으로 세었던 것은 **철회**했다 — offset 을 실제로 검증하는
    것은 원본 토큰을 비교 상대로 쓰는 `test_entity_token_match_holds` 다.
    """
    for record, _ in converted['records']:
        for entity in record['entities']:
            sliced = record['text'][
                entity['start_char']:entity['end_char']
            ]
            assert sliced == entity['text']


def test_exhaustiveness_gate_runs(converted):
    assert_exhaustive(source_types(converted['id2label']))


def test_undeclared_type_would_stop_conversion(converted):
    """선언되지 않은 타입이 조용히 지나가지 못한다."""
    from ner.augmenters.ontonotes_en.mapping import UndeclaredTypeError

    with pytest.raises(UndeclaredTypeError):
        assert_exhaustive(
            set(source_types(converted['id2label'])) | {'A_LATER_RELEASE_TYPE'}
        )


def test_source_span_counts_are_pinned(converted):
    """원본 타입별 span 수 고정 — 매핑과 독립인 층.

    canonical 쪽 수만 고정하면 BIO 디코드가 바뀐 것인지 매핑이 바뀐 것인지
    갈리지 않는다.
    """
    assert dict(converted['source_counts']) == FIXTURE_SOURCE_SPANS


def test_the_fac_span_is_split_by_the_verdict_table(converted):
    """`FAC` 가 타입이 아니라 표면으로 갈린다 — 무조건 층에서 태운다.

    픽스처의 `FAC` span 은 `Arthur Avenue` 하나이고 표가 `LOC` 로 판정했다.
    `FAC` 를 전량 `ORG` 로 되돌리면 이 검사가 먼저 붉어진다.
    """
    from ner.augmenters.ontonotes_en.mapping import resolve

    assert resolve('FAC', FIXTURE_FAC_SURFACE) == FIXTURE_FAC_VERDICT
    found = [
        entity for record, _ in converted['records']
        for entity in record['entities']
        if entity['text'] == FIXTURE_FAC_SURFACE
    ]
    assert len(found) == 1
    assert found[0]['label'] == FIXTURE_FAC_VERDICT


def test_an_unlisted_fac_surface_stops_conversion():
    """표에 없는 표면은 기본값을 못 받는다 — 미판정이 조용히 흐르지 않는다."""
    from ner.augmenters.ontonotes_en.mapping import (
        UnlistedSurfaceError, resolve,
    )

    with pytest.raises(UnlistedSurfaceError):
        resolve('FAC', 'a surface no corpus ever produced')


def test_canonical_counts_match_pinned_literals(converted):
    """canonical 타입별 수를 **손으로 박은 값**과 직접 비교한다.

    이전 판은 `FIXTURE_SOURCE_SPANS` 에 `resolve()` 를 먹여 기대치를 만들었다.
    그러면 `convert_record` 도 같은 `resolve()` 를 쓰므로 매핑축이 등식
    양변에 들어가, `PERSON`·`GPE`·`FAC`·`WORK_OF_ART` 를 `DROP` 으로 뒤집어도
    전부 통과했다. 리터럴로 박으면 그 상쇄가 사라진다.
    """
    actual: collections.Counter = collections.Counter()
    for record, _ in converted['records']:
        for entity in record['entities']:
            actual[entity['label']] += 1
    assert dict(actual) == FIXTURE_CANONICAL_SPANS


# 매핑을 뒤집는 방향 두 가지. 살아남을 것을 버리는 쪽만 보면 **드롭돼야 할
# 것이 새어 들어오는** 반대 방향이 무조건 층에서 안 잡힌다.
MAPPING_MUTATIONS = (
    ('PERSON', 'DROP'), ('GPE', 'DROP'), ('ORG', 'DROP'),
    ('WORK_OF_ART', 'DROP'),
    ('LAW', 'ORG'), ('NORP', 'LOC'), ('CARDINAL', 'PROD'),
)


@pytest.mark.parametrize('source_type,new_label', MAPPING_MUTATIONS)
def test_flipping_a_mapping_row_breaks_the_canonical_counts(
    monkeypatch, source_type, new_label,
):
    """매핑 한 행을 실제로 뒤집으면 변환 산출물이 박힌 값과 어긋난다.

    산술만 확인하지 않고 **매핑표를 진짜로 바꿔** 변환을 다시 돌린다. 앞선
    판은 루프에서 리터럴 `'EVENT'` 를 건너뛰는 산술 항등식이라 검출력이
    없었다.

    양방향을 본다 — 살아남을 타입을 `DROP` 으로 죽이는 쪽과, 버려야 할
    타입(`LAW`·`NORP`·`CARDINAL`)을 canonical 로 되살리는 쪽. 한쪽만 두면
    "드롭 목록에서 한 줄이 빠져 비-entity 가 학습 데이터에 섞이는" 사고가
    무조건 층을 그냥 지난다.
    """
    from ner.augmenters.ontonotes_en import mapping

    id2label = load_id2label(FIXTURE / 'label.json')
    mutated = dict(mapping.ONTONOTES_TO_CANONICAL)
    mutated[source_type] = (
        mapping.DROP if new_label == 'DROP' else new_label
    )
    monkeypatch.setattr(
        mapping, 'ONTONOTES_TO_CANONICAL', mutated,
    )
    actual: collections.Counter = collections.Counter()
    with (FIXTURE / 'sample.json').open(encoding='utf-8') as fh:
        for index, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            record = convert_record(
                row['tokens'], row['tags'], id2label,
                record_id=f'en-mut-{index:06d}', split='fixture',
            )
            for entity in record['entities']:
                actual[entity['label']] += 1
    assert dict(actual) != FIXTURE_CANONICAL_SPANS, (
        f'flipping {source_type} to {new_label} went unnoticed'
    )
