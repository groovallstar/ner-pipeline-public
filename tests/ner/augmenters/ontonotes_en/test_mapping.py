"""매핑표 전수성 게이트 — 수락 기준 3.

이 테스트가 지키는 것은 "18 종이 전부 선언돼 있다" 가 아니라 **"선언되지 않은
타입이 조용히 지나가지 못한다"** 이다. 앞엣것만 지키면 원천 버전이 올라
19 번째 타입이 생겼을 때 그 타입이 통째로 사라진 채 통과한다.
"""
import pytest

from ner.augmenters.ontonotes_en.mapping import (
    DROP,
    ONTONOTES_TO_CANONICAL,
    PRODUCED_LABELS,
    UndeclaredTypeError,
    assert_exhaustive,
    resolve,
)

# 원천 `tner/ontonotes5` 가 내놓는 엔티티 타입 전수. 이 목록은 매핑 모듈이
# 아니라 여기에 둔다 — 모듈에서 가져오면 모듈을 고칠 때 기대치도 함께
# 움직여 검사가 자기참조가 된다.
SOURCE_TYPES = frozenset({
    'PERSON', 'GPE', 'LOC', 'ORG', 'FAC', 'PRODUCT', 'WORK_OF_ART',
    'EVENT', 'DATE', 'LAW', 'NORP', 'LANGUAGE', 'TIME', 'QUANTITY',
    'MONEY', 'PERCENT', 'ORDINAL', 'CARDINAL',
})

# canonical 로 살아남는 매핑. 근거 절은 매핑 모듈 주석에 있다.
EXPECTED_KEPT = {
    'PERSON': 'PER',
    'GPE': 'LOC',
    'LOC': 'LOC',
    'ORG': 'ORG',
    'FAC': 'ORG',
    'PRODUCT': 'PROD',
    'WORK_OF_ART': 'PROD',
    'EVENT': 'EVT',
    'DATE': 'DAT',
}


def test_every_source_type_is_declared():
    assert set(ONTONOTES_TO_CANONICAL) == SOURCE_TYPES
    assert len(SOURCE_TYPES) == 18


def test_kept_types_map_as_expected():
    for src, canonical in EXPECTED_KEPT.items():
        assert resolve(src) == canonical


def test_dropped_types_resolve_to_none():
    for src in SOURCE_TYPES - set(EXPECTED_KEPT):
        assert resolve(src) is None
        assert ONTONOTES_TO_CANONICAL[src] == DROP


def test_produced_labels_are_the_six_kept():
    assert PRODUCED_LABELS == frozenset(EXPECTED_KEPT.values())
    assert PRODUCED_LABELS == frozenset(
        {'PER', 'LOC', 'ORG', 'PROD', 'EVT', 'DAT'}
    )


def test_undeclared_type_stops_conversion():
    """새 타입이 생기면 조용히 드롭되지 않고 즉시 터진다."""
    with pytest.raises(UndeclaredTypeError):
        resolve('NEW_TYPE_FROM_A_LATER_RELEASE')


def test_exhaustiveness_gate_reports_all_undeclared():
    with pytest.raises(UndeclaredTypeError) as excinfo:
        assert_exhaustive(SOURCE_TYPES | {'ALPHA', 'BETA'})
    message = str(excinfo.value)
    assert 'ALPHA' in message and 'BETA' in message


def test_exhaustiveness_gate_passes_on_known_types():
    assert_exhaustive(SOURCE_TYPES)


def test_schema_section_45_matches_this_table():
    """기준 파일 §4.5 의 매핑표가 코드와 갈리지 않는다.

    §4.5 는 이 표의 사본이다 — 사람이 "이 span 이 왜 ORG 인가" 를 되짚을 때
    읽는 자리이고, 기준 파일이라 바꾸려면 사람 승인과 반박자를 지나야 한다.
    사본이 조용히 낡으면 그 승인 절차가 낡은 표를 지키게 된다.

    지금까지 이 대조는 반박자가 라운드마다 **손으로** 했다. 손으로 하는 검사는
    누가 안 보면 그냥 지나간다.
    """
    import re
    from pathlib import Path

    schema = (Path(__file__).resolve().parents[4] / 'docs' / 'manual' /
              'data' / 'canonical-entity-schema.md').read_text(encoding='utf-8')
    body = schema[schema.index('### 4.5 EN'):]
    body = body[:body.index('####')]

    declared: dict[str, str] = {}
    for row in re.findall(r'^\|(.+?)\|(.+?)\|', body, re.M):
        raw, target = (c.strip() for c in row)
        if raw.startswith('---') or '원본 OntoNotes' in raw:
            continue
        types = re.findall(r'`([A-Z_]+)`', raw)
        if not types:
            continue  # "(OntoNotes 미커버)" 행 — PII 주입이 채우는 자리
        label = re.findall(r'`([A-Z]+)`', target)
        canonical = label[0] if label else DROP
        if '비-entity' in target:
            canonical = DROP
        for source_type in types:
            declared[source_type] = canonical

    assert declared == ONTONOTES_TO_CANONICAL, (
        '§4.5 와 mapping.py 가 갈렸다 — '
        f'문서만: {set(declared.items()) - set(ONTONOTES_TO_CANONICAL.items())} / '
        f'코드만: {set(ONTONOTES_TO_CANONICAL.items()) - set(declared.items())}'
    )
