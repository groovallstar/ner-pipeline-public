"""병합본 전량 불변식 — 학습에 들어가는 파일이 그대로인가.

병합본(`data/ontonotes_en/origin.jsonl`)이 없으면 통째로 건너뛴다. `data/` 는
gitignore 되므로 클론 직후에는 없고, 그때 실패시키면 무관한 변경까지 붉어진다.

**골든 넘버는 이 파일에 둔다.** 병합 모듈에서 가져오면 도구를 고칠 때 기대치도
함께 움직여 검사가 자기참조가 된다 — `test_corpus_invariants.py` 와 같은
이유다. 변환·주입을 의도적으로 다시 돌린 사람은 이 숫자를 손으로 고쳐야 하고,
그 마찰이 목적이다.
"""
from pathlib import Path

import pytest

from ner.classifier.data_utils import (
    group_stats,
    load_jsonl,
    validate_group_key,
)

MERGED = (
    Path(__file__).resolve().parents[4]
    / 'data' / 'ontonotes_en' / 'origin.jsonl'
)

pytestmark = pytest.mark.skipif(
    not MERGED.exists(),
    reason=f'merged EN corpus not present at {MERGED}',
)

GOLDEN_ROWS = 76378
GOLDEN_GROUPS = 70641
GOLDEN_SPANS = 167871
# 라벨별 span 수. NER 6 종은 변환이, PII 4 종은 주입이 만든다.
GOLDEN_LABELS = {
    'EMAIL': 24105,
    'PHONE': 23115,
    'CREDIT_CARD': 22950,
    'ID_NUM': 22529,
    'LOC': 21671,
    'PER': 19260,
    'ORG': 17379,
    'DAT': 13926,
    'PROD': 1990,
    'EVT': 946,
}


@pytest.fixture(scope='module')
def rows():
    return load_jsonl(str(MERGED))


def test_row_and_group_counts(rows):
    assert len(rows) == GOLDEN_ROWS
    assert group_stats(rows, 'orig') == (GOLDEN_ROWS, GOLDEN_GROUPS)


def test_group_key_is_usable(rows):
    """`--group-key orig` 가 실제로 통하는지 — 학습 진입점이 하는 검사 그대로."""
    validate_group_key(rows, 'orig')


def test_split_field_is_gone(rows):
    assert all('split' not in row for row in rows)


def test_label_counts(rows):
    counts = {}
    for row in rows:
        for entity in row['entities']:
            counts[entity['label']] = counts.get(entity['label'], 0) + 1
    assert counts == GOLDEN_LABELS
    assert sum(counts.values()) == GOLDEN_SPANS


def test_spans_are_self_consistent(rows):
    """오프셋이 가리키는 글자와 엔티티 문자열이 같은가 — 불일치 0 이어야 한다."""
    bad = [
        (row['id'], entity)
        for row in rows
        for entity in row['entities']
        if row['text'][entity['start_char']:entity['end_char']]
        != entity['text']
    ]
    assert bad == []
