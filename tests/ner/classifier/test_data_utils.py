"""classifier.data_utils 의 라벨 맵 / BIO 정렬 / span 디코드 단위 테스트."""

from typing import List

import pytest

from ner.classifier.data_utils import (
    CANONICAL_LABELS,
    NER_TYPES,
    PII_TYPES,
    _bio_labels_from_offsets,
    build_label_maps,
    decode_bio_to_spans,
    mask_pii_in_features,
    split_kfold_stratified,
    split_train_valid_test,
)


def test_canonical_labels_count():
    """canonical 은 NER 5 + PII 5 = 10종."""
    assert len(CANONICAL_LABELS) == 10
    ner = {'PER', 'LOC', 'ORG', 'PROD', 'EVT'}
    pii = {'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'}
    assert set(CANONICAL_LABELS) == ner | pii


def test_build_label_maps():
    """BIO 21 labels: O + 10*B + 10*I."""
    label2id, id2label = build_label_maps()
    assert len(label2id) == 21
    assert label2id['O'] == 0
    for t in CANONICAL_LABELS:
        assert f'B-{t}' in label2id
        assert f'I-{t}' in label2id
    # 역매핑 일관성
    for label, idx in label2id.items():
        assert id2label[idx] == label


def test_bio_labels_simple():
    """special token → -100, entity 내부는 B-/I-, 외부는 O."""
    label2id, _ = build_label_maps()
    # offsets: [CLS], 'abc'(0,3), 'de'(3,5), 'fg'(5,7), [SEP]
    offsets = [(0, 0), (0, 3), (3, 5), (5, 7), (0, 0)]
    entities = [{'label': 'PER', 'start_char': 0, 'end_char': 5, 'text': 'abcde'}]
    labels = _bio_labels_from_offsets(offsets, entities, label2id)
    assert labels == [
        -100,
        label2id['B-PER'],
        label2id['I-PER'],
        label2id['O'],
        -100,
    ]


def test_bio_labels_multiple_entities():
    """연속한 두 entity 는 각자 B 부터 시작."""
    label2id, _ = build_label_maps()
    offsets = [(0, 0), (0, 3), (4, 7), (8, 11), (0, 0)]
    entities = [
        {'label': 'PER', 'start_char': 0, 'end_char': 3, 'text': 'a'},
        {'label': 'ORG', 'start_char': 4, 'end_char': 11, 'text': 'b'},
    ]
    labels = _bio_labels_from_offsets(offsets, entities, label2id)
    assert labels == [
        -100,
        label2id['B-PER'],
        label2id['B-ORG'],
        label2id['I-ORG'],
        -100,
    ]


def test_decode_bio_to_spans():
    """BIO 시퀀스 → char-offset span (max(end) 병합)."""
    label2id, id2label = build_label_maps()
    label_ids = [
        label2id['B-PER'],
        label2id['I-PER'],
        label2id['O'],
        label2id['B-EMAIL'],
        label2id['I-EMAIL'],
    ]
    offsets = [(0, 3), (3, 6), (6, 7), (8, 13), (13, 18)]
    spans = decode_bio_to_spans(label_ids, offsets, id2label)
    assert spans == [
        {'type': 'PER', 'start': 0, 'end': 6},
        {'type': 'EMAIL', 'start': 8, 'end': 18},
    ]


def test_decode_special_tokens_skipped():
    """(0,0) special token 은 무시되고 진행 중 span 을 끊는다."""
    label2id, id2label = build_label_maps()
    label_ids = [
        label2id['B-PER'], label2id['I-PER'],
        label2id['O'],  # (0,0) on next
        label2id['B-PER'],
    ]
    offsets = [(0, 3), (3, 6), (0, 0), (10, 13)]
    spans = decode_bio_to_spans(label_ids, offsets, id2label)
    assert spans == [
        {'type': 'PER', 'start': 0, 'end': 6},
        {'type': 'PER', 'start': 10, 'end': 13},
    ]


def test_split_train_valid_test_partition():
    """3-way 분할: train/valid/test 가 disjoint 하고 union 이 전체."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    train, valid, test = split_train_valid_test(
        rows, valid_ratio=0.1, test_ratio=0.1, seed=42
    )
    assert len(test) == 10
    assert len(valid) == 10
    assert len(train) == 80
    train_ids = {r['id'] for r in train}
    valid_ids = {r['id'] for r in valid}
    test_ids = {r['id'] for r in test}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert train_ids | valid_ids | test_ids == {str(i) for i in range(100)}


def test_split_train_valid_test_deterministic():
    """같은 seed 면 train/valid/test 모두 동일."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    a = split_train_valid_test(rows, 0.1, 0.1, seed=7)
    b = split_train_valid_test(rows, 0.1, 0.1, seed=7)
    for left, right in zip(a, b):
        assert [r['id'] for r in left] == [r['id'] for r in right]


def test_kfold_partition():
    """단일 fold_index 에서 train/valid/test disjoint + union = 전체."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    train, valid, test = split_kfold_stratified(
        rows, n_folds=5, fold_index=0, seed=42
    )
    train_ids = {r['id'] for r in train}
    valid_ids = {r['id'] for r in valid}
    test_ids = {r['id'] for r in test}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert train_ids | valid_ids | test_ids == {str(i) for i in range(100)}


def test_kfold_every_row_tested_once():
    """fold_index 0..4 순회하며 test 집합 모으면 전체 rows 와 일치, 중복 없음."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    seen: List[str] = []
    for fi in range(5):
        _, _, test = split_kfold_stratified(
            rows, n_folds=5, fold_index=fi, seed=42
        )
        seen.extend(r['id'] for r in test)
    # 중복 없음
    assert len(seen) == len(set(seen))
    # 전체 rows 와 일치
    assert set(seen) == {str(i) for i in range(100)}


def test_kfold_stratification():
    """PROD 문장 20개 + 일반 80개 → 각 fold test 에 PROD 가 4개씩(±1)."""
    rows = []
    for i in range(20):
        rows.append({
            'text': '',
            'entities': [
                {'label': 'PROD', 'start_char': 0,
                 'end_char': 1, 'text': 'x'}
            ],
            'id': f'prod-{i}',
        })
    for i in range(80):
        rows.append({'text': '', 'entities': [], 'id': f'plain-{i}'})
    for fi in range(5):
        _, _, test = split_kfold_stratified(
            rows, n_folds=5, fold_index=fi, seed=42
        )
        n_prod = sum(
            1 for r in test
            if any(e['label'] == 'PROD' for e in r['entities'])
        )
        assert 3 <= n_prod <= 5


def test_kfold_deterministic():
    """같은 seed 로 두 번 호출 시 동일 분할."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    a = split_kfold_stratified(rows, n_folds=5, fold_index=2, seed=7)
    b = split_kfold_stratified(rows, n_folds=5, fold_index=2, seed=7)
    for left, right in zip(a, b):
        assert [r['id'] for r in left] == [r['id'] for r in right]


def test_kfold_invalid_fold_index():
    """범위 밖 fold_index 에 ValueError."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(10)]
    with pytest.raises(ValueError):
        split_kfold_stratified(rows, n_folds=5, fold_index=5, seed=42)
    with pytest.raises(ValueError):
        split_kfold_stratified(rows, n_folds=5, fold_index=-1, seed=42)


def test_kfold_n_folds_too_small():
    """n_folds < 3 이면 train 이 비므로 ValueError."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(10)]
    with pytest.raises(ValueError):
        split_kfold_stratified(rows, n_folds=2, fold_index=0, seed=42)


def test_kfold_n_folds_exceeds_rows():
    """n_folds 가 행 수보다 크면 ValueError."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(4)]
    with pytest.raises(ValueError):
        split_kfold_stratified(rows, n_folds=10, fold_index=0, seed=42)


def _grouped_rows() -> List[dict]:
    """원문 중복을 시뮬레이션한 rows: 30개 원문, 각 1~3행(주입텍스트 유니크)."""
    rows: List[dict] = []
    rid = 0
    for g in range(30):
        for _ in range(1 + g % 3):
            rows.append({
                'text': f'inj-{rid}',
                'orig': f'orig-{g}',
                'entities': [],
                'id': str(rid),
            })
            rid += 1
    return rows


def test_group_kfold_none_equals_rowlevel():
    """group_key=None 은 인자 생략(행 단위)과 완전히 동일한 분할."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    a = split_kfold_stratified(rows, 5, 1, 42)
    b = split_kfold_stratified(rows, 5, 1, 42, group_key=None)
    for left, right in zip(a, b):
        assert [r['id'] for r in left] == [r['id'] for r in right]


def test_group_kfold_no_cross_fold_leak():
    """group_key 분할 시 test 원문이 train·valid 원문과 절대 겹치지 않는다."""
    rows = _grouped_rows()
    for fi in range(5):
        tr, va, te = split_kfold_stratified(
            rows, n_folds=5, fold_index=fi, seed=42, group_key='orig'
        )
        test_orig = {r['orig'] for r in te}
        seen_orig = {r['orig'] for r in tr} | {r['orig'] for r in va}
        assert test_orig.isdisjoint(seen_orig)


def test_group_kfold_groups_intact():
    """같은 orig 의 모든 행은 항상 한 split 에만 속한다."""
    rows = _grouped_rows()
    where: dict = {}
    for fi in range(5):
        tr, va, te = split_kfold_stratified(
            rows, n_folds=5, fold_index=fi, seed=42, group_key='orig'
        )
        for name, split in (('train', tr), ('valid', va), ('test', te)):
            for r in split:
                where.setdefault((fi, r['orig']), set()).add(name)
    for (_, orig), splits in where.items():
        assert len(splits) == 1, f'{orig} split across {splits}'


def test_group_kfold_every_row_tested_once():
    """group 분할도 fold 순회 시 모든 행이 정확히 한 번 test 된다."""
    rows = _grouped_rows()
    seen: List[str] = []
    for fi in range(5):
        _, _, te = split_kfold_stratified(
            rows, n_folds=5, fold_index=fi, seed=42, group_key='orig'
        )
        seen.extend(r['id'] for r in te)
    assert len(seen) == len(set(seen))
    assert set(seen) == {r['id'] for r in rows}


def test_group_kfold_n_folds_exceeds_groups():
    """group 수보다 n_folds 가 크면 ValueError (행 수는 충분해도)."""
    rows = []
    for _ in range(20):
        rows.append({'text': '', 'orig': 'one', 'entities': [],
                     'id': str(len(rows))})
    with pytest.raises(ValueError):
        split_kfold_stratified(rows, n_folds=5, fold_index=0, seed=42,
                               group_key='orig')


def test_ner_pii_partition():
    """NER_TYPES + PII_TYPES = CANONICAL_LABELS, 교집합 없음."""
    assert set(NER_TYPES).isdisjoint(set(PII_TYPES))
    assert set(NER_TYPES) | set(PII_TYPES) == set(CANONICAL_LABELS)
    assert len(NER_TYPES) == 5
    assert len(PII_TYPES) == 5


def test_mask_pii_in_features():
    """mask_pii_in_features 가 PII BIO 라벨만 O 로 치환, NER/-100 보존."""
    label2id, _ = build_label_maps()
    o_id = label2id['O']
    features = [{
        'input_ids': [1, 2, 3, 4, 5, 6],
        'attention_mask': [1, 1, 1, 1, 1, 1],
        'labels': [
            -100,
            label2id['B-PER'],
            label2id['I-PER'],
            label2id['B-EMAIL'],
            label2id['I-EMAIL'],
            label2id['B-LOC'],
        ],
    }]
    out = mask_pii_in_features(features, label2id)
    assert out[0]['labels'] == [
        -100,
        label2id['B-PER'],
        label2id['I-PER'],
        o_id,
        o_id,
        label2id['B-LOC'],
    ]
    # 원본 변경 없음
    assert features[0]['labels'][3] == label2id['B-EMAIL']
    # input_ids/attention_mask 보존
    assert out[0]['input_ids'] == features[0]['input_ids']
