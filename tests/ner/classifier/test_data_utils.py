"""classifier.data_utils 의 라벨 맵 / BIO 정렬 / span 디코드 단위 테스트."""

from typing import List

import pytest

from ner.classifier.data_utils import (
    CANONICAL_LABELS,
    NER_TYPES,
    PII_TYPES,
    _bio_labels_from_offsets,
    _context_word_count,
    build_label_maps,
    decode_bio_to_spans,
    group_stats,
    mask_pii_in_features,
    split_holdout_deploy,
    split_kfold_stratified,
    split_train_valid_test,
    stronger_group_keys,
    validate_group_key,
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


def _before_after_relabel():
    """id·text 동일, PROD 라벨만 절반에 주입한 (before, after) 쌍."""
    before = [{'text': f't{i}', 'entities': [], 'id': str(i)}
              for i in range(100)]
    after = []
    for i in range(100):
        ents = ([{'label': 'PROD', 'start_char': 0, 'end_char': 1,
                  'text': 'x'}] if i % 2 == 0 else [])
        after.append({'text': f't{i}', 'entities': ents, 'id': str(i)})
    return before, after


def test_kfold_no_stratify_label_invariant():
    """strat_labels=() 면 fold 멤버십이 PROD/EVT 라벨과 무관 — relabel 전후
    test fold 의 id 집합이 동일하다(이슈 #140 고정 split 계약)."""
    before, after = _before_after_relabel()
    for fi in range(5):
        _, _, tb = split_kfold_stratified(
            before, n_folds=5, fold_index=fi, seed=42, strat_labels=())
        _, _, ta = split_kfold_stratified(
            after, n_folds=5, fold_index=fi, seed=42, strat_labels=())
        assert {r['id'] for r in tb} == {r['id'] for r in ta}


def test_kfold_default_stratify_shifts_on_relabel():
    """대조: 기본 PROD/EVT 층화는 라벨이 바뀌면 fold 멤버십이 흔들린다 —
    #140 이 고정 split(strat_labels=())을 쓰는 이유."""
    before, after = _before_after_relabel()
    shifted = False
    for fi in range(5):
        _, _, tb = split_kfold_stratified(
            before, n_folds=5, fold_index=fi, seed=42)
        _, _, ta = split_kfold_stratified(
            after, n_folds=5, fold_index=fi, seed=42)
        if {r['id'] for r in tb} != {r['id'] for r in ta}:
            shifted = True
    assert shifted


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


def test_holdout_deploy_partition():
    """test n_test 홀드아웃 + train/valid/test disjoint, union=전체, 8:2."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    train, valid, test = split_holdout_deploy(
        rows, n_test=10, valid_ratio=0.2, seed=42
    )
    assert len(test) == 10
    assert len(valid) == 18  # 남은 90행의 20%
    assert len(train) == 72
    train_ids = {r['id'] for r in train}
    valid_ids = {r['id'] for r in valid}
    test_ids = {r['id'] for r in test}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert train_ids | valid_ids | test_ids == {str(i) for i in range(100)}


def test_holdout_deploy_deterministic():
    """같은 seed 면 train/valid/test 모두 동일."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(100)]
    a = split_holdout_deploy(rows, n_test=10, valid_ratio=0.2, seed=7)
    b = split_holdout_deploy(rows, n_test=10, valid_ratio=0.2, seed=7)
    for left, right in zip(a, b):
        assert [r['id'] for r in left] == [r['id'] for r in right]


def test_holdout_deploy_group_no_leak():
    """group_key 홀드아웃: test 원문이 train·valid 와 안 겹치고 group 무결."""
    rows = _grouped_rows()
    train, valid, test = split_holdout_deploy(
        rows, n_test=10, valid_ratio=0.2, seed=42, group_key='orig'
    )
    assert len(test) >= 10
    test_orig = {r['orig'] for r in test}
    seen_orig = {r['orig'] for r in train} | {r['orig'] for r in valid}
    assert test_orig.isdisjoint(seen_orig)
    # 같은 원문은 정확히 한 split 에만
    where: dict = {}
    for name, split in (('train', train), ('valid', valid), ('test', test)):
        for r in split:
            where.setdefault(r['orig'], set()).add(name)
    for orig, names in where.items():
        assert len(names) == 1, f'{orig} split across {names}'
    # union = 전체
    got = (test_orig | {r['orig'] for r in train} | {r['orig'] for r in valid})
    assert got == {r['orig'] for r in rows}


def test_holdout_deploy_test_require_types():
    """test_require_types: test 는 해당 라벨 보유 unit 만, 빈 샘플은 제외."""
    rows = []
    for i in range(60):
        rows.append({'text': '', 'orig': f'o{i}',
                     'entities': [{'label': 'PER', 'start_char': 0,
                                   'end_char': 1, 'text': 'x'}],
                     'id': f'ent-{i}'})
    for i in range(40):  # 엔티티 없는 빈 샘플
        rows.append({'text': '', 'orig': f'e{i}',
                     'entities': [], 'id': f'empty-{i}'})
    train, valid, test = split_holdout_deploy(
        rows, n_test=10, valid_ratio=0.2, seed=42, group_key='orig',
        test_require_types=set(CANONICAL_LABELS),
    )
    assert len(test) >= 10
    # test 전원 엔티티 보유, 빈 샘플 0
    assert all(r['entities'] for r in test)
    assert all(not r['id'].startswith('empty') for r in test)
    # 빈 샘플은 train/valid 로
    assert all(r['entities'] == [] for r in train + valid
               if r['id'].startswith('empty'))
    # leak-free 유지
    test_orig = {r['orig'] for r in test}
    seen = {r['orig'] for r in train} | {r['orig'] for r in valid}
    assert test_orig.isdisjoint(seen)


def test_context_word_count():
    """엔티티 밖 문맥 토큰만 센다(엔티티 span 내부는 제외)."""
    # 'Anh ấy sống tại Ha Noi' — 'Ha Noi'(15-21)만 엔티티 → 문맥어 4
    row = {'text': 'Anh ấy sống tại Ha Noi',
           'entities': [{'label': 'LOC', 'start_char': 16,
                         'end_char': 22, 'text': 'Ha Noi'}]}
    assert _context_word_count(row) == 4
    # 전체가 엔티티 → 문맥어 0
    frag = {'text': 'Cục Điều tra',
            'entities': [{'label': 'ORG', 'start_char': 0,
                          'end_char': 12, 'text': 'Cục Điều tra'}]}
    assert _context_word_count(frag) == 0
    # 구두점·괄호는 단어 아님: 'Valve ( công ty )' → 문맥어 2(công, ty)
    disamb = {'text': 'Valve ( công ty )',
              'entities': [{'label': 'ORG', 'start_char': 0,
                            'end_char': 5, 'text': 'Valve'}]}
    assert _context_word_count(disamb) == 2


def test_holdout_deploy_min_context_words():
    """test_min_context_words: 문맥어 부족한 단편은 test 제외, 문장만 test."""
    frag = {'text': 'Cục Điều tra', 'orig': 'frag',
            'entities': [{'label': 'ORG', 'start_char': 0,
                          'end_char': 12, 'text': 'Cục Điều tra'}],
            'id': 'frag-0'}
    sents = [{'text': f'Anh ay song tai Ha Noi so {i}', 'orig': f's{i}',
              'entities': [{'label': 'LOC', 'start_char': 16,
                            'end_char': 22, 'text': 'Ha Noi'}],
              'id': f's-{i}'} for i in range(30)]
    rows = [frag] + sents
    train, valid, test = split_holdout_deploy(
        rows, n_test=5, valid_ratio=0.2, seed=42, group_key='orig',
        test_require_types=set(CANONICAL_LABELS), test_min_context_words=3,
    )
    assert len(test) >= 5
    # 단편은 test 에서 빠지고 train/valid 로
    assert all(r['id'] != 'frag-0' for r in test)
    assert any(r['id'] == 'frag-0' for r in train + valid)
    # test 전원 문맥어 ≥3
    assert all(_context_word_count(r) >= 3 for r in test)


def test_holdout_deploy_insufficient_rows():
    """n_test 가 전체 행보다 크면 ValueError."""
    rows = [{'text': '', 'entities': [], 'id': str(i)} for i in range(5)]
    with pytest.raises(ValueError):
        split_holdout_deploy(rows, n_test=10, seed=42)


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


# --- 그룹 키 검증 / 3-way 형제 묶기 ---


def _sibling_rows(n_orig: int = 20, per_orig: int = 3) -> List[dict]:
    """한 원문에서 per_orig 개 형제 행이 나오는 코퍼스 (문장은 전부 다르다)."""
    rows: List[dict] = []
    rid = 0
    for g in range(n_orig):
        for k in range(per_orig):
            rows.append({'text': f'sentence {g} variant {k}',
                         'entities': [], 'id': str(rid), 'orig': f'root{g}'})
            rid += 1
    return rows


def test_group_stats_counts_rows_and_groups():
    rows = _sibling_rows(10, 3)
    assert group_stats(rows, None) == (30, 30)
    assert group_stats(rows, 'id') == (30, 30)
    assert group_stats(rows, 'orig') == (30, 10)


def test_stronger_group_keys_finds_missed_sibling_field():
    """고유 키를 선언하면 형제를 묶는 orig 가 더 강한 후보로 잡힌다."""
    rows = _sibling_rows(10, 3)
    assert stronger_group_keys(rows, 'orig') == {}
    assert stronger_group_keys(rows, 'id') == {'orig': 10}
    assert stronger_group_keys(rows, None) == {'orig': 10}


def test_stronger_group_keys_none_when_no_siblings():
    """형제가 없는 코퍼스는 어떤 필드도 더 강하게 묶지 못한다."""
    rows = [{'text': f't{i}', 'entities': [], 'id': str(i)}
            for i in range(50)]
    assert stronger_group_keys(rows, 'id') == {}
    assert stronger_group_keys(rows, None) == {}


def test_stronger_group_keys_ignores_crosscutting_category_field():
    """범주 필드는 그룹 수가 적어도 형제 키가 아니다 — 그룹 경계를 가로지른다.

    domain 은 2개 그룹뿐이지만 같은 orig 형제들을 서로 다른 domain 으로
    쪼갠다. 그룹 수만 보면 orig 를 밀어내는 오탐이 난다.
    """
    rows = [{'text': f't{i}', 'entities': [], 'id': str(i),
             'orig': f'root{i // 3}',
             'domain': 'news' if i % 2 else 'blog'}
            for i in range(30)]
    assert stronger_group_keys(rows, 'orig') == {}
    validate_group_key(rows, 'orig')  # 통과해야 한다


def test_stronger_group_keys_accepts_coarsening_field():
    """선언 키의 그룹을 통째로 포함하는 필드는 진짜 후보다."""
    rows = [{'text': f't{i}', 'entities': [], 'id': str(i),
             'orig': f'root{i // 3}'}
            for i in range(30)]
    # id 그룹(싱글턴)은 orig 그룹 안에 완전히 들어간다
    assert stronger_group_keys(rows, 'id') == {'orig': 10}


def test_validate_group_key_rejects_unique_key_hiding_siblings():
    """고유 키는 그룹 보호를 no-op 으로 만들면서 누출 카운터를 0 으로 만든다."""
    rows = _sibling_rows(10, 3)
    with pytest.raises(ValueError, match='stronger key exists'):
        validate_group_key(rows, 'id')
    with pytest.raises(ValueError, match='stronger key exists'):
        validate_group_key(rows, None)
    validate_group_key(rows, 'orig')  # 통과


def test_validate_group_key_rejects_missing_and_null_field():
    rows = _sibling_rows(4, 2)
    rows[0].pop('orig')
    with pytest.raises(ValueError, match='missing in 1/8'):
        validate_group_key(rows, 'orig')
    rows[0]['orig'] = None
    with pytest.raises(ValueError, match='null in 1/8'):
        validate_group_key(rows, 'orig')


def test_validate_group_key_accepts_sibling_free_corpus():
    """형제가 없으면 고유 키도 none 도 통과한다 (증명적 0)."""
    rows = [{'text': f't{i}', 'entities': [], 'id': str(i)}
            for i in range(50)]
    validate_group_key(rows, 'id')
    validate_group_key(rows, None)


def test_split_3way_group_key_none_matches_legacy_rowlevel():
    """group_key=None 3-way 는 기존 행 단위 셔플 분할과 완전히 동일하다."""
    import random as _random
    rows = [{'text': f't{i}', 'entities': [], 'id': str(i)}
            for i in range(1000)]
    rng = _random.Random(7)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_test, n_valid = 100, 100
    legacy = (shuffled[n_test + n_valid:], shuffled[n_test:n_test + n_valid],
              shuffled[:n_test])
    got = split_train_valid_test(rows, 0.1, 0.1, seed=7)
    for left, right in zip(got, legacy):
        assert [r['id'] for r in left] == [r['id'] for r in right]


def test_split_3way_group_key_prevents_sibling_leak():
    """group_key 를 주면 test 형제가 train·valid 로 새지 않는다."""
    rows = _sibling_rows(100, 3)
    train, valid, test = split_train_valid_test(
        rows, 0.1, 0.1, seed=42, group_key='orig')
    assert len(train) + len(valid) + len(test) == len(rows)
    train_roots = {r['orig'] for r in train}
    valid_roots = {r['orig'] for r in valid}
    test_roots = {r['orig'] for r in test}
    assert not (test_roots & train_roots)
    assert not (test_roots & valid_roots)
    assert not (train_roots & valid_roots)


def test_split_3way_rowlevel_does_leak_siblings():
    """대조: 행 단위 분할은 같은 코퍼스에서 실제로 형제를 가른다."""
    rows = _sibling_rows(100, 3)
    train, _valid, test = split_train_valid_test(rows, 0.1, 0.1, seed=42)
    train_roots = {r['orig'] for r in train}
    assert any(r['orig'] in train_roots for r in test)
