"""ja_negative.oversampler 단위 테스트.

순수 함수 + 통합 (oversample_to_jsonl) 모두 cover. classifier 학습 경로는
별 통합 테스트로 분리.
"""

import json

from ner.augmenters.ja_negative.oversampler import (
    extract_seeds,
    filter_ambiguous_seeds,
    find_candidate_indices,
    oversample_to_jsonl,
)


def _write_diagnosis(tmp_path, entries):
    """diagnosis JSON 한 문장의 FP 리스트로 구성된 minimal dump 저장."""
    dump = {
        'sentences': [
            {'sent_idx': 0, 'text': '...', 'fp': entries, 'fn': []}
        ],
    }
    path = tmp_path / 'diag.json'
    path.write_text(json.dumps(dump), encoding='utf-8')
    return str(path)


def _fp(error_class, pred_type, text):
    return {
        'error_class': error_class,
        'pred': {'type': pred_type, 'start': 0, 'end': len(text), 'text': text},
    }


def test_extract_seeds_all_types(tmp_path):
    diag = _write_diagnosis(tmp_path, [
        _fp('HALLUCINATION', 'ORG', 'PC'),
        _fp('HALLUCINATION', 'PROD', 'Mac'),
        _fp('HALLUCINATION', 'EVT', 'FI'),
        _fp('BOUNDARY', 'ORG', 'KDE'),  # BOUNDARY 는 추출 안 됨
    ])
    seeds = extract_seeds(diag)
    assert seeds == {'PC', 'Mac', 'FI'}


def test_extract_seeds_filtered_by_pred_type(tmp_path):
    diag = _write_diagnosis(tmp_path, [
        _fp('HALLUCINATION', 'ORG', 'PC'),
        _fp('HALLUCINATION', 'CREDIT_CARD', '3468-9497'),
    ])
    ner_only = extract_seeds(diag, allowed_pred_types={'PER', 'LOC', 'ORG', 'PROD', 'EVT'})
    assert ner_only == {'PC'}


def test_filter_ambiguous_seeds():
    train_rows = [
        {'text': 'ドイツに住む', 'entities': [
            {'label': 'LOC', 'start_char': 0, 'end_char': 3},
        ]},
        {'text': '工場で働く', 'entities': []},  # 工場 은 entity 아님
    ]
    seeds = {'ドイツ', '工場', 'PC'}
    filtered, excluded = filter_ambiguous_seeds(seeds, train_rows)
    assert filtered == {'工場', 'PC'}
    assert excluded == {'ドイツ'}


def test_find_candidate_indices_basic():
    rows = [
        {'text': '私はPCを使う', 'entities': []},  # PC 부정 위치 → 후보
        {'text': 'PERSONAL COMPUTER のPC', 'entities': [
            {'label': 'ORG', 'start_char': 19, 'end_char': 21},  # 'PC' 가 ORG 라벨
        ]},  # PC 위치가 entity → 본 문장에서는 다른 unlabeled 위치도 없으므로 후보 아님
        {'text': '카드는Mac과 다르다', 'entities': []},  # Mac 부정 → 후보
    ]
    seeds = {'PC', 'Mac'}
    candidates = find_candidate_indices(rows, seeds)
    assert 0 in candidates
    assert 1 not in candidates
    assert 2 in candidates


def test_find_candidate_indices_skips_when_all_occurrences_are_entity():
    rows = [
        {'text': 'PC를쓴다', 'entities': [
            {'label': 'ORG', 'start_char': 0, 'end_char': 2},  # PC = ORG
        ]},
    ]
    candidates = find_candidate_indices(rows, {'PC'})
    assert candidates == []


def test_oversample_to_jsonl_extra_only(tmp_path):
    # 입력 jsonl: 3 row
    rows = [
        {'text': '문장1 PC', 'entities': []},
        {'text': '문장2 Mac', 'entities': []},
        {'text': '문장3 no_seed', 'entities': []},
    ]
    input_path = tmp_path / 'input.jsonl'
    with open(input_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    diag = _write_diagnosis(tmp_path, [
        _fp('HALLUCINATION', 'ORG', 'PC'),
        _fp('HALLUCINATION', 'PROD', 'Mac'),
    ])

    output_path = tmp_path / 'out.jsonl'
    stats = oversample_to_jsonl(
        input_path=str(input_path),
        diagnosis_path=diag,
        output_path=str(output_path),
        oversample=3,
        auto_exclude_ambiguous=False,
        extra_only=True,
        # split 옵션 영향 없음 (auto_exclude_ambiguous=False)
    )
    # candidates 2 (PC, Mac 문장) × oversample-1=2 = 4 extra row
    assert stats['candidates'] == 2
    assert stats['extra_rows'] == 4
    assert stats['output_rows'] == 4

    with open(output_path, encoding='utf-8') as fp:
        out_rows = [json.loads(line) for line in fp]
    assert len(out_rows) == 4
    # 출력 row 들이 PC 또는 Mac 문장임을 확인
    texts = [r['text'] for r in out_rows]
    for t in texts:
        assert 'PC' in t or 'Mac' in t


def test_oversample_to_jsonl_full_combo(tmp_path):
    # extra_only=False — 원본 + extra 합본 출력
    rows = [
        {'text': '문장1 PC', 'entities': []},
        {'text': '문장2 no_seed', 'entities': []},
    ]
    input_path = tmp_path / 'input.jsonl'
    with open(input_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    diag = _write_diagnosis(tmp_path, [
        _fp('HALLUCINATION', 'ORG', 'PC'),
    ])

    output_path = tmp_path / 'out.jsonl'
    stats = oversample_to_jsonl(
        input_path=str(input_path),
        diagnosis_path=diag,
        output_path=str(output_path),
        oversample=2,
        auto_exclude_ambiguous=False,
        extra_only=False,
    )
    # 원본 2 + extra 1 (candidate 1개 × oversample-1=1) = 3
    assert stats['candidates'] == 1
    assert stats['extra_rows'] == 1
    assert stats['output_rows'] == 3
