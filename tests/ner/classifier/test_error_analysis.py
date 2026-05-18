"""classify_span_errors / aggregate_errors / sample_for_review 단위 테스트.

추론·모델 로딩 경로는 별도 (느린) 통합 테스트로 분리. 본 파일은 순수 함수만.
"""

from ner.classifier.error_analysis import (
    NULL,
    aggregate_errors,
    build_confusion_matrix,
    classify_span_errors,
    sample_for_review,
    top_errors_by_type,
)


def _g(t, s, e, txt=''):
    return {'type': t, 'start': s, 'end': e, 'text': txt}


def test_classify_exact_match():
    gold = [_g('PER', 0, 3, 'abc')]
    pred = [_g('PER', 0, 3, 'abc')]
    r = classify_span_errors(gold, pred)
    assert r['counts']['exact'] == 1
    assert r['counts']['fn'] == 0
    assert r['counts']['fp'] == 0


def test_classify_boundary_error():
    # type 일치 + overlap 있으나 offset 다름
    gold = [_g('LOC', 0, 5, 'Tokyo')]
    pred = [_g('LOC', 0, 7, 'Tokyo S')]
    r = classify_span_errors(gold, pred)
    assert r['counts']['exact'] == 0
    assert len(r['fn']) == 1
    assert len(r['fp']) == 1
    assert r['fn'][0]['error_class'] == 'BOUNDARY'
    assert r['fp'][0]['error_class'] == 'BOUNDARY'


def test_classify_type_mismatch():
    # offset 동일하나 type 다름 → BOUNDARY 아님 (overlap 있고 type 다름 = TYPE_MISMATCH)
    gold = [_g('ORG', 10, 15)]
    pred = [_g('LOC', 10, 15)]
    r = classify_span_errors(gold, pred)
    assert r['fn'][0]['error_class'] == 'TYPE_MISMATCH'
    assert r['fp'][0]['error_class'] == 'TYPE_MISMATCH'


def test_classify_miss_and_hallucination():
    # gold 1, pred 1 — 둘은 무관 (overlap 0)
    gold = [_g('PER', 0, 3)]
    pred = [_g('LOC', 10, 13)]
    r = classify_span_errors(gold, pred)
    fn_classes = [e['error_class'] for e in r['fn']]
    fp_classes = [e['error_class'] for e in r['fp']]
    assert fn_classes == ['MISS']
    assert fp_classes == ['HALLUCINATION']


def test_classify_priority_exact_over_boundary():
    # 한 gold 가 두 pred 와 매칭 가능: EXACT 우선
    gold = [_g('PER', 0, 3)]
    pred = [_g('PER', 0, 4), _g('PER', 0, 3)]
    r = classify_span_errors(gold, pred)
    # EXACT 1개 + 남은 pred = HALLUCINATION
    assert r['counts']['exact'] == 1
    assert any(e['error_class'] == 'HALLUCINATION' for e in r['fp'])


def test_classify_boundary_picks_largest_overlap():
    # gold 1 + 두 BOUNDARY 후보 → 더 큰 overlap 선택
    gold = [_g('LOC', 0, 10)]
    pred = [_g('LOC', 0, 3), _g('LOC', 0, 9)]
    r = classify_span_errors(gold, pred)
    matched = [e for e in r['fn'] if e['error_class'] == 'BOUNDARY']
    assert len(matched) == 1
    assert matched[0]['matched_pred']['end'] == 9


def test_classify_overlap_diff_type_is_type_mismatch():
    gold = [_g('ORG', 0, 5)]
    pred = [_g('LOC', 2, 8)]
    r = classify_span_errors(gold, pred)
    assert r['fn'][0]['error_class'] == 'TYPE_MISMATCH'


def test_aggregate_counts():
    sr1 = {
        'counts': {'gold': 2, 'pred': 2, 'exact': 1, 'fn': 1, 'fp': 1},
        'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
        'fp': [{'error_class': 'HALLUCINATION', 'pred': _g('LOC', 5, 8)}],
    }
    sr2 = {
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
        'fn': [{'error_class': 'BOUNDARY', 'gold': _g('PER', 0, 3),
                'matched_pred': _g('PER', 0, 4)}],
        'fp': [{'error_class': 'BOUNDARY', 'pred': _g('PER', 0, 4),
                'matched_gold': _g('PER', 0, 3)}],
    }
    agg = aggregate_errors([sr1, sr2])
    assert agg['totals']['gold'] == 3
    assert agg['totals']['exact'] == 1
    assert agg['fn_by_class'] == {'MISS': 1, 'BOUNDARY': 1}
    assert agg['fp_by_class'] == {'HALLUCINATION': 1, 'BOUNDARY': 1}
    assert agg['fn_by_type']['PER'] == {'MISS': 1, 'BOUNDARY': 1}


def test_sample_for_review_deterministic():
    # 충분히 많은 오류 문장 생성
    sentence_results = []
    for i in range(100):
        sentence_results.append({
            'sent_idx': i,
            'text': f'sentence {i}',
            'gold_spans': [_g('PER', 0, 3)],
            'pred_spans': [],
            'exact': [],
            'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
            'fp': [],
            'counts': {'gold': 1, 'pred': 0, 'exact': 0, 'fn': 1, 'fp': 0},
        })
    s1 = sample_for_review(sentence_results, ratio=0.1, seed=42)
    s2 = sample_for_review(sentence_results, ratio=0.1, seed=42)
    assert [x['sent_idx'] for x in s1] == [x['sent_idx'] for x in s2]
    assert 5 <= len(s1) <= 15  # ratio + min_per_type 조합


def test_confusion_matrix_exact_and_boundary():
    # EXACT 1건 + BOUNDARY 1건 모두 대각선 (gold_type == pred_type)
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [{'gold': _g('PER', 0, 3), 'pred': _g('PER', 0, 3)}],
        'fn': [{'error_class': 'BOUNDARY', 'gold': _g('PER', 5, 10),
                'matched_pred': _g('PER', 5, 12)}],
        'fp': [{'error_class': 'BOUNDARY', 'pred': _g('PER', 5, 12),
                'matched_gold': _g('PER', 5, 10)}],
        'counts': {'gold': 2, 'pred': 2, 'exact': 1, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    # EXACT 1 + BOUNDARY 1 = 2 (모두 PER→PER)
    assert m['PER']['PER'] == 2
    assert NULL not in m  # MISS/HALL 없음
    assert 'PER' in m and len(m['PER']) == 1


def test_confusion_matrix_type_mismatch_no_double_count():
    # FN/FP 양쪽에 TYPE_MISMATCH entry 동일 — FN side 만 카운트되어 1번만 잡힘
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'TYPE_MISMATCH', 'gold': _g('ORG', 0, 5),
                'matched_pred': _g('PROD', 0, 5)}],
        'fp': [{'error_class': 'TYPE_MISMATCH', 'pred': _g('PROD', 0, 5),
                'matched_gold': _g('ORG', 0, 5)}],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    assert m['ORG']['PROD'] == 1
    assert m.get('PROD', {}).get('ORG', 0) == 0  # 반대 방향 카운트 없음


def test_confusion_matrix_miss_and_hallucination():
    # MISS → gold→NULL, HALLUCINATION → NULL→pred
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'MISS', 'gold': _g('LOC', 0, 3)}],
        'fp': [{'error_class': 'HALLUCINATION', 'pred': _g('PROD', 10, 13)}],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    assert m['LOC'][NULL] == 1
    assert m[NULL]['PROD'] == 1


def test_top_errors_by_type_surface_frequency_and_context():
    # 'abc' 3번 + 'def' 1번 → 'abc' 빈도 우선
    sr = []
    for i in range(3):
        sr.append({
            'sent_idx': i,
            'text': 'XXX abc YYY',
            'gold_spans': [], 'pred_spans': [],
            'exact': [], 'fn': [],
            'fp': [{
                'error_class': 'HALLUCINATION',
                'pred': {'type': 'PROD', 'start': 4, 'end': 7, 'text': 'abc'},
            }],
            'counts': {'gold': 0, 'pred': 1, 'exact': 0, 'fn': 0, 'fp': 1},
        })
    sr.append({
        'sent_idx': 99,
        'text': 'YYY def ZZZ',
        'gold_spans': [], 'pred_spans': [],
        'exact': [], 'fn': [],
        'fp': [{
            'error_class': 'TYPE_MISMATCH',
            'pred': {'type': 'PROD', 'start': 4, 'end': 7, 'text': 'def'},
            'matched_gold': {'type': 'ORG', 'start': 4, 'end': 7, 'text': 'def'},
        }],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 0, 'fp': 1},
    })
    top = top_errors_by_type(
        sr, side='fp', type_='PROD', n=10, ctx_chars=4,
    )
    assert len(top) == 4
    assert top[0]['surface'] == 'abc'  # 빈도 3 우선
    assert top[0]['left_ctx'] == 'XXX '  # ctx_chars=4
    assert top[0]['right_ctx'] == ' YYY'
    assert top[0]['counter_type'] == NULL  # HALLUCINATION
    # def 항목은 TYPE_MISMATCH 로 ORG counter
    def_entries = [r for r in top if r['surface'] == 'def']
    assert len(def_entries) == 1
    assert def_entries[0]['counter_type'] == 'ORG'


def test_sample_for_review_skips_clean_sentences():
    # FN/FP 가 없는 문장은 샘플 후보 아님
    clean = {
        'sent_idx': 0,
        'text': 'clean',
        'gold_spans': [], 'pred_spans': [],
        'exact': [], 'fn': [], 'fp': [],
        'counts': {'gold': 0, 'pred': 0, 'exact': 0, 'fn': 0, 'fp': 0},
    }
    err = {
        'sent_idx': 1,
        'text': 'err',
        'gold_spans': [_g('PER', 0, 3)], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
        'fp': [],
        'counts': {'gold': 1, 'pred': 0, 'exact': 0, 'fn': 1, 'fp': 0},
    }
    s = sample_for_review([clean, err], ratio=1.0, min_per_type=1, seed=0)
    assert all(x['sent_idx'] != 0 for x in s)
    assert any(x['sent_idx'] == 1 for x in s)
