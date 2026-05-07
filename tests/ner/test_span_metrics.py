"""Unit tests for char-offset span F1 metrics (strict + SemEval'13 partial)."""

from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)


def _spans(*triples):
    """헬퍼: (start, end, type) 튜플을 metrics 딕셔너리로 변환."""
    return [{'start': s, 'end': e, 'type': t} for s, e, t in triples]


# ---------------------------------------------------------------------------
# strict (회귀 방지)
# ---------------------------------------------------------------------------

def test_strict_perfect_match():
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 11, 'DAT'))]
    r = compute_offset_span_f1(gold, pred)
    assert r['overall']['f1'] == 1.0
    assert r['per_entity']['DAT']['support'] == 1


def test_strict_boundary_off_by_one_is_zero():
    """strict 는 1자 boundary 차이도 오답."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 10, 'DAT'))]
    r = compute_offset_span_f1(gold, pred)
    assert r['overall']['f1'] == 0.0


def test_strict_wrong_type():
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 11, 'LOC'))]
    r = compute_offset_span_f1(gold, pred)
    assert r['overall']['f1'] == 0.0


# ---------------------------------------------------------------------------
# relaxed (SemEval'13 Partial F1)
# ---------------------------------------------------------------------------

def test_relaxed_perfect_match_is_one():
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 11, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 1.0


def test_relaxed_boundary_off_end_yields_half():
    """type 일치 + overlap → COR=0, PAR=1 → P=R=0.5 → F1=0.5."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 10, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.5
    assert r['overall']['precision'] == 0.5
    assert r['overall']['recall'] == 0.5


def test_relaxed_boundary_off_start_yields_half():
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((1, 11, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.5


def test_relaxed_over_extend_yields_half():
    """gold ⊂ pred 도 partial (0.5)."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 14, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.5


def test_relaxed_one_char_overlap_yields_half():
    """1자만 겹쳐도 partial (임계 케이스)."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((10, 13, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.5


def test_relaxed_adjacent_no_overlap_is_zero():
    """반열림 [start, end) 기준 경계 접촉만 발생하면 overlap 없음 → 0."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((11, 14, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.0


def test_relaxed_type_mismatch_zero_even_with_full_overlap():
    """type 불일치는 overlap 무관 0점."""
    gold = [_spans((0, 11, 'DAT'))]
    pred = [_spans((0, 11, 'LOC'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['overall']['f1'] == 0.0


def test_relaxed_one_to_one_matching():
    """동일 gold 에 overlap 후보가 2개 있어도 1개만 매칭 (잔여 pred 는 spurious)."""
    gold = [_spans((0, 10, 'DAT'))]
    pred = [_spans((0, 5, 'DAT'), (5, 10, 'DAT'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    # COR=0, PAR=1, ACT=2, POS=1
    # P=(0+0.5)/2=0.25, R=(0+0.5)/1=0.5, F1=2·0.25·0.5/0.75=1/3
    assert abs(r['overall']['precision'] - 0.25) < 1e-9
    assert abs(r['overall']['recall'] - 0.5) < 1e-9
    assert abs(r['overall']['f1'] - 1.0 / 3.0) < 1e-9


def test_relaxed_mixed_exact_and_partial():
    """exact 1개 + partial 1개 → COR=1, PAR=1, POS=ACT=2.
    P=R=(1+0.5)/2=0.75, F1=0.75.
    """
    gold = [_spans((0, 5, 'PER'), (10, 15, 'LOC'))]
    pred = [_spans((0, 5, 'PER'), (10, 14, 'LOC'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert abs(r['overall']['f1'] - 0.75) < 1e-9


def test_relaxed_per_entity_breakdown():
    gold = [_spans((0, 5, 'PER'), (10, 15, 'LOC'))]
    pred = [_spans((0, 5, 'PER'), (10, 14, 'LOC'))]
    r = compute_offset_span_f1_relaxed(gold, pred)
    assert r['per_entity']['PER']['f1'] == 1.0
    assert r['per_entity']['LOC']['f1'] == 0.5


def test_relaxed_empty_inputs():
    r = compute_offset_span_f1_relaxed([], [])
    assert r['overall']['f1'] == 0.0
    assert r['per_entity'] == {}


def test_relaxed_gold_empty_pred_present():
    """gold 0, pred 존재 → POS=0 → recall=0 → F1=0 (모두 spurious)."""
    r = compute_offset_span_f1_relaxed([[]], [_spans((0, 5, 'PER'))])
    assert r['overall']['f1'] == 0.0


if __name__ == '__main__':
    import sys
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'OK  {name}')
            except Exception:
                failed += 1
                print(f'FAIL {name}')
                traceback.print_exc()
    sys.exit(failed)
