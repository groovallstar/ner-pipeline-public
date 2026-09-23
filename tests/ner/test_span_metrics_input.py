"""문장 누락이 offset F1 평가에서 숨겨지지 않는지 검증한다."""

from typing import Callable

import pytest

from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)


@pytest.mark.parametrize('metric', [
    compute_offset_span_f1, compute_offset_span_f1_relaxed,
], ids=['strict', 'relaxed'])
@pytest.mark.parametrize('gold_count,pred_count', [
    (2, 1), (1, 2), (1, 0), (0, 1),
])
@pytest.mark.parametrize('sentence', [
    [{'start': 0, 'end': 2, 'type': 'PER'}], [],
], ids=['with_entity', 'empty_sentence'])
def test_sentence_count_mismatch_is_rejected(
    metric: Callable, gold_count: int, pred_count: int, sentence: list,
) -> None:
    """양방향 문장 누락은 엔티티 유무와 무관하게 오류로 처리한다."""
    gold = [sentence for _ in range(gold_count)]
    pred = [sentence for _ in range(pred_count)]
    with pytest.raises(ValueError) as exc:
        metric(gold, pred)
    assert f'gold={gold_count}' in str(exc.value)
    assert f'pred={pred_count}' in str(exc.value)

@pytest.mark.parametrize('metric', [
    compute_offset_span_f1, compute_offset_span_f1_relaxed,
], ids=['strict', 'relaxed'])
@pytest.mark.parametrize('sentences', [[], [[], []]])
def test_empty_sentences_keep_zero_scores(
    metric: Callable, sentences: list,
) -> None:
    """같은 길이의 전체 빈 입력과 빈 문장 목록은 유효하다."""
    assert metric(sentences, sentences) == {
        'overall': {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0,
        },
        'per_entity': {},
    }

@pytest.mark.parametrize('metric', [
    compute_offset_span_f1, compute_offset_span_f1_relaxed,
], ids=['strict', 'relaxed'])
def test_different_entity_counts_keep_all_sentences(metric: Callable) -> None:
    """같은 문장 수에서 누락·추가 엔티티는 각각 FN·FP로 채점한다."""
    span = {'start': 0, 'end': 2, 'type': 'PER'}
    gold = [[span], [span], []]
    pred = [[span], [], [span]]
    assert metric(gold, pred) == {
        'overall': {
            'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'support': 2,
        },
        'per_entity': {'PER': {
            'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'support': 2,
        }},
    }
