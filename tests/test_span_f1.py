"""Unit tests for character-level span F1 metric (KLUE official)."""

import sys
sys.path.insert(0, "/work/git/ner_pipeline/src")

from evaluators.metrics import MetricsCalculator


def test_perfect_match_with_space_tokens():
    """KLUE-style test with whitespace tokens -- perfect prediction."""
    gold_tokens = [["경", "찰", "은", " ", "박", " ", "씨"]]
    gold_tags = [["B-OG", "I-OG", "O", "O", "B-PS", "O", "O"]]
    pred_tags = [["B-OG", "I-OG", "O", "O", "B-PS", "O", "O"]]

    result = MetricsCalculator.compute_span_f1(
        gold_tags, pred_tags, gold_tokens, gold_tokens
    )

    assert result["overall"]["f1"] == 1.0
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert "OG" in result["per_entity"]
    assert "PS" in result["per_entity"]
    assert result["per_entity"]["OG"]["f1"] == 1.0
    assert result["per_entity"]["PS"]["f1"] == 1.0


def test_partial_mismatch():
    """Pred misses one entity."""
    gold_tokens = [["경", "찰", "은", " ", "박", " ", "씨"]]
    gold_tags = [["B-OG", "I-OG", "O", "O", "B-PS", "O", "O"]]
    pred_tags = [["B-OG", "I-OG", "O", "O", "O", "O", "O"]]  # missed PS

    result = MetricsCalculator.compute_span_f1(
        gold_tags, pred_tags, gold_tokens, gold_tokens
    )

    # 1 correct (OG) out of 2 gold, 1 pred
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 0.5
    assert result["per_entity"]["PS"]["recall"] == 0.0


def test_wrong_entity_type():
    """Pred has correct span boundaries but wrong type."""
    gold_tokens = [["박"]]
    gold_tags = [["B-PS"]]
    pred_tags = [["B-OG"]]

    result = MetricsCalculator.compute_span_f1(
        gold_tags, pred_tags, gold_tokens, gold_tokens
    )

    # Different types => no match
    assert result["overall"]["f1"] == 0.0


def test_empty_input():
    """Empty inputs return zero metrics."""
    result = MetricsCalculator.compute_span_f1([], [], [], [])
    assert result["overall"]["f1"] == 0.0
    assert result["per_entity"] == {}


def test_multi_sentence():
    """Multiple sentences computed correctly."""
    gold_tokens = [["김", "철", "수"], ["서", "울"]]
    gold_tags = [["B-PS", "I-PS", "I-PS"], ["B-LC", "I-LC"]]
    pred_tags = [["B-PS", "I-PS", "I-PS"], ["B-LC", "I-LC"]]

    result = MetricsCalculator.compute_span_f1(
        gold_tags, pred_tags, gold_tokens, gold_tokens
    )

    assert result["overall"]["f1"] == 1.0
    assert result["per_entity"]["PS"]["support"] == 1
    assert result["per_entity"]["LC"]["support"] == 1


if __name__ == "__main__":
    test_perfect_match_with_space_tokens()
    test_partial_mismatch()
    test_wrong_entity_type()
    test_empty_input()
    test_multi_sentence()
    print("All tests passed.")
