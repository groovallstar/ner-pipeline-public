"""Tests for evaluators.span_evaluator module."""

import pytest

from evaluators.span_evaluator import evaluate


class MockLabeler:
    """Mock labeler that returns pre-configured spans per sentence."""

    def __init__(self, results: dict):
        """results: {sentence_text: [{"text": ..., "type": ...}, ...]}"""
        self._results = results

    def label_spans(self, text: str):
        return self._results.get(text, [])


class ErrorLabeler:
    """Mock labeler that raises on specific sentences."""

    def __init__(self, error_texts: set, fallback_spans: list):
        self._error_texts = error_texts
        self._fallback_spans = fallback_spans

    def label_spans(self, text: str):
        if text in self._error_texts:
            raise RuntimeError("labeling failed")
        return self._fallback_spans


def _make_record(id_, tokens, bio_tags, spans, sentence):
    return {
        "id": id_,
        "tokens": tokens,
        "bio_tags": bio_tags,
        "spans": spans,
        "sentence": sentence,
    }


class TestEvaluate:
    def test_empty_gold_records(self):
        labeler = MockLabeler({})
        result = evaluate([], labeler, show_progress=False)
        assert result["exact"]["overall"]["f1"] == 0.0
        assert result["relaxed"]["overall"]["f1"] == 0.0
        assert result["counts"]["gold_spans"] == 0

    def test_perfect_match(self):
        """Gold and pred spans identical → exact F1 = 1.0."""
        spans = [{"text": "경찰", "type": "OG"}, {"text": "서울", "type": "LC"}]
        rec = _make_record("0", ["경", "찰", " ", "서", "울"], ["B-OG", "I-OG", "O", "B-LC", "I-LC"], spans, "경찰 서울")
        labeler = MockLabeler({"경찰 서울": spans})
        result = evaluate([rec], labeler, show_progress=False)
        assert result["exact"]["overall"]["f1"] == 1.0
        assert result["exact"]["overall"]["precision"] == 1.0
        assert result["exact"]["overall"]["recall"] == 1.0

    def test_partial_match(self):
        """Gold 3 spans, pred gets 2 correct → check precision/recall."""
        gold_spans = [
            {"text": "경찰", "type": "OG"},
            {"text": "서울", "type": "LC"},
            {"text": "김철수", "type": "PS"},
        ]
        pred_spans = [
            {"text": "경찰", "type": "OG"},
            {"text": "서울", "type": "LC"},
        ]
        rec = _make_record("0", [], [], gold_spans, "경찰 서울 김철수")
        labeler = MockLabeler({"경찰 서울 김철수": pred_spans})
        result = evaluate([rec], labeler, show_progress=False)
        # precision = 2/2 = 1.0, recall = 2/3 ≈ 0.6667
        assert result["exact"]["overall"]["precision"] == 1.0
        assert 0.66 <= result["exact"]["overall"]["recall"] <= 0.67

    def test_space_normalization(self):
        """Gold '지난19일' vs pred '지난 19일' → exact match after normalization."""
        gold_spans = [{"text": "지난19일", "type": "DT"}]
        pred_spans = [{"text": "지난 19일", "type": "DT"}]
        rec = _make_record("0", [], [], gold_spans, "지난19일에")
        labeler = MockLabeler({"지난19일에": pred_spans})
        result = evaluate([rec], labeler, show_progress=False)
        assert result["exact"]["overall"]["f1"] == 1.0

    def test_relaxed_match(self):
        """Gold '박' vs pred '박씨' → relaxed match (containment)."""
        gold_spans = [{"text": "박", "type": "PS"}]
        pred_spans = [{"text": "박씨", "type": "PS"}]
        rec = _make_record("0", [], [], gold_spans, "박씨가")
        labeler = MockLabeler({"박씨가": pred_spans})
        result = evaluate([rec], labeler, show_progress=False)
        assert result["exact"]["overall"]["f1"] == 0.0  # not exact
        assert result["relaxed"]["overall"]["f1"] == 1.0  # but relaxed

    def test_labeler_error(self):
        """Labeler error on one record → skip it, evaluate the rest."""
        spans = [{"text": "서울", "type": "LC"}]
        rec_ok = _make_record("0", [], [], spans, "서울에서")
        rec_err = _make_record("1", [], [], [{"text": "경찰", "type": "OG"}], "경찰이")
        labeler = ErrorLabeler(error_texts={"경찰이"}, fallback_spans=spans)
        result = evaluate([rec_ok, rec_err], labeler, show_progress=False)
        assert result["counts"]["errors"] == 1
        assert result["exact"]["overall"]["f1"] == 1.0  # only rec_ok evaluated

    def test_max_samples(self):
        """max_samples=10 limits evaluation to first 10 records."""
        spans = [{"text": "서울", "type": "LC"}]
        records = [_make_record(str(i), [], [], spans, "서울") for i in range(100)]
        labeler = MockLabeler({"서울": spans})
        result = evaluate(records, labeler, max_samples=10, show_progress=False)
        evaluated = result["counts"]["gold_spans"]
        # 10 records × 1 span each = 10 gold spans
        assert evaluated == 10

    def test_return_structure(self):
        """Return dict has required top-level keys."""
        labeler = MockLabeler({})
        result = evaluate([], labeler, show_progress=False)
        assert "exact" in result
        assert "relaxed" in result
        assert "counts" in result
        assert "latency" in result
        assert "overall" in result["exact"]
        assert "per_entity" in result["exact"]
        assert "total_seconds" in result["latency"]

    def test_per_entity_metrics(self):
        """Per-entity breakdown computes independently for PS and OG."""
        gold_spans = [
            {"text": "경찰", "type": "OG"},
            {"text": "김철수", "type": "PS"},
            {"text": "소방서", "type": "OG"},
        ]
        pred_spans = [
            {"text": "경찰", "type": "OG"},
            {"text": "박영희", "type": "PS"},  # wrong PS
            {"text": "소방서", "type": "OG"},
        ]
        rec = _make_record("0", [], [], gold_spans, "경찰 김철수 소방서")
        labeler = MockLabeler({"경찰 김철수 소방서": pred_spans})
        result = evaluate([rec], labeler, show_progress=False)
        per_entity = result["exact"]["per_entity"]
        assert per_entity["OG"]["f1"] == 1.0  # 2/2 correct
        assert per_entity["PS"]["f1"] == 0.0  # 0/1 correct
