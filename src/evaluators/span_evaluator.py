"""Lightweight span-level evaluator for LLM NER labeling results.

Connects bio_dataset.load() gold records with labeler.label_spans() predictions,
using MetricsCalculator.compute_span_match() for exact/relaxed F1 computation.

Usage:
    from labelers.bio_dataset import load
    from evaluators.span_evaluator import evaluate

    gold = load("klue", "validation", max_samples=100)
    results = evaluate(gold, labeler)
"""

import logging
import time
from typing import Any, List, Optional

from evaluators.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def evaluate(
    gold_records: List[dict],
    labeler: Any,
    *,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> dict:
    """Evaluate LLM labeler against gold span annotations.

    Args:
        gold_records: Records from bio_dataset.load(), each with
                      "sentence", "spans", "tokens", "bio_tags", "id".
        labeler: Object with label_spans(text) -> list[{"text", "type"}].
        max_samples: Limit evaluation to first N records.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with "exact", "relaxed", "counts", "latency" keys.
    """
    if max_samples is not None:
        gold_records = gold_records[:max_samples]

    if not gold_records:
        return _empty_result()

    gold_spans_all: List[List[dict]] = []
    pred_spans_all: List[List[dict]] = []
    errors = 0

    iterator = gold_records
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(gold_records, desc="Evaluating", unit="sample")
        except ImportError:
            pass

    t_start = time.time()

    for record in iterator:
        sentence = record["sentence"]
        gold_spans = record["spans"]

        try:
            pred_spans = labeler.label_spans(sentence)
        except Exception as e:
            logger.warning("Labeling failed for record %s: %s", record.get("id", "?"), e)
            errors += 1
            continue

        gold_spans_all.append(gold_spans)
        pred_spans_all.append(pred_spans)

    t_total = time.time() - t_start

    if not gold_spans_all:
        result = _empty_result()
        result["counts"]["errors"] = errors
        result["latency"]["total_seconds"] = round(t_total, 4)
        return result

    span_match = MetricsCalculator.compute_span_match(gold_spans_all, pred_spans_all)

    return {
        "exact": span_match["exact"],
        "relaxed": span_match["relaxed"],
        "counts": {
            **span_match["counts"],
            "errors": errors,
        },
        "latency": {
            "total_seconds": round(t_total, 4),
        },
    }


def _empty_result() -> dict:
    """Return a zeroed-out result structure."""
    zero_overall = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "exact": {"overall": {**zero_overall}, "per_entity": {}},
        "relaxed": {"overall": {**zero_overall}, "per_entity": {}},
        "counts": {
            "gold_spans": 0,
            "pred_spans": 0,
            "exact_matches": 0,
            "relaxed_matches": 0,
            "errors": 0,
        },
        "latency": {
            "total_seconds": 0.0,
        },
    }
