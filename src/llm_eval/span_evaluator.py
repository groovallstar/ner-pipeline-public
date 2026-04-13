"""Lightweight span-level evaluator for LLM NER labeling results.

Connects bio_dataset.load() gold records with labeler.label_spans() predictions,
using MetricsCalculator.compute_span_match() for exact/relaxed F1 computation.

Usage:
    from labelers.bio_dataset import load
    from llm_eval.span_evaluator import evaluate

    gold = load("klue", "validation", max_samples=100)
    results = evaluate(gold, labeler)
"""

import logging
import time
from typing import Any, List, Optional

from metrics.bio_metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def evaluate(
    gold_records: List[dict],
    labeler: Any,
    *,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
    collect_diffs: bool = False,
) -> dict:
    """Evaluate LLM labeler against gold span annotations.

    Args:
        gold_records: Records from bio_dataset.load(), each with
                      "sentence", "spans", "tokens", "bio_tags", "id".
        labeler: Object with label_spans(text) -> list[{"text", "type"}].
        max_samples: Limit evaluation to first N records.
        show_progress: Show tqdm progress bar.
        collect_diffs: If True, include per-record diff analysis in result.

    Returns:
        Dict with "exact", "relaxed", "counts", "latency" keys.
        If collect_diffs=True, also includes "diffs" list.
    """
    if max_samples is not None:
        gold_records = gold_records[:max_samples]

    if not gold_records:
        result = _empty_result()
        if collect_diffs:
            result["diffs"] = []
        return result

    gold_spans_all: List[List[dict]] = []
    pred_spans_all: List[List[dict]] = []
    record_ids: List[str] = []
    sentences: List[str] = []
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
        if collect_diffs:
            record_ids.append(record.get("id", "?"))
            sentences.append(sentence)

    t_total = time.time() - t_start

    if not gold_spans_all:
        result = _empty_result()
        result["counts"]["errors"] = errors
        result["latency"]["total_seconds"] = round(t_total, 4)
        if collect_diffs:
            result["diffs"] = []
        return result

    span_match = MetricsCalculator.compute_span_match(gold_spans_all, pred_spans_all)

    result = {
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

    if collect_diffs:
        result["diffs"] = _compute_diffs(
            record_ids, sentences, gold_spans_all, pred_spans_all
        )

    return result


def _norm(text: str) -> str:
    """Normalize entity text: strip and remove internal spaces."""
    return text.strip().replace(" ", "")


def _compute_diffs(
    record_ids: List[str],
    sentences: List[str],
    gold_spans_all: List[List[dict]],
    pred_spans_all: List[List[dict]],
) -> List[dict]:
    """Compute per-record diff between gold and pred spans.

    Classifies each span as:
    - exact match: normalized text + type identical
    - boundary_error: relaxed match (containment) but not exact
    - false_negative: gold span with no match at all
    - false_positive: pred span with no match at all
    """
    diffs = []
    for rid, sentence, gold_spans, pred_spans in zip(
        record_ids, sentences, gold_spans_all, pred_spans_all
    ):
        # Normalize spans for comparison
        gold_normed = [(_norm(s["text"]), s["type"]) for s in gold_spans]
        pred_normed = [(_norm(s["text"]), s["type"]) for s in pred_spans]

        # Track which spans are matched
        gold_matched = [False] * len(gold_spans)
        pred_matched = [False] * len(pred_spans)

        # Pass 1: exact matches
        for pi, (pt, ptype) in enumerate(pred_normed):
            for gi, (gt, gtype) in enumerate(gold_normed):
                if gold_matched[gi] or pred_matched[pi]:
                    continue
                if pt == gt and ptype == gtype:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        # Pass 2: relaxed (containment) matches among remaining
        boundary_errors = []
        for pi, (pt, ptype) in enumerate(pred_normed):
            if pred_matched[pi]:
                continue
            for gi, (gt, gtype) in enumerate(gold_normed):
                if gold_matched[gi]:
                    continue
                if ptype == gtype and (pt in gt or gt in pt):
                    boundary_errors.append({
                        "gold": {"text": gold_spans[gi]["text"], "type": gtype},
                        "pred": {"text": pred_spans[pi]["text"], "type": ptype},
                    })
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        false_negatives = [
            {"text": gold_spans[gi]["text"], "type": gold_spans[gi]["type"]}
            for gi in range(len(gold_spans))
            if not gold_matched[gi]
        ]
        false_positives = [
            {"text": pred_spans[pi]["text"], "type": pred_spans[pi]["type"]}
            for pi in range(len(pred_spans))
            if not pred_matched[pi]
        ]

        diffs.append({
            "id": rid,
            "sentence": sentence,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "boundary_errors": boundary_errors,
        })

    return diffs


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
