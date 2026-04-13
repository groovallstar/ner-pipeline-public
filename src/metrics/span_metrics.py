"""Japanese NER span-based evaluation metrics.

Evaluates NER performance using character offset spans directly,
without BIO tag conversion. No seqeval dependency.
"""
from typing import Dict, List


def compute_offset_span_f1(
    gold_spans_list: List[List[dict]],
    pred_spans_list: List[List[dict]],
) -> dict:
    """Compute span-level F1 using character offsets.

    Each span dict must have keys: type, start, end.
    Exact match: (start, end, type) must all match.

    Args:
        gold_spans_list: Gold spans per sentence. Each span: {"type": str, "start": int, "end": int}
        pred_spans_list: Predicted spans per sentence. Same format.

    Returns:
        {"overall": {"f1", "precision", "recall", "support"},
         "per_entity": {"人名": {"f1", ...}, ...}}
    """
    all_gold: set = set()
    all_pred: set = set()
    per_entity_gold: Dict[str, set] = {}
    per_entity_pred: Dict[str, set] = {}

    for sent_idx, (g_spans, p_spans) in enumerate(zip(gold_spans_list, pred_spans_list)):
        for s in g_spans:
            key = (sent_idx, s["start"], s["end"], s["type"])
            all_gold.add(key)
            per_entity_gold.setdefault(s["type"], set()).add(key)
        for s in p_spans:
            key = (sent_idx, s["start"], s["end"], s["type"])
            all_pred.add(key)
            per_entity_pred.setdefault(s["type"], set()).add(key)

    # Overall micro-average
    tp = len(all_gold & all_pred)
    overall = _prf_from_counts(tp, len(all_pred), len(all_gold))

    # Per-entity breakdown
    all_types = sorted(set(list(per_entity_gold.keys()) + list(per_entity_pred.keys())))
    per_entity = {}
    for etype in all_types:
        g = per_entity_gold.get(etype, set())
        p = per_entity_pred.get(etype, set())
        tp_e = len(g & p)
        per_entity[etype] = _prf_from_counts(tp_e, len(p), len(g))

    return {"overall": overall, "per_entity": per_entity}


def _prf_from_counts(tp: int, pred_count: int, gold_count: int) -> dict:
    """Compute precision, recall, F1 from counts."""
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "support": gold_count}
