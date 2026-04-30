"""일본어 NER span 기반 평가 메트릭.

BIO 태그 변환 없이 문자 오프셋 span을 직접 사용하여 NER 성능을 평가한다.
seqeval 의존성이 없다.
"""
from typing import Dict, List


def compute_offset_span_f1(
    gold_spans_list: List[List[dict]],
    pred_spans_list: List[List[dict]],
) -> dict:
    """문자 오프셋을 사용하여 span 수준 F1을 계산한다.

    각 span dict는 type, start, end 키를 가져야 한다.
    정확 매칭: (start, end, type)이 모두 일치해야 한다.

    Args:
        gold_spans_list: 문장별 gold span. 각 span 형식: {"type": str, "start": int, "end": int}
        pred_spans_list: 문장별 예측 span. 동일 형식.

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

    # 전체 마이크로 평균
    tp = len(all_gold & all_pred)
    overall = _prf_from_counts(tp, len(all_pred), len(all_gold))

    # 엔티티별 분석
    all_types = sorted(set(list(per_entity_gold.keys()) + list(per_entity_pred.keys())))
    per_entity = {}
    for etype in all_types:
        g = per_entity_gold.get(etype, set())
        p = per_entity_pred.get(etype, set())
        tp_e = len(g & p)
        per_entity[etype] = _prf_from_counts(tp_e, len(p), len(g))

    return {"overall": overall, "per_entity": per_entity}


def _prf_from_counts(tp: int, pred_count: int, gold_count: int) -> dict:
    """카운트로부터 precision, recall, F1을 계산한다."""
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "support": gold_count}
