"""일본어 NER span 기반 평가 메트릭.

BIO 태그 변환 없이 문자 오프셋 span을 직접 사용하여 NER 성능을 평가한다.
seqeval 의존성이 없다.

두 가지 매칭 모드를 제공한다:
- strict: (start, end, type) 정확 일치만 정답 (기본 게이트 측정)
- relaxed: SemEval'13 Partial — type 일치 + char-offset overlap 시 0.5점,
  정확 일치 시 1.0점. boundary 오류 카테고리 분리·error analysis 용.
"""
from typing import Dict, List


def compute_offset_span_f1(
    gold_spans_list: List[List[dict]],
    pred_spans_list: List[List[dict]],
) -> dict:
    """문자 오프셋 strict span F1 (정확 일치).

    각 span dict는 type, start, end 키를 가져야 한다.
    정확 매칭: (start, end, type)이 모두 일치해야 한다.

    Args:
        gold_spans_list: 문장별 gold span. 각 span 형식: {"type": str, "start": int, "end": int}
        pred_spans_list: 문장별 예측 span. 동일 형식.

    Returns:
        {"overall": {"f1", "precision", "recall", "support"},
         "per_entity": {"人名": {"f1", ...}, ...}}

    Raises:
        ValueError: 정답과 예측의 문장 수가 다를 때 발생한다.
            엔티티가 없는 문장도 빈 리스트로 포함해야 한다.
    """
    _validate_sentence_counts(gold_spans_list, pred_spans_list)
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


def compute_offset_span_f1_relaxed(
    gold_spans_list: List[List[dict]],
    pred_spans_list: List[List[dict]],
) -> dict:
    """SemEval'13 Partial F1 (boundary-relaxed).

    채점 규칙:
    - COR (correct): type + (start, end) 정확 일치 → 1.0 점
    - PAR (partial): type 일치 + char-offset 반열림 구간이 1자 이상 overlap → 0.5 점
    - 그 외 (type 불일치 또는 overlap 없음): 0 점
    - 1:1 매칭 — 한 pred 가 두 gold 와 동시에 매칭되지 않는다 (greedy: exact 우선).

    카운트:
        POS = gold span 총수
        ACT = pred span 총수
    P = (COR + 0.5·PAR) / ACT, R = (COR + 0.5·PAR) / POS
    F1 = 2·P·R / (P+R)

    Args:
        gold_spans_list: 문장별 gold span. 각 span 형식: {"type", "start", "end"}.
        pred_spans_list: 문장별 예측 span. 동일 형식.

    Returns:
        {"overall": {"f1", "precision", "recall", "support"},
         "per_entity": {entity: {"f1", ...}, ...}}

    Raises:
        ValueError: 정답과 예측의 문장 수가 다를 때 발생한다.
            엔티티가 없는 문장도 빈 리스트로 포함해야 한다.
    """
    _validate_sentence_counts(gold_spans_list, pred_spans_list)
    overall_cor = 0
    overall_par = 0
    overall_pos = 0
    overall_act = 0
    per_entity: Dict[str, Dict[str, int]] = {}

    def _bucket(t: str) -> Dict[str, int]:
        return per_entity.setdefault(
            t, {"cor": 0, "par": 0, "pos": 0, "act": 0}
        )

    for g_spans, p_spans in zip(gold_spans_list, pred_spans_list):
        # POS / ACT 카운트 (매칭 결과와 무관)
        for g in g_spans:
            _bucket(g["type"])["pos"] += 1
            overall_pos += 1
        for p in p_spans:
            _bucket(p["type"])["act"] += 1
            overall_act += 1

        # 1:1 매칭: exact 우선, 그 다음 partial (type + overlap)
        pred_used = [False] * len(p_spans)
        for g in g_spans:
            t = g["type"]
            # exact match 시도
            matched_exact = False
            for i, p in enumerate(p_spans):
                if pred_used[i] or p["type"] != t:
                    continue
                if p["start"] == g["start"] and p["end"] == g["end"]:
                    _bucket(t)["cor"] += 1
                    overall_cor += 1
                    pred_used[i] = True
                    matched_exact = True
                    break
            if matched_exact:
                continue
            # partial match 시도 (반열림 구간 1자 이상 overlap)
            for i, p in enumerate(p_spans):
                if pred_used[i] or p["type"] != t:
                    continue
                if max(p["start"], g["start"]) < min(p["end"], g["end"]):
                    _bucket(t)["par"] += 1
                    overall_par += 1
                    pred_used[i] = True
                    break

    overall = _partial_prf(overall_cor, overall_par, overall_act, overall_pos)
    per_entity_out: Dict[str, dict] = {}
    for t in sorted(per_entity.keys()):
        c = per_entity[t]
        per_entity_out[t] = _partial_prf(c["cor"], c["par"], c["act"], c["pos"])

    return {"overall": overall, "per_entity": per_entity_out}


def _validate_sentence_counts(
    gold_spans_list: List[List[dict]],
    pred_spans_list: List[List[dict]],
) -> None:
    """문장 누락이 채점에서 숨겨지지 않도록 양쪽 문장 수를 확인한다."""
    if len(gold_spans_list) != len(pred_spans_list):
        raise ValueError(
            'Sentence count mismatch: '
            f'gold={len(gold_spans_list)}, pred={len(pred_spans_list)}'
        )

def _prf_from_counts(tp: int, pred_count: int, gold_count: int) -> dict:
    """카운트로부터 precision, recall, F1을 계산한다."""
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "support": gold_count}


def _partial_prf(cor: int, par: int, act: int, pos: int) -> dict:
    """SemEval'13 Partial precision/recall/F1.

    분자에 partial 카운트의 절반(0.5)이 가산된다. support 는 gold 수(POS).
    """
    score = cor + 0.5 * par
    precision = score / act if act > 0 else 0.0
    recall = score / pos if pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "support": pos}
