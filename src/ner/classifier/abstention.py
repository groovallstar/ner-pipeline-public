"""per-class 신뢰도 기권(abstention) 운영점: 임계값 fit·apply·저장.

span score(conf_mean, decode_bio_to_spans 가 부착) 기준으로 클래스별 임계값
미만 예측을 걸러 정밀도를 높이는 운영점이다. 임계값은 검증(valid) 세트에서
fit(greedy, overall P·R≥target)하고 추론 시 적용한다.

원리: 저신뢰 예측이 FP 에 편중돼 있어, 임계값으로 그 꼬리를 잘라내면 리콜
여유를 정밀도로 바꾼다. 모델 종속 — 재학습 시 자신감 분포가 달라지므로
임계값을 반드시 재-fit 해야 한다. NER 4종(ORG/LOC/EVT/PROD)만 대상이며
PII·PER 은 이미 포화라 임계값을 두지 않는다(score 없으면 통과).
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# 임계값 대상 = precision 헤드룸이 있는 NER 4종 (PII·PER 은 포화라 제외)
DEFAULT_ABSTAIN_TYPES: Tuple[str, ...] = ('ORG', 'LOC', 'EVT', 'PROD')


def apply_thresholds(pred_spans_list: List[List[dict]],
                     thresholds: Dict[str, float]) -> List[List[dict]]:
    """score < thresholds[type] 인 span 을 제거한 새 리스트를 반환.

    thresholds 에 없는 type, 또는 score 키가 없는 span 은 통과시킨다(BC):
    임계값 미대상(포화 클래스)·신뢰도 미포착 예측은 거르지 않는다.
    """
    out: List[List[dict]] = []
    for spans in pred_spans_list:
        kept = []
        for s in spans:
            thr = thresholds.get(s['type'])
            if thr is not None and 'score' in s and s['score'] < thr:
                continue
            kept.append(s)
        out.append(kept)
    return out


def _flatten(gold_spans_list: List[List[dict]],
             pred_spans_list: List[List[dict]]):
    """(type, score, is_tp) 평탄화 + type 별 gold 총수.

    set 기반 micro(문장 내 (type,start,end) dedup, score 는 max 유지).
    """
    gold_tot: Dict[str, int] = {}
    preds = []  # (type, score, is_tp)
    for gold, pred in zip(gold_spans_list, pred_spans_list):
        gset = {(s['type'], s['start'], s['end']) for s in gold}
        for t in gset:
            gold_tot[t[0]] = gold_tot.get(t[0], 0) + 1
        best: Dict[Tuple[str, int, int], float] = {}
        for s in pred:
            key = (s['type'], s['start'], s['end'])
            sc = float(s.get('score', 1.0))
            if key not in best or sc > best[key]:
                best[key] = sc
        for key, sc in best.items():
            preds.append((key[0], sc, key in gset))
    return preds, gold_tot


def _overall(preds, gold_tot: Dict[str, int],
             thr: Dict[str, float]) -> Tuple[float, float, float]:
    """평탄화 예측에 임계값 적용한 overall micro P/R/F1."""
    tp = fp = 0
    g = sum(gold_tot.values())
    for (typ, sc, is_tp) in preds:
        if sc < thr.get(typ, 0.0):
            continue
        if is_tp:
            tp += 1
        else:
            fp += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / g if g else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def fit_thresholds(gold_spans_list: List[List[dict]],
                   pred_spans_list: List[List[dict]],
                   *,
                   types: Tuple[str, ...] = DEFAULT_ABSTAIN_TYPES,
                   target_p: float = 0.93,
                   target_r: float = 0.93,
                   n_candidates: int = 120) -> Dict[str, float]:
    """검증 set 의 gold·scored pred → 클래스별 임계값 greedy 탐색.

    F1 최대화 s.t. overall P≥target_p ∧ R≥target_r (penalty). 각 type score
    의 분위수를 후보로 coordinate ascent. pred span 은 `score` 부착 필요
    (decode_bio_to_spans(confs=...) 산출). 결정적(난수 없음).
    """
    preds, gold_tot = _flatten(gold_spans_list, pred_spans_list)
    cand: Dict[str, List[float]] = {}
    for c in types:
        cs = sorted(sc for (t, sc, _) in preds if t == c)
        if cs:
            qs = np.quantile(cs, np.linspace(0, 0.98, n_candidates))
            cand[c] = sorted(set([0.0] + [float(q) for q in qs]))
        else:
            cand[c] = [0.0]

    def score(thr: Dict[str, float]) -> float:
        p, r, f1 = _overall(preds, gold_tot, thr)
        return (f1 - 100 * max(0.0, target_p - p)
                - 100 * max(0.0, target_r - r))

    thr = {c: 0.0 for c in types}
    best_s = score(thr)
    for _ in range(6):
        improved = False
        for c in types:
            keep, bs = thr[c], best_s
            for t in cand[c]:
                thr[c] = t
                s = score(thr)
                if s > bs:
                    bs, keep = s, t
            thr[c] = keep
            if bs > best_s + 1e-12:
                best_s, improved = bs, True
        if not improved:
            break
    return thr


def save_thresholds(path: str, thresholds: Dict[str, float],
                    *, meta: Optional[dict] = None) -> None:
    """thresholds.json 저장. meta 에 conf_key·target·fit-set 출처 등 기록."""
    payload: Dict[str, object] = {'thresholds': thresholds}
    if meta:
        payload['meta'] = meta
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_thresholds(path: str) -> Dict[str, float]:
    """thresholds.json → type→threshold dict."""
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    return {k: float(v) for k, v in payload['thresholds'].items()}
