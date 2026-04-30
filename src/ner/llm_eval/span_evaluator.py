"""LLM NER 라벨링 결과를 위한 경량 span 수준 평가기.

bio_dataset.load() gold 레코드와 labeler.label_spans() 예측을 연결하고,
MetricsCalculator.compute_span_match()로 정확/완화 F1을 계산한다.

사용 예:
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
    """LLM 라벨러를 gold span 어노테이션과 비교하여 평가한다.

    Args:
        gold_records: bio_dataset.load()의 레코드. 각 레코드는
                      "sentence", "spans", "tokens", "bio_tags", "id" 키를 가진다.
        labeler: label_spans(text) -> list[{"text", "type"}]를 가진 객체.
        max_samples: 평가를 처음 N개 레코드로 제한한다.
        show_progress: tqdm 진행 바 표시 여부.
        collect_diffs: True이면 결과에 레코드별 diff 분석을 포함한다.

    Returns:
        "exact", "relaxed", "counts", "latency" 키를 가진 dict.
        collect_diffs=True이면 "diffs" 리스트도 포함된다.
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
    """엔티티 텍스트를 정규화한다: 앞뒤 공백 및 내부 공백을 제거한다."""
    return text.strip().replace(" ", "")


def _compute_diffs(
    record_ids: List[str],
    sentences: List[str],
    gold_spans_all: List[List[dict]],
    pred_spans_all: List[List[dict]],
) -> List[dict]:
    """레코드별 gold span과 pred span의 diff를 계산한다.

    각 span을 다음으로 분류한다:
    - exact match: 정규화 텍스트 + 타입이 동일
    - boundary_error: 완화 매칭(포함 관계)이지만 정확 매칭은 아닌 경우
    - false_negative: 매칭되지 않은 gold span
    - false_positive: 매칭되지 않은 pred span
    """
    diffs = []
    for rid, sentence, gold_spans, pred_spans in zip(
        record_ids, sentences, gold_spans_all, pred_spans_all
    ):
        # 비교를 위해 span을 정규화한다
        gold_normed = [(_norm(s["text"]), s["type"]) for s in gold_spans]
        pred_normed = [(_norm(s["text"]), s["type"]) for s in pred_spans]

        # 매칭된 span을 추적한다
        gold_matched = [False] * len(gold_spans)
        pred_matched = [False] * len(pred_spans)

        # 1단계: 정확 매칭
        for pi, (pt, ptype) in enumerate(pred_normed):
            for gi, (gt, gtype) in enumerate(gold_normed):
                if gold_matched[gi] or pred_matched[pi]:
                    continue
                if pt == gt and ptype == gtype:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        # 2단계: 나머지에 대한 완화(포함 관계) 매칭
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
    """모든 값이 0으로 초기화된 결과 구조체를 반환한다."""
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
