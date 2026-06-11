"""NER 평가 메트릭: seqeval F1/Precision/Recall + span match."""

import logging
from typing import Dict, List

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """NER 평가 메트릭을 계산한다."""

    @staticmethod
    def compute_seqeval(
        gold_tags_list: List[List[str]],
        pred_tags_list: List[List[str]],
    ) -> Dict:
        """seqeval 메트릭을 계산한다 (엄격한 IOB2 모드).

        Args:
            gold_tags_list: gold BIO 태그 시퀀스 리스트.
            pred_tags_list: 예측 BIO 태그 시퀀스 리스트.

        Returns:
            전체 및 엔티티 타입별 메트릭이 포함된 dict.
        """
        # 동일 길이 시퀀스를 보장한다 (필요 시 잘라내거나 패딩)
        clean_gold, clean_pred = [], []
        for gold, pred in zip(gold_tags_list, pred_tags_list):
            min_len = min(len(gold), len(pred))
            if min_len == 0:
                continue
            clean_gold.append(gold[:min_len])
            clean_pred.append(pred[:min_len])

        if not clean_gold:
            return {"overall": {"f1": 0.0, "precision": 0.0, "recall": 0.0}, "per_entity": {}}

        try:
            overall_f1 = float(f1_score(clean_gold, clean_pred, mode="strict", scheme=IOB2))
            overall_prec = float(precision_score(clean_gold, clean_pred, mode="strict", scheme=IOB2))
            overall_rec = float(recall_score(clean_gold, clean_pred, mode="strict", scheme=IOB2))
        except Exception as e:
            logger.warning("Strict mode failed (%s), falling back to default", e)
            overall_f1 = float(f1_score(clean_gold, clean_pred))
            overall_prec = float(precision_score(clean_gold, clean_pred))
            overall_rec = float(recall_score(clean_gold, clean_pred))

        # classification_report에서 엔티티별 분석을 추출한다
        try:
            report_str = classification_report(clean_gold, clean_pred, output_dict=False, mode="strict", scheme=IOB2)
            report_dict = classification_report(clean_gold, clean_pred, output_dict=True, mode="strict", scheme=IOB2)
        except Exception:
            report_str = classification_report(clean_gold, clean_pred, output_dict=False)
            report_dict = classification_report(clean_gold, clean_pred, output_dict=True)

        per_entity = {}
        for key, val in report_dict.items():
            if isinstance(val, dict) and key not in ("micro avg", "macro avg", "weighted avg"):
                per_entity[key] = {
                    "precision": round(float(val.get("precision", 0.0)), 4),
                    "recall": round(float(val.get("recall", 0.0)), 4),
                    "f1": round(float(val.get("f1-score", 0.0)), 4),
                    "support": int(val.get("support", 0)),
                }

        return {
            "overall": {
                "f1": round(overall_f1, 4),
                "precision": round(overall_prec, 4),
                "recall": round(overall_rec, 4),
            },
            "per_entity": per_entity,
            "report": report_str,
        }

    @staticmethod
    def _extract_spans(tokens: List[str], tags: List[str]) -> set:
        """엔티티 span을 (char_start, char_end, entity_type) 튜플로 추출한다.

        BIO 토큰 수준 태그를 문자 수준 오프셋으로 변환하며,
        공백 전용 토큰(KLUE 음절 토큰화)은 건너뛴다.
        """
        spans = set()
        char_offset = 0
        current_start = None
        current_type = None

        for tok, tag in zip(tokens, tags):
            is_space = tok.strip() == ""
            tok_len = len(tok)

            if tag.startswith("B-"):
                # 이전 span이 있으면 닫는다
                if current_start is not None:
                    spans.add((current_start, char_offset, current_type))
                if not is_space:
                    current_start = char_offset
                    current_type = tag[2:]
                else:
                    current_start = None
                    current_type = None
            elif tag.startswith("I-") and current_type is not None and tag[2:] == current_type:
                # 현재 span을 계속한다 (공백 토큰은 건너뛰되 span을 끊지 않음)
                pass
            else:
                # O 태그 또는 타입 불일치: 현재 span을 닫는다
                if current_start is not None:
                    spans.add((current_start, char_offset, current_type))
                    current_start = None
                    current_type = None

            if not is_space:
                char_offset += tok_len

        # 남아 있는 span을 닫는다
        if current_start is not None:
            spans.add((current_start, char_offset, current_type))

        return spans

    @staticmethod
    def compute_span_f1(
        gold_tags_list: List[List[str]],
        pred_tags_list: List[List[str]],
        gold_tokens_list: List[List[str]],
        pred_tokens_list: List[List[str]],
    ) -> Dict:
        """문자 수준 엔티티 span F1을 계산한다 (KLUE 공식 메트릭).

        BIO 태그 시퀀스에서 엔티티 span을 추출하고, 문자 오프셋으로 변환하여
        (공백 토큰 제외) 정확 매칭 F1을 계산한다.

        Args:
            gold_tags_list: gold BIO 태그 시퀀스 리스트.
            pred_tags_list: 예측 BIO 태그 시퀀스 리스트.
            gold_tokens_list: gold 토큰 시퀀스 리스트.
            pred_tokens_list: 예측 토큰 시퀀스 리스트.

        Returns:
            전체 및 엔티티 타입별 precision/recall/F1이 포함된 dict.
        """
        total_gold = set()
        total_pred = set()
        # 문장 간 충돌을 방지하기 위해 문장 인덱스를 네임스페이스로 사용한다
        for idx, (g_tokens, g_tags, p_tokens, p_tags) in enumerate(
            zip(gold_tokens_list, gold_tags_list, pred_tokens_list, pred_tags_list)
        ):
            min_len_g = min(len(g_tokens), len(g_tags))
            min_len_p = min(len(p_tokens), len(p_tags))
            g_spans = MetricsCalculator._extract_spans(g_tokens[:min_len_g], g_tags[:min_len_g])
            p_spans = MetricsCalculator._extract_spans(p_tokens[:min_len_p], p_tags[:min_len_p])
            # 문장 인덱스로 네임스페이스를 지정한다
            for s in g_spans:
                total_gold.add((idx, *s))
            for s in p_spans:
                total_pred.add((idx, *s))

        correct = total_gold & total_pred
        n_correct = len(correct)
        n_gold = len(total_gold)
        n_pred = len(total_pred)

        precision = n_correct / n_pred if n_pred > 0 else 0.0
        recall = n_correct / n_gold if n_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 엔티티 타입별 분석
        entity_types = set()
        for item in total_gold | total_pred:
            entity_types.add(item[3])  # (idx, char_start, char_end, entity_type)

        per_entity: Dict[str, Dict] = {}
        for etype in sorted(entity_types):
            e_gold = {s for s in total_gold if s[3] == etype}
            e_pred = {s for s in total_pred if s[3] == etype}
            e_correct = len(e_gold & e_pred)
            e_prec = e_correct / len(e_pred) if len(e_pred) > 0 else 0.0
            e_rec = e_correct / len(e_gold) if len(e_gold) > 0 else 0.0
            e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0
            per_entity[etype] = {
                "precision": round(e_prec, 4),
                "recall": round(e_rec, 4),
                "f1": round(e_f1, 4),
                "support": len(e_gold),
            }

        return {
            "overall": {
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
            "per_entity": per_entity,
        }

    @staticmethod
    def compute_span_match(
        gold_spans_list: List[List[dict]],
        pred_spans_list: List[List[dict]],
    ) -> Dict:
        """BIO 정렬 없이 엔티티 span을 직접 비교하는 span 수준 평가.

        gold span과 pred span은 모두 [{"text": "경찰", "type": "OG"}, ...] 형식이다.
        정확 매칭 F1과 완화된(포함 관계) 매칭 F1을 계산한다.

        정확 매칭: 텍스트와 타입이 모두 동일 (공백 정규화 후).
        완화 매칭: 타입이 동일하고 한 텍스트가 다른 것을 포함
                   ("박" vs "박씨", "서울" vs "서울시" 처리).
        """
        from collections import Counter

        def _norm(text: str) -> str:
            """엔티티 텍스트를 정규화한다: 앞뒤 공백 제거 및 내부 공백 제거.

            음절 BIO에서 추출한 gold는 "지난19일"이지만 LLM은 "지난 19일"을 생성한다.
            공백 제거로 비교 가능하게 만든다.
            """
            return text.strip().replace(" ", "")

        # 문장 간 충돌을 방지하기 위해 문장 인덱스로 평탄화한다
        # 원문 텍스트(완화 매칭용)와 정규화 텍스트(정확 매칭용)를 모두 유지한다
        gold_raw = []    # (sent_idx, original_text, type)
        pred_raw = []
        gold_exact = []  # (sent_idx, normalized_text, type)
        pred_exact = []
        for idx, (g_spans, p_spans) in enumerate(zip(gold_spans_list, pred_spans_list)):
            for s in g_spans:
                text, etype = s["text"].strip(), s["type"].strip()
                gold_raw.append((idx, text, etype))
                gold_exact.append((idx, _norm(text), etype))
            for s in p_spans:
                text, etype = s["text"].strip(), s["type"].strip()
                pred_raw.append((idx, text, etype))
                pred_exact.append((idx, _norm(text), etype))

        # --- 정확 매칭 (공백 정규화, 멀티셋) ---
        gold_counter = Counter(gold_exact)
        pred_counter = Counter(pred_exact)
        exact_correct = sum((gold_counter & pred_counter).values())
        n_gold_exact = sum(gold_counter.values())
        n_pred_exact = sum(pred_counter.values())
        exact_prec = exact_correct / n_pred_exact if n_pred_exact > 0 else 0.0
        exact_rec = exact_correct / n_gold_exact if n_gold_exact > 0 else 0.0
        exact_f1 = 2 * exact_prec * exact_rec / (exact_prec + exact_rec) if (exact_prec + exact_rec) > 0 else 0.0

        # --- 완화 매칭 (포함 관계) ---
        # 각 pred span에 대해, 동일한 (sent_idx, type)을 가진 gold span 중
        # 한 텍스트가 다른 것을 포함하는지 확인한다
        gold_by_sent_type: Dict[tuple, List[str]] = {}
        for idx, text, etype in gold_exact:
            key = (idx, etype)
            gold_by_sent_type.setdefault(key, []).append(text)

        pred_by_sent_type: Dict[tuple, List[str]] = {}
        for idx, text, etype in pred_exact:
            key = (idx, etype)
            pred_by_sent_type.setdefault(key, []).append(text)

        relaxed_tp = 0
        matched_gold = set()  # 중복 집계 방지를 위해 매칭된 gold를 추적한다
        matched_pred = set()

        for (idx, etype), p_texts in pred_by_sent_type.items():
            g_texts = gold_by_sent_type.get((idx, etype), [])
            for pi, pt in enumerate(p_texts):
                for gi, gt in enumerate(g_texts):
                    g_key = (idx, etype, gi)
                    p_key = (idx, etype, pi)
                    if g_key in matched_gold or p_key in matched_pred:
                        continue
                    if pt == gt or pt in gt or gt in pt:
                        relaxed_tp += 1
                        matched_gold.add(g_key)
                        matched_pred.add(p_key)
                        break

        n_gold = len(gold_exact)
        n_pred = len(pred_exact)
        relaxed_prec = relaxed_tp / n_pred if n_pred > 0 else 0.0
        relaxed_rec = relaxed_tp / n_gold if n_gold > 0 else 0.0
        relaxed_f1 = 2 * relaxed_prec * relaxed_rec / (relaxed_prec + relaxed_rec) if (relaxed_prec + relaxed_rec) > 0 else 0.0

        # --- 엔티티별 분석 (정확 매칭, 멀티셋) ---
        entity_types = set(e[2] for e in gold_exact) | set(e[2] for e in pred_exact)
        per_entity: Dict[str, Dict] = {}
        for etype in sorted(entity_types):
            e_gold = Counter(s for s in gold_exact if s[2] == etype)
            e_pred = Counter(s for s in pred_exact if s[2] == etype)
            e_correct = sum((e_gold & e_pred).values())
            e_gold_n = sum(e_gold.values())
            e_pred_n = sum(e_pred.values())
            e_prec = e_correct / e_pred_n if e_pred_n > 0 else 0.0
            e_rec = e_correct / e_gold_n if e_gold_n > 0 else 0.0
            e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0
            per_entity[etype] = {
                "precision": round(e_prec, 4),
                "recall": round(e_rec, 4),
                "f1": round(e_f1, 4),
                "support": e_gold_n,
            }

        # --- 엔티티별 분석 (완화 매칭) ---
        per_entity_relaxed: Dict[str, Dict] = {}
        for etype in sorted(entity_types):
            e_gold_count = sum(1 for s in gold_exact if s[2] == etype)
            e_pred_count = sum(1 for s in pred_exact if s[2] == etype)
            e_tp = sum(1 for k in matched_gold if k[1] == etype)
            e_prec = e_tp / e_pred_count if e_pred_count > 0 else 0.0
            e_rec = e_tp / e_gold_count if e_gold_count > 0 else 0.0
            e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0
            per_entity_relaxed[etype] = {
                "precision": round(e_prec, 4),
                "recall": round(e_rec, 4),
                "f1": round(e_f1, 4),
                "support": e_gold_count,
            }

        return {
            "exact": {
                "overall": {
                    "f1": round(exact_f1, 4),
                    "precision": round(exact_prec, 4),
                    "recall": round(exact_rec, 4),
                },
                "per_entity": per_entity,
            },
            "relaxed": {
                "overall": {
                    "f1": round(relaxed_f1, 4),
                    "precision": round(relaxed_prec, 4),
                    "recall": round(relaxed_rec, 4),
                },
                "per_entity": per_entity_relaxed,
            },
            "counts": {
                "gold_spans": n_gold,
                "pred_spans": n_pred,
                "exact_matches": exact_correct,
                "relaxed_matches": relaxed_tp,
            },
        }
