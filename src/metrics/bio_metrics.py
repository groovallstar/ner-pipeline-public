"""NER evaluation metrics: seqeval F1/Precision/Recall + BERTScore."""

import logging
from typing import Dict, List, Optional

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Compute NER evaluation metrics."""

    @staticmethod
    def compute_seqeval(
        gold_tags_list: List[List[str]],
        pred_tags_list: List[List[str]],
    ) -> Dict:
        """Compute seqeval metrics (strict IOB2 mode).

        Args:
            gold_tags_list: List of gold BIO tag sequences.
            pred_tags_list: List of predicted BIO tag sequences.

        Returns:
            Dict with overall and per-entity-type metrics.
        """
        # Ensure same length sequences (truncate/pad if needed)
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

        # Per-entity breakdown from classification_report
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
    def extract_entities(tokens: List[str], tags: List[str]) -> List[str]:
        """Extract entity strings from BIO-tagged token sequence."""
        entities = []
        current = []
        for tok, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current:
                    entities.append(" ".join(current))
                current = [tok]
            elif tag.startswith("I-") and current:
                current.append(tok)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []
        if current:
            entities.append(" ".join(current))
        return entities

    @staticmethod
    def _extract_spans(tokens: List[str], tags: List[str]) -> set:
        """Extract entity spans as (char_start, char_end, entity_type) tuples.

        Converts BIO token-level tags to character-level offsets,
        skipping whitespace-only tokens (KLUE syllable tokenization).
        """
        spans = set()
        char_offset = 0
        current_start = None
        current_type = None

        for tok, tag in zip(tokens, tags):
            is_space = tok.strip() == ""
            tok_len = len(tok)

            if tag.startswith("B-"):
                # Close previous span if any
                if current_start is not None:
                    spans.add((current_start, char_offset, current_type))
                if not is_space:
                    current_start = char_offset
                    current_type = tag[2:]
                else:
                    current_start = None
                    current_type = None
            elif tag.startswith("I-") and current_type is not None and tag[2:] == current_type:
                # Continue current span (skip space tokens but don't break span)
                pass
            else:
                # O tag or type mismatch: close current span
                if current_start is not None:
                    spans.add((current_start, char_offset, current_type))
                    current_start = None
                    current_type = None

            if not is_space:
                char_offset += tok_len

        # Close any remaining span
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
        """Compute character-level entity span F1 (KLUE official metric).

        Extracts entity spans from BIO tag sequences, converts to character
        offsets (skipping whitespace tokens), and computes exact-match F1.

        Args:
            gold_tags_list: List of gold BIO tag sequences.
            pred_tags_list: List of predicted BIO tag sequences.
            gold_tokens_list: List of gold token sequences.
            pred_tokens_list: List of predicted token sequences.

        Returns:
            Dict with overall and per-entity-type precision/recall/F1.
        """
        total_gold = set()
        total_pred = set()
        # Use sentence index as namespace to avoid cross-sentence collisions
        for idx, (g_tokens, g_tags, p_tokens, p_tags) in enumerate(
            zip(gold_tokens_list, gold_tags_list, pred_tokens_list, pred_tags_list)
        ):
            min_len_g = min(len(g_tokens), len(g_tags))
            min_len_p = min(len(p_tokens), len(p_tags))
            g_spans = MetricsCalculator._extract_spans(g_tokens[:min_len_g], g_tags[:min_len_g])
            p_spans = MetricsCalculator._extract_spans(p_tokens[:min_len_p], p_tags[:min_len_p])
            # Namespace with sentence index
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

        # Per-entity type breakdown
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
        """Span-level evaluation: compare entity spans directly without BIO alignment.

        Gold spans and pred spans are both [{"text": "경찰", "type": "OG"}, ...].
        Computes exact match F1 and relaxed (containment) match F1.

        Exact match: text and type both identical (after space normalization).
        Relaxed match: type identical AND one text contains the other
                       (handles "박" vs "박씨", "서울" vs "서울시").
        """
        from collections import Counter

        def _norm(text: str) -> str:
            """Normalize entity text: strip and remove internal spaces.

            Gold (from syllable BIO) produces "지난19일" while LLM produces
            "지난 19일". Removing spaces makes them comparable.
            """
            return text.strip().replace(" ", "")

        # Flatten with sentence index to avoid cross-sentence collisions
        # Keep both original text (for relaxed match) and normalized (for exact match)
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

        # --- Exact Match (space-normalized, multiset) ---
        gold_counter = Counter(gold_exact)
        pred_counter = Counter(pred_exact)
        exact_correct = sum((gold_counter & pred_counter).values())
        n_gold_exact = sum(gold_counter.values())
        n_pred_exact = sum(pred_counter.values())
        exact_prec = exact_correct / n_pred_exact if n_pred_exact > 0 else 0.0
        exact_rec = exact_correct / n_gold_exact if n_gold_exact > 0 else 0.0
        exact_f1 = 2 * exact_prec * exact_rec / (exact_prec + exact_rec) if (exact_prec + exact_rec) > 0 else 0.0

        # --- Relaxed Match (containment) ---
        # For each pred span, check if any gold span has same (sent_idx, type)
        # and one text contains the other
        gold_by_sent_type: Dict[tuple, List[str]] = {}
        for idx, text, etype in gold_exact:
            key = (idx, etype)
            gold_by_sent_type.setdefault(key, []).append(text)

        pred_by_sent_type: Dict[tuple, List[str]] = {}
        for idx, text, etype in pred_exact:
            key = (idx, etype)
            pred_by_sent_type.setdefault(key, []).append(text)

        relaxed_tp = 0
        matched_gold = set()  # track matched gold to avoid double-counting
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

        # --- Per-entity breakdown (exact, multiset) ---
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

        # --- Per-entity breakdown (relaxed) ---
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

    @staticmethod
    def compute_bertscore(
        gold_tags_list: List[List[str]],
        pred_tags_list: List[List[str]],
        gold_tokens_list: List[List[str]],
        pred_tokens_list: List[List[str]],
        model_type: str = "klue/roberta-base",
    ) -> Dict:
        """Compute BERTScore between gold and predicted entity strings.

        Extracts entity spans from BIO tags, then computes BERTScore
        on the entity text lists.
        """
        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            logger.warning("bert-score not installed, skipping BERTScore")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        gold_entities_all = []
        pred_entities_all = []
        for g_tokens, g_tags, p_tokens, p_tags in zip(
            gold_tokens_list, gold_tags_list, pred_tokens_list, pred_tags_list
        ):
            gold_entities_all.extend(MetricsCalculator.extract_entities(g_tokens, g_tags))
            pred_entities_all.extend(MetricsCalculator.extract_entities(p_tokens, p_tags))

        if not gold_entities_all or not pred_entities_all:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Pad shorter list to same length for BERTScore
        max_len = max(len(gold_entities_all), len(pred_entities_all))
        gold_padded = gold_entities_all + [""] * (max_len - len(gold_entities_all))
        pred_padded = pred_entities_all + [""] * (max_len - len(pred_entities_all))

        P, R, F1 = bert_score_fn(
            pred_padded, gold_padded,
            model_type=model_type,
            lang="ko",
            verbose=False,
        )

        return {
            "precision": round(float(P.mean()), 4),
            "recall": round(float(R.mean()), 4),
            "f1": round(float(F1.mean()), 4),
        }
