"""Orchestrates NER benchmark: load gold data -> run labelers -> evaluate."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from labelers.tag_aligner import TagAligner, normalize_tags, extract_spans_from_bio
from evaluators.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    backend: str  # "ollama" | "vllm" | "openai" | "hf"
    metrics: Dict = field(default_factory=dict)
    latency: Dict = field(default_factory=dict)
    num_samples: int = 0
    errors: int = 0


class BenchmarkRunner:
    """Run NER benchmark against gold data."""

    def __init__(
        self,
        gold_records: List[dict],
        max_samples: Optional[int] = None,
        compute_bertscore: bool = True,
        lang: str = "ko",
    ) -> None:
        if max_samples is not None:
            gold_records = gold_records[:max_samples]
        self.gold_records = gold_records
        self.compute_bertscore = compute_bertscore
        self.lang = lang
        self._labelers: List[tuple] = []  # (name, backend, labeler)

    def add_labeler(self, name: str, backend: str, labeler: Any) -> None:
        self._labelers.append((name, backend, labeler))

    def run(self) -> List[BenchmarkResult]:
        results = []
        for name, backend, labeler in self._labelers:
            print(f"\n{'='*60}")
            print(f"Benchmarking: [{backend}] {name}")
            print(f"  Samples: {len(self.gold_records)}")
            print(f"{'='*60}")
            result = self._run_single(name, backend, labeler)
            results.append(result)
        return results

    def _run_single(self, name: str, backend: str, labeler: Any) -> BenchmarkResult:
        gold_tags_all = []
        pred_tags_all = []
        gold_tokens_all = []
        pred_tokens_all = []
        gold_spans_all = []  # [{"text": "경찰", "type": "OG"}, ...]
        pred_spans_all = []
        errors = 0

        t_start = time.time()
        pbar = tqdm(self.gold_records, desc=f"[{backend}] {name}", unit="sample")

        for i, record in enumerate(pbar):
            gold_tokens = record["tokens"]
            gold_tags = record["ner_tags"]
            # Use original sentence if available (preserves word boundaries)
            sentence = record.get("sentence")
            if sentence:
                text = re.sub(r'<([^:>]+):[A-Z]+>', r'\1', sentence)
            else:
                text = TagAligner.reconstruct_text(gold_tokens)

            try:
                has_space_tokens = any(t.strip() == "" for t in gold_tokens)

                # Extract gold spans from BIO tags (for span-level evaluation)
                g_spans = extract_spans_from_bio(gold_tokens, gold_tags, lang=self.lang)

                if has_space_tokens and hasattr(labeler, "label_spans"):
                    # LLM labelers: get raw spans directly (primary path)
                    p_spans = labeler.label_spans(text)
                    # Also produce syllable BIO for seqeval (secondary metric)
                    aligned_pred = TagAligner.spans_to_syllable_bio(text, gold_tokens, p_spans, lang=self.lang)
                    pred_tokens = gold_tokens
                    aligned_gold = normalize_tags(gold_tags, lang=self.lang)
                elif has_space_tokens and hasattr(labeler, "label_syllables"):
                    # HF models: direct syllable alignment via char offsets
                    pred_tags = labeler.label_syllables(text, gold_tokens)
                    pred_tokens = gold_tokens
                    aligned_gold = normalize_tags(gold_tags, lang=self.lang)
                    aligned_pred = pred_tags
                    # Extract pred spans from aligned BIO for span-level eval
                    p_spans = extract_spans_from_bio(gold_tokens, aligned_pred, lang=self.lang)
                else:
                    pred_records = labeler.label(text)
                    if not pred_records:
                        errors += 1
                        continue
                    pred_tokens = []
                    pred_tags = []
                    for pr in pred_records:
                        pred_tokens.extend(pr.get("tokens", []))
                        pred_tags.extend(pr.get("ner_tags", []))
                    aligned_gold, aligned_pred = TagAligner.align(
                        gold_tokens, gold_tags, pred_tokens, pred_tags, lang=self.lang
                    )
                    p_spans = extract_spans_from_bio(pred_tokens, aligned_pred, lang=self.lang)
            except Exception as e:
                logger.warning("Labeling failed for sample %d: %s", i, e)
                errors += 1
                continue

            gold_tags_all.append(aligned_gold)
            pred_tags_all.append(aligned_pred)
            gold_tokens_all.append(gold_tokens)
            pred_tokens_all.append(pred_tokens)
            gold_spans_all.append(g_spans)
            pred_spans_all.append(p_spans)

            # Update progress bar with running stats
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(speed=f"{speed:.1f}s/s", errors=errors)

        pbar.close()
        t_total = time.time() - t_start

        # Primary metric: Span-level match (bypasses syllable alignment entirely)
        span_match = MetricsCalculator.compute_span_match(gold_spans_all, pred_spans_all)

        # Secondary metric: seqeval on syllable BIO (for reference / BERT comparison)
        seqeval_metrics = MetricsCalculator.compute_seqeval(gold_tags_all, pred_tags_all)

        # Character-level span F1 (from BIO, KLUE official metric)
        char_span_f1 = MetricsCalculator.compute_span_f1(
            gold_tags_all, pred_tags_all, gold_tokens_all, pred_tokens_all
        )

        metrics = {
            "span_match": span_match,
            "overall": seqeval_metrics.get("overall", {}),
            "per_entity": seqeval_metrics.get("per_entity", {}),
            "report": seqeval_metrics.get("report", ""),
            "span_f1": char_span_f1,
        }

        if self.compute_bertscore:
            print(f"  Computing BERTScore...")
            bertscore = MetricsCalculator.compute_bertscore(
                gold_tags_all, pred_tags_all, gold_tokens_all, pred_tokens_all
            )
            metrics["bertscore"] = bertscore

        num_evaluated = len(gold_tags_all)

        # Collect token usage if the labeler tracks it
        prompt_tokens = getattr(labeler, "total_prompt_tokens", 0)
        completion_tokens = getattr(labeler, "total_completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        latency = {
            "total_seconds": round(t_total, 2),
            "samples_per_second": round(num_evaluated / t_total, 4) if t_total > 0 else 0,
            "avg_per_sample": round(t_total / num_evaluated, 4) if num_evaluated > 0 else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / t_total, 2) if t_total > 0 else 0,
            "output_tokens_per_second": round(completion_tokens / t_total, 2) if t_total > 0 else 0,
        }

        return BenchmarkResult(
            model_name=name,
            backend=backend,
            metrics=metrics,
            latency=latency,
            num_samples=num_evaluated,
            errors=errors,
        )
