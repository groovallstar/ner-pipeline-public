"""NER 벤치마크를 조율한다: gold 데이터 로드 -> 라벨러 실행 -> 평가."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from labelers.tag_aligner import TagAligner, normalize_tags, extract_spans_from_bio
from metrics.bio_metrics import MetricsCalculator
from metrics.span_metrics import compute_offset_span_f1

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
    """gold 데이터 대비 NER 벤치마크를 실행한다."""

    def __init__(
        self,
        gold_records: List[dict],
        max_samples: Optional[int] = None,
        compute_bertscore: bool = True,
        lang: str = "ko",
        eval_mode: str = "bio",
        sample_concurrency: int = 32,
    ) -> None:
        if max_samples is not None:
            gold_records = gold_records[:max_samples]
        self.gold_records = gold_records
        self.compute_bertscore = compute_bertscore
        self.lang = lang
        self.eval_mode = eval_mode  # "bio" (ko/vi) 또는 "offset_span" (ja)
        self.sample_concurrency = sample_concurrency  # offset_span 경로의 sample 단위 병렬도
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
            if self.eval_mode == "offset_span":
                result = self._run_offset_span(name, backend, labeler)
            else:
                result = self._run_single(name, backend, labeler)
            results.append(result)
        return results

    def _run_offset_span(self, name: str, backend: str, labeler: Any) -> BenchmarkResult:
        """문자 오프셋을 사용하는 Span F1 경로 (ja). 레코드 형식: {text, gold_spans}."""
        return asyncio.run(self._run_offset_span_async(name, backend, labeler))

    async def _run_offset_span_async(
        self, name: str, backend: str, labeler: Any
    ) -> BenchmarkResult:
        """sample 단위 concurrency로 병렬 라벨링한다."""
        from labelers.ja.span_matcher import match_spans

        sem = asyncio.Semaphore(self.sample_concurrency)

        has_async = hasattr(labeler, "alabel_spans")

        async def process_one(idx: int, record: dict):
            async with sem:
                text = record["text"]
                try:
                    if has_async:
                        raw_spans = await labeler.alabel_spans(text)
                    else:
                        raw_spans = await asyncio.to_thread(labeler.label_spans, text)
                    pred_spans = match_spans(text, raw_spans)
                    return idx, record["gold_spans"], pred_spans, None
                except Exception as e:  # noqa: BLE001
                    logger.warning("Labeling failed for sample %d: %s", idx, e)
                    return idx, None, None, e

        t_start = time.time()
        tasks = [process_one(i, r) for i, r in enumerate(self.gold_records)]

        results: List[tuple] = [None] * len(tasks)  # type: ignore[list-item]
        errors = 0
        pbar = tqdm(total=len(tasks), desc=f"[{backend}] {name}", unit="sample")
        for coro in asyncio.as_completed(tasks):
            idx, gold, pred, err = await coro
            if err is not None:
                errors += 1
            else:
                results[idx] = (gold, pred)
            pbar.update(1)
            pbar.set_postfix(errors=errors)
        pbar.close()
        t_total = time.time() - t_start

        gold_spans_all: List[List[dict]] = [r[0] for r in results if r is not None]
        pred_spans_all: List[List[dict]] = [r[1] for r in results if r is not None]

        span_f1 = compute_offset_span_f1(gold_spans_all, pred_spans_all)
        metrics = {"span_f1": span_f1}
        num_evaluated = len(gold_spans_all)

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
            model_name=name, backend=backend, metrics=metrics,
            latency=latency, num_samples=num_evaluated, errors=errors,
        )

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
            # 원문 문장이 있으면 사용한다 (단어 경계 보존)
            sentence = record.get("sentence")
            if sentence:
                text = re.sub(r'<([^:>]+):[A-Z]+>', r'\1', sentence)
            else:
                text = TagAligner.reconstruct_text(gold_tokens)

            try:
                has_space_tokens = any(t.strip() == "" for t in gold_tokens)

                # BIO 태그에서 gold span을 추출한다 (span 수준 평가용)
                g_spans = extract_spans_from_bio(gold_tokens, gold_tags, lang=self.lang)

                if has_space_tokens and hasattr(labeler, "label_spans"):
                    # LLM 라벨러: raw span을 직접 가져온다 (주 경로)
                    p_spans = labeler.label_spans(text)
                    # seqeval용 음절 BIO도 생성한다 (보조 메트릭)
                    aligned_pred = TagAligner.spans_to_syllable_bio(text, gold_tokens, p_spans, lang=self.lang)
                    pred_tokens = gold_tokens
                    aligned_gold = normalize_tags(gold_tags, lang=self.lang)
                elif has_space_tokens and hasattr(labeler, "label_syllables"):
                    # HF 모델: 문자 오프셋을 통한 직접 음절 정렬
                    pred_tags = labeler.label_syllables(text, gold_tokens)
                    pred_tokens = gold_tokens
                    aligned_gold = normalize_tags(gold_tags, lang=self.lang)
                    aligned_pred = pred_tags
                    # span 수준 평가를 위해 정렬된 BIO에서 pred span을 추출한다
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
            pbar.set_postfix(errors=errors)

        pbar.close()
        t_total = time.time() - t_start

        # 주 메트릭: span 수준 매칭 (음절 정렬을 완전히 우회)
        span_match = MetricsCalculator.compute_span_match(gold_spans_all, pred_spans_all)

        # 보조 메트릭: 음절 BIO에 대한 seqeval (참조용 / BERT 비교용)
        seqeval_metrics = MetricsCalculator.compute_seqeval(gold_tags_all, pred_tags_all)

        # 문자 수준 span F1 (BIO 기반, KLUE 공식 메트릭)
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

        # 라벨러가 토큰 사용량을 추적하는 경우 수집한다
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
