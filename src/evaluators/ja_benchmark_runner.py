"""Japanese NER benchmark runner using span-based evaluation.

No seqeval, no TagAligner, no BIO conversion.
Evaluates LLM output against Stockmark NER gold spans using character offsets.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from labelers.ja.span_matcher import match_spans
from evaluators.ja_metrics import compute_offset_span_f1

logger = logging.getLogger(__name__)


@dataclass
class JaBenchmarkResult:
    model_name: str
    backend: str
    metrics: Dict = field(default_factory=dict)
    latency: Dict = field(default_factory=dict)
    num_samples: int = 0
    errors: int = 0


class JaBenchmarkRunner:
    """Run Japanese NER benchmark against Stockmark gold data."""

    def __init__(
        self,
        gold_records: List[dict],
        max_samples: Optional[int] = None,
    ) -> None:
        if max_samples is not None:
            gold_records = gold_records[:max_samples]
        self.gold_records = gold_records
        self._labelers: List[tuple] = []

    def add_labeler(self, name: str, backend: str, labeler: Any) -> None:
        self._labelers.append((name, backend, labeler))

    def run(self) -> List[JaBenchmarkResult]:
        results = []
        for name, backend, labeler in self._labelers:
            print(f"\n{'='*60}")
            print(f"Benchmarking: [{backend}] {name}")
            print(f"  Samples: {len(self.gold_records)}")
            print(f"{'='*60}")
            result = self._run_single(name, backend, labeler)
            results.append(result)
        return results

    def _run_single(self, name: str, backend: str, labeler: Any) -> JaBenchmarkResult:
        gold_spans_all: List[List[dict]] = []
        pred_spans_all: List[List[dict]] = []
        errors = 0

        t_start = time.time()
        pbar = tqdm(self.gold_records, desc=f"[{backend}] {name}", unit="sample")

        for i, record in enumerate(pbar):
            text = record["text"]
            gold_spans = record["gold_spans"]

            try:
                # Get raw entity spans from LLM
                raw_spans = labeler.label_spans(text)
                # Convert text spans to character offset spans
                pred_spans = match_spans(text, raw_spans)
            except Exception as e:
                logger.warning("Labeling failed for sample %d: %s", i, e)
                errors += 1
                continue

            gold_spans_all.append(gold_spans)
            pred_spans_all.append(pred_spans)

            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(speed=f"{speed:.1f}s/s", errors=errors)

        pbar.close()
        t_total = time.time() - t_start

        # Compute span F1 (overall + per-entity)
        span_f1 = compute_offset_span_f1(gold_spans_all, pred_spans_all)

        metrics = {
            "span_f1": span_f1,
        }

        num_evaluated = len(gold_spans_all)

        # Collect token usage if available
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

        return JaBenchmarkResult(
            model_name=name,
            backend=backend,
            metrics=metrics,
            latency=latency,
            num_samples=num_evaluated,
            errors=errors,
        )


class JaReportGenerator:
    """Generate Japanese NER benchmark reports."""

    def __init__(self, results: List[JaBenchmarkResult]) -> None:
        self.results = results

    def print_table(self) -> None:
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 70)
        print("  JAPANESE NER — SPAN-LEVEL EVALUATION (Character Offset)")
        print("=" * 70)

        # Overall table
        headers = ["Model", "F1", "Precision", "Recall", "Speed (s/s)", "Tok/s", "Samples", "Errors"]
        rows = []
        for r in self.results:
            overall = r.metrics.get("span_f1", {}).get("overall", {})
            rows.append([
                f"[{r.backend}] {r.model_name}",
                f"{overall.get('f1', 0):.4f}",
                f"{overall.get('precision', 0):.4f}",
                f"{overall.get('recall', 0):.4f}",
                f"{r.latency.get('samples_per_second', 0):.2f}",
                f"{r.latency.get('tokens_per_second', 0):.1f}",
                str(r.num_samples),
                str(r.errors),
            ])
        self._print_rows(headers, rows)

        # Per-entity breakdown
        for r in self.results:
            per_entity = r.metrics.get("span_f1", {}).get("per_entity", {})
            if per_entity:
                print(f"  [{r.backend}] {r.model_name} — Per-Entity Breakdown:")
                e_headers = ["Entity", "F1", "Precision", "Recall", "Support"]
                e_rows = []
                for entity in sorted(per_entity.keys()):
                    ev = per_entity[entity]
                    e_rows.append([
                        entity,
                        f"{ev.get('f1', 0):.4f}",
                        f"{ev.get('precision', 0):.4f}",
                        f"{ev.get('recall', 0):.4f}",
                        str(ev.get("support", 0)),
                    ])
                self._print_rows(e_headers, e_rows, indent=4)

    @staticmethod
    def _print_rows(headers, rows, indent=0):
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        prefix = " " * indent

        def fmt_row(cells):
            parts = []
            for i, cell in enumerate(cells):
                if i == 0:
                    parts.append(cell.ljust(col_widths[i]))
                else:
                    parts.append(cell.rjust(col_widths[i]))
            return prefix + " | ".join(parts)

        sep = prefix + "-+-".join("-" * w for w in col_widths)
        print(fmt_row(headers))
        print(sep)
        for row in rows:
            print(fmt_row(row))
        print()

    def to_dict(self):
        from datetime import datetime, timezone
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": "ja",
            "num_models": len(self.results),
            "results": [
                {
                    "model_name": r.model_name,
                    "backend": r.backend,
                    "metrics": r.metrics,
                    "latency": r.latency,
                    "num_samples": r.num_samples,
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }

    def save_json(self, path: str):
        import json
        from pathlib import Path
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out}")
