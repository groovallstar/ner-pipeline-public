"""Benchmark report: CLI table + JSON output."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from evaluators.benchmark_runner import BenchmarkResult


class ReportGenerator:
    """Generate benchmark reports from results."""

    def __init__(self, results: List[BenchmarkResult]) -> None:
        self.results = results

    def print_table(self) -> None:
        """Print a formatted CLI table of benchmark results."""
        if not self.results:
            print("No results to display.")
            return

        has_bertscore = any(r.metrics.get("bertscore") for r in self.results)
        has_span_match = any(r.metrics.get("span_match") for r in self.results)

        # --- Primary: Span Match (LLM evaluation) ---
        if has_span_match:
            print("\n" + "=" * 70)
            print("  SPAN-LEVEL EVALUATION (Primary — no syllable alignment)")
            print("=" * 70)

            headers = ["Model", "Exact-F1", "Relaxed-F1", "E-Prec", "E-Rec", "R-Prec", "R-Rec", "Gold", "Pred"]
            rows = []
            for r in self.results:
                sm = r.metrics.get("span_match", {})
                exact = sm.get("exact", {}).get("overall", {})
                relaxed = sm.get("relaxed", {}).get("overall", {})
                counts = sm.get("counts", {})
                rows.append([
                    f"[{r.backend}] {r.model_name}",
                    f"{exact.get('f1', 0):.4f}",
                    f"{relaxed.get('f1', 0):.4f}",
                    f"{exact.get('precision', 0):.4f}",
                    f"{exact.get('recall', 0):.4f}",
                    f"{relaxed.get('precision', 0):.4f}",
                    f"{relaxed.get('recall', 0):.4f}",
                    str(counts.get("gold_spans", 0)),
                    str(counts.get("pred_spans", 0)),
                ])
            self._print_rows(headers, rows)

            # Per-entity span match breakdown
            for r in self.results:
                sm = r.metrics.get("span_match", {})
                exact_pe = sm.get("exact", {}).get("per_entity", {})
                relaxed_pe = sm.get("relaxed", {}).get("per_entity", {})
                if exact_pe:
                    print(f"  [{r.backend}] {r.model_name} — Span Match by Entity:")
                    e_headers = ["Entity", "Exact-F1", "Relaxed-F1", "E-Prec", "E-Rec", "Support"]
                    e_rows = []
                    for entity in sorted(exact_pe.keys()):
                        ev = exact_pe[entity]
                        rv = relaxed_pe.get(entity, {})
                        e_rows.append([
                            entity,
                            f"{ev.get('f1', 0):.4f}",
                            f"{rv.get('f1', 0):.4f}",
                            f"{ev.get('precision', 0):.4f}",
                            f"{ev.get('recall', 0):.4f}",
                            str(ev.get("support", 0)),
                        ])
                    self._print_rows(e_headers, e_rows, indent=4)

        # --- Secondary: Seqeval (syllable BIO, for reference) ---
        print("\n" + "-" * 70)
        print("  SEQEVAL (Secondary — syllable BIO alignment, for reference)")
        print("-" * 70)

        headers = ["Model", "F1", "Prec", "Recall"]
        if has_bertscore:
            headers.append("BERT-F1")
        headers.extend(["Speed (s/s)", "Tok/s", "Out Tok/s", "Tokens", "Samples", "Errors"])

        rows = []
        for r in self.results:
            overall = r.metrics.get("overall", {})
            row = [
                f"[{r.backend}] {r.model_name}",
                f"{overall.get('f1', 0):.4f}",
                f"{overall.get('precision', 0):.4f}",
                f"{overall.get('recall', 0):.4f}",
            ]
            if has_bertscore:
                bs = r.metrics.get("bertscore", {})
                row.append(f"{bs.get('f1', 0):.4f}")
            total_tokens = r.latency.get("total_tokens", 0)
            row.extend([
                f"{r.latency.get('samples_per_second', 0):.2f}",
                f"{r.latency.get('tokens_per_second', 0):.1f}",
                f"{r.latency.get('output_tokens_per_second', 0):.1f}",
                f"{total_tokens:,}",
                str(r.num_samples),
                str(r.errors),
            ])
            rows.append(row)
        self._print_rows(headers, rows)

    @staticmethod
    def _print_rows(headers: List[str], rows: List[List[str]], indent: int = 0) -> None:
        """Print aligned table rows."""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        prefix = " " * indent

        def fmt_row(cells: List[str]) -> str:
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

    def to_dict(self) -> Dict:
        """Convert results to a serializable dict."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    def save_json(self, path: str) -> None:
        """Save full results as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        # Remove non-serializable report string
        for r in data["results"]:
            r["metrics"].pop("report", None)
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out}")
