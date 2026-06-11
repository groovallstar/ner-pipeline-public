"""벤치마크 리포트: CLI 테이블 + JSON 출력."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from ner.llm_eval.benchmark_runner import BenchmarkResult


class ReportGenerator:
    """결과에서 벤치마크 리포트를 생성한다."""

    def __init__(self, results: List[BenchmarkResult], lang: str = "ko") -> None:
        self.results = results
        self.lang = lang

    def print_table(self) -> None:
        """벤치마크 결과를 정렬된 CLI 테이블로 출력한다."""
        if not self.results:
            print("No results to display.")
            return

        # 오프셋 span 모드 (ja): span_f1만 채워진다
        if all(not r.metrics.get("span_match") and r.metrics.get("span_f1", {}).get("overall") for r in self.results):
            self._print_offset_span_table()
            return

        has_span_match = any(r.metrics.get("span_match") for r in self.results)

        # --- 주 메트릭: Span 매칭 (LLM 평가) ---
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

            # 엔티티별 span 매칭 분석
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

        # --- 보조 메트릭: Seqeval (음절 BIO, 참조용) ---
        print("\n" + "-" * 70)
        print("  SEQEVAL (Secondary — syllable BIO alignment, for reference)")
        print("-" * 70)

        headers = ["Model", "F1", "Prec", "Recall"]
        headers.extend(["TPS", "Sec/Sample", "Out Tok/s", "Tokens", "Samples", "Errors"])

        rows = []
        for r in self.results:
            overall = r.metrics.get("overall", {})
            row = [
                f"[{r.backend}] {r.model_name}",
                f"{overall.get('f1', 0):.4f}",
                f"{overall.get('precision', 0):.4f}",
                f"{overall.get('recall', 0):.4f}",
            ]
            total_tokens = r.latency.get("total_tokens", 0)
            row.extend([
                f"{r.latency.get('tokens_per_second', 0):.1f}",
                f"{r.latency.get('avg_per_sample', 0):.3f}",
                f"{r.latency.get('output_tokens_per_second', 0):.1f}",
                f"{total_tokens:,}",
                str(r.num_samples),
                str(r.errors),
            ])
            rows.append(row)
        self._print_rows(headers, rows)

    def _print_offset_span_table(self) -> None:
        """일본어 방식의 문자 오프셋 span F1 테이블을 출력한다."""
        title = "JAPANESE NER" if self.lang == "ja" else "NER"
        print("\n" + "=" * 70)
        print(f"  {title} — SPAN-LEVEL EVALUATION (Character Offset)")
        print("=" * 70)
        headers = ["Model", "F1", "Precision", "Recall", "TPS", "Sec/Sample", "Samples", "Errors"]
        rows = []
        for r in self.results:
            overall = r.metrics.get("span_f1", {}).get("overall", {})
            rows.append([
                f"[{r.backend}] {r.model_name}",
                f"{overall.get('f1', 0):.4f}",
                f"{overall.get('precision', 0):.4f}",
                f"{overall.get('recall', 0):.4f}",
                f"{r.latency.get('tokens_per_second', 0):.1f}",
                f"{r.latency.get('avg_per_sample', 0):.3f}",
                str(r.num_samples),
                str(r.errors),
            ])
        self._print_rows(headers, rows)
        for r in self.results:
            per_entity = r.metrics.get("span_f1", {}).get("per_entity", {})
            if per_entity:
                print(f"  [{r.backend}] {r.model_name} — Per-Entity Breakdown:")
                e_headers = ["Entity", "F1", "Precision", "Recall", "Support"]
                e_rows = [
                    [entity,
                     f"{per_entity[entity].get('f1', 0):.4f}",
                     f"{per_entity[entity].get('precision', 0):.4f}",
                     f"{per_entity[entity].get('recall', 0):.4f}",
                     str(per_entity[entity].get('support', 0))]
                    for entity in sorted(per_entity.keys())
                ]
                self._print_rows(e_headers, e_rows, indent=4)

    @staticmethod
    def _print_rows(headers: List[str], rows: List[List[str]], indent: int = 0) -> None:
        """정렬된 테이블 행을 출력한다."""
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
        """결과를 직렬화 가능한 dict로 변환한다."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": self.lang,
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
        """전체 결과를 JSON으로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        # 직렬화할 수 없는 report 문자열을 제거한다
        for r in data["results"]:
            r["metrics"].pop("report", None)
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out}")
