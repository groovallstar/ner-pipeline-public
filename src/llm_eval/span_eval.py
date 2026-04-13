"""라벨링 결과 JSONL 파일을 위한 단독 실행 span 수준 평가기.

2단계 파이프라인의 2단계: 라벨러가 생성한 예측 JSONL을 소비하고,
문자 오프셋 span F1 + 지연 집계를 계산하여 비교 테이블을 출력한다.

입력 JSONL (레코드 한 줄):
    {"id": "...",
     "text": "...",
     "gold_spans": [{"start": 0, "end": 3, "label": "人名"}, ...],
     "pred_spans": [{"start": 0, "end": 3, "label": "人名"}, ...],
     "latency_seconds": 1.23,             # 선택 사항
     "prompt_tokens": 245,                # 선택 사항
     "completion_tokens": 18}             # 선택 사항

선택적 첫 줄 `_meta` 레코드 (평가에서 제외):
    {"_meta": {"model": "Qwen3.5-27B", "backend": "vllm", "lang": "ja"}}

사용 예:
    python -m llm_eval.span_eval --predictions f1.jsonl f2.jsonl [--output eval.json]
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from metrics.span_metrics import compute_offset_span_f1


@dataclass
class EvalResult:
    name: str
    backend: str
    lang: str
    metrics: Dict = field(default_factory=dict)
    latency: Dict = field(default_factory=dict)
    num_samples: int = 0


def _normalize_span(s: dict) -> dict:
    """{label}과 {type} 키를 모두 허용하며 start/end를 그대로 전달한다."""
    label = s.get("label", s.get("type"))
    return {"start": int(s["start"]), "end": int(s["end"]), "type": label}


def _load_jsonl(path: str) -> tuple:
    """(meta_dict_or_empty, records_list)를 반환한다."""
    meta = {}
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj and not records:
                meta = obj["_meta"]
                continue
            records.append(obj)
    return meta, records


def evaluate_file(path: str) -> EvalResult:
    meta, records = _load_jsonl(path)

    name = meta.get("model") or os.path.splitext(os.path.basename(path))[0]
    backend = meta.get("backend", "?")
    lang = meta.get("lang", "?")

    gold_spans = [[_normalize_span(s) for s in r.get("gold_spans", [])] for r in records]
    pred_spans = [[_normalize_span(s) for s in r.get("pred_spans", [])] for r in records]

    metrics = compute_offset_span_f1(gold_spans, pred_spans)

    total_seconds = sum(float(r.get("latency_seconds", 0) or 0) for r in records)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0) or 0) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens", 0) or 0) for r in records)
    total_tokens = prompt_tokens + completion_tokens
    n = len(records)

    latency = {
        "total_seconds": round(total_seconds, 3),
        "avg_per_sample": round(total_seconds / n, 4) if n else 0.0,
        "tokens_per_second": round(total_tokens / total_seconds, 2) if total_seconds > 0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return EvalResult(
        name=name, backend=backend, lang=lang,
        metrics=metrics, latency=latency, num_samples=n,
    )


def _print_table(results: List[EvalResult]) -> None:
    print("=" * 90)
    print("  SPAN-LEVEL EVALUATION (Character Offset)")
    print("=" * 90)
    headers = ["Model", "F1", "Precision", "Recall", "TPS", "Sec/Sample", "Samples"]
    rows = []
    for r in results:
        ov = r.metrics["overall"]
        rows.append([
            f"[{r.backend}] {r.name}",
            f"{ov['f1']:.4f}", f"{ov['precision']:.4f}", f"{ov['recall']:.4f}",
            f"{r.latency['tokens_per_second']:.1f}",
            f"{r.latency['avg_per_sample']:.3f}",
            str(r.num_samples),
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))

    for r in results:
        print(f"\n  [{r.backend}] {r.name} — Per-Entity Breakdown:")
        sub = [["Entity", "F1", "Precision", "Recall", "Support"]]
        for etype, m in sorted(r.metrics.get("per_entity", {}).items()):
            sub.append([etype, f"{m['f1']:.4f}", f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}", str(m["support"])])
        w = [max(len(row[i]) for row in sub) for i in range(5)]
        for i, row in enumerate(sub):
            print("    " + " | ".join(c.ljust(w[j]) for j, c in enumerate(row)))
            if i == 0:
                print("    " + "-+-".join("-" * x for x in w))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llm_eval.span_eval")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more JSONL prediction files")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = [evaluate_file(p) for p in args.predictions]
    _print_table(results)

    if args.output:
        payload = {
            "results": [
                {
                    "model": r.name, "backend": r.backend, "lang": r.lang,
                    "num_samples": r.num_samples,
                    "metrics": r.metrics, "latency": r.latency,
                }
                for r in results
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
