"""Collect Korean NER error diffs for prompt improvement analysis."""

import json
import sys

sys.path.insert(0, "src")

from labelers.bio_dataset import load
from labelers.ko.vllm_ner_labeler import VllmNERLabeler
from llm_eval.span_evaluator import evaluate

MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "results/ko_diffs_35b_200.json"

gold = load("klue", "validation", max_samples=MAX_SAMPLES)
print(f"Loaded {len(gold)} gold records")

labeler = VllmNERLabeler(
    base_url="http://localhost:8081/v1",
    model="Qwen/Qwen3.5-35B-A3B",
    concurrency=32,
)

result = evaluate(gold, labeler, collect_diffs=True, show_progress=True)

# Save full result
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Summary
diffs = result["diffs"]
total_fn = sum(len(d["false_negatives"]) for d in diffs)
total_fp = sum(len(d["false_positives"]) for d in diffs)
total_be = sum(len(d["boundary_errors"]) for d in diffs)

print(f"\n=== Summary ===")
print(f"Exact F1: {result['exact']['overall']['f1']:.4f}")
print(f"Relaxed F1: {result['relaxed']['overall']['f1']:.4f}")
print(f"Errors: {result['counts']['errors']}")
print(f"False Negatives (missed): {total_fn}")
print(f"False Positives (spurious): {total_fp}")
print(f"Boundary Errors: {total_be}")

# Per-entity exact F1
print(f"\n=== Per-entity Exact F1 ===")
for etype, metrics in sorted(result["exact"]["per_entity"].items()):
    print(f"  {etype}: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} (n={metrics['support']})")
