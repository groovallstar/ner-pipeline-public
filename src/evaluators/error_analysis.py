"""Per-sentence error analysis: extract FN/FP patterns from NER benchmark.

Usage:
    python -m evaluators.error_analysis \
        --models vllm:Qwen/Qwen3.5-27B \
        --max-samples 50 \
        --output results/error_analysis.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from labelers.dataset_loader import DatasetLoader
from evaluators.tag_aligner import TagAligner, normalize_tags, extract_spans_from_bio

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    """Space-normalize for comparison."""
    return text.strip().replace(" ", "")


def _classify_error(gold_text: str, pred_text: str, gold_type: str, pred_type: str) -> str:
    """Classify an error pattern."""
    gn, pn = _norm(gold_text), _norm(pred_text)
    if gn == pn and gold_type != pred_type:
        return "TYPE_MISMATCH"
    if gold_type == pred_type:
        if pn in gn:
            return "BOUNDARY_SUBSET"  # pred is subset of gold
        if gn in pn:
            return "BOUNDARY_SUPERSET"  # pred is superset of gold
    return "UNMATCHED"


def analyze_sentence(
    sent_idx: int,
    text: str,
    gold_spans: List[dict],
    pred_spans: List[dict],
) -> dict:
    """Analyze a single sentence's gold vs predicted spans.

    Returns dict with FN, FP, exact matches, and error classifications.
    """
    # Normalize for comparison
    gold_items = [(s["text"].strip(), s["type"].strip()) for s in gold_spans]
    pred_items = [(s["text"].strip(), s["type"].strip()) for s in pred_spans]

    gold_normed = [(_norm(t), ty) for t, ty in gold_items]
    pred_normed = [(_norm(t), ty) for t, ty in pred_items]

    # Exact match (space-normalized, multiset)
    gold_counter = Counter(gold_normed)
    pred_counter = Counter(pred_normed)
    exact_matches = list((gold_counter & pred_counter).elements())
    fn_counter = gold_counter - pred_counter  # in gold but not pred
    fp_counter = pred_counter - gold_counter  # in pred but not gold

    fn_items = list(fn_counter.elements())
    fp_items = list(fp_counter.elements())

    # Map back to original text
    gold_orig = {(_norm(t), ty): t for t, ty in gold_items}
    pred_orig = {(_norm(t), ty): t for t, ty in pred_items}

    # Classify errors
    fn_classified = []
    fp_classified = []
    matched_fp = set()

    for fn_norm_text, fn_type in fn_items:
        fn_orig = gold_orig.get((fn_norm_text, fn_type), fn_norm_text)
        best_match = None
        best_class = "MISS"  # default: completely missed

        # Try to find a related FP (same type, partial overlap)
        for j, (fp_norm_text, fp_type) in enumerate(fp_items):
            if j in matched_fp:
                continue
            cls = _classify_error(fn_norm_text, fp_norm_text, fn_type, fp_type)
            if cls != "UNMATCHED":
                fp_orig = pred_orig.get((fp_norm_text, fp_type), fp_norm_text)
                best_match = {"text": fp_orig, "type": fp_type, "normed": fp_norm_text}
                best_class = cls
                matched_fp.add(j)
                break

        entry = {
            "gold_text": fn_orig,
            "gold_type": fn_type,
            "normed": fn_norm_text,
            "error_class": best_class,
        }
        if best_match:
            entry["matched_pred"] = best_match
        fn_classified.append(entry)

    for j, (fp_norm_text, fp_type) in enumerate(fp_items):
        if j in matched_fp:
            continue
        fp_orig = pred_orig.get((fp_norm_text, fp_type), fp_norm_text)
        # Check if it's a type mismatch with any gold
        error_class = "HALLUCINATION"  # default: entity not in gold at all
        matched_gold_entry = None
        for gn_text, g_type in gold_normed:
            if gn_text == fp_norm_text and g_type != fp_type:
                error_class = "TYPE_MISMATCH"
                matched_gold_entry = {"text": gold_orig.get((gn_text, g_type), gn_text), "type": g_type}
                break
            if fp_norm_text in gn_text or gn_text in fp_norm_text:
                if g_type == fp_type:
                    error_class = "BOUNDARY_ERROR"
                    matched_gold_entry = {"text": gold_orig.get((gn_text, g_type), gn_text), "type": g_type}
                    break

        entry = {
            "pred_text": fp_orig,
            "pred_type": fp_type,
            "normed": fp_norm_text,
            "error_class": error_class,
        }
        if matched_gold_entry:
            entry["matched_gold"] = matched_gold_entry
        fp_classified.append(entry)

    return {
        "sent_idx": sent_idx,
        "text": text,
        "gold_spans": gold_items,
        "pred_spans": pred_items,
        "exact_matches": exact_matches,
        "fn": fn_classified,
        "fp": fp_classified,
        "counts": {
            "gold": len(gold_items),
            "pred": len(pred_items),
            "exact": len(exact_matches),
            "fn": len(fn_classified),
            "fp": len(fp_classified),
        },
    }


def aggregate_errors(sentence_results: List[dict]) -> dict:
    """Aggregate error patterns across all sentences."""
    # Error class counts
    fn_classes = Counter()
    fp_classes = Counter()
    fn_by_type = defaultdict(list)  # entity_type -> [error details]
    fp_by_type = defaultdict(list)
    total_gold = 0
    total_pred = 0
    total_exact = 0

    for sr in sentence_results:
        total_gold += sr["counts"]["gold"]
        total_pred += sr["counts"]["pred"]
        total_exact += sr["counts"]["exact"]
        for fn in sr["fn"]:
            fn_classes[fn["error_class"]] += 1
            fn_by_type[fn["gold_type"]].append({
                "text": fn["gold_text"],
                "class": fn["error_class"],
                "sentence": sr["text"][:80],
                "matched_pred": fn.get("matched_pred"),
            })
        for fp in sr["fp"]:
            fp_classes[fp["error_class"]] += 1
            fp_by_type[fp["pred_type"]].append({
                "text": fp["pred_text"],
                "class": fp["error_class"],
                "sentence": sr["text"][:80],
                "matched_gold": fp.get("matched_gold"),
            })

    return {
        "summary": {
            "total_gold": total_gold,
            "total_pred": total_pred,
            "total_exact": total_exact,
            "total_fn": sum(fn_classes.values()),
            "total_fp": sum(fp_classes.values()),
            "exact_f1": round(
                2 * total_exact / (total_gold + total_pred), 4
            ) if (total_gold + total_pred) > 0 else 0,
        },
        "fn_by_class": dict(fn_classes.most_common()),
        "fp_by_class": dict(fp_classes.most_common()),
        "fn_by_entity_type": {
            k: {"count": len(v), "examples": v[:5]}
            for k, v in sorted(fn_by_type.items(), key=lambda x: -len(x[1]))
        },
        "fp_by_entity_type": {
            k: {"count": len(v), "examples": v[:5]}
            for k, v in sorted(fp_by_type.items(), key=lambda x: -len(x[1]))
        },
    }


def print_error_report(agg: dict, sentence_results: List[dict]):
    """Print human-readable error analysis report."""
    s = agg["summary"]
    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"  Gold: {s['total_gold']}  Pred: {s['total_pred']}  "
          f"Exact: {s['total_exact']}  FN: {s['total_fn']}  FP: {s['total_fp']}")
    print(f"  Exact F1: {s['exact_f1']}")

    print(f"\n{'─'*70}")
    print(f"  FALSE NEGATIVES (missed entities) by class:")
    print(f"{'─'*70}")
    for cls, cnt in sorted(agg["fn_by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls:25s} {cnt:3d}")

    print(f"\n{'─'*70}")
    print(f"  FALSE POSITIVES (spurious entities) by class:")
    print(f"{'─'*70}")
    for cls, cnt in sorted(agg["fp_by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls:25s} {cnt:3d}")

    print(f"\n{'─'*70}")
    print(f"  FN BY ENTITY TYPE (top examples):")
    print(f"{'─'*70}")
    for etype, info in agg["fn_by_entity_type"].items():
        print(f"\n  [{etype}] — {info['count']} misses")
        for ex in info["examples"]:
            mp = ex.get("matched_pred")
            mp_str = f" → pred: \"{mp['text']}\"({mp['type']})" if mp else ""
            print(f"    [{ex['class']:20s}] \"{ex['text']}\"{mp_str}")
            print(f"      문장: {ex['sentence']}...")

    print(f"\n{'─'*70}")
    print(f"  FP BY ENTITY TYPE (top examples):")
    print(f"{'─'*70}")
    for etype, info in agg["fp_by_entity_type"].items():
        print(f"\n  [{etype}] — {info['count']} spurious")
        for ex in info["examples"]:
            mg = ex.get("matched_gold")
            mg_str = f" ← gold: \"{mg['text']}\"({mg['type']})" if mg else ""
            print(f"    [{ex['class']:20s}] \"{ex['text']}\"{mg_str}")
            print(f"      문장: {ex['sentence']}...")

    # Show worst sentences
    worst = sorted(sentence_results, key=lambda x: -(x["counts"]["fn"] + x["counts"]["fp"]))[:10]
    print(f"\n{'─'*70}")
    print(f"  WORST SENTENCES (most errors):")
    print(f"{'─'*70}")
    for sr in worst:
        if sr["counts"]["fn"] + sr["counts"]["fp"] == 0:
            continue
        print(f"\n  [{sr['sent_idx']}] FN={sr['counts']['fn']} FP={sr['counts']['fp']}")
        print(f"  문장: {sr['text'][:100]}...")
        print(f"  Gold: {sr['gold_spans']}")
        print(f"  Pred: {sr['pred_spans']}")
        for fn in sr["fn"]:
            mp = fn.get("matched_pred")
            mp_str = f" → pred:\"{mp['text']}\"({mp['type']})" if mp else ""
            print(f"    FN [{fn['error_class']}] \"{fn['gold_text']}\"({fn['gold_type']}){mp_str}")
        for fp in sr["fp"]:
            mg = fp.get("matched_gold")
            mg_str = f" ← gold:\"{mg['text']}\"({mg['type']})" if mg else ""
            print(f"    FP [{fp['error_class']}] \"{fp['pred_text']}\"({fp['pred_type']}){mg_str}")


def run_error_analysis(args) -> dict:
    """Run benchmark with per-sentence error tracking."""
    # Load gold data
    print(f"Loading gold data: {args.dataset}/{args.config} [{args.split}]")
    loader = DatasetLoader()
    gold_records = loader.load(args.dataset, config=args.config, split=args.split, max_samples=args.max_samples)
    print(f"  Loaded {len(gold_records)} records")

    # Create labeler
    from evaluators.__main__ import _create_labeler, _load_env
    _load_env()
    backend, labeler = _create_labeler(args.models[0], args)

    # Run per-sentence
    sentence_results = []
    errors = 0
    t_start = time.time()

    for i, record in enumerate(gold_records):
        gold_tokens = record["tokens"]
        gold_tags = record["ner_tags"]
        sentence = record.get("sentence")
        if sentence:
            text = re.sub(r'<([^:>]+):[A-Z]+>', r'\1', sentence)
        else:
            text = TagAligner.reconstruct_text(gold_tokens)

        try:
            has_space_tokens = any(t.strip() == "" for t in gold_tokens)
            g_spans = extract_spans_from_bio(gold_tokens, gold_tags)

            if has_space_tokens and hasattr(labeler, "label_spans"):
                p_spans = labeler.label_spans(text)
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
                p_spans = extract_spans_from_bio(pred_tokens, normalize_tags(pred_tags))

            result = analyze_sentence(i, text, g_spans, p_spans)
            sentence_results.append(result)

            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(gold_records)}] {elapsed:.1f}s elapsed, {errors} errors")

        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            errors += 1

    t_total = time.time() - t_start

    # Token usage / TPS
    prompt_tokens = getattr(labeler, "total_prompt_tokens", 0)
    completion_tokens = getattr(labeler, "total_completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    tps = round(total_tokens / t_total, 2) if t_total > 0 else 0
    out_tps = round(completion_tokens / t_total, 2) if t_total > 0 else 0
    sps = round(len(sentence_results) / t_total, 4) if t_total > 0 else 0

    print(f"\nDone: {len(sentence_results)} samples in {t_total:.1f}s ({errors} errors)")
    print(f"  Prompt tokens: {prompt_tokens:,}  Completion tokens: {completion_tokens:,}  Total: {total_tokens:,}")
    print(f"  TPS (total): {tps}  Output TPS: {out_tps}  Samples/s: {sps}")

    # Aggregate
    agg = aggregate_errors(sentence_results)
    print_error_report(agg, sentence_results)

    return {
        "model": args.models[0],
        "num_samples": len(sentence_results),
        "errors": errors,
        "elapsed_seconds": round(t_total, 2),
        "latency": {
            "total_seconds": round(t_total, 2),
            "samples_per_second": sps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tps,
            "output_tokens_per_second": out_tps,
        },
        "aggregate": agg,
        "sentences": sentence_results,
    }


def main():
    parser = argparse.ArgumentParser(description="NER Error Analysis")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", default="klue")
    parser.add_argument("--config", default="ner")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--vllm-url", default="http://vllm-server:8081/v1")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode for vLLM")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run_error_analysis(args)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
