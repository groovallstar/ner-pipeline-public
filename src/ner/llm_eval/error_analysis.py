"""문장별 오류 분석: NER 벤치마크에서 FN/FP 패턴을 추출한다.

사용 예:
    python -m ner.llm_eval.error_analysis \
        --models vllm:Qwen/Qwen3.5-27B \
        --max-samples 50 \
        --output results/error_analysis.json
"""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import List

from ner.labelers.dataset_loader import DatasetLoader
from ner.labelers.tag_aligner import TagAligner, normalize_tags, extract_spans_from_bio

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    """비교를 위해 공백을 정규화한다."""
    return text.strip().replace(" ", "")


def _classify_error(gold_text: str, pred_text: str, gold_type: str, pred_type: str) -> str:
    """오류 패턴을 분류한다."""
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
    """단일 문장의 gold span과 예측 span을 분석한다.

    FN, FP, 정확 매칭, 오류 분류가 포함된 dict를 반환한다.
    """
    # 비교를 위해 정규화한다
    gold_items = [(s["text"].strip(), s["type"].strip()) for s in gold_spans]
    pred_items = [(s["text"].strip(), s["type"].strip()) for s in pred_spans]

    gold_normed = [(_norm(t), ty) for t, ty in gold_items]
    pred_normed = [(_norm(t), ty) for t, ty in pred_items]

    # 정확 매칭 (공백 정규화, 멀티셋)
    gold_counter = Counter(gold_normed)
    pred_counter = Counter(pred_normed)
    exact_matches = list((gold_counter & pred_counter).elements())
    fn_counter = gold_counter - pred_counter  # in gold but not pred
    fp_counter = pred_counter - gold_counter  # in pred but not gold

    fn_items = list(fn_counter.elements())
    fp_items = list(fp_counter.elements())

    # 원문 텍스트로 역매핑한다
    gold_orig = {(_norm(t), ty): t for t, ty in gold_items}
    pred_orig = {(_norm(t), ty): t for t, ty in pred_items}

    # 오류를 분류한다
    fn_classified = []
    fp_classified = []
    matched_fp = set()

    for fn_norm_text, fn_type in fn_items:
        fn_orig = gold_orig.get((fn_norm_text, fn_type), fn_norm_text)
        best_match = None
        best_class = "MISS"  # default: completely missed

        # 관련 FP를 찾는다 (동일 타입, 부분 겹침)
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
        # gold와 타입 불일치인지 확인한다
        error_class = "HALLUCINATION"  # 기본값: gold에 없는 엔티티
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
    """전체 문장에 걸쳐 오류 패턴을 집계한다."""
    # 오류 클래스 카운트
    fn_classes = Counter()
    fp_classes = Counter()
    fn_by_type = defaultdict(list)  # entity_type -> [오류 상세]
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
    """사람이 읽기 쉬운 오류 분석 리포트를 출력한다."""
    s = agg["summary"]
    print(f"\n{'='*70}")
    print("  ERROR ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"  Gold: {s['total_gold']}  Pred: {s['total_pred']}  "
          f"Exact: {s['total_exact']}  FN: {s['total_fn']}  FP: {s['total_fp']}")
    print(f"  Exact F1: {s['exact_f1']}")

    print(f"\n{'─'*70}")
    print("  FALSE NEGATIVES (missed entities) by class:")
    print(f"{'─'*70}")
    for cls, cnt in sorted(agg["fn_by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls:25s} {cnt:3d}")

    print(f"\n{'─'*70}")
    print("  FALSE POSITIVES (spurious entities) by class:")
    print(f"{'─'*70}")
    for cls, cnt in sorted(agg["fp_by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls:25s} {cnt:3d}")

    print(f"\n{'─'*70}")
    print("  FN BY ENTITY TYPE (top examples):")
    print(f"{'─'*70}")
    for etype, info in agg["fn_by_entity_type"].items():
        print(f"\n  [{etype}] — {info['count']} misses")
        for ex in info["examples"]:
            mp = ex.get("matched_pred")
            mp_str = f" → pred: \"{mp['text']}\"({mp['type']})" if mp else ""
            print(f"    [{ex['class']:20s}] \"{ex['text']}\"{mp_str}")
            print(f"      문장: {ex['sentence']}...")

    print(f"\n{'─'*70}")
    print("  FP BY ENTITY TYPE (top examples):")
    print(f"{'─'*70}")
    for etype, info in agg["fp_by_entity_type"].items():
        print(f"\n  [{etype}] — {info['count']} spurious")
        for ex in info["examples"]:
            mg = ex.get("matched_gold")
            mg_str = f" ← gold: \"{mg['text']}\"({mg['type']})" if mg else ""
            print(f"    [{ex['class']:20s}] \"{ex['text']}\"{mg_str}")
            print(f"      문장: {ex['sentence']}...")

    # 최악의 문장을 표시한다 (오류 최다)
    worst = sorted(sentence_results, key=lambda x: -(x["counts"]["fn"] + x["counts"]["fp"]))[:10]
    print(f"\n{'─'*70}")
    print("  WORST SENTENCES (most errors):")
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
    """문장별 오류 추적을 포함한 벤치마크를 실행한다."""
    # gold 데이터를 로드한다
    print(f"Loading gold data: {args.dataset}/{args.config} [{args.split}]")
    loader = DatasetLoader()
    gold_records = loader.load(args.dataset, config=args.config, split=args.split, max_samples=args.max_samples)
    print(f"  Loaded {len(gold_records)} records")

    # 라벨러를 생성한다
    from ner.llm_eval.__main__ import _create_labeler, _load_env
    _load_env()
    backend, labeler = _create_labeler(args.models[0], args)

    # 문장별로 실행한다
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

            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(gold_records)}] {elapsed:.1f}s elapsed, {errors} errors")

        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            errors += 1

    t_total = time.time() - t_start

    # 토큰 사용량 / TPS 계산
    prompt_tokens = getattr(labeler, "total_prompt_tokens", 0)
    completion_tokens = getattr(labeler, "total_completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    tps = round(total_tokens / t_total, 2) if t_total > 0 else 0
    out_tps = round(completion_tokens / t_total, 2) if t_total > 0 else 0
    sps = round(len(sentence_results) / t_total, 4) if t_total > 0 else 0

    print(f"\nDone: {len(sentence_results)} samples in {t_total:.1f}s ({errors} errors)")
    print(f"  Prompt tokens: {prompt_tokens:,}  Completion tokens: {completion_tokens:,}  Total: {total_tokens:,}")
    print(f"  TPS (total): {tps}  Output TPS: {out_tps}  Samples/s: {sps}")

    # 집계한다
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
    parser.add_argument("--vllm-url", default="http://localhost:8081/v1")
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
