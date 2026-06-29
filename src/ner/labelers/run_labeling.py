"""라벨링 파이프라인 1단계: 라벨러를 데이터셋에 실행하고 JSONL을 출력한다.

출력 JSONL은 `llm_eval.span_evaluator_cli`에서 사용된다. 한 번 실행 시 모델 하나를 처리한다.

사용 예:
    python -m ner.labelers.run_labeling \
        --lang ja \
        --model vllm:Qwen/Qwen3.5-27B \
        --max-samples 500 \
        --output results/predictions/ja-vllm-27b.jsonl
"""
import argparse
import json
import os
import sys
import time

from tqdm import tqdm

from ner.labelers.span_matcher import match_spans


def _load_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
    env_path = os.path.abspath(env_path)
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


def _create_labeler(spec: str, lang: str, args):
    backend, model_name = spec.split(":", 1)
    if lang == "ja":
        if backend == "vllm":
            from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler
            return backend, VllmNERLabeler(base_url=args.vllm_url, model=model_name,
                                           concurrency=args.concurrency, thinking=args.thinking)
        if backend == "openai":
            from ner.labelers.ja.openai_ner_labeler import OpenAINERLabeler
            return backend, OpenAINERLabeler(model=model_name)
    raise ValueError(f"Unsupported lang/backend: {lang}/{backend}")


def _load_gold(lang: str, max_samples: int):
    if lang == "ja":
        from ner.labelers.ja.dataset_loader import JapaneseDatasetLoader
        return JapaneseDatasetLoader().load(split="test", max_samples=max_samples)
    raise ValueError(f"Lang not implemented yet: {lang}")


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser(prog="python -m ner.labelers.run_labeling")
    p.add_argument("--lang", default="ja", choices=["ja"])
    p.add_argument("--model", required=True, help="backend:model_name (e.g. vllm:Qwen/Qwen3.5-27B)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--vllm-url", default="http://localhost:8081/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()

    backend, labeler = _create_labeler(args.model, args.lang, args)
    model_name = args.model.split(":", 1)[1]

    print(f"Loading gold ({args.lang}) ...")
    records = _load_gold(args.lang, args.max_samples)
    print(f"  {len(records)} records")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fout = open(args.output, "w", encoding="utf-8")
    fout.write(json.dumps({"_meta": {"model": model_name, "backend": backend, "lang": args.lang}},
                          ensure_ascii=False) + "\n")

    prev_pt = getattr(labeler, "total_prompt_tokens", 0)
    prev_ct = getattr(labeler, "total_completion_tokens", 0)
    errors = 0

    for r in tqdm(records, desc=f"[{backend}] {model_name}", unit="sample"):
        text = r["text"]
        gold_spans = [{"start": g["start"], "end": g["end"], "label": g.get("type", g.get("label"))}
                      for g in r.get("gold_spans", [])]

        t0 = time.time()
        try:
            pred = labeler.label_spans(text)
        except Exception as e:
            errors += 1
            print(f"WARN: labeling failed for {r.get('id')}: {e}", file=sys.stderr)
            pred = []
        dt = time.time() - t0

        pt = getattr(labeler, "total_prompt_tokens", 0)
        ct = getattr(labeler, "total_completion_tokens", 0)
        d_pt, d_ct = pt - prev_pt, ct - prev_ct
        prev_pt, prev_ct = pt, ct

        pred = [p for p in pred if isinstance(p, dict) and p.get("text") and p.get("type")]
        matched = match_spans(text, pred)
        pred_spans = [{"start": p["start"], "end": p["end"], "label": p.get("type", p.get("label"))}
                      for p in matched]

        fout.write(json.dumps({
            "id": r.get("id"),
            "text": text,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
            "latency_seconds": round(dt, 4),
            "prompt_tokens": d_pt,
            "completion_tokens": d_ct,
        }, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Saved: {args.output}  (errors={errors})")


if __name__ == "__main__":
    main()
