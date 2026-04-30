"""Stockmark NER에서 BERT 모델을 파인튜닝하고 평가한다.

사용 예:
    python -m ner.classifier
    python -m ner.classifier --models "tohoku-nlp/bert-base-japanese-v3"
    python -m ner.classifier --epochs 3 --batch-size 8
"""

import argparse
import json
import logging
import os

from transformers import AutoTokenizer

from ner.classifier.data_utils import build_label_maps, load_stockmark, prepare_datasets
from ner.classifier.train_eval import NERDataset, fine_tune, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "tohoku-nlp/bert-base-japanese-v3",
    "xlm-roberta-base",
    "tohoku-nlp/bert-large-japanese-v2",
]


def main():
    parser = argparse.ArgumentParser(description="BERT NER fine-tune on Stockmark")
    parser.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/classifier")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    label2id, id2label = build_label_maps()

    # 원시 데이터를 한 번 로드한다
    logger.info("Loading Stockmark NER dataset...")
    train_raw, test_raw = load_stockmark()
    logger.info("Train: %d samples, Test: %d samples", len(train_raw), len(test_raw))

    results = {}

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}")

        short_name = model_name.split("/")[-1]
        model_output_dir = os.path.join(args.output_dir, short_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # 토크나이즈한다
        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Tokenizing and aligning labels...")
        train_items, test_items, test_offsets = prepare_datasets(
            train_raw, test_raw, tokenizer, label2id, args.max_length
        )

        train_ds = NERDataset(train_items)
        test_ds = NERDataset(test_items)
        logger.info("Train tokens: %d, Test tokens: %d", len(train_ds), len(test_ds))

        # 파인튜닝한다
        batch_size = args.batch_size
        if "large" in model_name.lower():
            batch_size = max(4, batch_size // 2)

        logger.info("Fine-tuning (epochs=%d, bs=%d, lr=%s)...", args.epochs, batch_size, args.lr)
        train_time = fine_tune(
            model_name=model_name,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            output_dir=model_output_dir,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
        )

        # 평가한다
        best_path = os.path.join(model_output_dir, "best")
        logger.info("Evaluating %s ...", best_path)
        metrics = evaluate_model(
            model_path=best_path,
            test_dataset=test_ds,
            test_offsets=test_offsets,
            test_raw=test_raw,
            id2label=id2label,
        )

        results[model_name] = {
            "overall": metrics["overall"],
            "per_entity": metrics["per_entity"],
            "train_time_sec": round(train_time, 1),
        }

        # 모델별 결과를 출력한다
        o = metrics["overall"]
        print(f"\n  F1={o['f1']:.4f}  Precision={o['precision']:.4f}  Recall={o['recall']:.4f}")
        print(f"  Training time: {train_time:.1f}s")
        print()
        print(f"  {'Entity':<20} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
        print(f"  {'-'*56}")
        for etype, em in sorted(metrics["per_entity"].items(), key=lambda x: -x[1]["f1"]):
            print(f"  {etype:<20} {em['f1']:>8.4f} {em['precision']:>8.4f} {em['recall']:>8.4f} {em['support']:>8}")

    # ── 비교 테이블 ──
    print(f"\n{'='*80}")
    print("  COMPARISON: BERT vs LLM (Stockmark NER, span-level F1)")
    print(f"{'='*80}")
    print(f"  {'Model':<45} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}")
    print(f"  {'-'*72}")

    for model_name, r in results.items():
        o = r["overall"]
        t = r["train_time_sec"]
        short = model_name.split("/")[-1]
        print(f"  {short:<45} {o['f1']:>8.4f} {o['precision']:>8.4f} {o['recall']:>8.4f} {t:>7.1f}s")

    print(f"  {'-'*72}")
    print(f"  {'vLLM Qwen3.5-27B + 2pass (LLM baseline)':<45} {'0.8488':>8} {'0.8488':>8} {'0.8488':>8} {'  N/A':>8}")
    print(f"  {'vLLM Qwen3.5-35B-A3B + 2pass (LLM)':<45} {'0.8108':>8} {'0.8117':>8} {'0.8099':>8} {'  N/A':>8}")
    print(f"  {'OpenAI gpt-5-mini + 2pass (LLM)':<45} {'0.7941':>8} {'0.7808':>8} {'0.8078':>8} {'  N/A':>8}")

    # 결과를 저장한다
    output_path = os.path.join(args.output_dir, "bert_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
