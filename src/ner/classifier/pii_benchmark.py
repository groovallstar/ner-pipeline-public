"""PII NER 벤치마크: 합성 PII 데이터 생성 → BERT 파인튜닝 → 평가.

사용 예:
    python -m ner.classifier.pii_benchmark
    python -m ner.classifier.pii_benchmark --models "tohoku-nlp/bert-base-japanese-v3"
    python -m ner.classifier.pii_benchmark --n-train 1000 --n-test 300
"""

import argparse
import json
import logging
import os
import random
import sys

from transformers import AutoTokenizer

from ner.classifier.data_utils import tokenize_and_align
from ner.classifier.train_eval import NERDataset, fine_tune, evaluate_model

# PII generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
from pii_gen import generate_dataset as generate_pii_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PII_ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DOB",
    "ID_NUMBER", "EMAIL", "CREDIT_CARD",
]

DEFAULT_MODELS = [
    "tohoku-nlp/bert-base-japanese-v3",
    "xlm-roberta-base",
]


def build_pii_label_maps():
    """PII 엔티티용 BIO 레이블 맵을 구성한다. O + 7 B + 7 I = 15개 레이블."""
    labels = ["O"]
    for etype in PII_ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def _convert_pii_to_ner_format(pii_samples):
    """pii_gen 형식을 classifier 형식으로 변환한다.

    pii_gen: {"text": ..., "entities": [{"label": "NAME", "start_char": 0, "end_char": 5, "text": "..."}]}
    classifier: {"text": ..., "entities": [{"type": "NAME", "span": [0, 5]}]}
    """
    converted = []
    for sample in pii_samples:
        entities = []
        for ent in sample["entities"]:
            entities.append({
                "type": ent["label"],
                "span": [ent["start_char"], ent["end_char"]],
            })
        converted.append({
            "text": sample["text"],
            "entities": entities,
        })
    return converted


def prepare_pii_datasets(train_raw, test_raw, tokenizer, label2id, max_length=512):
    """PII 데이터셋을 토크나이즈하고 정렬한다."""
    train_items = []
    for row in train_raw:
        enc, _ = tokenize_and_align(
            row["text"], row["entities"], tokenizer, label2id, max_length
        )
        train_items.append(enc)

    test_items = []
    test_offsets = []
    for row in test_raw:
        enc, offsets = tokenize_and_align(
            row["text"], row["entities"], tokenizer, label2id, max_length
        )
        test_items.append(enc)
        test_offsets.append(offsets)

    return train_items, test_items, test_offsets


def main():
    parser = argparse.ArgumentParser(description="PII NER Benchmark with BERT")
    parser.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names",
    )
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/pii_benchmark")
    args = parser.parse_args()

    random.seed(args.seed)

    models = [m.strip() for m in args.models.split(",")]
    label2id, id2label = build_pii_label_maps()

    # PII 데이터를 생성한다
    logger.info("Generating PII dataset: train=%d, test=%d", args.n_train, args.n_test)
    total = args.n_train + args.n_test
    all_pii = generate_pii_dataset(lang="ja", n_samples=total)

    # 셔플하고 분할한다
    random.shuffle(all_pii)
    train_pii = all_pii[:args.n_train]
    test_pii = all_pii[args.n_train:]

    # 형식을 변환한다
    train_raw = _convert_pii_to_ner_format(train_pii)
    test_raw = _convert_pii_to_ner_format(test_pii)

    logger.info("Train: %d samples, Test: %d samples", len(train_raw), len(test_raw))

    # 샘플을 출력한다
    sample = train_raw[0]
    logger.info("Sample text: %s", sample["text"][:80])
    logger.info("Sample entities: %s", sample["entities"][:3])

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
        train_items, test_items, test_offsets = prepare_pii_datasets(
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
            "train_samples": args.n_train,
            "test_samples": args.n_test,
        }

        # 모델별 결과를 출력한다
        o = metrics["overall"]
        print(f"\n  F1={o['f1']:.4f}  Precision={o['precision']:.4f}  Recall={o['recall']:.4f}")
        print(f"  Training time: {train_time:.1f}s")
        print()
        print(f"  {'Entity':<16} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
        print(f"  {'-'*52}")
        for etype, em in sorted(metrics["per_entity"].items(), key=lambda x: -x[1]["f1"]):
            print(f"  {etype:<16} {em['f1']:>8.4f} {em['precision']:>8.4f} {em['recall']:>8.4f} {em['support']:>8}")

    # 비교 테이블
    print(f"\n{'='*80}")
    print("  PII NER BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<40} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}")
    print(f"  {'-'*68}")

    for model_name, r in results.items():
        o = r["overall"]
        t = r["train_time_sec"]
        short = model_name.split("/")[-1]
        print(f"  {short:<40} {o['f1']:>8.4f} {o['precision']:>8.4f} {o['recall']:>8.4f} {t:>7.1f}s")

    # 결과를 저장한다
    output_path = os.path.join(args.output_dir, "pii_benchmark.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
