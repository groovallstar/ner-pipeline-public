"""classifier CLI: JA·VI canonical 10종 평면 BERT fine-tune.

augmenters 가 produce 한 PII 주입 JSONL 을 입력으로, BIO 21-class token classifier
를 학습하고 char-offset span F1 을 측정한다.

사용 예:
    python -m ner.classifier --lang ja
    python -m ner.classifier --lang vi --epochs 5 --batch-size 32
    python -m ner.classifier --lang ja --smoke    # 스모크 (100/50 sample, 1 epoch)
"""

import argparse
import json
import logging
import os

from transformers import AutoTokenizer

from ner.classifier.data_utils import (
    build_label_maps,
    class_weights_tensor,
    encode_dataset,
    load_jsonl,
    mask_pii_in_features,
    split_train_valid_test,
)
from ner.classifier.train_eval import evaluate_model, fine_tune

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# 언어별 기본값 (round 2 deep-interview lock-in)
DEFAULT_DATA = {
    'ja': 'data/stockmark/pii_all.jsonl',
    'vi': 'data/wikiann_vi/pii_all.jsonl',
}
DEFAULT_MODEL = {
    'ja': 'tohoku-nlp/bert-base-japanese-v3',
    'vi': 'xlm-roberta-base',
}

# F1 합격선 (deep-interview round 4 lock-in)
F1_GATE = 0.95


def main():
    parser = argparse.ArgumentParser(
        description='NER BERT fine-tune (canonical 10-class, JA·VI)'
    )
    parser.add_argument('--lang', choices=['ja', 'vi'], required=True)
    parser.add_argument('--data', help='Override JSONL path')
    parser.add_argument('--model-name', help='Override HF model name')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', help='Override output dir')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: 100 train / 50 test / 1 epoch',
    )
    # 변종 플래그 (class weight·hard-negative·multi-task 변종 실험용)
    parser.add_argument(
        '--class-weight-ner', type=float, default=None,
        help='Cross-entropy weight for NER 5-class BIO labels (default: 1.0)',
    )
    parser.add_argument(
        '--class-weight-pii', type=float, default=None,
        help='Cross-entropy weight for PII 5-class BIO labels (default: 1.0)',
    )
    parser.add_argument(
        '--metric-for-best', default='eval_loss',
        choices=['eval_loss', 'ner_f1', 'overall_f1'],
        help='Best-model selection metric',
    )
    parser.add_argument(
        '--curriculum', action='store_true',
        help='Two-stage NER warmup: stage 1 with PII masked to O, stage 2 full 21-class',
    )
    parser.add_argument(
        '--curriculum-stage1-epochs', type=int, default=3,
        help='Stage 1 epochs when --curriculum (stage 2 uses --epochs)',
    )
    parser.add_argument(
        '--precision', choices=['fp16', 'bf16'], default='fp16',
        help='Mixed precision (use bf16 for DeBERTa-v3 family)',
    )
    args = parser.parse_args()

    data_path = args.data or DEFAULT_DATA[args.lang]
    model_name = args.model_name or DEFAULT_MODEL[args.lang]
    output_dir = args.output_dir or f'results/classifier/{args.lang}'
    os.makedirs(output_dir, exist_ok=True)

    label2id, id2label = build_label_maps()

    logger.info('Loading data: %s', data_path)
    rows = load_jsonl(data_path)
    train_rows, valid_rows, test_rows = split_train_valid_test(
        rows, args.valid_ratio, args.test_ratio, args.seed
    )

    if args.smoke:
        train_rows = train_rows[:100]
        valid_rows = valid_rows[:25]
        test_rows = test_rows[:50]
        args.epochs = 1
        logger.info('Smoke mode active')
    logger.info(
        'Train=%d, Valid=%d, Test=%d, Epochs=%d, BS=%d, LR=%s, MaxLen=%d',
        len(train_rows), len(valid_rows), len(test_rows),
        args.epochs, args.batch_size, args.lr, args.max_length,
    )

    logger.info('Loading tokenizer: %s', model_name)
    # 우선 fast tokenizer 시도 → 실패 시 slow fallback (BertJapaneseTokenizer 등 MeCab 기반)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except (TypeError, ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    logger.info('Tokenizer fast=%s', tokenizer.is_fast)

    logger.info('Tokenizing and aligning labels...')
    train_features, _ = encode_dataset(
        train_rows, tokenizer, label2id, args.lang, args.max_length
    )
    valid_features, _ = encode_dataset(
        valid_rows, tokenizer, label2id, args.lang, args.max_length
    )
    test_features, test_offsets = encode_dataset(
        test_rows, tokenizer, label2id, args.lang, args.max_length
    )

    # class_weights tensor (None 이면 표준 CE)
    cw = None
    if args.class_weight_ner is not None or args.class_weight_pii is not None:
        cw = class_weights_tensor(
            label2id,
            w_ner=args.class_weight_ner if args.class_weight_ner is not None else 1.0,
            w_pii=args.class_weight_pii if args.class_weight_pii is not None else 1.0,
            w_o=1.0,
        )
        logger.info(
            'Class weights: NER=%s, PII=%s, O=1.0',
            args.class_weight_ner, args.class_weight_pii,
        )

    if args.curriculum:
        # Stage 1: PII 라벨을 O 로 마스킹하고 NER warmup
        stage1_dir = os.path.join(output_dir, 'stage1')
        os.makedirs(stage1_dir, exist_ok=True)
        train_features_s1 = mask_pii_in_features(train_features, label2id)
        valid_features_s1 = mask_pii_in_features(valid_features, label2id)
        logger.info(
            'Curriculum stage 1: NER-only warmup (PII labels masked to O), epochs=%d',
            args.curriculum_stage1_epochs,
        )
        s1_elapsed, s1_best = fine_tune(
            model_name=model_name,
            train_features=train_features_s1,
            eval_features=valid_features_s1,
            label2id=label2id,
            id2label=id2label,
            output_dir=stage1_dir,
            epochs=args.curriculum_stage1_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            class_weights=cw,
            metric_for_best=args.metric_for_best,
            precision=args.precision,
        )
        logger.info('Stage 1 time: %.1fs (best at %s)', s1_elapsed, s1_best)

        logger.info(
            'Curriculum stage 2: full 21-class fine-tune from stage1 weights, epochs=%d',
            args.epochs,
        )
        s2_elapsed, best_dir = fine_tune(
            model_name=model_name,
            train_features=train_features,
            eval_features=valid_features,
            label2id=label2id,
            id2label=id2label,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            class_weights=cw,
            metric_for_best=args.metric_for_best,
            init_model_path=s1_best,
            precision=args.precision,
        )
        elapsed = s1_elapsed + s2_elapsed
        logger.info('Curriculum total time: %.1fs (stage1=%.1f + stage2=%.1f)',
                    elapsed, s1_elapsed, s2_elapsed)
    else:
        logger.info('Fine-tuning %s ...', model_name)
        elapsed, best_dir = fine_tune(
            model_name=model_name,
            train_features=train_features,
            eval_features=valid_features,
            label2id=label2id,
            id2label=id2label,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            class_weights=cw,
            metric_for_best=args.metric_for_best,
            precision=args.precision,
        )
        logger.info('Train time: %.1fs', elapsed)

    logger.info('Evaluating on test split...')
    metrics = evaluate_model(
        model_path=best_dir,
        eval_features=test_features,
        eval_offsets=test_offsets,
        eval_rows=test_rows,
        id2label=id2label,
    )

    summary = {
        'lang': args.lang,
        'model_name': model_name,
        'data_path': data_path,
        'train_samples': len(train_rows),
        'valid_samples': len(valid_rows),
        'test_samples': len(test_rows),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_length': args.max_length,
        'valid_ratio': args.valid_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'class_weight_ner': args.class_weight_ner,
        'class_weight_pii': args.class_weight_pii,
        'metric_for_best': args.metric_for_best,
        'curriculum': args.curriculum,
        'curriculum_stage1_epochs': args.curriculum_stage1_epochs if args.curriculum else None,
        'train_time_sec': round(elapsed, 1),
        'overall': metrics['overall'],
        'per_entity': metrics['per_entity'],
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    o = metrics['overall']
    print(f"\n{'='*72}")
    print(f"  {args.lang.upper()} BERT NER (canonical 10-class)")
    print(f"{'='*72}")
    print(f"  Model    : {model_name}")
    print(f"  Data     : {data_path}")
    print(f"  Train/Valid/Test : {len(train_rows)} / {len(valid_rows)} / {len(test_rows)}  (epochs={args.epochs})")
    print(f"  F1={o['f1']:.4f}  Precision={o['precision']:.4f}  "
          f"Recall={o['recall']:.4f}  Support={o['support']}")
    print(f"  Train time: {elapsed:.1f}s")
    print()
    print(f"  {'Entity':<14} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
    print(f"  {'-'*52}")
    for etype, em in sorted(metrics['per_entity'].items(), key=lambda x: -x[1]['f1']):
        print(f"  {etype:<14} {em['f1']:>8.4f} {em['precision']:>8.4f} "
              f"{em['recall']:>8.4f} {em['support']:>8}")
    print(f"\n  Saved to {metrics_path}")

    if o['f1'] >= F1_GATE:
        print(f"\n  PASS  F1 {o['f1']:.4f} >= gate {F1_GATE}")
    else:
        print(f"\n  FAIL  F1 {o['f1']:.4f} <  gate {F1_GATE}  (gap {F1_GATE - o['f1']:.4f})")


if __name__ == '__main__':
    main()
