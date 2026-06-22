"""베트남어 NER 배포 패키지 빌드 — 100-홀드아웃 단일 모델 생성.

K-fold(교차검증 추정)와 별개로, 배포용 단일 모델을 만든다. test 100문장을
누출-free(orig 그룹 단위)로 홀드아웃하고 나머지를 train/valid 8:2 로 나눠
phobert 를 파인튜닝한 뒤, `data/stockmark/<lang>_ner_prod_seed*/` 포맷으로
패키지를 저장한다(abstention 미적용 — VI 는 임계값 안 씀).

산출 구조:
    <out>/
    ├── model/              # fine-tuned 가중치 + tokenizer
    ├── data/               # 학습/검증/평가 split (train/valid/test.jsonl)
    ├── metrics.json        # 학습 설정 + test strict/relaxed 메트릭 + 누출 가드
    └── MODEL_CARD.md

사용:
    python src/ner/scripts/build_vi_ner_prod.py
    python src/ner/scripts/build_vi_ner_prod.py \\
        --data data/wikiann_vi/origin.jsonl \\
        --out-dir data/wikiann_vi/vi_ner_prod_seed1 --seed 42
"""
import argparse
import json
import logging
import os
import shutil

from transformers import AutoTokenizer

from ner.classifier.data_utils import (
    CANONICAL_LABELS,
    build_label_maps,
    encode_dataset,
    load_jsonl,
    split_holdout_deploy,
)
from ner.classifier.train_eval import evaluate_model, fine_tune

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger('build_vi_ner_prod')


def dump_jsonl(rows, path):
    """row 리스트를 JSONL 로 저장(원본 필드 보존)."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_model_card(path, *, model_name, data_path, n_total, sizes,
                     hp, op, group_key, test_nonempty, test_min_ctx):
    """배포 모델 카드(MODEL_CARD.md) 작성 — JA 패키지 포맷."""
    train_n, valid_n, test_n = sizes
    leak = ('orig 그룹 단위(누출-free) — test 원문이 train/valid 에 0건'
            if group_key else '행 단위 random')
    parts = []
    if test_nonempty:
        parts.append('엔티티 보유(빈 샘플 제외)')
    if test_min_ctx > 0:
        parts.append(f'엔티티 밖 문맥어 ≥{test_min_ctx}(단편 제외)')
    test_comp = ' + '.join(parts) if parts else '제약 없음'
    card = f"""# VI NER production 모델

베트남어 canonical 10종 평면 NER BERT 분류기. 실제 추론 배포용.
abstention(임계값) 미적용 — raw 모델 출력 그대로.

## 구성

```
./
├── model/              # fine-tuned 가중치 (config.json, model.safetensors,
│                       #   tokenizer)
├── metrics.json        # 학습 설정 + test 메트릭 + 누출 가드
├── data/               # 학습/검증/평가 split
│   ├── train.jsonl     # {train_n} 문장
│   ├── valid.jsonl     # {valid_n} 문장 (best-epoch 선택)
│   └── test.jsonl      # {test_n} 문장 (최종 점검 — 학습 미노출)
└── MODEL_CARD.md
```

## 학습 설정

| 항목 | 값 |
|---|---|
| base 모델 | `{model_name}` |
| 토크나이저 | pyvi 단어분절 + word-level BPE (PhoBERT 계열) |
| 라벨 | canonical 10종 = NER 5(PER/LOC/ORG/PROD/EVT) + PII 5(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD), BIO 21 |
| gold | `{data_path}` ({n_total} 문장) |
| split | test {test_n} 홀드아웃 + 나머지 train/valid 8:2 = {train_n}/{valid_n}/{test_n} |
| 분할 단위 | {leak} |
| test 구성 | {test_comp} |
| 손실 | 표준 cross-entropy (boundary 가중 미적용) |
| 하이퍼파라미터 | epochs={hp['epochs']}, batch={hp['batch_size']}, lr={hp['lr']}, max_len={hp['max_length']}, {hp['precision']}, seed={hp['seed']} |

## 성능 (test {test_n}건, strict span F1)

| | P | R | F1 |
|---|---:|---:|---:|
| raw 모델 | {op['precision']:.4f} | {op['recall']:.4f} | {op['f1']:.4f} |

> 100-홀드아웃 단일 점검치 — support 가 작아 분산이 크다. 교차검증
> 추정치는 `docs/reports/vietnamese-bert-classifier-spec.md` 참조.

## 추론 사용법

1. `model/` + tokenizer 로드(pyvi 필요).
2. 토큰 분류 → BIO span 디코드(`ner.classifier.data_utils.decode_bio_to_spans`).
3. raw span 그대로 사용(임계값 없음).

평가 재현: `python src/ner/scripts/eval_vi_ner_test.py --model-dir
<abs>/model --test <abs>/data/test.jsonl`
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(card)


def main():
    p = argparse.ArgumentParser(
        description='Build a VI NER deployment package: hold out a fixed '
                    'test set (leak-free by orig group), split the rest '
                    'train/valid, fine-tune one model, and save the package '
                    'in the data/stockmark/<lang>_ner_prod_seed* layout. '
                    'No abstention (VI ships raw).')
    p.add_argument('--data', default='data/wikiann_vi/origin.jsonl',
                   help='Gold JSONL path')
    p.add_argument('--model-name', default='vinai/phobert-base-v2',
                   help='HF base model name')
    p.add_argument('--out-dir', default='data/wikiann_vi/vi_ner_prod_seed1',
                   help='Output package dir')
    p.add_argument('--n-test', type=int, default=100,
                   help='Held-out test sentences (min; whole orig groups)')
    p.add_argument('--valid-ratio', type=float, default=0.2,
                   help='Valid fraction of the non-test rows')
    p.add_argument('--group-key', default='orig',
                   help="Group field for leak-free split ('' to disable)")
    p.add_argument('--test-nonempty', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Restrict test to samples with >=1 entity (exclude '
                        'empty); --no-test-nonempty to allow empty')
    p.add_argument('--test-min-context-words', type=int, default=3,
                   help='Test samples must have >= N non-entity (context) '
                        'words (exclude entity-only fragments); 0 to disable')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    args = p.parse_args()

    group_key = args.group_key or None
    label2id, id2label = build_label_maps()
    rows = load_jsonl(args.data)
    logger.info('Loaded %d rows from %s', len(rows), args.data)

    req_types = set(CANONICAL_LABELS) if args.test_nonempty else None
    train_rows, valid_rows, test_rows = split_holdout_deploy(
        rows, n_test=args.n_test, valid_ratio=args.valid_ratio,
        seed=args.seed, group_key=group_key, test_require_types=req_types,
        test_min_context_words=args.test_min_context_words,
    )
    logger.info('Split: train=%d valid=%d test=%d (group_key=%s, '
                'test_nonempty=%s, min_ctx=%d)', len(train_rows),
                len(valid_rows), len(test_rows), group_key,
                args.test_nonempty, args.test_min_context_words)

    # 누출 가드: test 원문이 train/valid 에 있으면 즉시 실패
    leak = 0
    if group_key:
        test_g = {r[group_key] for r in test_rows}
        seen_g = ({r[group_key] for r in train_rows}
                  | {r[group_key] for r in valid_rows})
        leak = len(test_g & seen_g)
        if leak:
            raise SystemExit(f'Error: {leak} test groups leaked into train/'
                             'valid')

    # 토크나이저 (fast 우선, 실패 시 slow)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                  use_fast=True)
    except (TypeError, ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                  use_fast=False)

    logger.info('Tokenizing and aligning labels...')
    train_feats, _ = encode_dataset(
        train_rows, tokenizer, label2id, 'vi', args.max_length)
    valid_feats, _ = encode_dataset(
        valid_rows, tokenizer, label2id, 'vi', args.max_length)
    test_feats, test_offs = encode_dataset(
        test_rows, tokenizer, label2id, 'vi', args.max_length)

    train_dir = os.path.join(args.out_dir, '_train')
    os.makedirs(train_dir, exist_ok=True)
    logger.info('Fine-tuning %s ...', args.model_name)
    elapsed, best_dir = fine_tune(
        model_name=args.model_name, train_features=train_feats,
        eval_features=valid_feats, label2id=label2id, id2label=id2label,
        output_dir=train_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, precision=args.precision,
    )
    logger.info('Train time: %.1fs', elapsed)

    # 패키지 model/ 조립: best 가중치 복사 + tokenizer 저장
    model_dir = os.path.join(args.out_dir, 'model')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(best_dir, model_dir)
    tokenizer.save_pretrained(model_dir)
    shutil.rmtree(train_dir, ignore_errors=True)

    logger.info('Evaluating on held-out test...')
    metrics = evaluate_model(
        model_path=model_dir, eval_features=test_feats,
        eval_offsets=test_offs, eval_rows=test_rows, id2label=id2label,
    )
    strict_m = metrics['strict']
    relaxed_m = metrics['relaxed']
    op = strict_m['overall']

    # data/ split + metrics.json + MODEL_CARD.md
    data_dir = os.path.join(args.out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    dump_jsonl(train_rows, os.path.join(data_dir, 'train.jsonl'))
    dump_jsonl(valid_rows, os.path.join(data_dir, 'valid.jsonl'))
    dump_jsonl(test_rows, os.path.join(data_dir, 'test.jsonl'))

    hp = {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'max_length': args.max_length, 'precision': args.precision,
        'seed': args.seed,
    }
    summary = {
        'lang': 'vi', 'model_name': args.model_name, 'data_path': args.data,
        'group_key': group_key, 'leaked_test_groups': leak,
        'test_nonempty': args.test_nonempty,
        'test_min_context_words': args.test_min_context_words,
        'train_samples': len(train_rows), 'valid_samples': len(valid_rows),
        'test_samples': len(test_rows), 'train_time_sec': round(elapsed, 1),
        **hp,
        'overall_strict': strict_m['overall'],
        'per_entity_strict': strict_m['per_entity'],
        'overall_relaxed': relaxed_m['overall'],
        'per_entity_relaxed': relaxed_m['per_entity'],
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_model_card(
        os.path.join(args.out_dir, 'MODEL_CARD.md'),
        model_name=args.model_name, data_path=args.data, n_total=len(rows),
        sizes=(len(train_rows), len(valid_rows), len(test_rows)),
        hp=hp, op=op, group_key=group_key, test_nonempty=args.test_nonempty,
        test_min_ctx=args.test_min_context_words,
    )

    print(f"\n{'=' * 72}")
    print('  VI NER deployment package')
    print(f"{'=' * 72}")
    print(f'  Out      : {args.out_dir}')
    print(f'  Model    : {args.model_name}')
    print(f'  Split    : {len(train_rows)}/{len(valid_rows)}/'
          f'{len(test_rows)}  (leak={leak})')
    print(f"  Test     : P={op['precision']:.4f} R={op['recall']:.4f} "
          f"F1={op['f1']:.4f}")


if __name__ == '__main__':
    main()
