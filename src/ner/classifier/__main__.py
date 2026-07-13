"""classifier CLI: JA·VI·KO canonical 10종 평면 BERT fine-tune.

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
    boundary_weights_tensor,
    dataset_fingerprint,
    encode_dataset,
    group_stats,
    load_jsonl,
    mask_pii_in_features,
    split_kfold_stratified,
    split_train_valid_test,
    validate_group_key,
)
from ner.classifier.confidence_threshold import (
    DEFAULT_THRESHOLD_TYPES,
    apply_thresholds,
    fit_thresholds,
    load_thresholds,
    save_thresholds,
)
from ner.classifier.train_eval import evaluate_model, fine_tune
from ner.metrics.span_metrics import compute_offset_span_f1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# 언어별 기본값 (round 2 deep-interview lock-in)
DEFAULT_DATA = {
    'ja': 'data/stockmark/pii_all.jsonl',
    'vi': 'data/wikiann_vi/origin.jsonl',
    'ko': 'data/klue/pii_all.jsonl',
}
DEFAULT_MODEL = {
    'ja': 'tohoku-nlp/bert-base-japanese-v3',
    'vi': 'xlm-roberta-base',
    'ko': 'kakaobank/kf-deberta-base',
}


def main():
    parser = argparse.ArgumentParser(
        description='NER BERT fine-tune (canonical 10-class, JA·VI·KO)'
    )
    parser.add_argument('--lang', choices=['ja', 'vi', 'ko'], required=True)
    parser.add_argument('--data', help='Override JSONL path')
    parser.add_argument('--model-name', help='Override HF model name')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument(
        '--kfold', type=int, default=None,
        help='Number of folds for stratified K-fold CV. When set, '
             '--valid-ratio/--test-ratio are ignored.',
    )
    parser.add_argument(
        '--fold-index', type=int, default=None,
        help='Fold used as test split (0 <= fold-index < kfold). '
             'Required when --kfold is set.',
    )
    parser.add_argument(
        '--group-key', default=None,
        help='REQUIRED. Row field to group by for leak-free splits (e.g. '
             '"orig", "id"): rows sharing the value go to the same split, '
             'preventing sibling-row leakage across train/valid/test. Pass '
             '"none" to opt out explicitly (row-level split). The declared '
             'key is rejected if another field groups rows more strongly.',
    )
    parser.add_argument(
        '--no-stratify', action='store_true',
        help='Disable PROD/EVT stratification: split becomes label-invariant '
             '(seeded shuffle only), so fold membership stays identical before '
             'and after a gold relabel. Use for controlled before/after '
             'comparison on unchanged-gold types.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--train-seed', type=int, default=None,
        help='Training seed (head init, dropout, shuffle), decoupled from '
             '--seed (data split). Default None keeps the legacy unseeded '
             'head init (existing behavior); set it for a reproducible run.',
    )
    parser.add_argument('--output-dir', help='Override output dir')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: 100 train / 50 test / 1 epoch',
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
    parser.add_argument(
        '--legacy-no-offset-trim', action='store_true',
        help='Diagnostic: disable fast-tokenizer offset trim, reproducing '
             'the pre-fix SentencePiece misalignment collapse (F1 ~ 0). '
             'Only for as-is benchmark reproduction.',
    )
    parser.add_argument(
        '--metric-mode', choices=['strict', 'relaxed', 'both'], default='both',
        help='Span F1 mode shown in console. metrics.json always stores both. '
             'strict=(start,end,type) exact match (default gate). '
             "relaxed=SemEval'13 Partial (type match + overlap = 0.5).",
    )
    parser.add_argument(
        '--boundary-b-weight', type=float, default=None,
        help='Loss weight for B- tokens (entity start). Default 1.0 = no effect.',
    )
    parser.add_argument(
        '--boundary-i-weight', type=float, default=None,
        help='Loss weight for I- tokens (entity inside). Default 1.0 = no effect.',
    )
    parser.add_argument(
        '--data-extra-train-jsonl', default=None,
        help='Extra JSONL appended to TRAIN split only (valid/test stay '
             'from --data). Used for negative-oversampling experiments '
             'where leak-free evaluation is required.',
    )
    parser.add_argument(
        '--fit-threshold', action='store_true',
        help='Fit per-class confidence thresholds on the VALID split '
             '(greedy, overall P/R >= target) and save thresholds.json, '
             'then report the confidence-threshold operating point on '
             'test. NER-4 types only (ORG/LOC/EVT/PROD).',
    )
    parser.add_argument(
        '--confidence-thresholds', default=None,
        help='Path to a thresholds.json (from --fit-threshold) to apply '
             'at test eval. Mutually informative with --fit-threshold; if '
             'both given, --fit-threshold wins (re-fits on this run '
             'model).',
    )
    parser.add_argument(
        '--threshold-target', type=float, default=0.93,
        help='Target for both P and R when fitting confidence thresholds '
             '(default 0.93).',
    )
    args = parser.parse_args()

    if args.kfold is not None and args.fold_index is None:
        parser.error('--fold-index is required when --kfold is set')
    if args.kfold is not None and args.kfold < 3:
        parser.error('--kfold must be >= 3 (train needs at least one fold)')
    if args.group_key is None:
        parser.error(
            '--group-key is required. Pass the field that marks sibling rows '
            '(e.g. "orig" or "id"), or "none" to opt out of leak protection '
            'explicitly. An unmeasured split must not read as a clean one.'
        )
    # "none" 은 명시적 opt-out — 행 단위 분할이며 누출 카운터는 null 로 남는다.
    group_key = None if args.group_key == 'none' else args.group_key

    data_path = args.data or DEFAULT_DATA[args.lang]
    model_name = args.model_name or DEFAULT_MODEL[args.lang]
    output_dir = args.output_dir or f'results/classifier/{args.lang}'
    os.makedirs(output_dir, exist_ok=True)

    label2id, id2label = build_label_maps()

    logger.info('Loading data: %s', data_path)
    rows = load_jsonl(data_path)
    # 분할 전에 그룹 키를 검증한다 — 잘못 고른 키는 보호를 no-op 으로 만들면서
    # 누출 카운터는 0 을 내므로, 학습을 시작하기 전에 크게 실패해야 한다.
    validate_group_key(rows, group_key)
    n_rows, n_groups = group_stats(rows, group_key)
    # 지문·stratify 는 비교 유효성 게이트(validity)의 '자' 를 이룬다 —
    # 지문은 test gold 정체성(내용·순서), stratify 는 fold 멤버십을 결정한다.
    data_fp = dataset_fingerprint(rows)
    stratify = not args.no_stratify
    logger.info(
        'Split units: %d rows -> %d groups (group_key=%s), fingerprint=%s',
        n_rows, n_groups, args.group_key, data_fp,
    )
    if args.kfold is not None:
        strat_labels = () if args.no_stratify else ('PROD', 'EVT')
        train_rows, valid_rows, test_rows = split_kfold_stratified(
            rows, args.kfold, args.fold_index, args.seed,
            strat_labels=strat_labels, group_key=group_key,
        )
        logger.info(
            'K-fold: kfold=%d, fold_index=%d (test fold), valid fold=%d, '
            'group_key=%s, stratify=%s',
            args.kfold, args.fold_index,
            (args.fold_index + 1) % args.kfold, args.group_key,
            not args.no_stratify,
        )
    else:
        train_rows, valid_rows, test_rows = split_train_valid_test(
            rows, args.valid_ratio, args.test_ratio, args.seed,
            group_key=group_key,
        )

    if args.smoke:
        train_rows = train_rows[:100]
        valid_rows = valid_rows[:25]
        test_rows = test_rows[:50]
        args.epochs = 1
        logger.info('Smoke mode active')

    if args.data_extra_train_jsonl:
        extra_rows = load_jsonl(args.data_extra_train_jsonl)
        train_rows = train_rows + extra_rows
        logger.info(
            'Appended %d rows from extra JSONL to TRAIN only '
            '(valid/test from --data split unchanged): total train=%d',
            len(extra_rows), len(train_rows),
        )

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

    trim_offsets = not args.legacy_no_offset_trim
    if not trim_offsets:
        logger.info('Offset trim DISABLED (legacy as-is reproduction)')
    logger.info('Tokenizing and aligning labels...')
    train_features, _ = encode_dataset(
        train_rows, tokenizer, label2id, args.lang, args.max_length, trim_offsets
    )
    valid_features, valid_offsets = encode_dataset(
        valid_rows, tokenizer, label2id, args.lang, args.max_length, trim_offsets
    )
    test_features, test_offsets = encode_dataset(
        test_rows, tokenizer, label2id, args.lang, args.max_length, trim_offsets
    )

    # Boundary-aware weight (B/I/O 차등) — None 이면 표준 CE
    cw = None
    if args.boundary_b_weight is not None or args.boundary_i_weight is not None:
        cw = boundary_weights_tensor(
            label2id,
            w_b=args.boundary_b_weight if args.boundary_b_weight is not None else 1.0,
            w_i=args.boundary_i_weight if args.boundary_i_weight is not None else 1.0,
            w_o=1.0,
        )
        logger.info(
            'Boundary weights: B=%s, I=%s, O=1.0',
            args.boundary_b_weight, args.boundary_i_weight,
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
            precision=args.precision,
            train_seed=args.train_seed,
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
            init_model_path=s1_best,
            precision=args.precision,
            train_seed=args.train_seed,
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
            precision=args.precision,
            train_seed=args.train_seed,
        )
        logger.info('Train time: %.1fs', elapsed)

    is_kfold = args.kfold is not None

    # 신뢰도 임계값(confidence threshold) 운영점: valid 에서 per-class
    # 임계값 fit(저장) 또는 외부 load. 임계값은 모델 종속 — 이 run 모델의
    # valid 신뢰도로 fit 해야 정합적이라 --fit-threshold 가
    # --confidence-thresholds 보다 우선한다.
    conf_thr = None
    conf_meta = None
    if args.fit_threshold:
        logger.info('Fitting confidence thresholds on VALID split...')
        valid_scored = evaluate_model(
            model_path=best_dir,
            eval_features=valid_features,
            eval_offsets=valid_offsets,
            eval_rows=valid_rows,
            id2label=id2label,
            return_spans=True,
            capture_scores=True,
        )
        conf_thr = fit_thresholds(
            valid_scored['gold_spans_list'],
            valid_scored['pred_spans_list'],
            target_p=args.threshold_target,
            target_r=args.threshold_target,
        )
        conf_meta = {
            'conf_key': 'conf_mean', 'fit_set': 'valid',
            'target_p': args.threshold_target,
            'target_r': args.threshold_target,
            'types': list(DEFAULT_THRESHOLD_TYPES),
        }
        thr_path = os.path.join(output_dir, 'thresholds.json')
        save_thresholds(thr_path, conf_thr, meta=conf_meta)
        logger.info('Saved confidence thresholds: %s %s',
                    thr_path, conf_thr)
    elif args.confidence_thresholds:
        conf_thr = load_thresholds(args.confidence_thresholds)
        conf_meta = {'source': args.confidence_thresholds}
        logger.info('Loaded confidence thresholds: %s %s',
                    args.confidence_thresholds, conf_thr)

    logger.info('Evaluating on test split...')
    conf_active = conf_thr is not None
    metrics = evaluate_model(
        model_path=best_dir,
        eval_features=test_features,
        eval_offsets=test_offsets,
        eval_rows=test_rows,
        id2label=id2label,
        return_spans=is_kfold or conf_active,
        capture_scores=conf_active,
    )

    if is_kfold:
        # kfold pooled 평가용: test_rows 의 식별 필드와 gold/pred span 을 zip.
        # group 은 선언된 그룹 키의 값으로, pooling 단계 cross-fold 누출 검증의
        # 1순위 기준이다. group_key=none 이면 None 이라 누출은 미측정으로 남는다.
        # orig·text 는 그룹 키가 없는 레거시 예측 파일용 fallback 기준.
        preds_out = [
            {
                'id': row.get('id'),
                'text': row['text'],
                'orig': row.get('orig'),
                'group': row.get(group_key) if group_key else None,
                'group_key': args.group_key,
                'gold_spans': gold,
                'pred_spans': pred,
            }
            for row, gold, pred in zip(
                test_rows,
                metrics['gold_spans_list'],
                metrics['pred_spans_list'],
            )
        ]
        preds_path = os.path.join(output_dir, 'test_predictions.json')
        with open(preds_path, 'w', encoding='utf-8') as f:
            json.dump(preds_out, f, indent=2, ensure_ascii=False)
        logger.info('Saved test predictions: %s', preds_path)

    strict_m = metrics['strict']
    relaxed_m = metrics['relaxed']

    # 신뢰도 임계값 운영점 메트릭: 임계값 적용 결과. raw strict 는 baseline
    # 으로 보존(아래 'overall' 키)하고, 운영점은 별도 'confidence_threshold'
    # 블록에 둔다.
    conf_block = None
    if conf_active:
        filtered = apply_thresholds(metrics['pred_spans_list'], conf_thr)
        op = compute_offset_span_f1(metrics['gold_spans_list'], filtered)
        conf_block = {
            'thresholds': conf_thr,
            'meta': conf_meta,
            'overall_baseline': strict_m['overall'],
            'overall_operating': op['overall'],
            'per_entity_operating': op['per_entity'],
        }
        logger.info(
            'Confidence-threshold operating point: P=%.4f R=%.4f F1=%.4f '
            '(baseline F1=%.4f)',
            op['overall']['precision'], op['overall']['recall'],
            op['overall']['f1'], strict_m['overall']['f1'],
        )

    summary = {
        'lang': args.lang,
        'model_name': model_name,
        'data_path': data_path,
        'data_extra_train_jsonl': args.data_extra_train_jsonl,
        'train_samples': len(train_rows),
        'valid_samples': len(valid_rows),
        'test_samples': len(test_rows),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_length': args.max_length,
        'valid_ratio': args.valid_ratio,
        'test_ratio': args.test_ratio,
        'kfold': args.kfold,
        'fold_index': args.fold_index if is_kfold else None,
        # 두 분할 경로 모두에 적용되므로 kfold 여부와 무관하게 기록한다.
        # n_groups == n_rows 면 그룹 보호가 no-op 이었다는 뜻이다.
        'group_key': args.group_key,
        'n_rows': n_rows,
        'n_groups': n_groups,
        # 비교 유효성 게이트의 자 — 내용 지문 + fold 멤버십 결정 요소
        'data_fingerprint': data_fp,
        'stratify': stratify,
        'seed': args.seed,
        'train_seed': args.train_seed,
        'precision': args.precision,
        'offset_trim': trim_offsets,
        'metric_for_best': 'eval_loss',
        'curriculum': args.curriculum,
        'curriculum_stage1_epochs': args.curriculum_stage1_epochs if args.curriculum else None,
        'train_time_sec': round(elapsed, 1),
        # 게이트 lock-in (strict) 호환 키
        'overall': strict_m['overall'],
        'per_entity': strict_m['per_entity'],
        # 명시 키 (strict + SemEval'13 Partial)
        'overall_strict': strict_m['overall'],
        'per_entity_strict': strict_m['per_entity'],
        'overall_relaxed': relaxed_m['overall'],
        'per_entity_relaxed': relaxed_m['per_entity'],
    }
    if conf_block:
        summary['confidence_threshold'] = conf_block

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}")
    print(f"  {args.lang.upper()} BERT NER (canonical 10-class)")
    print(f"{'='*72}")
    print(f"  Model    : {model_name}")
    print(f"  Data     : {data_path}")
    print(f"  Train/Valid/Test : {len(train_rows)} / {len(valid_rows)} / {len(test_rows)}  (epochs={args.epochs})")
    print(f"  Train time: {elapsed:.1f}s")
    show_strict = args.metric_mode in ('strict', 'both')
    show_relaxed = args.metric_mode in ('relaxed', 'both')
    if show_strict:
        _print_metrics_block('strict', strict_m)
    if show_relaxed:
        _print_metrics_block("relaxed (SemEval'13 Partial)", relaxed_m)
    if conf_block:
        op = conf_block['overall_operating']
        bl = conf_block['overall_baseline']
        print(f"\n  [confidence-threshold operating point] thresholds="
              f"{conf_thr}")
        print(f"  P={op['precision']:.4f}  R={op['recall']:.4f}  "
              f"F1={op['f1']:.4f}   (baseline F1={bl['f1']:.4f}, "
              f"ΔF1={op['f1'] - bl['f1']:+.4f})")
    print(f"\n  Saved to {metrics_path}")

    strict_o = strict_m['overall']
    print(f"\n  relaxed F1 {relaxed_m['overall']['f1']:.4f}  "
          f"(Δ vs strict {relaxed_m['overall']['f1'] - strict_o['f1']:+.4f})")


def _print_metrics_block(label: str, m: dict) -> None:
    """strict / relaxed 메트릭 블록 콘솔 출력."""
    o = m['overall']
    print(f"\n  [{label}]")
    print(f"  F1={o['f1']:.4f}  Precision={o['precision']:.4f}  "
          f"Recall={o['recall']:.4f}  Support={o['support']}")
    print(f"  {'Entity':<14} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
    print(f"  {'-'*52}")
    for etype, em in sorted(m['per_entity'].items(), key=lambda x: -x[1]['f1']):
        print(f"  {etype:<14} {em['f1']:>8.4f} {em['precision']:>8.4f} "
              f"{em['recall']:>8.4f} {em['support']:>8}")


if __name__ == '__main__':
    main()
