"""환각 부정 예시 oversample CLI.

학습 흐름:
    1. classifier error_analysis 로 production 모델 진단
    2. 본 CLI 로 보강 jsonl 생성 (--extra-only 권장)
    3. classifier 학습에 --data-extra-train-jsonl 로 주입

예 (이슈 #58 v3 셋업, JA production 후보):
    python -m ner.augmenters.ja_negative \\
        --input data/stockmark/pii_all.jsonl \\
        --diagnosis results/classifier/ja_sweep/v3_boundary/error_analysis.json \\
        --output data/stockmark/pii_neg_aug_N2_extra.jsonl \\
        --oversample 2 --auto-exclude-ambiguous --extra-only

    python -m ner.classifier --lang ja \\
        --data data/stockmark/pii_all.jsonl \\
        --data-extra-train-jsonl data/stockmark/pii_neg_aug_N2_extra.jsonl \\
        --boundary-b-weight 1.5 --boundary-i-weight 1.2 \\
        --output-dir results/classifier/ja_sweep/s7_promote
"""

import argparse
import logging

from ner.augmenters.ja_negative.oversampler import oversample_to_jsonl


def main():
    parser = argparse.ArgumentParser(
        description='Hallucination-negative oversampler for JA NER classifier.',
    )
    parser.add_argument('--input', required=True,
                        help='Source JSONL (augmenters contract).')
    parser.add_argument('--diagnosis', required=True,
                        help='error_analysis.json from classifier --with-diagnosis.')
    parser.add_argument('--output', required=True,
                        help='Output JSONL path.')
    parser.add_argument(
        '--oversample', type=int, default=2,
        help='Each candidate sentence is included N times in non-extra-only '
             'output (1 original + N-1 extra copies). With --extra-only, only '
             'the N-1 extra copies are written. Default: 2.',
    )
    parser.add_argument(
        '--auto-exclude-ambiguous', action='store_true',
        help='Drop seeds whose surface ever appears as entity in TRAIN '
             'split (avoids recall regression from over-suppressing real '
             'entities).',
    )
    parser.add_argument(
        '--seed-pred-types', default=None,
        help='Comma-separated entity types to keep as seed source '
             '(e.g. PER,LOC,ORG,PROD,EVT for NER-only). Default: all types.',
    )
    parser.add_argument(
        '--extra-only', action='store_true',
        help='Output ONLY the extra copies (not the original rows). '
             'Combine with classifier --data-extra-train-jsonl for leak-free '
             'evaluation.',
    )
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    seed_pred_types = None
    if args.seed_pred_types:
        seed_pred_types = {
            t.strip() for t in args.seed_pred_types.split(',') if t.strip()
        }

    stats = oversample_to_jsonl(
        input_path=args.input,
        diagnosis_path=args.diagnosis,
        output_path=args.output,
        oversample=args.oversample,
        auto_exclude_ambiguous=args.auto_exclude_ambiguous,
        seed_pred_types=seed_pred_types,
        extra_only=args.extra_only,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print()
    print(f'Seeds initial: {stats["seeds_initial"]}')
    print(f'Seeds excluded (ambiguous): {stats["seeds_excluded"]}')
    print(f'Seeds final: {stats["seeds_final"]}')
    print(f'Candidate sentences: {stats["candidates"]}')
    print(f'Extra rows added: {stats["extra_rows"]}')
    print(f'Output rows: {stats["output_rows"]}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
