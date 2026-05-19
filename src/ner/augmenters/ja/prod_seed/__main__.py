"""PROD 도메인 휴리스틱 seed oversample CLI.

예 (V1 production 셋업, base extra = S7N2 negative):
    python -m ner.augmenters.ja.prod_seed \\
        --input data/stockmark/pii_all_phonediv.jsonl \\
        --base-extra data/stockmark/pii_neg_aug_N2_extra.jsonl \\
        --output data/stockmark/pii_extra_s7n2_prodfnaware_heur_N5.jsonl \\
        --oversample 5 --include-domain --extra-only

    python -m ner.classifier --lang ja \\
        --data data/stockmark/pii_all_phonediv.jsonl \\
        --data-extra-train-jsonl \\
            data/stockmark/pii_extra_s7n2_prodfnaware_heur_N5.jsonl \\
        --boundary-b-weight 1.5 --boundary-i-weight 1.2 \\
        --output-dir results/classifier/ja_sweep/prod_seed_v1
"""

import argparse
import logging

from ner.augmenters.ja.prod_seed.seed_selector import oversample_to_jsonl


def main():
    parser = argparse.ArgumentParser(
        description=(
            'PROD domain-heuristic seed oversampler for JA NER classifier.'
        ),
    )
    parser.add_argument('--input', required=True,
                        help='Source JSONL (augmenters contract).')
    parser.add_argument('--output', required=True,
                        help='Output JSONL path.')
    parser.add_argument(
        '--base-extra', default=None,
        help='Optional base extra JSONL prepended verbatim (e.g. S7N2 '
             'negative-oversample extra) before PROD-pos extra rows.',
    )
    parser.add_argument(
        '--oversample', type=int, default=5,
        help='Number of copies per candidate sentence (default: 5). With '
             '--extra-only, N copies are written; otherwise 1 original + '
             '(N-1) extra copies.',
    )
    parser.add_argument(
        '--include-domain', action='store_true', default=True,
        help='Include domain-heuristic seed pool (law/book/food/transit/'
             'music). Enabled by default.',
    )
    parser.add_argument(
        '--no-include-domain', dest='include_domain',
        action='store_false',
        help='Disable domain-heuristic seed pool.',
    )
    parser.add_argument(
        '--include-long', action='store_true', default=False,
        help='Also include LONG-surface seed pool (PROD surface >= '
             '--long-min-length, excluding famous-media keywords).',
    )
    parser.add_argument(
        '--long-min-length', type=int, default=6,
        help='Minimum PROD surface length for the LONG seed pool '
             '(default: 6).',
    )
    parser.add_argument(
        '--extra-only', action='store_true', default=True,
        help='Output ONLY the extra copies (default). Combine with '
             'classifier --data-extra-train-jsonl for leak-free evaluation.',
    )
    parser.add_argument(
        '--no-extra-only', dest='extra_only', action='store_false',
        help='Output original rows + extra copies (mind split leak).',
    )
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    stats = oversample_to_jsonl(
        input_path=args.input,
        output_path=args.output,
        oversample=args.oversample,
        use_domain=args.include_domain,
        use_long=args.include_long,
        long_min_length=args.long_min_length,
        base_extra_path=args.base_extra,
        extra_only=args.extra_only,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print()
    print(f'Domain-heuristic seed sentences: {stats["domain_seed_sentences"]}')
    print(f'LONG-surface seed sentences: {stats["long_seed_sentences"]}')
    print(f'Total candidate sentences: {stats["candidates"]}')
    print(f'Base extra rows (prepended): {stats["base_extra_rows"]}')
    print(f'PROD-pos extra rows: {stats["extra_rows"]}')
    print(f'Output rows: {stats["output_rows"]}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
