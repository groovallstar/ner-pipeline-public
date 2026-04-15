"""PII 주입 CLI.

예:
    python -m augmenters.pii --source stockmark --lang ja \\
        --output /data/ner/ja_stockmark_pii.jsonl --n-samples 100
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from augmenters.pii.config import InjectionConfig
from augmenters.pii.injector import PIIInjector
from augmenters.pii.stats import compute_stats

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Inject synthetic PII into NER datasets'
    )
    p.add_argument(
        '--source', choices=['stockmark', 'jsonl', 'hf'], required=True,
        help='Input dataset source',
    )
    p.add_argument('--input', type=str, default=None,
                   help='Input path (for --source jsonl)')
    p.add_argument('--hf-name', type=str, default=None,
                   help='HF dataset name (for --source hf)')
    p.add_argument('--hf-split', type=str, default='train')
    p.add_argument('--lang', choices=['ja', 'vi'], default='ja')
    p.add_argument('--output', type=str, required=True,
                   help='Output JSONL path')
    p.add_argument('--n-samples', type=int, default=None,
                   help='Limit the number of input samples')
    p.add_argument('--pii-max', type=int, default=3,
                   help='Max PII injected per sample (truncates density)')
    p.add_argument('--seed', type=int, default=42)
    return p


def _truncate_density(
    density: dict[int, float], pii_max: int
) -> dict[int, float]:
    """density 분포를 pii_max 상한까지 잘라낸 뒤 재정규화한다."""
    kept = {k: v for k, v in density.items() if k <= pii_max}
    if not kept:
        return {0: 1.0}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


def _load_records(args: argparse.Namespace):
    from augmenters.pii.loaders import load_hf, load_jsonl, load_stockmark
    if args.source == 'stockmark':
        return load_stockmark(
            split='train', max_samples=args.n_samples, seed=args.seed,
        )
    if args.source == 'jsonl':
        if not args.input:
            raise SystemExit('--input is required for --source jsonl')
        return load_jsonl(args.input, max_samples=args.n_samples)
    if args.source == 'hf':
        if not args.hf_name:
            raise SystemExit('--hf-name is required for --source hf')
        return load_hf(
            args.hf_name, split=args.hf_split,
            max_samples=args.n_samples,
        )
    raise SystemExit(f'unknown source: {args.source}')


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args(argv)

    cfg = InjectionConfig(lang=args.lang, seed=args.seed)
    cfg.density = _truncate_density(cfg.density, args.pii_max)
    cfg.validate()

    logger.info('Loading records from source=%s', args.source)
    records = _load_records(args)
    logger.info('Loaded %d records', len(records))

    injector = PIIInjector(cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    injected = []
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in injector.inject_dataset(records):
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + '\n')
            injected.append(rec)
    logger.info('Wrote %d records to %s', len(injected), out_path)

    stats = compute_stats(injected)
    stats_path = out_path.with_name(out_path.stem + '.stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info('Wrote stats to %s', stats_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
