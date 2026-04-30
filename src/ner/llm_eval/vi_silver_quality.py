"""VI silver(Gemma·Qwen 단독 또는 merge 결과) 의 PER/LOC/ORG 부분을
WikiANN-vi 인간 주석 gold 와 비교해 span F1 을 측정한다.

CLI 사용 예::

    python -m ner.llm_eval.vi_silver_quality \
        --silver-dir data/wikiann_vi \
        --silver-prefix gemma \
        --splits test validation train \
        --output results/vi_silver_quality_gemma.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from ner.llm_eval.wikiann_vi_gold import GOLD_TYPES, load_wikiann_vi_gold
from ner.metrics.span_metrics import compute_offset_span_f1

logger = logging.getLogger(__name__)

# silver JSONL 안에서 span 리스트가 들어 있는 키 후보 (입력 형태별).
SILVER_SPAN_KEYS = ('gold_spans_8type_merged', 'gold_spans_8type')


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_silver_spans(record: dict) -> List[dict]:
    """silver 레코드에서 span 리스트를 꺼내 PER/LOC/ORG 만 필터링."""
    spans: List[dict] = []
    for key in SILVER_SPAN_KEYS:
        if key in record:
            spans = record[key] or []
            break
    return [s for s in spans if s.get('type') in GOLD_TYPES]


def evaluate_split(
    silver_path: Path,
    split: str,
    *,
    max_samples: int | None = None,
) -> dict:
    """한 split 의 silver vs gold span F1 결과를 반환."""
    silver_records = _read_jsonl(silver_path)
    silver_by_id = {str(r['id']): r for r in silver_records}

    gold_records = load_wikiann_vi_gold(split, max_samples=max_samples)
    gold_by_id = {str(r['id']): r for r in gold_records}

    common_ids = sorted(
        set(silver_by_id) & set(gold_by_id), key=lambda x: int(x),
    )

    gold_lists: List[List[dict]] = []
    pred_lists: List[List[dict]] = []
    for rid in common_ids:
        g = gold_by_id[rid]['gold_spans']
        p = _extract_silver_spans(silver_by_id[rid])
        gold_lists.append(g)
        pred_lists.append(p)

    f1_result = compute_offset_span_f1(gold_lists, pred_lists)

    return {
        'split': split,
        'silver_path': str(silver_path),
        'matched_records': len(common_ids),
        'silver_records': len(silver_records),
        'gold_records': len(gold_records),
        'overall': f1_result['overall'],
        'per_entity': {
            t: f1_result['per_entity'].get(t, _zero_prf())
            for t in GOLD_TYPES
        },
    }


def _zero_prf() -> dict:
    return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 0}


def evaluate_silver_set(
    silver_dir: Path,
    silver_prefix: str,
    splits: List[str],
    *,
    max_samples: int | None = None,
) -> Dict[str, dict]:
    """{prefix}_{split}.jsonl 파일들을 일괄 평가."""
    results: Dict[str, dict] = {}
    for split in splits:
        path = silver_dir / f'{silver_prefix}_{split}.jsonl'
        if not path.exists():
            logger.warning('silver missing: %s', path)
            continue
        results[split] = evaluate_split(
            path, split, max_samples=max_samples,
        )
    return results


def _format_summary(results: Dict[str, dict]) -> str:
    lines: List[str] = []
    lines.append(
        f"{'split':<12} {'overall_f1':>10} {'PER_f1':>8} "
        f"{'LOC_f1':>8} {'ORG_f1':>8} {'matched':>8}"
    )
    for split, r in results.items():
        lines.append(
            f"{split:<12} {r['overall']['f1']:>10.4f} "
            f"{r['per_entity']['PER']['f1']:>8.4f} "
            f"{r['per_entity']['LOC']['f1']:>8.4f} "
            f"{r['per_entity']['ORG']['f1']:>8.4f} "
            f"{r['matched_records']:>8d}"
        )
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.llm_eval.vi_silver_quality',
        description='Evaluate VI silver vs WikiANN-vi 3-gold span F1',
    )
    p.add_argument(
        '--silver-dir', required=True,
        help='Directory containing {prefix}_{split}.jsonl files',
    )
    p.add_argument(
        '--silver-prefix', required=True,
        help='File prefix (e.g. gemma, qwen, vi_wikiann_recall)',
    )
    p.add_argument(
        '--splits', nargs='+',
        default=['test', 'validation', 'train'],
        choices=['train', 'validation', 'test'],
    )
    p.add_argument(
        '--max-samples', type=int, default=None,
        help='Subsample N gold records (debug only)',
    )
    p.add_argument(
        '--output', required=True,
        help='Output JSON path with per-split per-type results',
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    silver_dir = Path(args.silver_dir)
    results = evaluate_silver_set(
        silver_dir, args.silver_prefix, args.splits,
        max_samples=args.max_samples,
    )
    out = {
        'silver_prefix': args.silver_prefix,
        'silver_dir': str(silver_dir),
        'splits': args.splits,
        'results': results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'Wrote -> {out_path}')
    print(_format_summary(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
