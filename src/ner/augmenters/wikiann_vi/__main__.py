"""WikiANN-vi → canonical 5종 재라벨 CLI.

사용 예::

    python -m ner.augmenters.wikiann_vi \
        --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
        --base-url http://localhost:8081/v1 \
        --split test --max-samples 1000 \
        --output data/wikiann_vi/gemma_test.jsonl

JSONL 출력 스키마(레코드별):
    {id, text, gold_spans, gold_spans_8type, relabel_model}

파일·필드의 ``8type`` 리터럴은 데이터 호환성을 위해 유지한다
(의미는 canonical 5종: PER·LOC·ORG·PROD·EVT).
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from datasets import ClassLabel, load_dataset

from ner.augmenters.wikiann_vi.relabel_8type import Relabeler
from ner.labelers.vi.dataset_loader import bio_to_offset_spans


def _load_wikiann_hf(
    hf_name: str, hf_config: str, split: str,
    max_samples: int | None, cache_dir: str | None,
) -> list[dict]:
    """HF WikiANN 원본(BIO 3종)을 재라벨 입력용 레코드로 변환.

    labelers/vi 로더는 canonical 덤프 전용이므로, 재라벨 파이프라인
    고유의 HF 원본 읽기는 본 CLI가 직접 책임진다.
    """
    import os
    cache = cache_dir or os.environ.get(
        'HF_DATASETS_CACHE', '/work/.huggingface/datasets',
    )
    hf_dataset = load_dataset(
        hf_name, hf_config, split=split,
        cache_dir=cache, trust_remote_code=False,
    )
    if max_samples is not None:
        hf_dataset = hf_dataset.select(
            range(min(max_samples, len(hf_dataset)))
        )
    label_feature = hf_dataset.features['ner_tags'].feature
    is_class_label = isinstance(label_feature, ClassLabel)
    records: list[dict] = []
    for i, row in enumerate(hf_dataset):
        tokens = list(row['tokens'])
        raw_tags = row['ner_tags']
        tags = (
            [label_feature.int2str(t) for t in raw_tags]
            if is_class_label
            else [str(t) for t in raw_tags]
        )
        text, gold_spans = bio_to_offset_spans(tokens, tags)
        records.append({
            'id': str(i),
            'text': text,
            'gold_spans': gold_spans,
        })
    return records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m ner.augmenters.wikiann_vi',
        description='Relabel WikiANN-vi to canonical 5-type schema',
    )
    parser.add_argument(
        '--hf-name', default='unimelb-nlp/wikiann',
        help='HuggingFace dataset name',
    )
    parser.add_argument(
        '--hf-config', default='vi',
        help='HuggingFace dataset config (language)',
    )
    parser.add_argument(
        '--split', default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split',
    )
    parser.add_argument(
        '--max-samples', type=int, default=1000,
        help='Limit number of records',
    )
    parser.add_argument(
        '--base-url', default='http://localhost:8081/v1',
        help='vLLM OpenAI-compatible endpoint',
    )
    parser.add_argument(
        '--model', default='cyankiwi/gemma-4-31B-it-AWQ-8bit',
        help='vLLM model id',
    )
    parser.add_argument(
        '--concurrency', type=int, default=16,
        help='Parallel LLM request limit',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Group N records per BATCH prompt call (1=SINGLE)',
    )
    parser.add_argument(
        '--max-tokens', type=int, default=2048,
        help='max_tokens per LLM completion',
    )
    parser.add_argument(
        '--timeout', type=float, default=120.0,
        help='Per-request timeout seconds',
    )
    parser.add_argument(
        '--output', required=True,
        help='Output JSONL path',
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    return parser


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _summarize(records: list[dict]) -> None:
    total = len(records)
    empty = sum(1 for r in records if not r.get('gold_spans_8type'))
    error = sum(1 for r in records if r.get('relabel_error'))
    type_counter: Counter[str] = Counter()
    for rec in records:
        for span in rec.get('gold_spans_8type', []):
            type_counter[span.get('type', '')] += 1

    print('=== Relabel Summary ===')
    print(f'Total records: {total}')
    print(f'Records with >=1 5-type span: {total - empty}')
    print(f'Records with relabel error: {error}')
    print(f'Total 5-type spans: {sum(type_counter.values())}')
    print('Per-type counts:')
    for type_name, count in type_counter.most_common():
        print(f'  {type_name}: {count}')


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    print(f'Loading {args.hf_name}/{args.hf_config} split={args.split}')
    records = _load_wikiann_hf(
        hf_name=args.hf_name,
        hf_config=args.hf_config,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=None,
    )
    print(f'Loaded {len(records)} records')

    relabeler = Relabeler(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        timeout=args.timeout,
        batch_size=args.batch_size,
    )

    print(
        f'Relabeling with {args.model} @ {args.base_url} '
        f'(concurrency={args.concurrency})'
    )
    started = perf_counter()
    relabeled = relabeler.relabel_sync(records)
    elapsed = perf_counter() - started
    print(f'Relabel done in {elapsed:.1f}s')
    print(
        f'Token usage: prompt={relabeler.total_prompt_tokens} '
        f'completion={relabeler.total_completion_tokens}'
    )

    out_path = Path(args.output)
    _write_jsonl(out_path, relabeled)
    print(f'Wrote {len(relabeled)} records -> {out_path}')

    _summarize(relabeled)
    return 0


if __name__ == '__main__':
    sys.exit(main())
