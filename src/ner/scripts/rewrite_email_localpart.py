"""디스크 코퍼스의 EMAIL 로컬파트를 목표 표면형 분포로 다시 만든다.

주입기(`augmenters/pii/generators/base.py`)는 고쳤지만 이미 만들어져 있는
`data/*/origin.jsonl` 은 옛 분포 그대로다. 이 스크립트가 그 어긋남을 닫는다.

사용 예::

    python -m ner.scripts.rewrite_email_localpart \\
        --input data/ontonotes_en/origin.jsonl \\
        --output data/ontonotes_en/origin.jsonl \\
        --lang en --seed 42

## 로컬파트만 바꾸는 이유

LLM 자연 주입을 다시 돌리면 문맥·다른 라벨·도메인이 전부 함께 바뀌고, 그
코퍼스로 다시 학습한 모델의 점수 차이를 EMAIL 표면형 때문이라고 말할 근거가
사라진다. 그래서 `@` 앞 로컬파트 하나만 바꾸고 도메인·문맥·다른 엔티티는 한
자도 건드리지 않는다. 바뀐 문자열 길이만큼 뒤따르는 엔티티 오프셋을 민다.

## 멈추는 자리

gold span 이 가리키는 표면과 `text` 필드가 어긋나거나, 엔티티가 오프셋
오름차순이 아니면 예외로 멈춘다. 어긋난 코퍼스를 조용히 고치면 그 손상이
치환 결과에 섞여 들어가 어느 쪽이 원인인지 가릴 수 없다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import random

from ner.augmenters.pii.generators.base import (
    LOCAL_PART_CAPITALIZED_RATE,
    LOCAL_PART_DIGIT_PREFIX_RATE,
    LOCAL_PART_DIGIT_SUFFIX_RATE,
    LOCAL_PART_FORM_WEIGHTS,
    email_name_hint,
    random_local_part,
)

logger = logging.getLogger(__name__)

EMAIL_LABEL = 'EMAIL'


def local_part_rates(values: list[str]) -> dict[str, float]:
    """로컬파트 표본의 표면형 비율."""
    if not values:
        return {}
    n = len(values)
    return {
        'digit_prefix': sum(v[:1].isdigit() for v in values) / n,
        'dot': sum('.' in v for v in values) / n,
        'capitalized': sum(any(c.isupper() for c in v) for v in values) / n,
        'any_digit': sum(
            any(c.isdigit() for c in v) for v in values
        ) / n,
        'dotless_lower': sum(
            '.' not in v and v == v.lower() for v in values
        ) / n,
    }


def rewrite_rows(
    rows: list[dict[str, Any]], lang: str, rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """레코드 목록의 EMAIL 로컬파트를 치환한 새 목록과 통계를 돌려준다."""
    out: list[dict[str, Any]] = []
    before: list[str] = []
    after: list[str] = []

    for row in rows:
        row = copy.deepcopy(row)
        entities = row.get('entities', [])
        if not any(e.get('label') == EMAIL_LABEL for e in entities):
            out.append(row)
            continue

        text = row['text']
        shift = 0
        prev_start = -1
        for entity in entities:
            if entity['start_char'] < prev_start:
                raise ValueError(
                    f'entities are not sorted by offset: id={row.get("id")}'
                )
            prev_start = entity['start_char']
            start = entity['start_char'] + shift
            end = entity['end_char'] + shift
            entity['start_char'], entity['end_char'] = start, end
            if entity['label'] != EMAIL_LABEL:
                continue

            old = entity['text']
            if text[start:end] != old:
                raise ValueError(
                    f'span text mismatch: id={row.get("id")} '
                    f'gold={old!r} text={text[start:end]!r}'
                )
            if '@' not in old:
                raise ValueError(
                    f'EMAIL span without a domain: id={row.get("id")} '
                    f'value={old!r}'
                )

            domain = old.rsplit('@', 1)[1]
            local = random_local_part(email_name_hint(lang, rng), rng)
            new = f'{local}@{domain}'
            text = text[:start] + new + text[end:]
            entity['end_char'] = start + len(new)
            entity['text'] = new
            shift += len(new) - len(old)
            before.append(old.rsplit('@', 1)[0])
            after.append(local)

        row['text'] = text
        out.append(row)

    stats = {
        'rows': len(rows),
        'email_spans': len(before),
        'before': local_part_rates(before),
        'after': local_part_rates(after),
    }
    return out, stats


def read_rows(path: Path) -> list[dict[str, Any]]:
    """JSONL 을 읽는다."""
    with path.open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dump_rows(rows: list[dict[str, Any]]) -> str:
    """JSONL 문자열로 직렬화한다."""
    return ''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in rows
    )


def sha256_of_text(text: str) -> str:
    """치환 전후를 견주기 위한 지문."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Rewrite EMAIL local parts in a disk corpus'
    )
    p.add_argument('--input', required=True, type=Path,
                   help='Input JSONL corpus')
    p.add_argument('--output', required=True, type=Path,
                   help='Output JSONL corpus (may equal --input)')
    p.add_argument('--lang', required=True,
                   choices=['en', 'ko', 'ja', 'vi'],
                   help='Locale that supplies the name pools')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducible rewrites')
    p.add_argument('--ledger', type=Path, default=None,
                   help='Where to write the JSON ledger '
                        '(default: <output>.email-rewrite.json)')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute the ledger without writing the corpus')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)

    source = args.input.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    rewritten, stats = rewrite_rows(rows, args.lang, random.Random(args.seed))
    payload = dump_rows(rewritten)

    ledger = {
        'lang': args.lang,
        'seed': args.seed,
        'input': str(args.input),
        'input_sha256': sha256_of_text(source),
        'output': str(args.output),
        'output_sha256': sha256_of_text(payload),
        'target': {
            'form_weights': LOCAL_PART_FORM_WEIGHTS,
            'capitalized_rate': LOCAL_PART_CAPITALIZED_RATE,
            'digit_prefix_rate': LOCAL_PART_DIGIT_PREFIX_RATE,
            'digit_suffix_rate': LOCAL_PART_DIGIT_SUFFIX_RATE,
        },
        **stats,
    }
    ledger_path = args.ledger or args.output.with_suffix(
        args.output.suffix + '.email-rewrite.json'
    )

    if args.dry_run:
        logger.info('dry run, corpus not written')
    else:
        args.output.write_text(payload, encoding='utf-8')
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        logger.info('wrote %s and %s', args.output, ledger_path)

    logger.info(json.dumps(
        {k: ledger[k] for k in ('rows', 'email_spans', 'before', 'after')},
        ensure_ascii=False,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
