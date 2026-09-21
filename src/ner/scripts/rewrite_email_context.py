"""디스크 코퍼스의 EMAIL 주소 일부를 공백이 아닌 글자 뒤로 붙인다.

바이트 BPE 모델(en)이 괄호·따옴표·`mailto:`·콜론 뒤 주소의 앞쪽을 놓치는
결함을 학습 분포에서 메운다. 감싸는 규칙과 비율은
`ner.augmenters.pii.email_context` 가 단일 출처다.

사용 예::

    python -m ner.scripts.rewrite_email_context \\
        --input data/ontonotes_en/origin.jsonl \\
        --output data/ontonotes_en/origin.jsonl --seed 42

## 감싸기만 하는 이유

LLM 주입을 다시 돌리면 문맥·다른 라벨·주소가 전부 함께 바뀌어, 재학습 전후
차이를 붙은 문맥 때문이라고 말할 근거가 사라진다. 그래서 주소와 다른 엔티티의
표면은 한 자도 바꾸지 않고, 주소 둘레에 글자를 더한 만큼 뒤 엔티티 오프셋만
민다. 입력·출력 지문은 원장(`<output>.email-context.json`)에 남긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

from ner.augmenters.pii.email_context import (
    EMAIL_CUE_COLON_RATE,
    EMAIL_WRAP_WEIGHTS,
    glue_email_contexts,
    glued_share,
)

logger = logging.getLogger(__name__)


def sha256_of_text(text: str) -> str:
    """치환 전후를 견주기 위한 지문."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Glue part of the EMAIL spans in a disk corpus to '
                    'non-space context (brackets, quotes, mailto:, colon)'
    )
    p.add_argument('--input', required=True, type=Path,
                   help='Input JSONL corpus')
    p.add_argument('--output', required=True, type=Path,
                   help='Output JSONL corpus (may equal --input)')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducible rewrites')
    p.add_argument('--ledger', type=Path, default=None,
                   help='Where to write the JSON ledger '
                        '(default: <output>.email-context.json)')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute the ledger without writing the corpus')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)

    source = args.input.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rewritten, forms = [], Counter()
    for row in rows:
        new, applied = glue_email_contexts(row, rng)
        rewritten.append(new)
        forms.update(applied)
    payload = ''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in rewritten)

    ledger = {
        'seed': args.seed,
        'input': str(args.input),
        'input_sha256': sha256_of_text(source),
        'output': str(args.output),
        'output_sha256': sha256_of_text(payload),
        'target': {
            'wrap_weights': EMAIL_WRAP_WEIGHTS,
            'cue_colon_rate': EMAIL_CUE_COLON_RATE,
        },
        'rows': len(rows),
        'email_spans': sum(forms.values()),
        'forms': dict(sorted(forms.items())),
        'glued_share': {
            'before': glued_share(rows),
            'after': glued_share(rewritten),
        },
    }
    ledger_path = args.ledger or args.output.with_suffix(
        args.output.suffix + '.email-context.json')

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
        {k: ledger[k] for k in ('rows', 'email_spans', 'forms', 'glued_share')},
        ensure_ascii=False,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
