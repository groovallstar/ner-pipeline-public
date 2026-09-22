"""디스크 코퍼스의 VI DAT 표면형을 목표 분포로 다시 만든다.

주입기(`augmenters/pii/generators/vi.py`)는 고쳤지만 이미 만들어져 있는
`data/wikiann_vi/origin.jsonl` 은 옛 분포 그대로다. 이 스크립트가 그 어긋남을 닫는다.

사용 예::

    python -m ner.scripts.rewrite_vi_dat_surface \\
        --input data/wikiann_vi/origin.jsonl \\
        --output data/wikiann_vi/origin.jsonl --seed 42

## 표면형만 바꾸는 이유

LLM 자연 주입을 다시 돌리면 문맥·다른 라벨·도메인이 전부 함께 바뀌고, 그
코퍼스로 다시 학습한 모델의 점수 차이를 DAT 표면형 때문이라고 말할 근거가
사라진다. 그래서 DAT span 의 표면 하나만 바꾸고 다른 엔티티와 문맥은 한 자도
건드리지 않는다. 바뀐 문자열 길이만큼 뒤따르는 엔티티 오프셋을 민다.

## `ngày` 를 다루는 두 규칙

앞에 붙는 시간 낱말은 그것을 떼고도 날짜로 읽히면 span 밖에 둔다. 현 코퍼스가
`ngày` + 숫자 날짜 3,708 자리를 예외 없이 그렇게 두고 있어 관례를 따른다.

- **서술형 `D tháng M năm Y`** — span 만 바꾸고 앞의 `ngày` 는 그대로 둔다.
  결과는 `được cấp ngày 26 tháng 5 năm 1988` 이다.
- **월-연 `tháng M năm Y`** — 앞의 `ngày` 를 먹고 들어간다. 남겨두면
  `ngày tháng 5 năm 1988` 이 되어 "날 5월 1988년" 이라 말이 안 되기 때문이다.
  결과는 `được cấp tháng 5 năm 1988` 이다. 먹을 `ngày` 가 없는 자리에서는
  숫자 표기로 남긴다.

## 생성기와 다른 점

`vi.generate_dat` 은 월-연 표기를 내지 않는다. 생성기는 값 문자열만 돌려주고 그
앞에 `ngày` 가 붙을지는 LLM 주입기가 정하는데, 생성기가 그것을 통제하지 못해
깨진 문맥이 나올 수 있기 때문이다. 앞 글자까지 함께 보는 이 스크립트만 월-연
표기를 안전하게 만들 수 있어 그 형식은 여기에만 있다.

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
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DAT_LABEL = 'DAT'

# 형식별 목표 비율. 숫자 표기를 과반으로 남기는 것은 그것이 서식·증서·기록에
# 실제로 쓰이는 표기이고, 현 DAT metric 이 그 표기 위에서 나온 값이라 너무
# 깎으면 무회귀 기준이 흔들리기 때문이다.
DAT_FORM_WEIGHTS = {'numeric': 60, 'descriptive_full': 30, 'month_year': 10}

_NUMERIC_PATTERNS = (
    (re.compile(r'^(\d{2})/(\d{2})/(\d{4})$'), ('d', 'm', 'y')),
    (re.compile(r'^(\d{4})-(\d{2})-(\d{2})$'), ('y', 'm', 'd')),
    (re.compile(r'^(\d{2})-(\d{2})-(\d{4})$'), ('d', 'm', 'y')),
)

_DESCRIPTIVE_FULL = re.compile(r'^\d{1,2} tháng \d{1,2} năm \d{4}$')
_MONTH_YEAR = re.compile(r'^tháng \d{1,2} năm \d{4}$', re.IGNORECASE)

# span 바로 앞에 붙은 시간 낱말. `ngày` 는 먹을 수 있어 따로 본다.
_NGAY_BEFORE = re.compile(r'ngày[ \t]+$', re.IGNORECASE)
_TIME_WORD_BEFORE = re.compile(r'(?:năm|tháng)[ \t]+$', re.IGNORECASE)

_FALLBACK_KEYS = (
    'unparsed',
    'preceded_by_time_word',
    'month_year_without_ngay',
    'blocked_by_previous_entity',
)


def parse_numeric_date(value: str) -> tuple[int, int, int] | None:
    """숫자 날짜 표기를 (일, 월, 연)으로 읽는다. 못 읽으면 None."""
    for pattern, order in _NUMERIC_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue
        parts = dict(zip(order, (int(g) for g in match.groups())))
        return parts['d'], parts['m'], parts['y']
    return None


def dat_form(value: str) -> str:
    """DAT 표면 하나가 어느 형식인지 돌려준다."""
    if parse_numeric_date(value) is not None:
        return 'numeric'
    if _DESCRIPTIVE_FULL.match(value):
        return 'descriptive_full'
    if _MONTH_YEAR.match(value):
        return 'month_year'
    return 'other'


def form_rates(values: list[str]) -> dict[str, float]:
    """DAT 표면 표본의 형식별 비율."""
    if not values:
        return {}
    n = len(values)
    forms = [dat_form(v) for v in values]
    return {
        key: forms.count(key) / n
        for key in ('numeric', 'descriptive_full', 'month_year', 'other')
    }


def _draw_form(rng: random.Random, weights: dict[str, int]) -> str:
    keys = ['numeric', 'descriptive_full', 'month_year']
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def rewrite_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
    weights: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """레코드 목록의 DAT 표면형을 치환한 새 목록과 통계를 돌려준다."""
    weights = weights or DAT_FORM_WEIGHTS
    out: list[dict[str, Any]] = []
    before: list[str] = []
    after: list[str] = []
    fallback = dict.fromkeys(_FALLBACK_KEYS, 0)

    for row in rows:
        row = copy.deepcopy(row)
        entities = row.get('entities', [])
        if not any(e.get('label') == DAT_LABEL for e in entities):
            out.append(row)
            continue

        text = row['text']
        shift = 0
        prev_start = -1
        prev_end = 0
        for entity in entities:
            if entity['start_char'] < prev_start:
                raise ValueError(
                    f'entities are not sorted by offset: id={row.get("id")}'
                )
            prev_start = entity['start_char']
            start = entity['start_char'] + shift
            end = entity['end_char'] + shift
            entity['start_char'], entity['end_char'] = start, end
            if entity['label'] != DAT_LABEL:
                prev_end = end
                continue

            old = entity['text']
            if text[start:end] != old:
                raise ValueError(
                    f'span text mismatch: id={row.get("id")} '
                    f'gold={old!r} text={text[start:end]!r}'
                )
            before.append(old)

            parsed = parse_numeric_date(old)
            if parsed is None:
                fallback['unparsed'] += 1
                after.append(old)
                prev_end = end
                continue

            day, month, year = parsed
            form = _draw_form(rng, weights)
            prefix = text[:start]

            if form != 'numeric' and _TIME_WORD_BEFORE.search(prefix):
                # `năm 26 tháng 5 năm 1988` 처럼 같은 낱말이 두 번 난다.
                fallback['preceded_by_time_word'] += 1
                form = 'numeric'

            if form == 'month_year':
                ngay = _NGAY_BEFORE.search(prefix)
                if ngay is None:
                    fallback['month_year_without_ngay'] += 1
                    form = 'numeric'
                elif ngay.start() < prev_end:
                    fallback['blocked_by_previous_entity'] += 1
                    form = 'numeric'

            if form == 'numeric':
                after.append(old)
                prev_end = end
                continue

            if form == 'descriptive_full':
                new = f'{day} tháng {month} năm {year}'
                cut_start = start
            else:
                ngay = _NGAY_BEFORE.search(prefix)
                cut_start = ngay.start()
                new = f'tháng {month} năm {year}'
                if prefix[cut_start].isupper():
                    new = new[0].upper() + new[1:]

            text = text[:cut_start] + new + text[end:]
            entity['start_char'] = cut_start
            entity['end_char'] = cut_start + len(new)
            entity['text'] = new
            shift += len(new) - (end - cut_start)
            after.append(new)
            prev_end = entity['end_char']

        row['text'] = text
        out.append(row)

    stats = {
        'rows': len(rows),
        'dat_spans': len(before),
        'before': form_rates(before),
        'after': form_rates(after),
        'fallback': fallback,
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
        description='Rewrite VI DAT surface forms in a disk corpus'
    )
    p.add_argument('--input', required=True, type=Path,
                   help='Input JSONL corpus')
    p.add_argument('--output', required=True, type=Path,
                   help='Output JSONL corpus (may equal --input)')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducible rewrites')
    p.add_argument('--ledger', type=Path, default=None,
                   help='Where to write the JSON ledger '
                        '(default: <output>.dat-surface.json)')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute the ledger without writing the corpus')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)

    source = args.input.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    rewritten, stats = rewrite_rows(rows, random.Random(args.seed))
    payload = dump_rows(rewritten)

    ledger = {
        'lang': 'vi',
        'seed': args.seed,
        'input': str(args.input),
        'input_sha256': sha256_of_text(source),
        'output': str(args.output),
        'output_sha256': sha256_of_text(payload),
        'target': {'form_weights': DAT_FORM_WEIGHTS},
        **stats,
    }
    ledger_path = args.ledger or args.output.with_suffix(
        args.output.suffix + '.dat-surface.json'
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
        {k: ledger[k]
         for k in ('rows', 'dat_spans', 'before', 'after', 'fallback')},
        ensure_ascii=False,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
