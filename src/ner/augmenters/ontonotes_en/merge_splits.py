"""PII 주입까지 끝난 split 파일 셋을 학습용 단일 JSONL 로 합친다.

사용 예::

    python -m ner.augmenters.ontonotes_en.merge_splits \
        --input-dir data/ontonotes_en/pii \
        --output data/ontonotes_en/pii_all.jsonl

## 왜 합치나

`classifier` 는 파일 하나를 받아 스스로 쪼갠다(`--data`). ko·ja·vi 가 모두 그
경로를 쓰므로, 네 언어를 같은 자로 재려면 en 도 한 파일이어야 한다. 원본
split 파일을 그대로 넘기는 경로를 새로 뚫는 쪽은 분할 코드를 건드리게 되고,
그 코드는 비교 유효성의 '자' 라 함부로 바꾸지 않는다.

## 왜 `split` 필드를 떨어뜨리나

합치면 그 필드가 한 파일 안에서 세 값을 갖는다. `validate_group_key` 는 값이
적으면서 선언한 키의 그룹을 가르지 않는 필드를 "더 강한 그룹 키" 로 보고
거부하는데, `split` 이 정확히 그 모양이다 — 같은 `orig` 의 행은 언제나 같은
split 에 있어(경계를 하나도 가르지 않는다) 3 그룹짜리 후보로 잡히고,
`--group-key orig` 가 통째로 막힌다.

재분할이 전제라 원 split 은 학습에 쓰이지 않으며, 어느 split 이었는지는
`id`·`orig` 접두사(`en-train-`/`en-valid-`/`en-test-`)에 그대로 남는다. 즉
필드를 빼도 정보는 잃지 않는다.

## 검사

합치기 전에 세 가지를 본다 — span 자기정합(문자열 오프셋이 실제 그 글자를
가리키나), 라벨이 canonical 10 종 안인가, `orig` 가 모든 행에 있고 split
사이에서 값이 겹치지 않는가. 하나라도 어긋나면 비영으로 끝나고 산출물을
남기지 않는다. 검사를 나중에 따로 돌리게 두면 안 돌린 산출물이 섞인다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from ner.augmenters.ontonotes_en.mapping import PRODUCED_LABELS

logger = logging.getLogger(__name__)

# 병합 순서. 학습 전에 다시 섞이지만 파일 내용을 결정적으로 만든다.
SPLITS: tuple[str, ...] = ('train', 'valid', 'test')

# 주입 단계가 채우는 PII 4 종. 변환이 만드는 6 종(`PRODUCED_LABELS`)과 합쳐
# canonical 10 종 평면이 된다.
INJECTED_LABELS: frozenset[str] = frozenset(
    {'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'}
)
CANONICAL_LABELS: frozenset[str] = PRODUCED_LABELS | INJECTED_LABELS

# 병합본에서 떨어뜨리는 필드. 이유는 모듈 docstring.
DROPPED_FIELDS: tuple[str, ...] = ('split',)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ontonotes_en.merge_splits',
        description='Merge per-split canonical JSONL files into a single '
                    'training input, dropping the split field.',
    )
    p.add_argument('--input-dir', type=Path, required=True,
                   help='Directory holding {train,valid,test}.jsonl')
    p.add_argument('--output', type=Path, required=True,
                   help='Path of the merged JSONL to write')
    p.add_argument('--splits', nargs='+', default=list(SPLITS),
                   choices=list(SPLITS),
                   help='Splits to merge (default: all)')
    return p


def check_row(row: dict, where: str) -> list[str]:
    """한 행의 span 자기정합과 라벨 유효성을 본다.

    반환은 사람이 읽는 문제 문자열 목록이며, 비어 있으면 통과다.
    """
    problems: list[str] = []
    text = row.get('text')
    if not isinstance(text, str):
        return [f'{where}: missing or non-string text field']
    if not row.get('orig'):
        problems.append(f'{where}: missing or empty group key "orig"')
    for i, entity in enumerate(row.get('entities', ())):
        label = entity.get('label')
        if label not in CANONICAL_LABELS:
            problems.append(
                f'{where}: entity[{i}] label {label!r} is outside the '
                f'canonical 10-type set'
            )
        start, end = entity.get('start_char'), entity.get('end_char')
        if not isinstance(start, int) or not isinstance(end, int):
            problems.append(f'{where}: entity[{i}] has non-integer offsets')
            continue
        if not 0 <= start < end <= len(text):
            problems.append(
                f'{where}: entity[{i}] offsets ({start}, {end}) fall outside '
                f'text of length {len(text)}'
            )
            continue
        if text[start:end] != entity.get('text'):
            problems.append(
                f'{where}: entity[{i}] offsets point at '
                f'{text[start:end]!r} but the entity text is '
                f'{entity.get("text")!r}'
            )
    return problems


def load_split(path: Path, split: str) -> tuple[list[dict], list[str]]:
    """한 split 파일을 읽어 `split` 필드를 뗀 행 목록과 문제 목록을 준다."""
    rows: list[dict] = []
    problems: list[str] = []
    with path.open(encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problems += check_row(row, f'{split}.jsonl:{lineno}')
            for field in DROPPED_FIELDS:
                row.pop(field, None)
            rows.append(row)
    return rows, problems


def check_group_keys_disjoint(by_split: dict[str, list[dict]]) -> list[str]:
    """split 끼리 `orig` 값을 공유하지 않는지 본다.

    공유하면 병합 뒤 서로 다른 split 의 행이 한 그룹으로 묶여, 재분할 때
    의도치 않은 덩어리가 생긴다. 접두사가 split 을 담고 있어 실제로는 일어나지
    않지만, 조용히 어긋나면 누출로 드러나므로 여기서 크게 실패시킨다.
    """
    problems: list[str] = []
    seen: dict[object, str] = {}
    for split, rows in by_split.items():
        for row in rows:
            key = row.get('orig')
            owner = seen.setdefault(key, split)
            if owner != split:
                problems.append(
                    f'group key {key!r} appears in both {owner} and {split}'
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(message)s',
    )
    args = build_parser().parse_args(argv)
    args.splits = list(dict.fromkeys(args.splits))

    by_split: dict[str, list[dict]] = {}
    problems: list[str] = []
    for split in args.splits:
        path = args.input_dir / f'{split}.jsonl'
        if not path.exists():
            logger.error('Missing input file: %s', path)
            return 1
        rows, split_problems = load_split(path, split)
        by_split[split] = rows
        problems += split_problems
        logger.info('%s: %d rows, %d spans', split, len(rows),
                    sum(len(r['entities']) for r in rows))

    problems += check_group_keys_disjoint(by_split)

    if problems:
        logger.error('%d integrity problems found; nothing was written',
                     len(problems))
        for line in problems[:20]:
            logger.error('  %s', line)
        if len(problems) > 20:
            logger.error('  ... and %d more', len(problems) - 20)
        return 1

    merged = [row for split in args.splits for row in by_split[split]]
    labels: Counter = Counter(
        entity['label'] for row in merged for entity in row['entities']
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 검사를 지난 뒤에 제자리로 옮긴다 — 곧장 쓰면 중간에 실패해도 완결돼
    # 보이는 파일이 남아 다음 사람이 검사를 지난 것으로 읽는다.
    staged = args.output.with_suffix('.jsonl.partial')
    with staged.open('w', encoding='utf-8') as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=False))
            out.write('\n')
    staged.replace(args.output)

    groups = len({row['orig'] for row in merged})
    print(f'\nmerged -> {args.output}')
    print(f'  rows      : {len(merged):,}')
    print(f'  groups    : {groups:,} (orig)')
    print(f'  spans     : {sum(labels.values()):,}')
    print(f'  dropped   : {", ".join(DROPPED_FIELDS)}')
    print('  labels    :')
    for label, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        print(f'    {label:<12} {count:>7,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
