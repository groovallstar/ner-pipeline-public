"""EN OntoNotes 디스크 코퍼스를 `FAC` 판정 표에 맞추고 원장을 남긴다.

변환 계층은 표면별 `FAC` 판정을 이미 갖고 있지만(`augmenters/ontonotes_en`),
`data/ontonotes_en/` 에 이미 만들어져 있던 주입·병합 산출물은 그 판정 이전
라벨(원본 `FAC` 를 전량 `ORG`)을 갖는다. 이 스크립트가 그 어긋남을 닫는다.

사용 예::

    # 1) 변환 산출물을 현행 코드로 다시 만든다
    python -m ner.augmenters.ontonotes_en \
        --raw-dir data/ontonotes_en/raw --output-dir data/ontonotes_en

    # 2) 주입 산출물을 옮기고 병합본을 다시 만들고 원장을 쓴다
    python -m ner.scripts.migrate_en_fac_disk_corpus \
        --data-dir data/ontonotes_en

## 하는 일이 둘뿐인 이유

주입 산출물의 NER 엔티티를 `extract_spans`+`merge_entities` 로 다시 도출하면
`test_injection_replays_exactly` 가 항진명제가 된다 — 그 테스트가 바로 그
경로로 기대치를 만들어 산출물과 대조하기 때문이다. 그래서 여기서는 **라벨
제자리 치환**과 **엔티티 삭제** 둘만 하고 offset 은 한 자리도 다시 계산하지
않는다. 주입 모듈을 import 하지 않는 것 자체를 테스트가 정적으로 확인한다.

## 대상을 어떻게 정하나

낡은 디스크 파일을 참조하지 않고 **원천 + 판정 표**에서 다시 도출한다.
디스크 파일을 기준으로 삼으면 그 파일이 이미 손상돼 있어도 알 수 없다.

판정 표의 키는 원본 토큰의 공백 조인인데 주입 산출물에는 자연문 표면만
남으므로, 표를 `text` 에 직접 먹이면 복원이 손대는 표면(`Vermont - Slauson`·
`U.S. 460`)이 통째로 안 걸린다. 그래서 원천을 변환기와 같은 경로로 지나
`(행 id, 자연문 표면)` 을 얻고, 그 쌍으로 주입 산출물의 행을 찾는다.

**`id` 는 행만 정하고 행 안 어느 엔티티인지는 안 정한다.** `(id, offset)` 은
주입이 문장을 다시 써 어긋나고, `(id, text)` 는 공존 표면에서 모호하다 —
`Wall Street` 가 한 행에 `FAC` 유래와 원본 `ORG` 유래로 함께 있으면 손대기
전 라벨도 `text` 도 같다. 그래서 판정 뒤에도 같은 `text` 의 `ORG` 가 그 행에
남는 자리는 **모호**로 보고 건드리지 않으며, 원장에 남긴다.

## 원장이 적는 범위

`pii/{split}.jsonl` 과 병합본에 **실제로 적용한 변경**만 적는다. 변환 쪽
(`{split}.jsonl`)까지 적으면 주입이 떨군 gold 만큼 어긋나 개수 등식이
깨진다. 변환 산출물은 원천 + 코드에서 결정적으로 다시 만들어지므로 지문만
남겨 **재현성 앵커**로 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ner.augmenters.ontonotes_en import merge_splits
from ner.augmenters.ontonotes_en.__main__ import SPLIT_FILES, convert_one_split
from ner.augmenters.ontonotes_en.convert import (
    BY_SURFACE_TYPES,
    decode_bio,
    fac_surface,
    load_id2label,
)
from ner.augmenters.ontonotes_en.detokenize import detokenize
from ner.augmenters.ontonotes_en.mapping import FAC_VERDICTS, resolve

logger = logging.getLogger(__name__)

# 원장의 기본 자리. `data/` 는 gitignore 라 원장이 커밋되는 유일한 기록이다.
DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / 'augmenters' / 'ontonotes_en' / 'data' / 'fac_disk_migration_ledger.json'
)

# 마이그레이션 전 주입 산출물이 갖고 있던 라벨. 원본 `FAC` 를 전량 `ORG` 로
# 태우던 판이라 손댈 대상은 모두 여기서 출발한다.
LABEL_BEFORE = 'ORG'

SPLITS: tuple[str, ...] = ('train', 'valid', 'test')


@dataclass(frozen=True)
class Target:
    """판정 표가 `ORG` 아닌 답을 낸 원본 `FAC` span 하나."""

    split: str
    id: str
    token_surface: str
    verdict: str        # 'LOC' (경로 → 이동) 또는 'DROP' (비-entity → 제거)
    start_char: int     # 변환 산출물 기준. 주입 산출물 기준이 아니다
    end_char: int
    text: str


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_rows(rows: list[dict]) -> str:
    """JSONL 직렬화 — 코퍼스를 쓰는 모든 자리와 같은 형태여야 한다.

    지문이 파일 바이트의 해시라, 손대지 않은 행이 한 글자라도 달라지면
    역적용이 마이그레이션 전 파일을 재현하지 못한다.
    """
    return ''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in rows
    )


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def derive_targets(
    raw_dir: Path,
) -> tuple[list[Target], dict[str, Counter]]:
    """원천 + 판정 표에서 손댈 span 과 "행에 남는 `ORG` 표면" 을 함께 낸다.

    두 번째 값이 모호성 판정의 재료다 — 판정 뒤에도 같은 `text` 의 `ORG` 가
    그 행에 남으면, 주입 산출물에서 둘을 가를 수단이 없다.
    """
    id2label = load_id2label(raw_dir / 'label.json')
    targets: list[Target] = []
    surviving_org: dict[str, Counter] = defaultdict(Counter)

    for split, filenames in SPLIT_FILES.items():
        index = 0
        for filename in filenames:
            with (raw_dir / filename).open(encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens, tags = row['tokens'], row['tags']
                    text, offsets = detokenize(tokens)
                    record_id = f'en-{split}-{index:06d}'
                    index += 1
                    for tok_start, tok_end, src_type in decode_bio(
                        tags, id2label,
                    ):
                        surface = fac_surface(tokens, tok_start, tok_end)
                        label = resolve(src_type, surface)
                        start = offsets[tok_start][0]
                        end = offsets[tok_end][1]
                        if label == LABEL_BEFORE:
                            surviving_org[record_id][text[start:end]] += 1
                        if src_type not in BY_SURFACE_TYPES:
                            continue
                        verdict = FAC_VERDICTS[surface]
                        if verdict == LABEL_BEFORE:
                            continue
                        targets.append(Target(
                            split=split, id=record_id, token_surface=surface,
                            verdict=verdict, start_char=start, end_char=end,
                            text=text[start:end],
                        ))
    return targets, surviving_org


@dataclass
class Plan:
    """한 split 에 실제로 적용할 변경과, 적용하지 않기로 한 자리."""

    move: list[dict]
    remove: list[dict]
    ambiguous: list[dict]
    absent: list[dict]


def plan_split(
    targets: list[Target],
    surviving_org: dict[str, Counter],
    rows: list[dict],
) -> Plan:
    """행 `id` + 행 안 유일 대응으로 손댈 엔티티 인덱스를 정한다.

    인덱스는 **손대기 전 행** 기준이다. 제거가 인덱스를 밀기 때문에 적용은
    내림차순, 역적용은 오름차순으로 돌아야 한다(`apply_to_rows`).
    """
    by_id = {row['id']: row for row in rows}
    plan = Plan(move=[], remove=[], ambiguous=[], absent=[])

    groups: dict[tuple[str, str], list[Target]] = defaultdict(list)
    for target in targets:
        groups[(target.id, target.text)].append(target)

    for (record_id, text), group in groups.items():
        def record(bucket: list[dict], **extra) -> None:
            for target in group:
                bucket.append({**asdict(target), **extra})

        if surviving_org[record_id][text]:
            # 같은 표면의 `ORG` 가 그 행에 남는다 — 주입 산출물에서 어느 쪽이
            # 어느 쪽인지 가를 수단이 없다. 건드리지 않고 기록만 한다.
            record(plan.ambiguous, why='same-text ORG stays in the row')
            continue
        row = by_id.get(record_id)
        hits = [] if row is None else [
            i for i, entity in enumerate(row['entities'])
            if entity['label'] == LABEL_BEFORE and entity['text'] == text
        ]
        if not hits:
            record(plan.absent, row_present=row is not None)
            continue
        if len(hits) != len(group):
            # 개수가 어긋나면 어느 자리를 손댈지가 임의 선택이 된다.
            record(plan.ambiguous,
                   why=f'{len(group)} verdict(s) but {len(hits)} ORG in row')
            continue
        for target, entity_index in zip(group, hits):
            entry = {**asdict(target), 'entity_index': entity_index,
                     'before': LABEL_BEFORE}
            if target.verdict == 'DROP':
                plan.remove.append(entry)
            else:
                entry['after'] = target.verdict
                plan.move.append(entry)
    return plan


def apply_to_rows(rows: list[dict], plan_move, plan_remove) -> None:
    """제자리 적용 — 라벨 치환 먼저, 제거는 인덱스 내림차순으로 (제자리 수정).

    치환을 먼저 하는 것은 제거가 뒤 인덱스를 밀기 때문이다. 원장 인덱스는
    손대기 전 행 기준이라, 순서를 바꾸면 엉뚱한 자리를 손댄다.
    """
    by_id = {row['id']: row for row in rows}
    for entry in plan_move:
        entity = by_id[entry['id']]['entities'][entry['entity_index']]
        if entity['label'] != entry['before'] or entity['text'] != entry['text']:
            raise ValueError(f'move target moved under us: {entry}')
        entity['label'] = entry['after']
    per_row: dict[str, list[int]] = defaultdict(list)
    for entry in plan_remove:
        per_row[entry['id']].append(entry['entity_index'])
    for record_id, indexes in per_row.items():
        entities = by_id[record_id]['entities']
        for i in sorted(indexes, reverse=True):
            del entities[i]


def reverse_apply_to_rows(rows: list[dict], plan_move, plan_remove) -> None:
    """역적용 — 제거를 인덱스 오름차순으로 되꽂고, 라벨을 되돌린다.

    `apply_to_rows` 의 정확한 역이다. 이것이 마이그레이션 전 파일을 **바이트
    단위로** 재현해야 원장의 전 지문이 증거가 된다 — 지문이 없으면 역적용이
    원장 자신을 재현하는 정의상 참이 된다.
    """
    by_id = {row['id']: row for row in rows}
    per_row: dict[str, list[dict]] = defaultdict(list)
    for entry in plan_remove:
        per_row[entry['id']].append(entry)
    for record_id, entries in per_row.items():
        entities = by_id[record_id]['entities']
        for entry in sorted(entries, key=lambda e: e['entity_index']):
            entities.insert(entry['entity_index'], {
                'label': entry['before'],
                'start_char': entry['pii_start_char'],
                'end_char': entry['pii_end_char'],
                'text': entry['text'],
            })
    for entry in plan_move:
        entity = by_id[entry['id']]['entities'][entry['entity_index']]
        entity['label'] = entry['before']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.scripts.migrate_en_fac_disk_corpus',
        description='Re-label the EN OntoNotes PII corpus to match the FAC '
                    'verdict table and write a fingerprinted ledger.',
    )
    p.add_argument('--data-dir', type=Path, required=True,
                   help='Directory holding raw/, pii/ and the merged corpus')
    p.add_argument('--merged-name', default='origin.jsonl',
                   help='Filename of the merged corpus inside --data-dir')
    p.add_argument('--ledger', type=Path, default=DEFAULT_LEDGER,
                   help=f'Where to write the ledger (default: {DEFAULT_LEDGER})')
    p.add_argument('--dry-run', action='store_true',
                   help='Report what would change and write nothing')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(message)s',
    )
    args = build_parser().parse_args(argv)
    raw_dir = args.data_dir / 'raw'
    pii_dir = args.data_dir / 'pii'
    merged_path = args.data_dir / args.merged_name

    targets, surviving_org = derive_targets(raw_dir)
    verdict_counts = Counter(target.verdict for target in targets)
    logger.info('verdict table asks for %d change(s): %s',
                len(targets), dict(verdict_counts))

    # 변환 산출물이 현행 코드의 산물인지 먼저 본다. 낡아 있으면 주입 쪽만
    # 옮겨도 재생 대조가 어긋나고, 그 실패는 여기서 멈추는 편이 읽기 쉽다.
    id2label = load_id2label(raw_dir / 'label.json')
    conversion_sha: dict[str, str] = {}
    for split in SPLITS:
        records, problems, _, _ = convert_one_split(raw_dir, split, id2label)
        if problems:
            logger.error('%s: conversion reports %d problem(s)',
                         split, len(problems))
            return 1
        expected = hashlib.sha256(
            dump_rows(records).encode('utf-8')
        ).hexdigest()
        path = args.data_dir / f'{split}.jsonl'
        if not path.exists() or sha256_of(path) != expected:
            logger.error(
                '%s is not what the current converter produces. Run '
                '`python -m ner.augmenters.ontonotes_en --raw-dir %s '
                '--output-dir %s` first.', path, raw_dir, args.data_dir)
            return 1
        conversion_sha[split] = expected

    before_sha: dict[str, str] = {}
    after_sha: dict[str, str] = {}
    ledger_move: list[dict] = []
    ledger_remove: list[dict] = []
    ledger_ambiguous: list[dict] = []
    ledger_absent: list[dict] = []
    staged: list[tuple[Path, Path]] = []

    for split in SPLITS:
        path = pii_dir / f'{split}.jsonl'
        rows = read_rows(path)
        if dump_rows(rows) != path.read_text(encoding='utf-8'):
            # 왕복이 안 되면 손대지 않은 행까지 바이트가 달라져, 역적용이
            # 마이그레이션 전 파일을 재현하지 못한다.
            logger.error('%s does not round-trip through json.dumps', path)
            return 1
        before_sha[f'pii/{split}.jsonl'] = sha256_of(path)

        plan = plan_split(
            [t for t in targets if t.split == split], surviving_org, rows,
        )
        # 원장은 주입 산출물 기준 offset 도 함께 적는다 — 제거한 엔티티를
        # 되꽂으려면 그 행 안에서의 자리가 필요하기 때문이다.
        by_id = {row['id']: row for row in rows}
        for entry in plan.move + plan.remove:
            entity = by_id[entry['id']]['entities'][entry['entity_index']]
            entry['pii_start_char'] = entity['start_char']
            entry['pii_end_char'] = entity['end_char']

        logger.info('%s: move=%d remove=%d ambiguous=%d absent=%d',
                    split, len(plan.move), len(plan.remove),
                    len(plan.ambiguous), len(plan.absent))
        ledger_move += plan.move
        ledger_remove += plan.remove
        ledger_ambiguous += plan.ambiguous
        ledger_absent += plan.absent

        if args.dry_run:
            continue
        apply_to_rows(rows, plan.move, plan.remove)
        # 검사를 지난 뒤에 제자리로 옮긴다 — 곧장 쓰면 중간에 실패해도
        # 완결돼 보이는 파일이 남는다.
        partial = path.with_suffix('.jsonl.partial')
        partial.write_text(dump_rows(rows), encoding='utf-8')
        staged.append((partial, path))

    if args.dry_run:
        print(f'\ndry run — nothing written\n'
              f'  move      : {len(ledger_move):,}\n'
              f'  remove    : {len(ledger_remove):,}\n'
              f'  ambiguous : {len(ledger_ambiguous):,}\n'
              f'  absent    : {len(ledger_absent):,}')
        return 0

    before_sha[args.merged_name] = sha256_of(merged_path)
    for partial, path in staged:
        partial.replace(path)

    if merge_splits.main([
        '--input-dir', str(pii_dir), '--output', str(merged_path),
    ]) != 0:
        logger.error('merge failed after migration; the merged corpus is '
                     'now older than the per-split files')
        return 1

    for split in SPLITS:
        after_sha[f'pii/{split}.jsonl'] = sha256_of(pii_dir / f'{split}.jsonl')
    after_sha[args.merged_name] = sha256_of(merged_path)

    ledger = {
        'why': 'EN OntoNotes FAC 판정 표(#236)를 디스크 코퍼스에 적용한 기록. '
               'data/ 는 gitignore 이고 주입 텍스트는 재생 불가라, 이 원장이 '
               '"어떤 span 이 왜 손대졌는가" 의 유일한 커밋된 기록이다.',
        'verdict_table': 'src/ner/augmenters/ontonotes_en/data/fac_labels.json',
        'generated_by': 'python -m ner.scripts.migrate_en_fac_disk_corpus',
        'merged_name': args.merged_name,
        'verdict_counts': dict(sorted(verdict_counts.items())),
        'conversion_sha256': conversion_sha,
        'before_sha256': before_sha,
        'after_sha256': after_sha,
        'move': ledger_move,
        'remove': ledger_remove,
        'not_applied': {
            'ambiguous': ledger_ambiguous,
            'absent_from_pii': ledger_absent,
        },
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8',
    )

    print(f'\nledger -> {args.ledger}')
    print(f'  move      : {len(ledger_move):,} ORG -> LOC')
    print(f'  remove    : {len(ledger_remove):,} ORG dropped by the table')
    print(f'  ambiguous : {len(ledger_ambiguous):,} (left untouched)')
    print(f'  absent    : {len(ledger_absent):,} (injection had dropped them)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
