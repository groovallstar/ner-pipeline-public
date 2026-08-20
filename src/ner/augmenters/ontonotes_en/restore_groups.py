"""주입 산출물에 `orig`·`split` 를 되돌린다.

## 왜 필요한가

`augmenters.pii` 의 `Record` 스키마는 `text`·`entities`·`id` 세 필드만 담는다
(`pii/schema.py`). 그래서 주입을 지나면 변환 단계가 심어둔 `orig`(형제 묶음
키)와 `split`(원본 배정)이 **사라진다.** JA 는 `--group-key id` 를 써서 이
문제를 겪지 않았지만 EN 은 `orig` 를 쓴다 — OntoNotes 가 같은 문장을 여러 번
담기 때문이다(train 59,924 행 중 고유 55,154).

`orig` 가 없으면 후속 학습에서 `--group-key` 를 줄 수단이 없고, 같은 문장에서
나온 행이 train 과 test 로 갈려 점수가 부푼다.

## 왜 id 로 이을 수 있나

주입은 한 입력에서 최대 한 행을 낸다(실패 시 0 행). 그래서 `id` 가 산출물에서
그대로 유일하며 원본과 1:1 로 붙는다 — 실측에서 세 split 모두 원본에 없는
`id` 가 0 이었다.

주입은 문장을 다시 쓰므로 `text` 는 달라진다. 그래서 **`orig` 로 묶는 것이
여전히 옳다** — 같은 원문에서 나온 행들은 주입 결과가 서로 달라도 한 unit 이다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 주입이 버리는, 변환 단계가 심어둔 필드.
CARRIED_FIELDS = ('orig', 'split')


def build_index(source: Path) -> dict[str, dict[str, str]]:
    """변환 산출물에서 `id` → 옮길 필드 표를 만든다.

    원본 행에 옮길 필드가 없으면 중단한다. 빠진 것을 건너뛰면 표에 빈 칸이
    실리고, `restore` 의 `row.update({})` 는 아무 일도 안 하면서 성공으로
    보고한다 — 복원했다는 로그가 찍힌 채 `orig` 없는 행이 학습에 들어간다.
    `restore` 가 모르는 `id` 에 대해 막는 것과 같은 위험이라 같은 자리에서
    막는다.
    """
    index: dict[str, dict[str, str]] = {}
    with source.open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in CARRIED_FIELDS if key not in row]
            if missing:
                raise KeyError(
                    f'source record {row.get("id")!r} lacks {missing}; '
                    f'cannot build the restore index'
                )
            index[row['id']] = {key: row[key] for key in CARRIED_FIELDS}
    return index


def restore(injected: Path, index: dict[str, dict[str, str]],
            out_path: Path) -> dict[str, int]:
    """주입 산출물에 필드를 되돌려 쓴다.

    원본에 없는 `id` 를 만나면 중단한다 — 조용히 건너뛰면 그 행만 그룹 키가
    없는 채로 학습에 들어가고, `validate_group_key` 는 그때서야 터진다.

    **임시 파일에 쓴 뒤 옮긴다.** 제자리 갱신(`out_path == injected`)이 흔한
    쓰임인데, 읽는 파일을 그대로 쓰기 모드로 열면 읽기 전에 비워져 원본이
    사라진다. 중간에 터져도 원본이 남는 이점도 같이 온다.
    """
    written = 0
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    try:
        with injected.open(encoding='utf-8') as src, \
                tmp_path.open('w', encoding='utf-8') as out:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = row.get('id')
                if record_id not in index:
                    raise KeyError(
                        f'injected record id {record_id!r} not found in '
                        f'source; cannot restore {CARRIED_FIELDS}'
                    )
                row.update(index[record_id])
                out.write(json.dumps(row, ensure_ascii=False))
                out.write('\n')
                written += 1
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {'written': written, 'source_rows': len(index)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ontonotes_en.restore_groups',
        description='Restore orig/split fields that PII injection drops.',
    )
    p.add_argument('--source-dir', type=Path, required=True,
                   help='Directory holding the converted {split}.jsonl files')
    p.add_argument('--injected-dir', type=Path, required=True,
                   help='Directory holding the PII-injected {split}.jsonl')
    p.add_argument('--splits', nargs='+',
                   default=['train', 'valid', 'test'],
                   help='Splits to restore (default: all three)')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(message)s',
    )
    args = build_parser().parse_args(argv)
    for split in args.splits:
        source = args.source_dir / f'{split}.jsonl'
        injected = args.injected_dir / f'{split}.jsonl'
        index = build_index(source)
        stats = restore(injected, index, injected)
        logger.info('%s: restored %s on %d rows (source %d)',
                    split, ','.join(CARRIED_FIELDS),
                    stats['written'], stats['source_rows'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
