"""OntoNotes5 → canonical JSONL 변환 CLI.

사용 예::

    python -m ner.augmenters.ontonotes_en \
        --raw-dir data/ontonotes_en/raw \
        --output-dir data/ontonotes_en

원본은 `tner/ontonotes5` 의 `dataset/` 파일을 그대로 받아 `--raw-dir` 에
둔다(`train00..03.json`·`valid.json`·`test.json`·`label.json`). `data/` 는
gitignore 되므로 산출물은 커밋되지 않는다 — 재생성 경로가 이 CLI 다.

## split 을 필드가 아니라 파일로 가르는 이유

원본 split 을 그대로 쓰기로 했으므로 어느 split 이었는지가 데이터에 남아야
한다. 그런데 `split` 을 **한 파일 안의 필드**로 두면 값이 3 종뿐이라
`validate_group_key` 가 이를 "더 강한 그룹 키 후보" 로 잡아 올바른 형제 키
(`orig`)를 밀어낸다 — `split` 은 파티션 라벨이지 형제 표시가 아닌데도.
split 별로 파일을 가르면 파일 안에서 `split` 이 상수가 되어(그룹 수 1) 후보
검사에서 빠지고, 필드는 그대로 남아 정보도 잃지 않는다. 출하 번들 관례
(`data/{train,valid,test}.jsonl`)와도 같은 모양이다.

변환과 동시에 엔티티↔원본 토큰 대조를 돌리고, 하나라도 어긋나면 비영으로
끝난다. 검사를 나중에 따로 돌리게 두면 안 돌린 산출물이 섞인다.

`FAC` 판정 표의 피복도 같은 자리에서 본다 — 표에 없는 표면은 `resolve` 가
span 단위로 세우고, 표에만 있고 코퍼스에 없는 줄은 끝에서 관측 집합과 맞춰
잡는다. 뒤엣것만 `--partial-corpus` 로 끌 수 있고(픽스처처럼 부분 입력일
때), 껐다는 사실이 산출 meta 에 남는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from ner.augmenters.ontonotes_en.convert import (
    assign_text_groups,
    check_entity_token_match,
    convert_record,
    count_fac_surfaces,
    count_source_spans,
    load_id2label,
    source_types,
)
from ner.augmenters.ontonotes_en.mapping import (
    FacCoverageError,
    assert_exhaustive,
    assert_fac_coverage,
)

logger = logging.getLogger(__name__)

# split → 원본 파일. train 은 원천이 4 개로 쪼개 배포한다.
SPLIT_FILES: dict[str, tuple[str, ...]] = {
    'train': ('train00.json', 'train01.json', 'train02.json', 'train03.json'),
    'valid': ('valid.json',),
    'test': ('test.json',),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ontonotes_en',
        description='Convert OntoNotes5 (tner distribution) to canonical '
                    'NER JSONL with char-offset spans, one file per split.',
    )
    p.add_argument('--raw-dir', type=Path, required=True,
                   help='Directory holding the tner ontonotes5 dataset files')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Directory to write {train,valid,test}.jsonl into')
    p.add_argument('--splits', nargs='+', default=list(SPLIT_FILES),
                   choices=list(SPLIT_FILES),
                   help='Splits to convert (default: all)')
    p.add_argument('--partial-corpus', action='store_true',
                   help='Input is a subset of the corpus (a fixture), so '
                        'skip the check that every FAC verdict-table row is '
                        'observed. The forward check (a surface with no '
                        'verdict) still stops the run, and the skip is '
                        'recorded in conversion_meta.json')
    return p


def convert_one_split(
    raw_dir: Path, split: str, id2label: dict[int, str],
) -> tuple[list[dict], list[str], Counter, Counter]:
    """한 split 을 레코드로 옮기고 대조 결과·원본 span 수·표면 수를 준다.

    원본 span 수는 매핑표를 거치지 않고 태그에서 직접 센다. 쓰임은 테스트가
    그 수를 고정하는 것(`GOLDEN_SOURCE_SPANS`)과 `conversion_meta.json` 에
    남기는 것이다 — canonical 쪽 골든과 함께 두면 실패가 어느 단계에서
    났는지 갈린다.

    네 번째 값은 표면별 판정 타입(`FAC`)의 관측 표면이다. 판정 표에만 있고
    코퍼스에 없는 줄은 span 이 하나도 안 지나가 변환 도중에는 안 걸리므로,
    끝에서 양쪽을 맞추려면 관측 집합이 따로 있어야 한다.
    """
    records: list[dict] = []
    problems: list[str] = []
    source_counts: Counter = Counter()
    fac_surfaces: Counter = Counter()
    index = 0
    for filename in SPLIT_FILES[split]:
        with (raw_dir / filename).open(encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_counts.update(count_source_spans(row['tags'], id2label))
                fac_surfaces.update(count_fac_surfaces(
                    row['tokens'], row['tags'], id2label,
                ))
                record = convert_record(
                    row['tokens'], row['tags'], id2label,
                    record_id=f'en-{split}-{index:06d}', split=split,
                )
                problems += check_entity_token_match(
                    record, row['tokens'], row['tags'], id2label,
                )
                records.append(record)
                index += 1
    assign_text_groups(records)
    return records, problems, source_counts, fac_surfaces


def report_duplicate_texts(by_split: dict[str, list[dict]]) -> None:
    """중복 문장 실태를 매 실행 보고한다.

    OntoNotes 공식 split 이 원래 가진 성질이라 제거하지 않는다. 근거는 "외부
    공개 수치와 비교" 가 **아니다** — 18→6 매핑으로 라벨 공간이 이미
    달라져 published OntoNotes NER F1 과 나란히 못 놓는다. 남은 근거는 공식
    split 이 재현·인용에 유리하고 노출이 test span 의 0.81% 로 작다는 것이며,
    규모가 커지면 재검토 대상이다. 그래서 매 실행 눈앞에 둔다.
    """
    train_texts = {r['text'] for r in by_split.get('train', ())}
    print('\nduplicate texts (inherited from the official split):')
    for split, records in by_split.items():
        unique = len({r['text'] for r in records})
        line = (f'  {split:<6} {len(records):>7,} rows, '
                f'{unique:>7,} unique texts')
        if split != 'train' and train_texts:
            shared = [r for r in records if r['text'] in train_texts]
            spans_shared = sum(len(r['entities']) for r in shared)
            spans_all = sum(len(r['entities']) for r in records)
            ratio = spans_shared / spans_all if spans_all else 0.0
            line += (f' | also in train: {len(shared):,} rows, '
                     f'{spans_shared:,}/{spans_all:,} spans ({ratio:.2%})')
        print(line)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(message)s',
    )
    args = build_parser().parse_args(argv)
    # 같은 split 을 두 번 주면 스테이징 파일을 두 번 옮기려다 두 번째에서
    # FileNotFoundError 로 끝난다 — 산출물은 제자리에 있는데 meta 만 안 써져
    # 실패가 애매해진다. 순서는 지키고 중복만 없앤다.
    args.splits = list(dict.fromkeys(args.splits))

    id2label = load_id2label(args.raw_dir / 'label.json')
    assert_exhaustive(source_types(id2label))
    logger.info('label table: %d ids, %d source types',
                len(id2label), len(source_types(id2label)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[tuple[Path, Path]] = []
    by_split: dict[str, list[dict]] = {}
    per_split_source: dict[str, Counter] = {}
    per_split_labels: dict[str, Counter] = {}
    problems: list[str] = []
    fac_surfaces: Counter = Counter()

    for split in args.splits:
        records, split_problems, source_counts, split_surfaces = (
            convert_one_split(args.raw_dir, split, id2label)
        )
        fac_surfaces.update(split_surfaces)
        problems += split_problems
        by_split[split] = records
        labels: Counter = Counter()
        path = args.output_dir / f'{split}.jsonl'
        # 임시 파일에 쓰고 대조가 끝난 뒤에 옮긴다. 곧장 쓰면 대조가 실패해
        # 비영으로 끝나도 완결돼 보이는 산출물이 디스크에 남아, 다음 사람이
        # "검사를 지난 파일" 로 읽는다.
        staged = path.with_suffix('.jsonl.partial')
        staged_paths.append((staged, path))
        with staged.open('w', encoding='utf-8') as out:
            for record in records:
                for entity in record['entities']:
                    labels[entity['label']] += 1
                out.write(json.dumps(record, ensure_ascii=False))
                out.write('\n')
        per_split_labels[split] = labels
        per_split_source[split] = source_counts
        logger.info('%s: %d sentences, %d spans -> %s',
                    split, len(records), sum(labels.values()), path)

    # 판정 표 피복 검사 — 표에만 있고 코퍼스에 없는 줄을 잡는 자리다.
    # 반대 방향(코퍼스에만 있는 표면)은 `resolve` 가 span 단위로 이미 세웠다.
    #
    # `--partial-corpus` 는 이 방향만 끈다. 부분 입력(픽스처·표본)에서는 표의
    # 대부분이 관측되지 않는 것이 정상이라 켜 두면 늘 붉기 때문이다. 끈 사실은
    # 화면과 `conversion_meta.json` 양쪽에 남긴다 — 안 남기면 통과가
    # "검사를 지났다" 인지 "검사가 안 돌았다" 인지 구별되지 않는다.
    coverage_error: str | None = None
    if args.partial_corpus:
        logger.warning(
            'partial corpus: FAC verdict-table coverage NOT checked '
            '(rows in the table with no observed span are allowed)'
        )
    else:
        try:
            assert_fac_coverage(fac_surfaces, args.splits)
        except FacCoverageError as exc:
            coverage_error = str(exc)

    if problems or coverage_error:
        if coverage_error:
            logger.error('%s', coverage_error)
        if problems:
            logger.error('entity/token mismatch: %d', len(problems))
            for problem in problems[:20]:
                logger.error('  %s', problem)
        for staged, _ in staged_paths:
            staged.unlink(missing_ok=True)
        logger.error('no output written (staged files removed)')
        return 1
    for staged, final in staged_paths:
        staged.replace(final)

    print('\nlabel  ' + ''.join(f'{s:>10}' for s in args.splits) + '     total')
    all_labels = sorted({
        label for counter in per_split_labels.values() for label in counter
    })
    for label in all_labels:
        row = [per_split_labels[s][label] for s in args.splits]
        print(f'{label:<7}' + ''.join(f'{v:>10,}' for v in row)
              + f'{sum(row):>10,}')
    print(f'{"TOTAL":<7}'
          + ''.join(f'{sum(per_split_labels[s].values()):>10,}'
                    for s in args.splits)
          + f'{sum(sum(c.values()) for c in per_split_labels.values()):>10,}')
    # 판정이 모델 판에 걸려 있으므로 산출물에 판을 남긴다 — 골든 넘버를
    # 지키는 마찰이 근거를 갖게 하려면 무엇으로 쟀는지가 데이터에 있어야 한다.
    meta = {
        'source': str(args.raw_dir),
        'splits': list(args.splits),
        'source_span_counts': {
            split: dict(counter)
            for split, counter in per_split_source.items()
        },
        'label_counts': {
            split: dict(counter)
            for split, counter in per_split_labels.items()
        },
        'sentence_counts': {
            split: len(records) for split, records in by_split.items()
        },
        'fac_surface_counts': dict(sorted(fac_surfaces.items())),
        'fac_coverage_checked': not args.partial_corpus,
    }
    meta_path = args.output_dir / 'conversion_meta.json'
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'\nwrote {meta_path}')

    report_duplicate_texts(by_split)
    return 0


if __name__ == '__main__':
    sys.exit(main())
