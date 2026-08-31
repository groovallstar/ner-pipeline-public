"""원본 `FAC` 표면 전량과 각 표면의 문맥 문장을 뽑는다.

판정 표(`augmenters/ontonotes_en/data/fac_labels.json`)를 만들 때 쓰는
입력 생성기다. 표면만으로는 실체를 못 가리는 자리가 있어(`Broadway` 는
거리이자 극장가, `Vermont - Slauson` 은 교차로 이름의 쇼핑센터) 판정에
문맥이 필요하다.

표면 키는 **원본 토큰의 공백 조인**이다. 자연문 복원(`detokenize`)이
바뀌어도 키가 안 흔들리는 자리라, 표와 변환이 같은 것을 본다.

    python -m ner.scripts.extract_fac_surfaces data/ontonotes_en/raw out.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from ner.augmenters.ontonotes_en.convert import decode_bio, load_id2label

SPLIT_FILES = {
    'train': ['train00.json', 'train01.json', 'train02.json', 'train03.json'],
    'valid': ['valid.json'],
    'test': ['test.json'],
}


def collect(raw_dir: Path, max_contexts: int) -> list[dict]:
    """`FAC` 표면별로 등장 수·문맥·원본 타입 분포를 모은다.

    원본 타입 분포를 함께 남기는 것은 같은 표면을 원본이 `FAC` 와 다른
    타입으로도 태그했는지가 판정에 쓰이기 때문이다 — 그런 표면은 최종
    데이터에서 두 라벨로 갈린다.
    """
    id2label = load_id2label(raw_dir / 'label.json')
    occurrences: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter,
    )
    contexts: dict[str, list[str]] = collections.defaultdict(list)
    source_types: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter,
    )

    for split, names in SPLIT_FILES.items():
        for name in names:
            with (raw_dir / name).open(encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens = row['tokens']
                    for start, end, src in decode_bio(row['tags'], id2label):
                        surface = ' '.join(tokens[start:end + 1])
                        source_types[surface][src] += 1
                        if src != 'FAC':
                            continue
                        occurrences[surface][split] += 1
                        if len(contexts[surface]) < max_contexts:
                            contexts[surface].append(' '.join(tokens))

    entries = []
    for surface in sorted(occurrences):
        counts = occurrences[surface]
        entries.append({
            'surface': surface,
            'occurrences': {
                split: counts.get(split, 0) for split in SPLIT_FILES
            },
            'contexts': contexts[surface],
            'source_types': dict(source_types[surface]),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', type=Path, help='OntoNotes raw directory')
    parser.add_argument('output', type=Path, help='output JSON path')
    parser.add_argument(
        '--max-contexts', type=int, default=5,
        help='max context sentences kept per surface',
    )
    args = parser.parse_args()

    entries = collect(args.raw_dir, args.max_contexts)
    args.output.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding='utf-8',
    )
    total = sum(sum(e['occurrences'].values()) for e in entries)
    per_split = {
        split: sum(e['occurrences'][split] for e in entries)
        for split in SPLIT_FILES
    }
    print(f'surfaces={len(entries)} spans={total} {per_split}')


if __name__ == '__main__':
    main()
