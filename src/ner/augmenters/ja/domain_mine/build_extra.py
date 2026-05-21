"""B5: relabeled 레코드를 train/test extra JSONL 로 분배·출력.

anchor_status 기준 분배 (leak-free):
- confirmed (2모델 PROD 합의)  → test 확장 (검증 등급)
- anchor_only (모델 미검출)    → train 보강 (FN 회복 신호)
- conflict (합의가 non-PROD)   → 드롭

merged_spans 를 classifier contract entity 로 변환하고 validate_record
(offset 무결성·라벨·중첩) 를 통과한 레코드만 출력한다. 같은 문장이 양쪽
풀에 들어가면 test 우선으로 남겨 leak 을 막는다. 분배·변환·검증은 순수
함수 build_extra 로 분리해 단위 테스트한다.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ner.augmenters.ja.domain_mine import schema

logger = logging.getLogger(__name__)


def _valid_entities(text: str, spans: List[Dict]) -> Optional[List[Dict]]:
    """merged_spans → contract entity 변환 후 검증. 실패 시 None."""
    entities = schema.spans_to_entities(spans)
    rec = schema.make_record('_', text, entities)
    errors = schema.validate_record(rec)
    if errors:
        logger.warning('skip invalid record: %s', errors[:2])
        return None
    return entities


def build_extra(records: List[Dict], id_prefix: str = 'dm'
                ) -> Tuple[List[Dict], List[Dict]]:
    """relabeled 레코드 → (test_records, train_records). 순수 함수.

    confirmed → test, anchor_only → train, conflict → drop. 문장 중복은
    test 우선으로 dedup 해 train 과 leak 되지 않게 한다.
    """
    test_seen = set()
    test_pairs: List[Tuple[str, List[Dict]]] = []
    train_pairs: List[Tuple[str, List[Dict]]] = []
    for r in records:
        status = r.get('anchor_status')
        text = r['text']
        if status == 'confirmed':
            if text in test_seen:
                continue
            entities = _valid_entities(text, r.get('merged_spans', []))
            if entities is None:
                continue
            test_seen.add(text)
            test_pairs.append((text, entities))
        elif status == 'anchor_only':
            entities = _valid_entities(text, r.get('merged_spans', []))
            if entities is None:
                continue
            train_pairs.append((text, entities))
        # conflict: drop

    train_seen = set()
    train_dedup: List[Tuple[str, List[Dict]]] = []
    for text, entities in train_pairs:
        if text in test_seen or text in train_seen:
            continue
        train_seen.add(text)
        train_dedup.append((text, entities))

    test_out = [schema.make_record(f'{id_prefix}-test-{i}', t, e)
                for i, (t, e) in enumerate(test_pairs)]
    train_out = [schema.make_record(f'{id_prefix}-train-{i}', t, e)
                 for i, (t, e) in enumerate(train_dedup)]
    return test_out, train_out


def _prod_count(records: List[Dict]) -> int:
    """레코드 리스트의 PROD entity 총 개수."""
    return sum(1 for r in records for e in r['entities']
               if e['label'] == 'PROD')


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ja.domain_mine.build_extra',
        description='Partition relabeled records into train/test extra',
    )
    p.add_argument('--in-jsonl', nargs='+', required=True,
                   help='B4 relabeled JSONL(s) (per-domain)')
    p.add_argument('--out-train', required=True,
                   help='Train extra JSONL (anchor_only)')
    p.add_argument('--out-test', required=True,
                   help='Test expansion JSONL (confirmed)')
    p.add_argument('--id-prefix', default='dm')
    p.add_argument('--log-level', default='INFO')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    records: List[Dict] = []
    for path in args.in_jsonl:
        records.extend(
            json.loads(ln) for ln
            in Path(path).read_text(encoding='utf-8').splitlines()
            if ln.strip())
    test_out, train_out = build_extra(records, args.id_prefix)
    _write_jsonl(Path(args.out_test), test_out)
    _write_jsonl(Path(args.out_train), train_out)
    print(f'test : {len(test_out)} sents / {_prod_count(test_out)} PROD '
          f'-> {args.out_test}')
    print(f'train: {len(train_out)} sents / {_prod_count(train_out)} PROD '
          f'-> {args.out_train}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
