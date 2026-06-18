"""EVT learning curve용 train gold 서브샘플 — EVT span을 rate%만 유지.

각 EVT span에 안정적 해시 기반 uniform 값을 부여해 임계 처리하므로
25% ⊂ 50% ⊂ 75% ⊂ 100% 의 nested subset이 된다(곡선 형태 깨끗).
유지 안 된 EVT는 비-entity(O)로 강등. EVT 외 라벨은 불변.

사용:
    python -m ner.scripts.subsample_evt --input data/klue/pii_all.jsonl \
        --rate 50 --output data/klue/pii_all.evt50.jsonl
"""
import argparse
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def _keep(rec_id: str, ent: dict, rate: int) -> bool:
    """span별 안정 해시 uniform < rate/100 이면 유지(nested)."""
    key = f"{rec_id}:{ent['start_char']}:{ent['end_char']}".encode()
    h = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate / 100.0


def main():
    parser = argparse.ArgumentParser(
        description='Subsample EVT spans in ko gold for a learning curve.'
    )
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rate', type=int, required=True, help='keep %% (1-100)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    kept = dropped = 0
    with open(args.input) as fin, open(args.output, 'w') as fout:
        for line in fin:
            rec = json.loads(line)
            ents = []
            for ent in rec['entities']:
                if ent['label'] == 'EVT':
                    if _keep(rec['id'], ent, args.rate):
                        kept += 1
                    else:
                        dropped += 1
                        continue
                ents.append(ent)
            rec['entities'] = ents
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
    logger.info('rate=%d%%  EVT kept=%d dropped=%d', args.rate, kept, dropped)


if __name__ == '__main__':
    main()
