"""자연 PII 주입용 소스 추출 — suffix gold → (원문, 원문 NER).

현 VI gold(`pii_all.jsonl`)는 suffix 모드로 PII가 문장 끝에 스캐폴딩
주입돼 있다. 자연 주입(`--mode llm`)으로 재생성하려면 *주입 전 원문*과
거기 달린 *정제 NER 라벨*(PER/LOC/ORG/PROD/EVT)만 필요하다.

원문 경계는 `resilver_vi_isolated.build_source`(all.jsonl prefix 매칭 →
suffix marker → 전체 텍스트 폴백)로 복원한다. 경계 이전 영역의 NER 만
남기고, 경계 이후의 주입 PII(EMAIL·PHONE·NAME→PER·ADDRESS→LOC 등)는
버린다. 원문 좌표는 `text[:boundary]` 와 동일하므로 offset 재계산이
불필요하다.

출력은 `{text, entities:[{label,start_char,end_char,text}], id}` 스키마로,
`python -m ner.augmenters.pii --source jsonl` 가 그대로 소비한다.

사용:
    python src/ner/augmenters/wikiann_vi/build_vi_natural_source.py \
        --pii data/wikiann_vi/pii_all.jsonl \
        --all data/wikiann_vi/all.jsonl \
        --out results/classifier/vi/audit/natural_source.jsonl
"""
import argparse
import json
import os

from ner.augmenters.wikiann_vi.resilver_vi_isolated import (
    _load_jsonl,
    build_source,
)

NER_TYPES = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')


def build_natural_source(pii: list[dict], all_rows: list[dict]) -> list[dict]:
    """행별 (원문, 원문 NER) 산출. 경계 이후 주입 PII 는 제거한다."""
    src = build_source(pii, all_rows)
    out: list[dict] = []
    mismatch = 0
    # 원본 id 는 중복이 많아 조인 키로 못 쓴다 → 행 인덱스로 유니크 id 부여.
    for i, (row, s) in enumerate(zip(pii, src)):
        original = s['original_text']
        boundary = s['boundary']
        ner: list[dict] = []
        for e in row['entities']:
            if e['end_char'] > boundary:
                continue  # 주입 PII 영역
            if e['label'] not in NER_TYPES:
                continue  # 경계 안쪽의 PII(이론상 없음) 방어
            # 원문 좌표 정합 검증 (text[:boundary] == original)
            if original[e['start_char']:e['end_char']] != e['text']:
                mismatch += 1
                continue
            ner.append({
                'label': e['label'],
                'start_char': e['start_char'],
                'end_char': e['end_char'],
                'text': e['text'],
            })
        out.append({
            'text': original,
            'entities': ner,
            'id': str(i),
        })
    if mismatch:
        print(f'WARNING: dropped {mismatch} offset-mismatched NER spans')
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pii', required=True, help='suffix gold JSONL')
    p.add_argument('--all', dest='all_path', required=True,
                   help='pre-injection corpus (all.jsonl)')
    p.add_argument('--out', required=True, help='output natural source JSONL')
    args = p.parse_args()

    pii = _load_jsonl(args.pii)
    all_rows = _load_jsonl(args.all_path)
    out = build_natural_source(pii, all_rows)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n_ner = sum(len(r['entities']) for r in out)
    n_empty = sum(1 for r in out if not r['entities'])
    print(f'wrote {len(out)} rows ({n_ner} NER spans, '
          f'{n_empty} no-entity) -> {args.out}')


if __name__ == '__main__':
    main()
