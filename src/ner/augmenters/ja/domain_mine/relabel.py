"""B4: 2모델 교차검증 재라벨 (Gemma + Qwen 합의 + mined anchor 병합).

마이닝 문장(B3)을 두 vLLM 모델(JA 라벨러)로 각각 canonical 10종 재라벨한
뒤, wikiann_vi.merge_confidence 로 offset 기반 합의(confidence 태그)를
만든다. 마이닝으로 알아낸 PROD anchor 는 gold 로 간주해 합의 결과에
강제 병합한다 (wikiann_vi 의 gold_spans anchor 패턴).

LLM span(parse_spans)은 {text,type} 만 가지므로 assign_offsets 로 문장에
매칭해 {text,type,start,end} 로 만든 뒤 merge 에 넘긴다. offset 부여·anchor
병합은 순수 함수로 분리해 단위 테스트한다.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ner.augmenters.wikiann_vi.merge_confidence import merge_records
from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler

logger = logging.getLogger(__name__)

# 현재 vLLM 배치 (docker: vllm-gemma :8081, vllm-qwen :8082).
GEMMA_BASE = 'http://localhost:8081/v1'
GEMMA_MODEL = 'cyankiwi/gemma-4-31B-it-AWQ-8bit'
QWEN_BASE = 'http://localhost:8082/v1'
QWEN_MODEL = 'cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit'


def assign_offsets(text: str, llm_spans: List[Dict]) -> List[Dict]:
    """LLM span {text,type} 을 문장에 매칭해 {text,type,start,end} 로 변환.

    같은 surface 가 여러 번 나오면 비중첩 첫 미사용 위치에 배정한다.
    문장에 없는 surface 는 버린다 (환각·조사 부착 등). 순수 함수.
    """
    out: List[Dict] = []
    used: List = []
    for sp in llm_spans:
        t = (sp.get('text') or '').strip()
        ty = (sp.get('type') or '').strip()
        if not t or not ty:
            continue
        start = 0
        while True:
            idx = text.find(t, start)
            if idx < 0:
                break
            s, e = idx, idx + len(t)
            if not any(not (e <= u0 or s >= u1) for u0, u1 in used):
                out.append({'text': t, 'type': ty, 'start': s, 'end': e})
                used.append((s, e))
                break
            start = idx + 1
    return out


def _overlaps(a_s: int, a_e: int, span: Dict) -> bool:
    """span 이 [a_s, a_e) 와 겹치면 True."""
    return not (span['end'] <= a_s or span['start'] >= a_e)


def apply_anchor(spans: List[Dict], anchors: List, text: str):
    """mined PROD anchor 를 2모델 합의로 중재한다 (합의 우선).

    anchor 는 gold 가 아닌 hint 다. 사용자가 택한 2모델 교차검증이 최종
    판정한다. anchor 별 판정:
    - 합의에 PROD 가 anchor 와 겹침 → confirmed (모델이 PROD 확인)
    - 합의에 non-PROD 만 겹침 → conflict (예: 회사 ORG → anchor 거짓)
    - 합의가 anchor 에 침묵 → anchor_only (소스만 보증, FN 회복 후보)

    Returns:
        (spans, status). status ∈ {confirmed, anchor_only, conflict}.
        conflict 가 하나라도 있으면 conflict, 아니면 anchor_only 가 있으면
        anchor_only, 모두 PROD 합의면 confirmed. 순수 함수.
    """
    out = [dict(s) for s in spans]
    conflict = False
    anchor_only = False
    for a_s, a_e in anchors:
        ov = [s for s in out if _overlaps(a_s, a_e, s)]
        if any(s['type'] == 'PROD' for s in ov):
            continue
        if ov:
            conflict = True
        else:
            out.append({
                'text': text[a_s:a_e], 'type': 'PROD',
                'start': a_s, 'end': a_e,
                'confidence': 'anchor_only', 'source': 'mined_anchor',
            })
            anchor_only = True
    out.sort(key=lambda s: (s['start'], s['end']))
    if conflict:
        status = 'conflict'
    elif anchor_only:
        status = 'anchor_only'
    else:
        status = 'confirmed'
    return out, status


def _model_record(rec_id: str, text: str, labeler: VllmNERLabeler,
                  model_name: str) -> Dict:
    """단일 모델로 문장을 재라벨해 gold_spans_8type 레코드를 만든다."""
    raw = labeler.label_spans(text, split=False)
    spans = assign_offsets(text, raw)
    return {'id': rec_id, 'text': text, 'gold_spans_8type': spans,
            'relabel_model': model_name}


def relabel_and_merge(mined: List[Dict], gemma: VllmNERLabeler,
                      qwen: VllmNERLabeler,
                      policy: str = 'recall_strict') -> List[Dict]:
    """마이닝 레코드를 2모델 재라벨 + 합의 + anchor 병합한다.

    출력: {id, text, domain, name, source_title, merged_spans:[...]}.
    merged_spans 는 confidence 태그 포함 (B5 가 정책 필터에 사용).
    """
    gemma_recs = []
    qwen_recs = []
    for i, m in enumerate(mined):
        rid = str(i)
        gemma_recs.append(_model_record(rid, m['text'], gemma, GEMMA_MODEL))
        qwen_recs.append(_model_record(rid, m['text'], qwen, QWEN_MODEL))
    merged = merge_records(gemma_recs, qwen_recs, policy=policy)
    merged_by_id = {r['id']: r for r in merged}

    out: List[Dict] = []
    for i, m in enumerate(mined):
        rec = merged_by_id.get(str(i))
        if rec is None:
            continue
        anchors = [tuple(a) for a in m.get('anchors', [])]
        spans, status = apply_anchor(
            rec.get('gold_spans_8type_merged', []), anchors, m['text'])
        out.append({
            'id': str(i), 'text': m['text'], 'domain': m['domain'],
            'name': m['name'], 'source_title': m.get('source_title'),
            'anchor_status': status, 'merged_spans': spans,
        })
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ja.domain_mine.relabel',
        description='Cross-verify relabel mined sentences (Gemma+Qwen)',
    )
    p.add_argument('--in-jsonl', required=True,
                   help='B3 mined JSONL (one domain)')
    p.add_argument('--out-jsonl', required=True,
                   help='Output relabeled JSONL with merged_spans')
    p.add_argument('--policy', default='recall_strict')
    p.add_argument('--gemma-base', default=GEMMA_BASE)
    p.add_argument('--gemma-model', default=GEMMA_MODEL)
    p.add_argument('--qwen-base', default=QWEN_BASE)
    p.add_argument('--qwen-model', default=QWEN_MODEL)
    p.add_argument('--max-records', type=int, default=0,
                   help='Cap input records (0 = all)')
    p.add_argument('--log-level', default='INFO')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    mined = [json.loads(ln) for ln
             in Path(args.in_jsonl).read_text(encoding='utf-8').splitlines()
             if ln.strip()]
    if args.max_records:
        mined = mined[:args.max_records]
    gemma = VllmNERLabeler(base_url=args.gemma_base, model=args.gemma_model)
    qwen = VllmNERLabeler(base_url=args.qwen_base, model=args.qwen_model)
    out = relabel_and_merge(mined, gemma, qwen, args.policy)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'{args.in_jsonl}: {len(out)} relabeled -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
