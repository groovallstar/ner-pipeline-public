"""두 모델 재라벨 결과를 confidence 태그 부가해 단일 파일로 병합한다.

각 span에 두 모델 합의 수준을 나타내는 `confidence`·`source` 필드를 붙여,
학습 시 정책(보수 vs recall 강화 등)을 필터 한 줄로 바꿀 수 있게 한다.

카테고리 정의 (Gemma=A, Qwen=B 전제):
- `high`          source=both         : 동일 offset + 동일 타입 (양쪽 합의)
- `conflict`      source=both_disagree: 동일 offset + 타입 다름
- `medium_recall` source=gemma_only   : A만 라벨, B는 skip
- `medium_prec`   source=qwen_only    : B만 라벨, A는 skip

정책 (span 선택):
- `recall`        : high + gemma_only (A의 넓은 커버리지, conflict 제외)
- `precision`     : high + qwen_only  (B의 보수적 커버리지, conflict 제외)
- `high_only`     : high 만             (양쪽 합의, 최고신뢰)
- `full`          : 전부 포함 (conflict 포함)
- `recall_strict` : recall 정책 + PROD/EVT 는 high 만 (PROD/EVT 합의율
                    낮은 점을 보정하기 위해 신규 type 의 medium 을 drop)
- `recall_strict_evt` : recall_strict 와 동일하되 EVT single-model 중 §3
                    명시 legit 카테고리(연도대회·조약·전쟁·재해·선거) 매칭분만
                    구제. Qwen 의 구조적 EVT recall 병목(보통명사-핵 서술구
                    누락)이 strict 교집합을 천장 걸어 legit EVT 를 떨구는 회귀
                    보정. 구제 = §3 결정론 규칙의 regex 재적용(self-confirm 아님)

신뢰도 계층별 학습 데이터 활용 구조의 단일-파일 구현.
타입별 신뢰도 격차를 반영한 type-aware 필터(`recall_strict`)를 함께 제공한다.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

POLICIES = (
    'recall', 'precision', 'high_only', 'full', 'recall_strict',
    'recall_strict_evt',
)

# recall_strict 에서 high 만 허용할 type — 신규 5종 중 합의율 낮은 두 type
HIGH_ONLY_TYPES = frozenset({'PROD', 'EVT'})

# ── EVT legit-카테고리 패턴 (recall_strict_evt 구제 게이트) ──────────────
# Qwen 은 베트남어 EVT(보통명사-핵 서술구: Cúp·Trận·Công ước·Bão…)를
# 자유 생성에서 누락하는 구조적 recall 병목이 있어, strict 교집합이 EVT 를
# Qwen recall 에 천장 건다. §3 가 *명시적으로* EVT 로 규정한 결정론 규칙의
# 표면형에 매칭되는 single-model EVT 만 구제 — Gemma/Qwen 이 surface 한 span
# 에 §3 규칙을 regex 로 독립 재적용하는 것이라 self-confirm/유도 모두 회피한다.
# 미매칭 long-tail(투어·금융위기·영화제 등)은 의도적 보수 drop.
_EVT_YEAR = re.compile(r'(?:19|20)\d{2}|\d{4}-\d{2,4}')
_EVT_LEGIT_PATTERNS = (
    # 연도별 대회 에디션 — 정기 대회의 특정 연도/회차 (§3: "World Cup 2022")
    lambda t: bool(_EVT_YEAR.search(t)) and bool(re.search(
        r'Cúp|Giải|Đại hội|Olympic|World Cup|UEFA|League|Siêu cúp|'
        r'Thế vận|Vô địch', t)),
    # 조약/협약/공의회/회의/협상 (§3: hiệp ước = EVT)
    lambda t: bool(re.match(
        r'(Công ước|Hiệp định|Hiệp ước|Công đồng|Hội nghị|Đàm phán|'
        r'Vòng đàm phán)\b', t)),
    # 전쟁/전투/작전/내전 (§3: chiến tranh = EVT)
    lambda t: bool(re.search(
        r'\b(Chiến tranh|Trận|Chiến dịch|Nội chiến|Xung đột)\b', t)),
    # 명명 재해/테러 (§3: thảm họa có tên = EVT)
    lambda t: bool(re.match(
        r'(Bão|Động đất|Sóng thần|Đánh bom|Vụ (đánh bom|tấn công)|'
        r'Thảm họa)\b', t)),
    # 선거/봉기/쿠데타/혁명 (§3: bầu cử·phong trào có mốc = EVT)
    lambda t: bool(re.search(
        r'\b(Bầu cử|Trưng cầu|Phong trào|Khởi nghĩa|Đảo chính|Cách mạng)\b',
        t)),
)


def _is_evt_legit(text: str) -> bool:
    """EVT 표면형이 §3 명시 legit 카테고리(연도대회·조약·전쟁·재해·선거)에
    매칭되는지 — single-model EVT 구제 게이트."""
    if not text:
        return False
    return any(p(text) for p in _EVT_LEGIT_PATTERNS)


def _span_map(spans: List[dict]) -> Dict[Tuple[int, int], dict]:
    """(start, end) → span dict. 중복 offset은 마지막이 승리."""
    return {
        (int(s['start']), int(s['end'])): s
        for s in spans
        if 'start' in s and 'end' in s and 'type' in s
    }


def categorize_spans(
    gemma_spans: List[dict],
    qwen_spans: List[dict],
) -> List[dict]:
    """두 모델의 span 리스트를 비교해 confidence 태그 부가된 병합 리스트 반환.

    반환 span 스키마: {text, type, start, end, confidence, source}
    conflict 케이스는 gemma_type·qwen_type 필드 추가, primary(Gemma) 타입을
    `type`으로 채택.
    """
    a = _span_map(gemma_spans)
    b = _span_map(qwen_spans)
    merged: List[dict] = []

    # A 기준 순회
    for key in sorted(a):
        ga = a[key]
        gb = b.get(key)
        start, end = key
        if gb is None:
            merged.append({
                'text': ga['text'],
                'type': ga['type'],
                'start': start,
                'end': end,
                'confidence': 'medium_recall',
                'source': 'gemma_only',
            })
        elif ga['type'] == gb['type']:
            merged.append({
                'text': ga['text'],
                'type': ga['type'],
                'start': start,
                'end': end,
                'confidence': 'high',
                'source': 'both',
            })
        else:
            merged.append({
                'text': ga['text'],
                'type': ga['type'],
                'start': start,
                'end': end,
                'confidence': 'conflict',
                'source': 'both_disagree',
                'gemma_type': ga['type'],
                'qwen_type': gb['type'],
            })
    # B에만 있는 span
    for key in sorted(b):
        if key in a:
            continue
        gb = b[key]
        start, end = key
        merged.append({
            'text': gb['text'],
            'type': gb['type'],
            'start': start,
            'end': end,
            'confidence': 'medium_prec',
            'source': 'qwen_only',
        })
    # start 오프셋 순으로 재정렬
    merged.sort(key=lambda s: (int(s['start']), int(s['end'])))
    return merged


def _filter_by_policy(
    spans: List[dict], policy: str,
) -> List[dict]:
    if policy == 'recall':
        allowed = {'high', 'medium_recall'}
    elif policy == 'precision':
        allowed = {'high', 'medium_prec'}
    elif policy == 'high_only':
        allowed = {'high'}
    elif policy == 'full':
        return list(spans)
    elif policy == 'recall_strict':
        # PER/LOC/ORG 는 recall 정책 (high+medium_recall),
        # PROD/EVT 는 high 만 (합의율 낮은 신규 type 보수 처리).
        return [
            s for s in spans
            if (
                s['type'] in HIGH_ONLY_TYPES
                and s['confidence'] == 'high'
            ) or (
                s['type'] not in HIGH_ONLY_TYPES
                and s['confidence'] in {'high', 'medium_recall'}
            )
        ]
    elif policy == 'recall_strict_evt':
        # recall_strict 와 동일하되, EVT single-model(gemma_only/qwen_only)
        # 중 §3 legit 카테고리 매칭분을 구제 (Qwen EVT recall 병목 우회).
        # PROD 는 여전히 high 만, conflict 는 전 type drop.
        out: List[dict] = []
        for s in spans:
            t = s['type']
            c = s['confidence']
            if t == 'EVT':
                if c == 'high' or (
                    c in {'medium_recall', 'medium_prec'}
                    and _is_evt_legit(s.get('text', ''))
                ):
                    out.append(s)
            elif t == 'PROD':
                if c == 'high':
                    out.append(s)
            elif c in {'high', 'medium_recall'}:
                out.append(s)
        return out
    else:
        raise ValueError(f'unknown policy: {policy}')
    return [s for s in spans if s['confidence'] in allowed]


def merge_records(
    gemma_records: List[dict],
    qwen_records: List[dict],
    policy: str = 'recall',
) -> List[dict]:
    """두 JSONL 레코드 리스트를 id 기준으로 매칭해 단일 레코드 리스트 반환.

    출력 스키마: 기존 필드 + `gold_spans_8type_merged` + `merge_policy`.
    Gemma 레코드를 기준으로 순회하며, Qwen에만 있는 id는 드물지만 안전망으로
    후단에 추가한다.
    """
    a_by_id = {str(r['id']): r for r in gemma_records}
    b_by_id = {str(r['id']): r for r in qwen_records}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    merged: List[dict] = []
    for rid in common_ids:
        ga = a_by_id[rid]
        gb = b_by_id[rid]
        all_merged = categorize_spans(
            ga.get('gold_spans_8type', []),
            gb.get('gold_spans_8type', []),
        )
        filtered = _filter_by_policy(all_merged, policy)
        rec = {
            k: v for k, v in ga.items()
            if k not in ('gold_spans_8type',)
        }
        rec['gold_spans_8type_merged'] = filtered
        rec['merge_policy'] = policy
        rec['merge_sources'] = {
            'a': ga.get('relabel_model', 'gemma'),
            'b': gb.get('relabel_model', 'qwen'),
        }
        merged.append(rec)
    return merged


def _load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _summarize(records: List[dict]) -> None:
    total = len(records)
    total_spans = sum(
        len(r.get('gold_spans_8type_merged', [])) for r in records
    )
    from collections import Counter
    conf_counter: Counter = Counter()
    type_counter: Counter = Counter()
    for r in records:
        for s in r.get('gold_spans_8type_merged', []):
            conf_counter[s.get('confidence', '?')] += 1
            type_counter[s.get('type', '?')] += 1
    print('=== Merge Summary ===')
    print(f'Records: {total}')
    print(f'Total merged spans: {total_spans}')
    print('Confidence breakdown:')
    for c, n in conf_counter.most_common():
        print(f'  {c}: {n}')
    print('Per-type counts:')
    for t, n in type_counter.most_common():
        print(f'  {t}: {n}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.wikiann_vi.merge_confidence',
        description='Merge two relabel JSONLs with confidence tagging',
    )
    p.add_argument('--gemma', required=True, help='Gemma JSONL path')
    p.add_argument('--qwen', required=True, help='Qwen JSONL path')
    p.add_argument(
        '--policy', default='recall', choices=POLICIES,
        help='Span selection policy',
    )
    p.add_argument(
        '--output', required=True, help='Merged JSONL output path',
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    gemma = _load_jsonl(Path(args.gemma))
    qwen = _load_jsonl(Path(args.qwen))
    print(f'Gemma: {len(gemma)} records, Qwen: {len(qwen)} records')
    merged = merge_records(gemma, qwen, policy=args.policy)
    print(f'Merged policy={args.policy}: {len(merged)} records')
    _write_jsonl(Path(args.output), merged)
    print(f'Wrote -> {args.output}')
    _summarize(merged)
    return 0


if __name__ == '__main__':
    sys.exit(main())
