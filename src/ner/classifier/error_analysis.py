"""classifier 모델의 test-set 오류 분류 도구.

baseline 모델을 test split 에 대해 추론한 뒤, gold/pred span 비교 결과를
오류 카테고리(boundary/type_mismatch/miss/hallucination)로 분류하고
사람 검수용 stratified sample 을 추출한다.

사용 예 (모델 천장 vs gold 라벨 천장 분리):
    python -m ner.classifier.error_analysis \
        --lang ja \
        --model-path results/classifier/ja_sweep/baseline/best \
        --output-dir results/classifier/ja_sweep/baseline

산출물:
    error_analysis.json        # 전체 문장별 결과 + 집계
    review_sample.jsonl        # 사람 검수용 stratified ~10% 샘플

검수 후 다음 단계는 corrections.jsonl 입력 → recompute baseline metrics.
"""

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ner.classifier.data_utils import CANONICAL_LABELS

logger = logging.getLogger(__name__)

# confusion matrix 에서 'gold 누락 (HALL)' / 'pred 누락 (MISS)' 자리 표시자
NULL = '∅'


def _overlap_len(a: dict, b: dict) -> int:
    """반열림 구간 [start,end) 간 겹치는 char 수. 음수면 0."""
    return max(0, min(a['end'], b['end']) - max(a['start'], b['start']))


def _classify_match_pair(g: dict, p: dict) -> str:
    """동일 문장 내 한 gold·pred 쌍의 매칭 카테고리.

    EXACT: (start,end,type) 모두 일치
    BOUNDARY: type 일치 + overlap > 0 + offset 다름
    TYPE_MISMATCH: type 다름 + overlap > 0
    UNRELATED: overlap 0 (매칭 후보 아님)
    """
    if g['start'] == p['start'] and g['end'] == p['end'] and g['type'] == p['type']:
        return 'EXACT'
    if _overlap_len(g, p) <= 0:
        return 'UNRELATED'
    if g['type'] == p['type']:
        return 'BOUNDARY'
    return 'TYPE_MISMATCH'


def classify_span_errors(
    gold_spans: List[dict],
    pred_spans: List[dict],
) -> dict:
    """단일 문장의 gold·pred span 비교 → 카테고리화된 오류 분류.

    1:1 매칭 우선순위: EXACT > BOUNDARY > TYPE_MISMATCH. 매칭된 pair 는 양쪽에서 제거.
    남은 gold = (MISS), 남은 pred = (HALLUCINATION).

    각 span dict 는 {'type', 'start', 'end', 'text'} (text 는 옵션이지만 사람 검수에 필요).

    Returns:
        {
            'exact': [{gold, pred}],
            'fn': [{gold, error_class, matched_pred?}],
            'fp': [{pred, error_class, matched_gold?}],
            'counts': {gold, pred, exact, fn, fp},
        }
        error_class:
            FN: BOUNDARY / TYPE_MISMATCH / MISS
            FP: BOUNDARY / TYPE_MISMATCH / HALLUCINATION
    """
    n_g = len(gold_spans)
    n_p = len(pred_spans)
    g_used = [False] * n_g
    p_used = [False] * n_p

    exact: List[dict] = []
    fn: List[dict] = []
    fp: List[dict] = []

    # 1) EXACT 우선 매칭
    for i, g in enumerate(gold_spans):
        for j, p in enumerate(pred_spans):
            if g_used[i] or p_used[j]:
                continue
            if _classify_match_pair(g, p) == 'EXACT':
                exact.append({'gold': g, 'pred': p})
                g_used[i] = True
                p_used[j] = True
                break

    # 2) BOUNDARY (type 일치 + overlap)
    for i, g in enumerate(gold_spans):
        if g_used[i]:
            continue
        best_j = -1
        best_overlap = 0
        for j, p in enumerate(pred_spans):
            if p_used[j]:
                continue
            if _classify_match_pair(g, p) == 'BOUNDARY':
                ov = _overlap_len(g, p)
                if ov > best_overlap:
                    best_j = j
                    best_overlap = ov
        if best_j >= 0:
            fn.append({
                'gold': g,
                'error_class': 'BOUNDARY',
                'matched_pred': pred_spans[best_j],
            })
            fp.append({
                'pred': pred_spans[best_j],
                'error_class': 'BOUNDARY',
                'matched_gold': g,
            })
            g_used[i] = True
            p_used[best_j] = True

    # 3) TYPE_MISMATCH (overlap 있으나 type 다름)
    for i, g in enumerate(gold_spans):
        if g_used[i]:
            continue
        best_j = -1
        best_overlap = 0
        for j, p in enumerate(pred_spans):
            if p_used[j]:
                continue
            if _classify_match_pair(g, p) == 'TYPE_MISMATCH':
                ov = _overlap_len(g, p)
                if ov > best_overlap:
                    best_j = j
                    best_overlap = ov
        if best_j >= 0:
            fn.append({
                'gold': g,
                'error_class': 'TYPE_MISMATCH',
                'matched_pred': pred_spans[best_j],
            })
            fp.append({
                'pred': pred_spans[best_j],
                'error_class': 'TYPE_MISMATCH',
                'matched_gold': g,
            })
            g_used[i] = True
            p_used[best_j] = True

    # 4) 남은 gold = MISS
    for i, g in enumerate(gold_spans):
        if not g_used[i]:
            fn.append({'gold': g, 'error_class': 'MISS'})

    # 5) 남은 pred = HALLUCINATION
    for j, p in enumerate(pred_spans):
        if not p_used[j]:
            fp.append({'pred': p, 'error_class': 'HALLUCINATION'})

    return {
        'exact': exact,
        'fn': fn,
        'fp': fp,
        'counts': {
            'gold': n_g,
            'pred': n_p,
            'exact': len(exact),
            'fn': len(fn),
            'fp': len(fp),
        },
    }


def aggregate_errors(sentence_results: List[dict]) -> dict:
    """전체 문장 결과 → 카테고리별·entity type 별 집계."""
    fn_classes = Counter()
    fp_classes = Counter()
    fn_by_type: Dict[str, Counter] = defaultdict(Counter)
    fp_by_type: Dict[str, Counter] = defaultdict(Counter)
    total_gold = 0
    total_pred = 0
    total_exact = 0

    for sr in sentence_results:
        c = sr['counts']
        total_gold += c['gold']
        total_pred += c['pred']
        total_exact += c['exact']
        for entry in sr['fn']:
            cls = entry['error_class']
            etype = entry['gold']['type']
            fn_classes[cls] += 1
            fn_by_type[etype][cls] += 1
        for entry in sr['fp']:
            cls = entry['error_class']
            etype = entry['pred']['type']
            fp_classes[cls] += 1
            fp_by_type[etype][cls] += 1

    return {
        'totals': {
            'gold': total_gold,
            'pred': total_pred,
            'exact': total_exact,
            'fn': sum(fn_classes.values()),
            'fp': sum(fp_classes.values()),
        },
        'fn_by_class': dict(fn_classes.most_common()),
        'fp_by_class': dict(fp_classes.most_common()),
        'fn_by_type': {t: dict(c) for t, c in fn_by_type.items()},
        'fp_by_type': {t: dict(c) for t, c in fp_by_type.items()},
    }


def build_confusion_matrix(sentence_results: List[dict]) -> Dict[str, Dict[str, int]]:
    """전체 sentence_results → (gold_type × pred_type) span-level confusion matrix.

    매칭 카운트 규칙 (이중계산 방지):
    - EXACT: matrix[t][t] += 1
    - BOUNDARY: matrix[t][t] += 1 (type 동일, offset 차이만 무시한 대각선)
    - TYPE_MISMATCH: matrix[gold_type][pred_type] += 1 (FN side 만 카운트)
    - MISS: matrix[gold_type][NULL] += 1
    - HALLUCINATION: matrix[NULL][pred_type] += 1

    NULL ('∅') 은 'gold 부재 (FP 환각)' / 'pred 부재 (FN 누락)' 자리 표시자.
    TYPE_MISMATCH 와 BOUNDARY 는 FN/FP 양쪽에 동일 entry 가 들어가므로 FN side
    에서만 집계해 이중계산을 막는다.
    """
    matrix: Dict[str, Counter] = defaultdict(Counter)

    for sr in sentence_results:
        for ex in sr.get('exact', []):
            t = ex['gold']['type']
            matrix[t][t] += 1

        for entry in sr.get('fn', []):
            cls = entry['error_class']
            gold_type = entry['gold']['type']
            if cls == 'MISS':
                matrix[gold_type][NULL] += 1
            elif cls == 'BOUNDARY':
                matrix[gold_type][gold_type] += 1
            elif cls == 'TYPE_MISMATCH':
                pred_type = entry['matched_pred']['type']
                matrix[gold_type][pred_type] += 1

        for entry in sr.get('fp', []):
            cls = entry['error_class']
            if cls == 'HALLUCINATION':
                matrix[NULL][entry['pred']['type']] += 1
            # BOUNDARY/TYPE_MISMATCH 는 FN side 에서 카운트 완료

    return {t: dict(c) for t, c in matrix.items()}


def top_errors_by_type(
    sentence_results: List[dict],
    side: str,
    type_: str,
    n: int = 50,
    ctx_chars: int = 20,
) -> List[dict]:
    """특정 entity type 의 FP 또는 FN 항목을 surface + 좌우 context 와 함께 추출.

    Args:
        side: 'fp' 또는 'fn'
        type_: 추출 대상 entity type. side='fp' 시 pred type, side='fn' 시
               gold type 기준
        n: 최대 추출 건수
        ctx_chars: 좌우 context 문자 수

    Returns:
        [{'sent_idx', 'surface', 'left_ctx', 'right_ctx', 'error_class',
          'counter_type', 'span': (start, end)}, ...]
        counter_type: TYPE_MISMATCH/BOUNDARY 면 매칭 상대 type, MISS/HALLUCINATION
        면 NULL.

    정렬: 같은 surface 빈도가 높은 항목 우선 (반복 패턴 식별), 동률 시 sent_idx.
    """
    if side not in ('fp', 'fn'):
        raise ValueError(f"side must be 'fp' or 'fn', got {side!r}")

    primary_key = 'pred' if side == 'fp' else 'gold'
    counter_key = 'matched_gold' if side == 'fp' else 'matched_pred'

    out: List[dict] = []
    for sr in sentence_results:
        text = sr.get('text', '')
        for entry in sr.get(side, []):
            primary = entry.get(primary_key)
            if primary is None or primary['type'] != type_:
                continue
            cls = entry['error_class']
            if cls in ('TYPE_MISMATCH', 'BOUNDARY'):
                counter = entry.get(counter_key)
                counter_type = counter['type'] if counter else NULL
            else:
                counter_type = NULL

            s, e = primary['start'], primary['end']
            out.append({
                'sent_idx': sr.get('sent_idx', -1),
                'surface': text[s:e],
                'left_ctx': text[max(0, s - ctx_chars):s],
                'right_ctx': text[e:e + ctx_chars],
                'error_class': cls,
                'counter_type': counter_type,
                'span': (s, e),
            })

    freq = Counter(r['surface'] for r in out)
    out.sort(key=lambda r: (-freq[r['surface']], r['sent_idx']))
    return out[:n]


def render_confusion_matrix_md(
    matrix: Dict[str, Dict[str, int]],
    types: Optional[List[str]] = None,
) -> str:
    """confusion matrix → markdown 표 (gold 행 × pred 열, NULL 컬럼 포함)."""
    if types is None:
        types = list(CANONICAL_LABELS)
    rows = list(types) + [NULL]
    cols = list(types) + [NULL]

    lines: List[str] = []
    lines.append('| gold \\ pred | ' + ' | '.join(cols) + ' | row_sum |')
    lines.append('|' + '|'.join(['---'] * (len(cols) + 2)) + '|')

    col_sums = {c: 0 for c in cols}
    for r in rows:
        cells = []
        row_sum = 0
        for c in cols:
            v = matrix.get(r, {}).get(c, 0)
            cells.append(str(v) if v else '-')
            row_sum += v
            col_sums[c] += v
        cells.append(str(row_sum))
        lines.append(f'| {r} | ' + ' | '.join(cells) + ' |')

    total = sum(col_sums.values())
    bottom = [str(col_sums[c]) for c in cols] + [str(total)]
    lines.append('| **col_sum** | ' + ' | '.join(bottom) + ' |')
    return '\n'.join(lines)


def render_top_errors_md(top: List[dict], title: str) -> str:
    """top_errors_by_type 결과 → markdown 표 (surface + context + counter)."""
    lines = [f'# {title}', '']
    lines.append('| # | surface | error_class | counter | context |')
    lines.append('|---|---|---|---|---|')
    for i, r in enumerate(top, 1):
        surface_safe = r['surface'].replace('|', '\\|').replace('\n', ' ')
        left = r['left_ctx'].replace('|', '\\|').replace('\n', ' ')
        right = r['right_ctx'].replace('|', '\\|').replace('\n', ' ')
        ctx = f'{left}【{surface_safe}】{right}'
        lines.append(
            f"| {i} | {surface_safe} | {r['error_class']} | "
            f"{r['counter_type']} | {ctx} |"
        )
    return '\n'.join(lines)


def sample_for_review(
    sentence_results: List[dict],
    ratio: float = 0.1,
    min_per_type: int = 3,
    seed: int = 42,
) -> List[dict]:
    """사람 검수용 stratified sample 을 추출한다.

    오류(FN 또는 FP)가 있는 문장 중에서 entity type 별로 비례 샘플링한다.
    각 type 마다 최소 `min_per_type` 문장을 보장(가능한 경우).

    Args:
        sentence_results: classify_span_errors 결과를 sent 단위로 모은 리스트
        ratio: 전체 오류 문장 중 샘플링 비율 (0.1 = 10%)
        min_per_type: type 별 최소 샘플 수
        seed: 결정적 샘플링용 random seed

    Returns:
        샘플된 sentence_result 리스트 (원본 reference, 복사 아님)
    """
    rng = random.Random(seed)
    error_sents = [
        sr for sr in sentence_results
        if sr['counts']['fn'] > 0 or sr['counts']['fp'] > 0
    ]
    target_n = max(1, int(round(len(error_sents) * ratio)))

    # type 별 후보 그룹화 (한 문장이 여러 type 에 등장하면 모두 후보)
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for sr in error_sents:
        types = set()
        for e in sr['fn']:
            types.add(e['gold']['type'])
        for e in sr['fp']:
            types.add(e['pred']['type'])
        for t in types:
            by_type[t].append(sr)

    selected_ids = set()
    selected: List[dict] = []

    # min_per_type 보장
    for t, lst in sorted(by_type.items()):
        rng.shuffle(lst)
        for sr in lst:
            if len(selected) >= target_n and t in {x for sr2 in selected for x in _types_in(sr2)}:
                break
            sid = sr['sent_idx']
            if sid in selected_ids:
                continue
            selected_ids.add(sid)
            selected.append(sr)
            type_count = sum(1 for x in selected if t in _types_in(x))
            if type_count >= min_per_type:
                break

    # 부족분 채우기 (랜덤)
    remaining = [sr for sr in error_sents if sr['sent_idx'] not in selected_ids]
    rng.shuffle(remaining)
    for sr in remaining:
        if len(selected) >= target_n:
            break
        selected.append(sr)
        selected_ids.add(sr['sent_idx'])

    selected.sort(key=lambda x: x['sent_idx'])
    return selected


def _types_in(sr: dict) -> set:
    """sentence_result 의 모든 entity type (gold + pred)."""
    out = set()
    for e in sr['fn']:
        out.add(e['gold']['type'])
    for e in sr['fp']:
        out.add(e['pred']['type'])
    return out


def _build_review_record(sr: dict) -> dict:
    """검수용 1행: 사람이 읽고 라벨링하기 좋은 형식으로 펼침."""
    return {
        'sent_idx': sr['sent_idx'],
        'text': sr['text'],
        'gold_spans': [
            {'type': g['type'], 'start': g['start'], 'end': g['end'],
             'text': g.get('text', '')}
            for g in sr['gold_spans']
        ],
        'pred_spans': [
            {'type': p['type'], 'start': p['start'], 'end': p['end'],
             'text': p.get('text', '')}
            for p in sr['pred_spans']
        ],
        'errors': [
            {
                'side': 'fn',
                'class': e['error_class'],
                'gold': e['gold'],
                'matched_pred': e.get('matched_pred'),
                # 검수자 입력란 (빈 값으로 남김)
                'verdict': '',  # 'model_error' | 'gold_error' | 'ambiguous'
                'corrected_label': '',
                'corrected_start': None,
                'corrected_end': None,
                'note': '',
            }
            for e in sr['fn']
        ] + [
            {
                'side': 'fp',
                'class': e['error_class'],
                'pred': e['pred'],
                'matched_gold': e.get('matched_gold'),
                'verdict': '',
                'corrected_label': '',
                'corrected_start': None,
                'corrected_end': None,
                'note': '',
            }
            for e in sr['fp']
        ],
    }


def run_inference(
    *,
    lang: str,
    model_path: str,
    data_path: str,
    tokenizer_name: Optional[str] = None,
    split: str = 'test',
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_length: int = 256,
    batch_size: int = 32,
) -> List[dict]:
    """baseline best 모델을 test split 에 적용 → 문장별 (text, gold, pred) 리스트.

    Returns:
        [{'sent_idx', 'text', 'gold_spans', 'pred_spans'}, ...]
        gold/pred span dict 형식: {'type', 'start', 'end', 'text'}
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from ner.classifier.data_utils import (
        build_label_maps,
        encode_dataset,
        load_jsonl,
        split_train_valid_test,
    )
    from ner.classifier.data_utils import decode_bio_to_spans

    label2id, id2label = build_label_maps()

    rows = load_jsonl(data_path)
    train_rows, valid_rows, test_rows = split_train_valid_test(
        rows, valid_ratio, test_ratio, seed,
    )
    split_map = {'train': train_rows, 'valid': valid_rows, 'test': test_rows}
    if split not in split_map:
        raise ValueError(f'unknown split: {split!r}')
    target_rows = split_map[split]

    # trainer.save_model() 산출물은 토크나이저를 포함하지 않으므로
    # 학습 시 사용한 base 모델 이름을 별도로 받는다
    if tokenizer_name is None:
        raise ValueError(
            'tokenizer_name is required (model_path saves model only, '
            'not tokenizer). pass --tokenizer-name or the original HF '
            'model id (e.g., tohoku-nlp/bert-base-japanese-v3).'
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except (TypeError, ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    target_features, target_offsets = encode_dataset(
        target_rows, tokenizer, label2id, lang, max_length
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device).eval()

    results: List[dict] = []
    n = len(target_features)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = target_features[i:i + batch_size]
            input_ids = torch.tensor(
                [f['input_ids'] for f in batch], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask'] for f in batch], dtype=torch.long
            ).to(device)
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits
            preds = logits.argmax(dim=-1).cpu().numpy()

            for j, pred_ids in enumerate(preds):
                idx = i + j
                offs = target_offsets[idx]
                pred_ids_list = [int(x) for x in pred_ids[:len(offs)]]
                pred_spans_raw = decode_bio_to_spans(
                    pred_ids_list, offs, id2label,
                )
                row = target_rows[idx]
                text = row['text']
                gold_spans = [
                    {'type': e['label'],
                     'start': e['start_char'],
                     'end': e['end_char'],
                     'text': text[e['start_char']:e['end_char']]}
                    for e in row['entities']
                ]
                pred_spans = [
                    {'type': s['type'],
                     'start': s['start'],
                     'end': s['end'],
                     'text': text[s['start']:s['end']]}
                    for s in pred_spans_raw
                ]
                results.append({
                    'sent_idx': idx,
                    'text': text,
                    'gold_spans': gold_spans,
                    'pred_spans': pred_spans,
                })

    return results


def _analyze_sentences(
    sentences: List[dict],
    *,
    lang: str,
    output_dir: str,
    split: str,
    seed: int,
    meta: Optional[dict] = None,
    review_ratio: float = 0.1,
    review_min_per_type: int = 3,
    with_diagnosis: bool = False,
    diagnosis_fp_types: Optional[List[str]] = None,
    diagnosis_fn_types: Optional[List[str]] = None,
    diagnosis_top_n: int = 50,
    diagnosis_ctx_chars: int = 20,
) -> dict:
    """추론·로딩 이후 공통 파이프라인: 분류 → 집계 → 검수 샘플 → 산출물.

    `sentences` 는 {'sent_idx', 'text', 'gold_spans', 'pred_spans'} 리스트.
    추론 경로(`run_error_analysis`)와 K-fold pooled 예측 경로
    (`run_pooled_error_analysis`)가 공유한다. `meta` 는 산출 JSON 헤더에
    합칠 출처 정보(model_path / fold_dirs 등).
    """
    sentence_results: List[dict] = []
    for s in sentences:
        cls = classify_span_errors(s['gold_spans'], s['pred_spans'])
        sentence_results.append({
            'sent_idx': s['sent_idx'],
            'text': s['text'],
            'gold_spans': s['gold_spans'],
            'pred_spans': s['pred_spans'],
            **cls,
        })

    agg = aggregate_errors(sentence_results)

    os.makedirs(output_dir, exist_ok=True)
    suffix = '' if split == 'test' else f'_{split}'
    full_path = os.path.join(output_dir, f'error_analysis{suffix}.json')
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump({
            'lang': lang,
            'split': split,
            'seed': seed,
            **(meta or {}),
            'aggregate': agg,
            'sentences': sentence_results,
        }, f, ensure_ascii=False, indent=2)
    logger.info('Saved %s', full_path)

    if split == 'test':
        # test split: 기존 sample 기반 검수 흐름 유지
        review = sample_for_review(
            sentence_results, review_ratio, review_min_per_type, seed,
        )
        review_path = os.path.join(output_dir, 'review_sample.jsonl')
        review_label = 'sample'
    else:
        # test 외(valid/train/pooled): 전체 오류 문장을 직접 출력
        review = [
            sr for sr in sentence_results
            if sr['counts']['fn'] > 0 or sr['counts']['fp'] > 0
        ]
        review.sort(key=lambda x: x['sent_idx'])
        review_path = os.path.join(output_dir, f'review_full_{split}.jsonl')
        review_label = 'full'

    with open(review_path, 'w', encoding='utf-8') as f:
        for sr in review:
            f.write(json.dumps(_build_review_record(sr), ensure_ascii=False))
            f.write('\n')
    logger.info('Saved %s (%d sentences)', review_path, len(review))

    print(f"\n{'='*70}")
    print(f'  ERROR ANALYSIS [split={split}]')
    print(f"{'='*70}")
    t = agg['totals']
    print(
        f"  Sentences={len(sentence_results)}  "
        f"Gold={t['gold']}  Pred={t['pred']}  "
        f"Exact={t['exact']}  FN={t['fn']}  FP={t['fp']}"
    )
    print('\n  FN by class:')
    for k, v in agg['fn_by_class'].items():
        print(f"    {k:18s} {v:4d}")
    print('  FP by class:')
    for k, v in agg['fp_by_class'].items():
        print(f"    {k:18s} {v:4d}")
    print(f"\n  Review {review_label}: {len(review)} sentences → {review_path}")

    diagnosis_paths: List[str] = []
    if with_diagnosis:
        fp_types = diagnosis_fp_types or ['PROD', 'EVT', 'ORG']
        fn_types = diagnosis_fn_types or ['LOC']

        matrix = build_confusion_matrix(sentence_results)
        cm_path = os.path.join(output_dir, f'confusion_matrix{suffix}.md')
        with open(cm_path, 'w', encoding='utf-8') as f:
            f.write('# Confusion matrix — gold (행) × pred (열)\n\n')
            f.write(
                '대각선 = (EXACT + BOUNDARY) 합. off-diagonal = TYPE_MISMATCH '
                '(FN side 만 카운트). 우측 마지막 열 ∅ = MISS, 하단 마지막 행 '
                '∅ = HALLUCINATION.\n\n'
            )
            f.write(render_confusion_matrix_md(matrix) + '\n')
        logger.info('Saved %s', cm_path)
        diagnosis_paths.append(cm_path)

        for t in fp_types:
            top = top_errors_by_type(
                sentence_results, side='fp', type_=t,
                n=diagnosis_top_n, ctx_chars=diagnosis_ctx_chars,
            )
            path = os.path.join(output_dir, f'fp_top_{t}{suffix}.md')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(render_top_errors_md(top, f'{t} FP top {len(top)}'))
                f.write('\n')
            logger.info('Saved %s', path)
            diagnosis_paths.append(path)

        for t in fn_types:
            top = top_errors_by_type(
                sentence_results, side='fn', type_=t,
                n=diagnosis_top_n, ctx_chars=diagnosis_ctx_chars,
            )
            path = os.path.join(output_dir, f'fn_top_{t}{suffix}.md')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(render_top_errors_md(top, f'{t} FN top {len(top)}'))
                f.write('\n')
            logger.info('Saved %s', path)
            diagnosis_paths.append(path)

        print(f"\n  Diagnosis: {len(diagnosis_paths)} files → {output_dir}")

    return {
        'aggregate': agg,
        'review_size': len(review),
        'diagnosis_paths': diagnosis_paths,
    }


def load_pooled_predictions(fold_dirs: List[str]) -> List[dict]:
    """K-fold 각 fold 의 test_predictions.json 을 합쳐 문장 리스트로 변환.

    재추론 없이 저장된 fold 예측을 진단 파이프라인 입력 형식
    ({'sent_idx', 'text', 'gold_spans', 'pred_spans'}) 으로 펼친다.
    sent_idx 는 fold 를 가로지르는 전역 일련번호. span dict 에 'text' 키가
    없으면 문장 text 에서 surface 를 복원해 채운다 (검수 산출용).

    무결성: fold 간 중복 text 가 있으면 ValueError (kfold_pool 과 동일
    규칙 — 같은 문장이 두 번 test 되면 분할 오류 또는 leak 신호).

    Args:
        fold_dirs: 각 fold output_dir (안에 test_predictions.json 존재)

    Returns:
        문장 리스트. 호출자는 fold_dirs 가 전체 fold 를 빠짐없이 포함하는지
        직접 확인해야 한다 (누락 fold 는 감지되지 않음).
    """
    sentences: List[dict] = []
    seen_texts: set = set()
    idx = 0
    for fold_dir in fold_dirs:
        preds_path = os.path.join(fold_dir, 'test_predictions.json')
        with open(preds_path, encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            text = rec.get('text', '')
            if text:
                if text in seen_texts:
                    raise ValueError(
                        f'duplicate test sentence across folds '
                        f'(in {preds_path}): {text[:50]!r}'
                    )
                seen_texts.add(text)
            sentences.append({
                'sent_idx': idx,
                'text': text,
                'gold_spans': [_with_surface(s, text) for s in rec['gold_spans']],
                'pred_spans': [_with_surface(s, text) for s in rec['pred_spans']],
            })
            idx += 1
    return sentences


def _with_surface(span: dict, text: str) -> dict:
    """span dict 에 'text' 키가 없으면 문장 text 에서 surface 복원."""
    if 'text' in span:
        return span
    return {**span, 'text': text[span['start']:span['end']]}


def run_pooled_error_analysis(
    *,
    lang: str,
    fold_dirs: List[str],
    output_dir: str,
    seed: int = 42,
    with_diagnosis: bool = False,
    diagnosis_fp_types: Optional[List[str]] = None,
    diagnosis_fn_types: Optional[List[str]] = None,
    diagnosis_top_n: int = 50,
    diagnosis_ctx_chars: int = 20,
) -> dict:
    """K-fold pooled 예측 → 오류 분류·집계·진단 (재추론 없음).

    저장된 fold 별 test_predictions.json 을 합쳐 전체 코퍼스에 대한 단일
    진단을 낸다. split 라벨은 'pooled' (산출 파일 suffix = '_pooled').
    """
    logger.info('Loading pooled predictions from %d folds', len(fold_dirs))
    sentences = load_pooled_predictions(fold_dirs)
    logger.info('Loaded %d pooled sentences', len(sentences))
    return _analyze_sentences(
        sentences,
        lang=lang,
        output_dir=output_dir,
        split='pooled',
        seed=seed,
        meta={'fold_dirs': list(fold_dirs)},
        with_diagnosis=with_diagnosis,
        diagnosis_fp_types=diagnosis_fp_types,
        diagnosis_fn_types=diagnosis_fn_types,
        diagnosis_top_n=diagnosis_top_n,
        diagnosis_ctx_chars=diagnosis_ctx_chars,
    )


def run_error_analysis(
    *,
    lang: str,
    model_path: str,
    data_path: str,
    output_dir: str,
    tokenizer_name: str,
    split: str = 'test',
    review_ratio: float = 0.1,
    review_min_per_type: int = 3,
    seed: int = 42,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_length: int = 256,
    batch_size: int = 32,
    with_diagnosis: bool = False,
    diagnosis_fp_types: Optional[List[str]] = None,
    diagnosis_fn_types: Optional[List[str]] = None,
    diagnosis_top_n: int = 50,
    diagnosis_ctx_chars: int = 20,
) -> dict:
    """전체 파이프라인: 추론 → 분류 → 집계 → 검수 샘플 → 산출물 저장.

    출력 파일명은 split 별로 분리된다:
    - 'test' (default): 기존 호환을 위해 suffix 없음 (error_analysis.json,
      review_sample.jsonl)
    - 'valid' / 'train': suffix 부여 + 샘플링 생략, 모든 오류 문장을
      review_full_{split}.jsonl 에 직접 출력 (full review 기본 가정)
    """
    logger.info('Loading model and running inference (lang=%s, split=%s)',
                lang, split)
    sentences = run_inference(
        lang=lang,
        model_path=model_path,
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        split=split,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
        max_length=max_length,
        batch_size=batch_size,
    )
    logger.info('Inference done: %d sentences', len(sentences))

    return _analyze_sentences(
        sentences,
        lang=lang,
        output_dir=output_dir,
        split=split,
        seed=seed,
        meta={'model_path': model_path, 'data_path': data_path},
        review_ratio=review_ratio,
        review_min_per_type=review_min_per_type,
        with_diagnosis=with_diagnosis,
        diagnosis_fp_types=diagnosis_fp_types,
        diagnosis_fn_types=diagnosis_fn_types,
        diagnosis_top_n=diagnosis_top_n,
        diagnosis_ctx_chars=diagnosis_ctx_chars,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Test-set error analysis (classifier).'
    )
    parser.add_argument('--lang', choices=['ja', 'vi'], required=True)
    parser.add_argument(
        '--from-predictions', action='store_true',
        help='Diagnose saved K-fold pooled predictions instead of running '
             'inference. Requires --fold-dirs; ignores --model-path/'
             '--tokenizer-name/--data/--split.',
    )
    parser.add_argument(
        '--fold-dirs', nargs='+',
        help='K-fold output dirs (each with test_predictions.json) for '
             '--from-predictions mode.',
    )
    parser.add_argument(
        '--model-path',
        help='HF model dir (e.g., results/classifier/ja_sweep/baseline/best). '
             'Required unless --from-predictions.',
    )
    parser.add_argument(
        '--tokenizer-name',
        help='Original HF tokenizer id used during training '
             '(trainer.save_model only stores model weights/config, not '
             'tokenizer). e.g., tohoku-nlp/bert-base-japanese-v3. '
             'Required unless --from-predictions.',
    )
    parser.add_argument(
        '--data',
        help='Override JSONL path (default: stockmark JA / wikiann VI)',
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory to write error_analysis.json + review_sample.jsonl',
    )
    parser.add_argument(
        '--split', choices=['test', 'valid', 'train'], default='test',
        help='Dataset split to analyze (default: test). valid/train write '
             'review_full_{split}.jsonl with all error sentences (no sampling).',
    )
    parser.add_argument('--review-ratio', type=float, default=0.1)
    parser.add_argument('--review-min-per-type', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--with-diagnosis', action='store_true',
        help='Generate confusion_matrix.md + fp_top/fn_top per-type dumps '
             '(default fp=PROD,EVT,ORG / fn=LOC).',
    )
    parser.add_argument(
        '--diagnosis-fp-types', default='PROD,EVT,ORG',
        help='Comma-separated entity types for FP top dump (with --with-diagnosis).',
    )
    parser.add_argument(
        '--diagnosis-fn-types', default='LOC',
        help='Comma-separated entity types for FN top dump (with --with-diagnosis).',
    )
    parser.add_argument(
        '--diagnosis-top-n', type=int, default=50,
        help='Max entries per top-N dump (default: 50).',
    )
    parser.add_argument(
        '--diagnosis-ctx-chars', type=int, default=20,
        help='Surrounding context chars per side in top dumps (default: 20).',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    fp_types = [t.strip() for t in args.diagnosis_fp_types.split(',') if t.strip()]
    fn_types = [t.strip() for t in args.diagnosis_fn_types.split(',') if t.strip()]

    if args.from_predictions:
        if not args.fold_dirs:
            parser.error('--from-predictions requires --fold-dirs')
        run_pooled_error_analysis(
            lang=args.lang,
            fold_dirs=args.fold_dirs,
            output_dir=args.output_dir,
            seed=args.seed,
            with_diagnosis=args.with_diagnosis,
            diagnosis_fp_types=fp_types,
            diagnosis_fn_types=fn_types,
            diagnosis_top_n=args.diagnosis_top_n,
            diagnosis_ctx_chars=args.diagnosis_ctx_chars,
        )
        return

    if not args.model_path or not args.tokenizer_name:
        parser.error(
            '--model-path and --tokenizer-name are required '
            '(unless --from-predictions)'
        )

    default_data = {
        'ja': 'data/stockmark/pii_all.jsonl',
        'vi': 'data/wikiann_vi/pii_all.jsonl',
    }
    data_path = args.data or default_data[args.lang]

    run_error_analysis(
        lang=args.lang,
        model_path=args.model_path,
        data_path=data_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        split=args.split,
        review_ratio=args.review_ratio,
        review_min_per_type=args.review_min_per_type,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        batch_size=args.batch_size,
        with_diagnosis=args.with_diagnosis,
        diagnosis_fp_types=fp_types,
        diagnosis_fn_types=fn_types,
        diagnosis_top_n=args.diagnosis_top_n,
        diagnosis_ctx_chars=args.diagnosis_ctx_chars,
    )


if __name__ == '__main__':
    main()
