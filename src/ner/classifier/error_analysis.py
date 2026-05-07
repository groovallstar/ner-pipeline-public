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

logger = logging.getLogger(__name__)


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
            'model_path': model_path,
            'data_path': data_path,
            'split': split,
            'seed': seed,
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
        # valid/train: 전체 오류 문장을 review_full_{split}.jsonl 로 직접 출력
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

    return {'aggregate': agg, 'review_size': len(review)}


def main():
    parser = argparse.ArgumentParser(
        description='Test-set error analysis (classifier).'
    )
    parser.add_argument('--lang', choices=['ja', 'vi'], required=True)
    parser.add_argument(
        '--model-path', required=True,
        help='HF model dir (e.g., results/classifier/ja_sweep/baseline/best)',
    )
    parser.add_argument(
        '--tokenizer-name', required=True,
        help='Original HF tokenizer id used during training '
             '(trainer.save_model only stores model weights/config, not '
             'tokenizer). e.g., tohoku-nlp/bert-base-japanese-v3',
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
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
    )


if __name__ == '__main__':
    main()
