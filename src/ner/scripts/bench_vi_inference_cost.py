"""VI classifier 추론 비용 비교 — pyvi 전처리 오버헤드 + 레이턴시·처리량·메모리.

accuracy 동률(phobert ≈ xlm-r-base) 상황에서 배포 선택을 가르는 비용 축을
모델별로 측정한다. 학습/평가는 하지 않고, 고정 fold0 test 문장셋에 대해
전처리(encode)·forward 추론·메모리 비용만 잰다. 정확도는 #103 확정값 인용.

핵심: PhoBERT의 pyvi 단어분절 비용은 forward 가 아니라 encode 단계에 있으므로
encode 시간을 별도 계측한다(evaluate_model.capture_timing 은 forward 만 잰다).

사용:
    CUDA_VISIBLE_DEVICES=0 python src/ner/scripts/bench_vi_inference_cost.py \
        --out results/classifier/vi/canonical5fold/inference_cost.json
"""
import argparse
import json
import os
import statistics
import time
from typing import List

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from ner.classifier.data_utils import (
    build_label_maps,
    encode_dataset,
    load_jsonl,
    split_kfold_stratified,
)

DEFAULT_DATA = 'data/wikiann_vi/pii_all.jsonl'
DEFAULT_ROOT = 'results/classifier/vi/canonical5fold'
# (라벨, 체크포인트 디렉토리명, 토크나이저 hub id)
DEFAULT_MODELS = [
    ('phobert-base-v2', 'phobert-base-v2', 'vinai/phobert-base-v2'),
    ('xlm-roberta-base', 'xlm-roberta-base', 'xlm-roberta-base'),
    ('mmbert-base', 'mmbert-base', 'jhu-clsp/mmBERT-base'),
    ('cafebert', 'cafebert', 'uitnlp/CafeBERT'),
    ('xlm-roberta-large', 'xlm-roberta-large', 'xlm-roberta-large'),
]


def load_tokenizer(hub_id: str):
    """학습 경로 미러 — fast 우선, 실패 시 slow fallback.

    vinai/phobert-base-v2 는 fast 토크나이저가 없어 자동 slow 로 떨어지고
    data_utils 의 pyvi(_encode_phobert) 경로를 탄다.
    """
    try:
        return AutoTokenizer.from_pretrained(hub_id, use_fast=True)
    except (TypeError, ValueError, OSError):
        return AutoTokenizer.from_pretrained(hub_id, use_fast=False)


def measure_throughput(model, features: List[dict], device,
                       batch_size: int):
    """batch_size forward 루프 전체 시간 + peak GPU memory.

    argmax 까지(.cpu()) 수행해 실제 추론 1회와 동일한 동기화 지점을 만든다.
    반환: (seconds, peak_mem_mb).
    """
    n = len(features)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = features[i:i + batch_size]
            input_ids = torch.tensor(
                [f['input_ids'] for f in batch], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask'] for f in batch], dtype=torch.long
            ).to(device)
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits.argmax(dim=-1).cpu().numpy()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - t0
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mb = 0.0
    return seconds, peak_mb


def measure_latency(model, features: List[dict], device,
                    n_samples: int, warmup: int = 10):
    """batch=1 단문 추론 레이턴시(ms) — warmup 후 문장별 동기화 측정 median.

    온라인 서빙(1문장씩)의 현실적 지연을 잰다.
    """
    samples = features[:n_samples]
    times_ms: List[float] = []
    with torch.no_grad():
        for k, f in enumerate(samples):
            input_ids = torch.tensor(
                [f['input_ids']], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask']], dtype=torch.long
            ).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits.argmax(dim=-1).cpu().numpy()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1000.0
            if k >= warmup:
                times_ms.append(dt)
    return {
        'median_ms': statistics.median(times_ms),
        'mean_ms': statistics.fmean(times_ms),
        'p90_ms': sorted(times_ms)[int(len(times_ms) * 0.9)],
    }


def bench_one(label: str, ckpt_dir: str, hub_id: str, test_rows,
              label2id, device, batch_size: int, latency_samples: int):
    """단일 모델 비용 측정 → dict."""
    best = os.path.join(ckpt_dir, 'best')
    print(f'\n=== {label} ===')

    t0 = time.perf_counter()
    tok = load_tokenizer(hub_id)
    tok_load_s = time.perf_counter() - t0
    print(f'tokenizer fast={tok.is_fast} load={tok_load_s:.2f}s')

    # encode — pyvi(phobert)는 여기 포함
    t0 = time.perf_counter()
    feats, _ = encode_dataset(test_rows, tok, label2id, 'vi', 256)
    encode_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = AutoModelForTokenClassification.from_pretrained(
        best, dtype=torch.float32
    ).to(device).eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    model_load_s = time.perf_counter() - t0

    n = len(test_rows)
    param_count = sum(p.numel() for p in model.parameters())
    disk_mb = os.path.getsize(
        os.path.join(best, 'model.safetensors')) / (1024 ** 2)

    thr_s, peak_mb = measure_throughput(model, feats, device, batch_size)
    lat = measure_latency(model, feats, device, latency_samples)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    res = {
        'label': label,
        'hub_id': hub_id,
        'tokenizer_fast': tok.is_fast,
        'n_test': n,
        'param_count_m': round(param_count / 1e6, 1),
        'disk_mb': round(disk_mb, 1),
        'tok_load_s': round(tok_load_s, 2),
        'model_load_s': round(model_load_s, 2),
        'encode_s_total': round(encode_s, 2),
        'encode_ms_per_sent': round(encode_s / n * 1000, 3),
        'throughput_sent_s': round(n / thr_s, 1),
        'forward_total_s': round(thr_s, 2),
        'latency_b1': {k: round(v, 3) for k, v in lat.items()},
        'peak_gpu_mem_mb': round(peak_mb, 1),
    }
    print(f"encode={res['encode_ms_per_sent']}ms/sent "
          f"throughput={res['throughput_sent_s']}/s "
          f"latency_b1={res['latency_b1']['median_ms']}ms "
          f"peak_mem={res['peak_gpu_mem_mb']}MB "
          f"params={res['param_count_m']}M disk={res['disk_mb']}MB")
    return res


def main():
    p = argparse.ArgumentParser(
        description='Benchmark VI classifier inference cost (encode/pyvi '
                    'overhead, throughput, latency, peak memory) on the '
                    'fixed fold0 test set. Measurement only; no training.')
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--results-root', default=DEFAULT_ROOT)
    p.add_argument('--fold-index', type=int, default=0)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--latency-samples', type=int, default=200)
    p.add_argument('--out', default=None,
                   help='Output JSON path (default: print only)')
    args = p.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} visible={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    label2id, _ = build_label_maps()
    rows = load_jsonl(args.data)
    _, _, test_rows = split_kfold_stratified(
        rows, n_folds=args.n_folds, fold_index=args.fold_index, seed=42)
    print(f'fold{args.fold_index} test = {len(test_rows)} sentences')

    results = []
    for label, sub, hub_id in DEFAULT_MODELS:
        ckpt = os.path.join(
            args.results_root, sub, f'fold{args.fold_index}')
        if not os.path.isdir(os.path.join(ckpt, 'best')):
            print(f'SKIP {label}: no checkpoint at {ckpt}/best')
            continue
        results.append(bench_one(
            label, ckpt, hub_id, test_rows, label2id, device,
            args.batch_size, args.latency_samples))

    out = {
        'fold_index': args.fold_index,
        'n_folds': args.n_folds,
        'n_test': len(test_rows),
        'batch_size': args.batch_size,
        'latency_samples': args.latency_samples,
        'models': results,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
