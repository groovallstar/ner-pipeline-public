"""서버 추론 처리량 벤치 — production 경로(predict) end-to-end 측정.

배치 엔드포인트의 현행 경로(seq=텍스트별 단건 predict / batch=predict_many
cross-text 묶음)의 처리량을 잰다. 측정 경계는 tokenize→forward→decode→
threshold(apply_threshold) 까지 end-to-end 다. 입력은 배포 test set
(`/data/ner/{lang}/data/test.jsonl`) 의 고정 분포를 쓴다.

서빙은 fp32 로만 돈다(운영점 정합·결정성) — precision 은 벤치 인자가 아니다.

`--concurrency N` 은 같은 작업량을 N 스레드로 나눠 돌린다. 서버가 추론을
`run_in_threadpool` 로 돌리므로 운영에서는 이쪽이 실제 경로이고, 토크나이저
직렬화처럼 **경합이 있어야 드러나는** 비용은 N=1 에서는 아예 측정되지 않는다.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from server.config import ServerConfig
from server.inference import LangModel


def load_test_texts(lang: str, model_root: str) -> List[str]:
    """배포 test set 의 원문 텍스트(파일 순서 고정)."""
    path = f'{model_root}/{lang}/data/test.jsonl'
    texts: List[str] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)['text'])
    return texts


def _run_seq(lm: LangModel, texts: List[str], apply_threshold: bool) -> None:
    """순차 경로 — 텍스트별 단건 predict."""
    for t in texts:
        lm.predict(t, apply_threshold=apply_threshold)


def _run_batch(lm: LangModel, texts: List[str], apply_threshold: bool,
               batch_size: int) -> None:
    """배치 경로 — batch_size 씩 묶어 predict_many(cross-text forward)."""
    for i in range(0, len(texts), batch_size):
        lm.predict_many(texts[i:i + batch_size],
                        apply_threshold=apply_threshold)


def _run_concurrent(lm: LangModel, texts: List[str], apply_threshold: bool,
                    mode: str, batch_size: int, concurrency: int) -> None:
    """같은 텍스트 셋을 concurrency 스레드로 나눠 돌린다(총 작업량 불변).

    스레드마다 다른 몫을 맡아 전체 셋을 정확히 한 번 처리하므로, N=1 과 N=8 의
    전체 처리시간을 그대로 견줄 수 있다. 나누기는 stride(`texts[i::N]`)라 길이
    분포가 스레드에 고르게 퍼진다 — 앞뒤로 자르면 긴 문장이 한 스레드에 몰려
    꼬리가 전체 시간을 지배한다.
    """
    def shard(texts_part: List[str]) -> None:
        if mode == 'batch':
            _run_batch(lm, texts_part, apply_threshold, batch_size)
        else:
            _run_seq(lm, texts_part, apply_threshold)

    shards = [texts[i::concurrency] for i in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in ex.map(shard, shards):
            pass


def measure(lm: LangModel, texts: List[str], apply_threshold: bool,
            warmup: int, reps: int, mode: str,
            batch_size: int, concurrency: int = 1) -> List[float]:
    """warmup 후 reps 회, 전체 셋 처리시간(ms) 분포를 잰다(seq|batch)."""
    def warm() -> None:
        if mode == 'batch':
            lm.predict_many(texts[:batch_size],
                            apply_threshold=apply_threshold)
        else:
            lm.predict(texts[0], apply_threshold=apply_threshold)

    def run() -> None:
        if concurrency > 1:
            _run_concurrent(lm, texts, apply_threshold, mode, batch_size,
                            concurrency)
        elif mode == 'batch':
            _run_batch(lm, texts, apply_threshold, batch_size)
        else:
            _run_seq(lm, texts, apply_threshold)

    for _ in range(warmup):
        warm()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def count_tokens(lm: LangModel, texts: List[str]) -> int:
    """입력 토큰 총수(special 제외 근사) — tokens/sec 보조지표용."""
    total = 0
    for t in texts:
        enc = lm.tokenizer(t, add_special_tokens=False)
        total += len(enc['input_ids'])
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description='NER server throughput bench')
    ap.add_argument('--lang', required=True, choices=['ja', 'ko', 'vi', 'en'])
    ap.add_argument('--mode', default='seq', choices=['seq', 'batch'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--apply-threshold', default='true',
                    choices=['true', 'false'])
    ap.add_argument('--concurrency', type=int, default=1,
                    help='threads sharing the model (server path uses >1)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default=None, help='write result JSON to path')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for throughput bench')

    apply_threshold = args.apply_threshold == 'true'
    cfg = ServerConfig.from_env()
    lm = LangModel(args.lang, cfg.model_dir(args.lang),
                   cfg.thresholds_path(args.lang), cfg.max_length)
    texts = load_test_texts(args.lang, cfg.model_root)

    if args.concurrency < 1:
        raise SystemExit('--concurrency must be >= 1')
    times = measure(lm, texts, apply_threshold, args.warmup, args.reps,
                    args.mode, args.batch_size, args.concurrency)
    n = len(texts)
    med = statistics.median(times)
    tokens = count_tokens(lm, texts)
    result = {
        'lang': args.lang,
        'config': {
            'mode': args.mode,
            'concurrency': args.concurrency,
            'batch_size': args.batch_size if args.mode == 'batch' else None,
            'apply_threshold': apply_threshold,
            'thresholds_applied': lm.has_thresholds and apply_threshold,
            'n_texts': n,
            'tokens_total': tokens,
            'max_length': cfg.max_length,
            'warmup': args.warmup,
            'reps': args.reps,
            'device': torch.cuda.get_device_name(0),
        },
        'throughput_texts_per_sec': round(n / (med / 1000), 2),
        'throughput_tokens_per_sec': round(tokens / (med / 1000), 1),
        'total_ms': {
            'median': round(med, 2),
            'min': round(min(times), 2),
            'max': round(max(times), 2),
        },
        'per_text_ms_median': round(med / n, 3),
        'reps_ms': [round(x, 2) for x in times],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
