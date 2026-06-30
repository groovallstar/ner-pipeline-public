"""서버 추론 처리량 벤치 — production 경로(predict) end-to-end 측정.

배치 엔드포인트의 현행 경로(텍스트별 순차 predict)를 baseline 으로 동결하고,
정밀도(fp32/tf32/bf16)를 바꿔가며 처리량을 비교한다. 측정 경계는
tokenize→forward→decode→threshold(abstain) 까지 end-to-end 다. 입력은 배포
test set(`/data/ner/{lang}/data/test.jsonl`) 의 고정 분포를 쓴다 — parity
측정과 동일 입력이라 처리량·정밀도를 같은 기준으로 비교할 수 있다.

`--mode` 로 seq(텍스트별 단건 predict)·batch(predict_many cross-text 묶음)를
고른다 — bf16 은 배치(B>1)에서만 켜진다.
"""

import argparse
import json
import statistics
import time
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


def _set_tf32(enabled: bool) -> None:
    """matmul/cudnn TF32 토글(fp32 baseline 은 off 로 동결)."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _run_seq(lm: LangModel, texts: List[str], abstain: bool) -> None:
    """순차 경로 — 텍스트별 단건 predict(배치 핸들러의 옛 경로 = baseline).

    정밀도는 lm.autocast_dtype(predict 내장 autocast)이 결정한다 — 측정과
    출시 런타임이 같은 코드 경로를 타도록 바깥에서 감싸지 않는다.
    """
    for t in texts:
        lm.predict(t, abstain=abstain)


def _run_batch(lm: LangModel, texts: List[str], abstain: bool,
               batch_size: int) -> None:
    """배치 경로 — batch_size 씩 묶어 predict_many(cross-text forward)."""
    for i in range(0, len(texts), batch_size):
        lm.predict_many(texts[i:i + batch_size], abstain=abstain)


def measure(lm: LangModel, texts: List[str], abstain: bool, warmup: int,
            reps: int, mode: str, batch_size: int) -> List[float]:
    """warmup 후 reps 회, 전체 셋 처리시간(ms) 분포를 잰다(seq|batch)."""
    def warm() -> None:
        if mode == 'batch':
            lm.predict_many(texts[:batch_size], abstain=abstain)
        else:
            lm.predict(texts[0], abstain=abstain)

    def run() -> None:
        if mode == 'batch':
            _run_batch(lm, texts, abstain, batch_size)
        else:
            _run_seq(lm, texts, abstain)

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


def _resolve_precision(lm: LangModel, precision: str) -> None:
    """정밀도 인자 → TF32 부수효과 + lm.autocast_dtype 설정."""
    if precision == 'tf32':
        _set_tf32(True)
        lm.autocast_dtype = None
    elif precision == 'fp32':
        _set_tf32(False)
        lm.autocast_dtype = None
    else:  # bf16
        _set_tf32(False)
        lm.autocast_dtype = torch.bfloat16


def main() -> None:
    ap = argparse.ArgumentParser(description='NER server throughput bench')
    ap.add_argument('--lang', required=True, choices=['ja', 'vi'])
    ap.add_argument('--precision', default='fp32',
                    choices=['fp32', 'tf32', 'bf16'])
    ap.add_argument('--mode', default='seq', choices=['seq', 'batch'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--abstain', default='true', choices=['true', 'false'])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default=None, help='write result JSON to path')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for throughput bench')

    abstain = args.abstain == 'true'
    cfg = ServerConfig.from_env()
    lm = LangModel(args.lang, cfg.model_dir(args.lang),
                   cfg.thresholds_path(args.lang), cfg.max_length)
    texts = load_test_texts(args.lang, cfg.model_root)
    _resolve_precision(lm, args.precision)

    times = measure(lm, texts, abstain, args.warmup, args.reps,
                    args.mode, args.batch_size)
    n = len(texts)
    med = statistics.median(times)
    tokens = count_tokens(lm, texts)
    result = {
        'lang': args.lang,
        'config': {
            'precision': args.precision,
            'tf32': args.precision == 'tf32',
            'mode': args.mode,
            'batch_size': args.batch_size if args.mode == 'batch' else None,
            'abstain': abstain,
            'thresholds_applied': lm.has_thresholds and abstain,
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
