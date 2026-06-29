"""정밀도 parity 측정 — production 경로 결정-불일치 카운트(축 분해).

fp32(레퍼런스) vs bf16(autocast) 의 `predict(abstain=True)` 출력 span 집합을
엔티티별로 비교해 등장·소멸·라벨변경된 span 의 절대 개수를 센다. 게이트는
엔티티별 변경 span ≤ max(2, 0.5%×support) — support 는 fp32 레퍼런스 span 수.
bf16 run-to-run 재현성(>=3 run 대칭차 worst-case)도 측정한다. micro-F1 델타는
진단용(게이트 아님)이다.

측정 경로는 출시 구성 — ref=fp32 순차, cmp=bf16 배치(predict_many 경로). 단건은
fp32, bf16 은 배치(B>1) forward 에만 적용되므로 게이트는 fp32순차 대 bf16배치
2지점이다(축 라벨 `fp32_seq -> bf16_batch`).
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from server.config import ServerConfig
from server.inference import LangModel

SpanKey = Tuple[str, int, int]  # (label, start_char, end_char)


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


def predict_all(lm: LangModel, texts: List[str], abstain: bool,
                autocast_dtype: Optional[torch.dtype], mode: str = 'seq',
                batch_size: int = 32) -> List[Set[SpanKey]]:
    """텍스트별 predict 결과를 (label,start,end) 집합 리스트로 반환.

    정밀도는 lm.autocast_dtype 로, 경로는 mode(seq=단건 predict /
    batch=predict_many cross-text 묶음)로 제어한다 — 측정과 출시 런타임이
    동일 코드 경로를 탄다. bf16 은 batch(B>1)에서만 발동한다.
    """
    lm.autocast_dtype = autocast_dtype

    def to_keys(spans) -> Set[SpanKey]:
        return {(s['label'], s['start_char'], s['end_char']) for s in spans}

    rows: List[Set[SpanKey]] = []
    if mode == 'batch':
        for i in range(0, len(texts), batch_size):
            for spans in lm.predict_many(texts[i:i + batch_size],
                                         abstain=abstain):
                rows.append(to_keys(spans))
    else:
        for t in texts:
            rows.append(to_keys(lm.predict(t, abstain=abstain)))
    return rows


def decision_mismatch(ref: List[Set[SpanKey]], cmp: List[Set[SpanKey]]):
    """엔티티별 등장·소멸·라벨변경 span 절대수 + fp32 support 를 센다.

    위치(start,end) 가 같고 label 만 다르면 라벨변경 1회(ref 라벨 귀속).
    위치가 ref 에만/ cmp 에만 있으면 각각 소멸/등장 1회.
    """
    changed: Dict[str, int] = defaultdict(int)
    support: Dict[str, int] = defaultdict(int)
    for r_set, c_set in zip(ref, cmp):
        r_pos = {(s, e): lab for (lab, s, e) in r_set}
        c_pos = {(s, e): lab for (lab, s, e) in c_set}
        for (lab, _s, _e) in r_set:
            support[lab] += 1
        for pos, lab in r_pos.items():
            if pos not in c_pos:
                changed[lab] += 1            # 소멸
            elif c_pos[pos] != lab:
                changed[lab] += 1            # 라벨변경(ref 라벨)
        for pos, lab in c_pos.items():
            if pos not in r_pos:
                changed[lab] += 1            # 등장(cmp 라벨)
    return changed, support


def gate(changed: Dict[str, int], support: Dict[str, int]):
    """엔티티별 변경 span ≤ max(2, 0.5%×support) 게이트 판정."""
    rows = []
    ok = True
    for lab in sorted(set(changed) | set(support)):
        sup = support.get(lab, 0)
        chg = changed.get(lab, 0)
        thr = max(2.0, 0.005 * sup)
        passed = chg <= thr
        ok = ok and passed
        rows.append({'label': lab, 'support': sup, 'changed': chg,
                     'threshold': round(thr, 2), 'passed': passed})
    return ok, rows


def reproducibility(lm: LangModel, texts: List[str], abstain: bool,
                    dtype: torch.dtype, runs: int, mode: str = 'seq',
                    batch_size: int = 32) -> int:
    """run-to-run 대칭차 worst-case(0 이면 결정적). 출시 경로(mode)로 잰다."""
    base = predict_all(lm, texts, abstain, dtype, mode, batch_size)
    worst = 0
    for _ in range(max(0, runs - 1)):
        nxt = predict_all(lm, texts, abstain, dtype, mode, batch_size)
        diff = sum(len(a ^ b) for a, b in zip(base, nxt))
        worst = max(worst, diff)
    return worst


def micro_f1(ref: List[Set[SpanKey]], cmp: List[Set[SpanKey]]):
    """fp32 를 기준으로 본 bf16 의 span micro-F1(진단)."""
    tp = fp = fn = 0
    for r, c in zip(ref, cmp):
        tp += len(r & c)
        fp += len(c - r)
        fn += len(r - c)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'micro_f1': round(f1, 4), 'tp': tp, 'fp': fp, 'fn': fn}


def main() -> None:
    ap = argparse.ArgumentParser(description='bf16 parity vs fp32')
    ap.add_argument('--lang', required=True, choices=['ja', 'vi'])
    ap.add_argument('--repro-runs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for parity measurement')

    cfg = ServerConfig.from_env()
    lm = LangModel(args.lang, cfg.model_dir(args.lang),
                   cfg.thresholds_path(args.lang), cfg.max_length)
    texts = load_test_texts(args.lang, cfg.model_root)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = predict_all(lm, texts, abstain=True, autocast_dtype=None)
    bf16 = predict_all(lm, texts, abstain=True,
                       autocast_dtype=torch.bfloat16, mode='batch',
                       batch_size=args.batch_size)
    changed, support = decision_mismatch(ref, bf16)
    ok, per_label = gate(changed, support)
    worst = reproducibility(lm, texts, True, torch.bfloat16,
                            args.repro_runs, mode='batch',
                            batch_size=args.batch_size)

    result = {
        'lang': args.lang,
        'axis': 'fp32_seq -> bf16_batch',
        'config': {
            'abstain': True,
            'thresholds_applied': lm.has_thresholds,
            'n_texts': len(texts),
            'batch_size': args.batch_size,
            'repro_runs': args.repro_runs,
            'device': torch.cuda.get_device_name(0),
        },
        'gate': {
            'passed': ok,
            'rule': 'changed <= max(2, 0.5%*support)',
            'total_changed': sum(changed.values()),
            'per_label': per_label,
        },
        'reproducibility': {
            'runs': args.repro_runs,
            'worst_symdiff': worst,
            'deterministic': worst == 0,
        },
        'diagnostic': micro_f1(ref, bf16),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
