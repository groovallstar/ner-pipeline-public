"""영어 NER 배포 패키지 빌드 — 학습 산출물을 배포 레이아웃으로 포장한다.

**학습은 하지 않는다.** `python -m ner.classifier` 가 낸 run 디렉토리를 읽어
ja·vi 와 같은 배포 레이아웃으로 옮긴다. 학습 경로를 여기서 복제하면 배포
체크포인트가 CLI 가 아닌 이 스크립트의 산물이 돼, 원장(certified)에 올라간
벤치마크 수치와 맞춰 보는 일이 서로 다른 코드 경로를 비교하는 것이 된다.

분할 JSONL 은 run 이 저장하지 않으므로 같은 인자로 다시 유도한다
(`split_train_valid_test` 는 결정적이다). 유도한 크기가 run 의 `metrics.json`
이 기록한 train/valid/test 수와 어긋나면 중단한다 — 실제로 학습에 쓰이지 않은
분할을 배포 데이터로 적어 두는 것이 여기서 가능한 가장 조용한 실패다.

산출 구조 (ja·vi 동형):

    <out>/
    ├── model/          # best 가중치 + tokenizer (자립 로드)
    ├── data/           # train/valid/test.jsonl — 실제 학습에 쓰인 분할
    ├── metrics.json    # run 의 metrics.json + deploy_package 블록
    └── MODEL_CARD.md

사용:
    python src/ner/scripts/build_en_ner_prod.py \\
        --run-dir results/classifier/en/deploy-trainseed42
    python src/ner/scripts/build_en_ner_prod.py \\
        --run-dir <run> --out-dir /abs/out --force
"""
import argparse
import json
import logging
import os
import shutil

from transformers import AutoTokenizer

from ner.classifier.data_utils import (
    dataset_fingerprint,
    load_jsonl,
    split_train_valid_test,
)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger('build_en_ner_prod')

DEFAULT_OUT_DIR = '/data/ner/en'


def dump_jsonl(rows, path):
    """row 리스트를 JSONL 로 저장(원본 필드 보존)."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_run_metrics(run_dir):
    """run 의 metrics.json 을 읽고, 배포 포장이 가능한 run 인지 검사한다.

    포장 가능한 run 은 단일 분할·단일 스테이지다. k-fold 는 교차검증 추정이라
    배포할 체크포인트 하나를 가리키지 않고, curriculum·extra-train 은 기록된
    분할만으로 학습 데이터를 재현할 수 없다.
    """
    path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(path):
        raise SystemExit(f'Error: run metrics not found: {path}')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)

    if m.get('lang') != 'en':
        raise SystemExit(
            f"Error: run lang is {m.get('lang')!r}, expected 'en'")
    if m.get('kfold') is not None:
        raise SystemExit(
            'Error: k-fold run cannot be packaged for deployment '
            '(no single checkpoint); use a single-split run')
    if m.get('curriculum'):
        raise SystemExit(
            'Error: curriculum run cannot be packaged — the recorded split '
            'does not reproduce stage-1 training data')
    if m.get('data_extra_train_jsonl'):
        raise SystemExit(
            'Error: run used --data-extra-train-jsonl; the train split '
            'cannot be reproduced from the recorded arguments alone')
    return m


def rederive_split(m):
    """run 이 기록한 인자로 분할을 다시 유도하고 run 과 일치하는지 검사한다.

    지문은 데이터 내용(정답·순서)의 동일성을, 크기 대조는 유도 경로의 동일성을
    본다. 둘 중 하나만 봐서는 "같은 파일을 다르게 잘랐다"와 "다른 파일을 같은
    비율로 잘랐다"를 구별하지 못한다.
    """
    data_path = m['data_path']
    if not os.path.exists(data_path):
        raise SystemExit(f'Error: dataset not found: {data_path}')
    rows = load_jsonl(data_path)

    fp = dataset_fingerprint(rows)
    if fp != m['data_fingerprint']:
        raise SystemExit(
            f'Error: dataset fingerprint mismatch — run recorded '
            f"{m['data_fingerprint']}, current file is {fp}. The dataset "
            'changed since the run; re-train before packaging')

    group_key = m['group_key']
    group_key = None if group_key in (None, 'none') else group_key
    train_rows, valid_rows, test_rows = split_train_valid_test(
        rows, m['valid_ratio'], m['test_ratio'], m['seed'],
        group_key=group_key,
    )

    got = (len(train_rows), len(valid_rows), len(test_rows))
    want = (m['train_samples'], m['valid_samples'], m['test_samples'])
    if got != want:
        raise SystemExit(
            f'Error: re-derived split {got} does not match the run '
            f'{want}; refusing to ship a split the model was not trained on')

    # 누출 가드 — test 그룹이 train/valid 에 있으면 배포 데이터가 이미 오염이다.
    leak = 0
    if group_key:
        test_g = {r[group_key] for r in test_rows}
        seen_g = ({r[group_key] for r in train_rows}
                  | {r[group_key] for r in valid_rows})
        leak = len(test_g & seen_g)
        if leak:
            raise SystemExit(
                f'Error: {leak} test groups leaked into train/valid')

    return rows, train_rows, valid_rows, test_rows, leak


def build_model_dir(run_dir, out_dir, model_name, force):
    """best 가중치를 복사하고 tokenizer 를 함께 저장해 자립 로드를 만든다.

    `Trainer.save_model()` 은 가중치·config 만 남기므로, 그대로 배포하면
    `model/` 만으로는 토크나이즈가 안 된다. base 모델에서 tokenizer 를 받아
    같은 디렉토리에 저장해 소비자가 경로 하나만 알면 되게 한다.
    """
    best_dir = os.path.join(run_dir, 'best')
    if not os.path.isdir(best_dir):
        raise SystemExit(f'Error: best checkpoint not found: {best_dir}')

    model_dir = os.path.join(out_dir, 'model')
    if os.path.exists(model_dir):
        if not force:
            raise SystemExit(
                f'Error: {model_dir} already exists; pass --force to replace')
        shutil.rmtree(model_dir)
    shutil.copytree(best_dir, model_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except (TypeError, ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def write_model_card(path, *, m, leak, out_dir, run_dir, notes):
    """배포 모델 카드(MODEL_CARD.md) 작성 — ja·vi 패키지 포맷.

    운영 주의는 인자(`notes`)로 받는다. 이 체크포인트에만 해당하는 한계를
    템플릿에 박아 두면 다음 빌드에도 따라붙어, 카드가 사실이 아닌 것을
    말하게 된다.
    """
    op = m['overall_strict']
    per = m['per_entity_strict']
    ner5 = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')
    pii5 = ('DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD')

    def rows_for(types):
        """타입 목록을 마크다운 표 행으로 — run 에 없는 타입은 건너뛴다."""
        out = []
        for t in types:
            e = per.get(t)
            if e is None:
                continue
            out.append(
                f"| {t} | {e['precision']:.4f} | {e['recall']:.4f} "
                f"| {e['f1']:.4f} | {e['support']} |")
        return '\n'.join(out)

    leak_line = (f'`{m["group_key"]}` 그룹 단위(누출-free) — test 원문이 '
                 f'train/valid 에 {leak}건'
                 if m['group_key'] not in (None, 'none') else '행 단위 random')

    card = f"""# EN NER production 모델

영어 canonical 10종 평면 NER BERT 분류기. 실제 추론 배포용.
신뢰도 임계값(confidence threshold) 미적용 — raw 모델 출력 그대로.

## 구성

```
{out_dir}/
├── model/              # fine-tuned 가중치 + tokenizer (자립 로드)
│   ├── config.json · model.safetensors    # 모델 (BIO 21라벨)
│   └── tokenizer.json · tokenizer_config.json  # 토크나이저 (RoBERTa BPE)
├── metrics.json        # 학습 설정 + test 메트릭 + 누출 가드
├── data/               # 학습/검증/평가 split
│   ├── train.jsonl     # {m['train_samples']:,} 문장
│   ├── valid.jsonl     # {m['valid_samples']:,} 문장 (best-epoch 선택)
│   └── test.jsonl      # {m['test_samples']:,} 문장 (학습 미노출)
└── MODEL_CARD.md
```

## 학습 설정

| 항목 | 값 |
|---|---|
| base 모델 | `{m['model_name']}` |
| 토크나이저 | fast(RoBERTa BPE) — **`model/` 에 동봉**. 별도 런타임 의존 없음 |
| 라벨 | canonical 10종 = NER 5(PER/LOC/ORG/PROD/EVT) + PII 5(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD), BIO 21 |
| gold | `{m['data_path']}` ({m['n_rows']:,} 문장 · {m['n_groups']:,} 그룹) |
| 데이터 지문 | `{m['data_fingerprint']}` |
| split | valid_ratio={m['valid_ratio']} · test_ratio={m['test_ratio']} → {m['train_samples']}/{m['valid_samples']}/{m['test_samples']} |
| 분할 단위 | {leak_line} |
| 손실 | 표준 cross-entropy (boundary 가중 미적용) |
| 하이퍼파라미터 | epochs={m['epochs']} · batch={m['batch_size']} · lr={m['lr']} · max_len={m['max_length']} · {m['precision']} · seed={m['seed']} · train_seed={m['train_seed']} |
| best 선택 | valid `{m['metric_for_best']}` (낮을수록 좋음) |

## 성능 (test {m['test_samples']:,}건, strict span F1)

| | P | R | F1 | support |
|---|---:|---:|---:|---:|
| 전체(10종) | {op['precision']:.4f} | {op['recall']:.4f} | {op['f1']:.4f} | {op['support']} |

### NER 5종

| 타입 | P | R | F1 | support |
|---|---:|---:|---:|---:|
{rows_for(ner5)}

### PII 5종

| 타입 | P | R | F1 | support |
|---|---:|---:|---:|---:|
{rows_for(pii5)}

> PII 5종은 합성 주입이라 포화된다 — 모델 실력의 지표로 읽지 않는다.
> 실질 판정은 NER 5종, 그중에서도 support 가 작은 PROD·EVT 가 가른다.

## 추론 사용법

1. `model/` 을 `AutoTokenizer` · `AutoModelForTokenClassification` 로 로드
   (별도 tokenizer 다운로드 불필요).
2. char offset 디코드는 `ner.classifier.data_utils` 의 인코딩 경로와 짝을
   이룬다 — `src/ner/scripts/eval_en_ner_test.py` 가 그 사용 예다.

## 프로비넌스

`python -m ner.classifier --lang en` 산출물(`{run_dir}`)을
`src/ner/scripts/build_en_ner_prod.py` 로 포장했다. 학습 경로는 CLI 이고 이
카드의 수치는 그 run 의 `metrics.json` 에서 옮긴 값이다.
"""
    if notes:
        card += '\n## 운영 주의\n\n'
        card += '\n\n'.join(f'- {n}' for n in notes) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(card)


def main():
    p = argparse.ArgumentParser(
        description='Package a trained EN NER run into the deployment '
                    'layout (model/ + data/ + metrics.json + MODEL_CARD.md). '
                    'Does not train: point --run-dir at the output of '
                    '`python -m ner.classifier --lang en`.')
    p.add_argument('--run-dir', required=True,
                   help='Training run dir containing best/ and metrics.json')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help=f'Deployment package dir (default {DEFAULT_OUT_DIR})')
    p.add_argument('--force', action='store_true',
                   help='Replace an existing package at --out-dir')
    p.add_argument('--note', action='append', default=[],
                   help='Operational caveat for this checkpoint; repeatable. '
                        'Rendered as a MODEL_CARD section instead of being '
                        'baked into the template.')
    args = p.parse_args()

    m = load_run_metrics(args.run_dir)
    logger.info('Run: model=%s split=%s/%s/%s fingerprint=%s',
                m['model_name'], m['train_samples'], m['valid_samples'],
                m['test_samples'], m['data_fingerprint'])

    rows, train_rows, valid_rows, test_rows, leak = rederive_split(m)
    logger.info('Re-derived split matches the run (%d rows, leak=%d)',
                len(rows), leak)

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = build_model_dir(
        args.run_dir, args.out_dir, m['model_name'], args.force)
    logger.info('Model dir ready: %s', model_dir)

    data_dir = os.path.join(args.out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    dump_jsonl(train_rows, os.path.join(data_dir, 'train.jsonl'))
    dump_jsonl(valid_rows, os.path.join(data_dir, 'valid.jsonl'))
    dump_jsonl(test_rows, os.path.join(data_dir, 'test.jsonl'))

    # run 의 metrics 를 그대로 싣고 포장 사실만 덧붙인다 — 수치를 손대면
    # 원장 대조가 이 스크립트를 거친 값을 보게 된다.
    packaged = dict(m)
    packaged['deploy_package'] = {
        'source_run_dir': args.run_dir,
        'leaked_test_groups': leak,
        'split_rederived': True,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(packaged, f, indent=2, ensure_ascii=False)

    write_model_card(
        os.path.join(args.out_dir, 'MODEL_CARD.md'),
        m=m, leak=leak, out_dir=args.out_dir, run_dir=args.run_dir,
        notes=args.note)

    op = m['overall_strict']
    print(f"\n{'=' * 72}")
    print('  EN NER deployment package')
    print(f"{'=' * 72}")
    print(f'  Out      : {args.out_dir}')
    print(f"  Model    : {m['model_name']}")
    print(f'  Split    : {len(train_rows)}/{len(valid_rows)}/'
          f'{len(test_rows)}  (leak={leak})')
    print(f"  Test     : P={op['precision']:.4f} R={op['recall']:.4f} "
          f"F1={op['f1']:.4f}")


if __name__ == '__main__':
    main()
