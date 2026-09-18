"""NER 배포 패키지 빌드 — 학습 산출물을 배포 레이아웃으로 포장한다.

**학습은 하지 않는다.** `python -m ner.classifier` 가 낸 run 디렉토리를 읽어
`/data/ner/{lang}/` 배포 레이아웃으로 옮긴다. 학습 경로를 여기서 복제하면 배포
체크포인트가 CLI 가 아닌 이 스크립트의 산물이 돼, 기존 학습 run의
벤치마크 수치와 맞춰 보는 일이 서로 다른 코드 경로를 비교하는 것이 된다.

**언어는 run 이 정한다.** 언어별 사본을 두지 않는 이유는 분할 재유도·지문
대조·누출 가드가 언어와 무관한 안전장치이기 때문이다 — 복사본을 만들면 그
장치가 여러 벌이 되고 한쪽만 고쳐지는 순간 조용히 갈린다. 언어에 딸린 것은
출력 경로와 카드 문구뿐이라 `metrics.json` 의 `lang` 에서 끌어온다.

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
    python src/ner/scripts/build_ner_prod.py \\
        --run-dir results/classifier/ko/deploy-trainseed42
    python src/ner/scripts/build_ner_prod.py \\
        --run-dir <run> --lang ko --out-dir /abs/out --force
"""
import argparse
import json
import logging
import os
import shutil

from ner.classifier.data_utils import (
    dataset_fingerprint,
    load_jsonl,
    load_tokenizer,
    split_train_valid_test,
)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger('build_ner_prod')

# 배포 루트 — 언어 디렉토리가 그 아래 붙는다(`/data/ner/{lang}`).
DEPLOY_ROOT = '/data/ner'

# 카드 산문에 쓰는 언어 이름. 포장 가능한 언어의 목록이기도 하다 — 이름이
# 없는 언어는 카드를 쓸 수 없으므로 여기 없으면 포장을 거절한다.
LANG_LABEL = {'ja': '일본어', 'ko': '한국어', 'vi': '베트남어', 'en': '영어'}


def dump_jsonl(rows, path):
    """row 리스트를 JSONL 로 저장(원본 필드 보존)."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_run_metrics(run_dir, expect_lang=None):
    """run 의 metrics.json 을 읽고, 배포 포장이 가능한 run 인지 검사한다.

    포장 가능한 run 은 단일 분할·단일 스테이지다. k-fold 는 교차검증 추정이라
    배포할 체크포인트 하나를 가리키지 않고, curriculum·extra-train 은 기록된
    분할만으로 학습 데이터를 재현할 수 없다.

    언어는 run 이 선언한 값을 쓰되, `expect_lang` 이 주어지면 대조한다 —
    출력 경로가 언어에서 유도되므로, 잘못된 run 을 가리켰을 때 조용히 다른
    언어 자리에 쓰는 대신 여기서 멈춘다.
    """
    path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(path):
        raise SystemExit(f'Error: run metrics not found: {path}')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)

    lang = m.get('lang')
    if lang not in LANG_LABEL:
        raise SystemExit(
            f'Error: run lang is {lang!r}; packageable languages are '
            f'{sorted(LANG_LABEL)}')
    if expect_lang is not None and lang != expect_lang:
        raise SystemExit(
            f'Error: run lang is {lang!r} but --lang says {expect_lang!r}; '
            'refusing to ship a run into another language\'s slot')
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

    # 학습과 같은 로더로 연다. 바이트 BPE 의 앞 공백 설정이 여기서 저장돼
    # 서버가 학습 때와 같은 토큰을 받는다.
    tokenizer = load_tokenizer(model_name)
    tokenizer.save_pretrained(model_dir)
    # 토크나이저를 돌려주는 것은 카드가 *실제로 동봉된 것*을 적게 하려는
    # 것이다. 언어별 표를 따로 두면 백본을 바꿨을 때 카드만 옛말이 된다.
    return model_dir, tokenizer


def describe_tokenizer(tokenizer):
    """카드에 적을 토크나이저 한 줄 — 실제 클래스와 fast 여부에서 만든다."""
    kind = 'fast' if getattr(tokenizer, 'is_fast', False) else 'slow'
    return f'{kind}(`{type(tokenizer).__name__}`)'


def tokenizer_files(model_dir):
    """`model/` 에 저장된 토크나이저 파일 이름들(가중치·config 제외)."""
    weights = {'config.json', 'model.safetensors', 'pytorch_model.bin',
               'training_args.bin'}
    names = sorted(n for n in os.listdir(model_dir) if n not in weights)
    return ' · '.join(names) if names else '(none)'


def write_model_card(path, *, m, leak, out_dir, run_dir, notes,
                     tokenizer_desc, tokenizer_names):
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

    lang = m['lang']
    label = LANG_LABEL[lang]
    threshold_line = (
        '신뢰도 임계값(confidence threshold) 미적용 — raw 모델 출력 그대로.'
        if not os.path.exists(os.path.join(out_dir, 'thresholds.json'))
        else '신뢰도 임계값(`thresholds.json`) 적용 — 운영점 그대로 낸다.')

    inference_example = (
        f' `src/ner/scripts/eval_{lang}_ner_test.py`가 그 사용 예다.'
        if lang in ('ja', 'vi') else '')

    card = f"""# {lang.upper()} NER production 모델

{label} canonical 10종 평면 NER BERT 분류기. 실제 추론 배포용.
{threshold_line}

## 구성

```
{out_dir}/
├── model/              # fine-tuned 가중치 + tokenizer (자립 로드)
│   ├── config.json · model.safetensors    # 모델 (BIO 21라벨)
│   └── {tokenizer_names}  # 토크나이저
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
| 토크나이저 | {tokenizer_desc} — **`model/` 에 동봉**. 별도 런타임 의존 없음 |
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
   이룬다.{inference_example}

## 프로비넌스

`python -m ner.classifier --lang {lang}` 산출물(`{run_dir}`)을
`src/ner/scripts/build_ner_prod.py` 로 포장했다. 학습 경로는 CLI 이고 이
카드의 수치는 그 run 의 `metrics.json` 에서 옮긴 값이다.
"""
    if notes:
        card += '\n## 운영 주의\n\n'
        card += '\n\n'.join(f'- {n}' for n in notes) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(card)


def main():
    p = argparse.ArgumentParser(
        description='Package a trained NER run into the deployment layout '
                    '(model/ + data/ + metrics.json + MODEL_CARD.md). Does '
                    'not train: point --run-dir at the output of '
                    '`python -m ner.classifier --lang <lang>`. The language '
                    'comes from the run; --lang only cross-checks it.')
    p.add_argument('--run-dir', required=True,
                   help='Training run dir containing best/ and metrics.json')
    p.add_argument('--lang', choices=sorted(LANG_LABEL),
                   help='Expected language; fails if the run disagrees')
    p.add_argument('--out-dir',
                   help=f'Deployment package dir '
                        f'(default {DEPLOY_ROOT}/<lang>)')
    p.add_argument('--force', action='store_true',
                   help='Replace an existing package at --out-dir')
    p.add_argument('--note', action='append', default=[],
                   help='Operational caveat for this checkpoint; repeatable. '
                        'Rendered as a MODEL_CARD section instead of being '
                        'baked into the template.')
    args = p.parse_args()

    m = load_run_metrics(args.run_dir, expect_lang=args.lang)
    lang = m['lang']
    out_dir = args.out_dir or os.path.join(DEPLOY_ROOT, lang)
    logger.info('Run: lang=%s model=%s split=%s/%s/%s fingerprint=%s',
                lang, m['model_name'], m['train_samples'],
                m['valid_samples'], m['test_samples'], m['data_fingerprint'])

    rows, train_rows, valid_rows, test_rows, leak = rederive_split(m)
    logger.info('Re-derived split matches the run (%d rows, leak=%d)',
                len(rows), leak)

    os.makedirs(out_dir, exist_ok=True)
    model_dir, tokenizer = build_model_dir(
        args.run_dir, out_dir, m['model_name'], args.force)
    logger.info('Model dir ready: %s (%s)',
                model_dir, describe_tokenizer(tokenizer))

    data_dir = os.path.join(out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    dump_jsonl(train_rows, os.path.join(data_dir, 'train.jsonl'))
    dump_jsonl(valid_rows, os.path.join(data_dir, 'valid.jsonl'))
    dump_jsonl(test_rows, os.path.join(data_dir, 'test.jsonl'))

    # run 의 metrics를 그대로 싣고 포장 사실만 덧붙여 학습 결과를 보존한다.
    packaged = dict(m)
    packaged['deploy_package'] = {
        'source_run_dir': args.run_dir,
        'leaked_test_groups': leak,
        'split_rederived': True,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(packaged, f, indent=2, ensure_ascii=False)

    write_model_card(
        os.path.join(out_dir, 'MODEL_CARD.md'),
        m=m, leak=leak, out_dir=out_dir, run_dir=args.run_dir,
        notes=args.note,
        tokenizer_desc=describe_tokenizer(tokenizer),
        tokenizer_names=tokenizer_files(model_dir))

    op = m['overall_strict']
    print(f"\n{'=' * 72}")
    print(f'  {lang.upper()} NER deployment package')
    print(f"{'=' * 72}")
    print(f'  Out      : {out_dir}')
    print(f"  Model    : {m['model_name']}")
    print(f'  Split    : {len(train_rows)}/{len(valid_rows)}/'
          f'{len(test_rows)}  (leak={leak})')
    print(f"  Test     : P={op['precision']:.4f} R={op['recall']:.4f} "
          f"F1={op['f1']:.4f}")


if __name__ == '__main__':
    main()
