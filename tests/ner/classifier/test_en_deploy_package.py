"""EN 배포 패키지의 분할·누출·성능 하한과 자체 로드 가능성을 검증한다.

패키지(`/data/ner/en/`)가 없으면 건너뛴다. 과거 실험 결과와의 대조나
재현 검증은 수행하지 않는다.
"""
import json
import os
from pathlib import Path

import pytest

PACKAGE = Path('/data/ner/en')
PKG_METRICS = PACKAGE / 'metrics.json'

pytestmark = pytest.mark.skipif(
    not PKG_METRICS.exists(),
    reason=f'EN deployment package not present at {PACKAGE}',
)

# 붕괴 검출용 바닥 — 이 아래면 체크포인트가 망가진 것이다.
USABLE_F1_FLOOR = 0.85


@pytest.fixture(scope='module')
def pkg():
    with open(PKG_METRICS, encoding='utf-8') as f:
        return json.load(f)


def test_split_sizes_are_the_locked_ones(pkg):
    """분할 크기와 데이터 지문은 그 자체가 고정값이다.

    **네 상수의 출처는 코퍼스이지 run 이 아니다.** 배포된 `metrics.json` 에서
    베껴 오면 이 검사가 검사 대상의 전사가 되어, 잘못된 코퍼스로 학습한
    패키지도 자기 자신과는 언제나 일치한다.

    - `data_fingerprint`: `dataset_fingerprint(load_jsonl(
      'data/ontonotes_en/origin.jsonl'))` 를 직접 돌린 값
    - 분할 셋: 코퍼스 76,378 행을 CLI 가 선언한 비율(valid 0.1 / test 0.1)로
      나눈 값이다. 10% 두 몫은 내림해 각 7,637 이고 나머지가 train 61,104 다

    어긋나면 상수를 고칠 것이 아니라 조사한다 — 코퍼스가 바뀌었거나, 분할
    인자가 바뀌었거나, 패키지가 다른 데이터로 학습된 run 에서 나온 것이다.
    지문은 EMAIL 붙은 문맥 치환 이후 값이다. 분할 크기는 그 치환과 앞선
    로컬파트 재생성이 행 수를 바꾸지 않아 그대로다.
    """
    assert pkg['train_samples'] == 61104
    assert pkg['valid_samples'] == 7637
    assert pkg['test_samples'] == 7637
    assert pkg['data_fingerprint'] == '17a7936b5dec7a98'


def test_no_group_leak(pkg):
    """빌더가 test 그룹 누출을 실제로 셌고 0 이어야 한다."""
    deploy = pkg.get('deploy_package')
    assert deploy is not None, 'metrics.json has no deploy_package block'
    assert deploy['leaked_test_groups'] == 0
    assert deploy['split_rederived'] is True


def test_model_is_usable(pkg):
    """붕괴한 체크포인트가 출하되는 것만 막는다(재현 문턱이 아니다)."""
    f1 = pkg['overall_strict']['f1']
    assert f1 >= USABLE_F1_FLOOR, (
        f'deployed checkpoint scores {f1:.4f} strict micro-F1, below the '
        f'usability floor {USABLE_F1_FLOOR}. This gate catches a collapsed '
        'checkpoint, not seed-to-seed variation.'
    )


def test_package_layout():
    """ja·vi 동형 레이아웃 — 소비자가 경로 하나만 알면 되게."""
    assert (PACKAGE / 'model').is_dir()
    assert (PACKAGE / 'MODEL_CARD.md').is_file()
    for split in ('train', 'valid', 'test'):
        assert (PACKAGE / 'data' / f'{split}.jsonl').is_file()


def test_split_files_match_recorded_sizes(pkg):
    """배포된 split 파일의 행 수가 metrics 의 기록과 같아야 한다."""
    for split, key in (('train', 'train_samples'),
                       ('valid', 'valid_samples'),
                       ('test', 'test_samples')):
        path = PACKAGE / 'data' / f'{split}.jsonl'
        with open(path, encoding='utf-8') as f:
            n = sum(1 for _ in f)
        assert n == pkg[key], f'{split}.jsonl has {n} rows, expected {pkg[key]}'


@pytest.mark.integration
def test_model_dir_is_self_contained():
    """`model/` 만으로 토크나이저·모델이 로드되고 1건 추론이 나와야 한다.

    배포의 실질은 "이 디렉토리 하나면 된다"이므로, 레포의 학습 코드나 원격
    허브를 다시 타야 한다면 배포가 안 된 것이다. `trainer.save_model()` 은
    토크나이저를 남기지 않으므로 빌더가 동봉하며, 그 동봉이 빠지면 예외가
    아니라 **망가진 토크나이저가 조용히 돌아와** 예측이 전량 비는 것으로
    나타난다 — 그래서 로드만 보지 않고 추론까지 태운다.
    """
    torch = pytest.importorskip('torch')
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = str(PACKAGE / 'model')
    prev = os.environ.get('HF_HUB_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    finally:
        if prev is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = prev

    assert len(model.config.id2label) == 21, 'expected BIO 21-label head'

    enc = tok('Barack Obama visited Seoul.', return_tensors='pt')
    # 토크나이저가 망가지면 여기서 대부분 unk 로 뭉개진다 — 조각 수로 잡는다.
    assert enc['input_ids'].shape[1] >= 6, 'tokenizer produced too few tokens'
    with torch.no_grad():
        logits = model(**enc).logits
    assert logits.shape[-1] == 21
    assert logits.shape[1] == enc['input_ids'].shape[1]
