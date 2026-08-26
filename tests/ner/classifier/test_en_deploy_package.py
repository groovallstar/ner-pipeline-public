"""EN 배포 패키지가 스스로 정합하고 쓸 만한가.

**재현 대조는 하지 않는다.** 처음에는 "같은 데이터 시드·같은 학습 시드면
원장의 seed42 run 을 재현한다"를 기준으로 삼았으나, 그 전제가 거짓임이
드러났다 — best 에포크 선택이 valid `eval_loss` 기준이라 미세한 수치 차이가
어느 체크포인트를 출하할지를 뒤집고, 예측 span 복원이 라벨 생성과 같은
offset 배열을 쓰므로 채점 경로도 완전히 결정적이지 않다. 그래서 이 파일은
재현이 아니라 **세 가지**만 본다:

1. 패키지가 자기가 나온 run 과 어긋나지 않는가 (프로비넌스 정합)
2. 원장 run 과 **같은 자로** 쟀는가 (데이터·분할·설정 동일 — 수치를 나란히
   놓을 자격)
3. 모델이 쓸 만한가 (붕괴한 체크포인트가 출하되는 것만 막는 바닥)

바닥은 재현 문턱이 아니라 sanity 문턱이다. 좁게 조이면 seed 뽑기를 통과
조건으로 만드는 셈이라, 붕괴(F1 이 0 에 가까움)만 걸리게 둔다.

패키지(`/data/ner/en/`)가 없으면 통째로 건너뛴다. `/data` 는 저장소 밖이라
클론 직후에는 없고, 그때 실패시키면 무관한 변경까지 붉어진다.
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LEDGER = (
    REPO / 'certified' / 'classifier' / 'en' / 'backbone-bench'
    / 'roberta_base_seed42' / 'metrics.json'
)
PACKAGE = Path('/data/ner/en')
PKG_METRICS = PACKAGE / 'metrics.json'

pytestmark = pytest.mark.skipif(
    not PKG_METRICS.exists(),
    reason=f'EN deployment package not present at {PACKAGE}',
)

# 붕괴 검출용 바닥 — 이 아래면 체크포인트가 망가진 것이다.
USABLE_F1_FLOOR = 0.85

# 두 run 이 "같은 자로 잰다"를 이루는 필드. 하나라도 어긋나면 수치를 나란히
# 놓을 자격이 없다. train_time_sec 처럼 실행 환경에 딸린 값은 뺀다.
RULER_FIELDS = (
    'lang',
    'model_name',
    'data_path',
    'data_fingerprint',
    'n_rows',
    'n_groups',
    'group_key',
    'train_samples',
    'valid_samples',
    'test_samples',
    'valid_ratio',
    'test_ratio',
    'seed',
    'precision',
    'epochs',
    'batch_size',
    'lr',
    'max_length',
    'metric_for_best',
)


@pytest.fixture(scope='module')
def pkg():
    with open(PKG_METRICS, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def ledger():
    with open(LEDGER, encoding='utf-8') as f:
        return json.load(f)


def test_ledger_run_is_committed():
    """비교 상대가 원장에 있어야 한다 — 없으면 대조가 허울이다."""
    assert LEDGER.exists(), f'ledger run missing: {LEDGER}'


def test_ruler_matches_ledger(pkg, ledger):
    """데이터·분할·설정이 원장 run 과 같아야 수치를 나란히 놓을 수 있다."""
    mismatched = {
        k: (pkg.get(k), ledger.get(k))
        for k in RULER_FIELDS
        if pkg.get(k) != ledger.get(k)
    }
    assert not mismatched, f'package/ledger ruler mismatch: {mismatched}'


def test_split_sizes_are_the_locked_ones(pkg):
    """분할 크기와 데이터 지문은 그 자체가 고정값이다."""
    assert pkg['train_samples'] == 61104
    assert pkg['valid_samples'] == 7637
    assert pkg['test_samples'] == 7637
    assert pkg['data_fingerprint'] == '39cb0f9e9c16f265'


def test_test_gold_is_the_same_gold(pkg, ledger):
    """test gold 의 span 수가 같아야 같은 정답을 재고 있는 것이다."""
    assert (pkg['overall_strict']['support']
            == ledger['overall_strict']['support'])


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
