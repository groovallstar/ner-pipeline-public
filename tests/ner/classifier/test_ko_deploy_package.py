"""KO 배포 패키지가 스스로 정합하고 쓸 만한가.

**재현 대조도, 독립 원장과의 대조도 하지 않는다.** en(#221)은 백본 벤치라는
선행 원장이 있어 패키지를 그것과 견줬지만, ko 에 있는 원장은 10-fold pooled
뿐이라 단일 분할 배포 run 과 자가 다르다 — 교차검증 추정치와 홀드아웃 수치를
같은 표에 놓는 것은 대조가 아니라 혼동이다. 그래서 ko 는 **배포 run 자신을
원장으로 승격**하고, 이 파일은 넷만 본다:

1. 패키지가 자기가 나온 run 과 어긋나지 않는가 (프로비넌스 정합 — 원장과
   verbatim, 다른 것은 포장이 덧붙인 `deploy_package` 블록뿐)
2. 분할이 기록된 그대로 출하됐는가 (행 수·지문·누출)
3. 모델이 쓸 만한가 (붕괴한 체크포인트가 나가는 것만 막는 바닥)
4. `model/` 하나로 로드·추론이 되는가 (배포의 실질)

바닥은 재현 문턱이 아니라 sanity 문턱이다. 좁게 조이면 seed 뽑기를 통과
조건으로 만드는 셈이라, 붕괴(F1 이 0 에 가까움)만 걸리게 둔다.

패키지(`/data/ner/ko/`)가 없으면 통째로 건너뛴다. `/data` 는 저장소 밖이라
클론 직후에는 없고, 그때 실패시키면 무관한 변경까지 붉어진다.
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LEDGER = (
    REPO / 'certified' / 'classifier' / 'ko' / 'deploy-trainseed42'
    / 'metrics.json'
)
PACKAGE = Path('/data/ner/ko')
PKG_METRICS = PACKAGE / 'metrics.json'

pytestmark = pytest.mark.skipif(
    not PKG_METRICS.exists(),
    reason=f'KO deployment package not present at {PACKAGE}',
)

# 붕괴 검출용 바닥 — 이 아래면 체크포인트가 망가진 것이다. en 과 같은 값을
# 쓴다(참고: ko 10-fold pooled strict overall F1 ≈ 0.92).
USABLE_F1_FLOOR = 0.85

# 포장이 덧붙이는 유일한 키. 이것만 빼면 패키지와 원장은 글자까지 같아야
# 한다 — 수치를 손대는 순간 원장 대조가 포장 스크립트를 거친 값을 보게 된다.
PACKAGING_ONLY_KEY = 'deploy_package'


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


def test_package_is_the_ledger_run_verbatim(pkg, ledger):
    """패키지 metrics 는 원장 run + 포장 블록, 그 외엔 한 글자도 다르지 않다."""
    stripped = {k: v for k, v in pkg.items() if k != PACKAGING_ONLY_KEY}
    assert stripped == ledger


def test_package_declares_its_source_run(pkg):
    """어느 run 에서 나왔는지가 패키지 안에 남아 있어야 한다."""
    deploy = pkg.get(PACKAGING_ONLY_KEY)
    assert deploy is not None, 'metrics.json has no deploy_package block'
    assert deploy['source_run_dir']
    assert deploy['split_rederived'] is True


def test_split_sizes_are_the_locked_ones(pkg):
    """분할 크기와 데이터 지문은 그 자체가 고정값이다."""
    assert pkg['lang'] == 'ko'
    assert pkg['train_samples'] == 20793
    assert pkg['valid_samples'] == 2598
    assert pkg['test_samples'] == 2598
    assert pkg['data_fingerprint'] == 'a6aaa31ba099c9fd'
    assert pkg['group_key'] == 'id'


def test_no_group_leak(pkg):
    """빌더가 test 그룹 누출을 실제로 셌고 0 이어야 한다."""
    assert pkg[PACKAGING_ONLY_KEY]['leaked_test_groups'] == 0


def test_model_is_usable(pkg):
    """붕괴한 체크포인트가 출하되는 것만 막는다(재현 문턱이 아니다)."""
    f1 = pkg['overall_strict']['f1']
    assert f1 >= USABLE_F1_FLOOR, (
        f'deployed checkpoint scores {f1:.4f} strict micro-F1, below the '
        f'usability floor {USABLE_F1_FLOOR}. This gate catches a collapsed '
        'checkpoint, not seed-to-seed variation.'
    )


def test_package_layout():
    """ja·vi·en 동형 레이아웃 — 소비자가 경로 하나만 알면 되게."""
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


def test_ships_without_thresholds(pkg):
    """vi·en 동형으로 raw 출력 — 서버의 graceful 폴백이 정상 경로다."""
    assert not (PACKAGE / 'thresholds.json').exists()


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
    # 서버가 char offset 을 쓰므로 fast 경로(offset_mapping)가 살아 있어야 한다.
    assert tok.is_fast, 'shipped tokenizer lost its fast/offset capability'

    enc = tok('김민준은 서울대학교를 졸업했다.', return_tensors='pt')
    # 토크나이저가 망가지면 여기서 대부분 unk 로 뭉개진다 — 조각 수로 잡는다.
    assert enc['input_ids'].shape[1] >= 6, 'tokenizer produced too few tokens'
    with torch.no_grad():
        logits = model(**enc).logits
    assert logits.shape[-1] == 21
    assert logits.shape[1] == enc['input_ids'].shape[1]
