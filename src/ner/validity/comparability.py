"""비교 가능성 축 — 두 실험이 '같은 자'로 측정됐는지 판정.

RULER_FIELDS 가 자다. data_fingerprint 는 test gold 내용의 지문이라 파일 경로가
아니라 정답 정체성을 비교하고, seed·stratify 는 fold 멤버십을 결정하므로 paired
비교(variance 축)의 전제가 된다 — 그래서 variance 는 comparability 에 의존한다.
"""
from pathlib import Path
from typing import List, Tuple

from ner.validity._common import _fold_dirs, _load_json

# 비교 가능성(같은 자) 판정에 쓰는 필드 — 이 값이 하나라도 다르면 두 실험은
# 서로 다른 자로 잰 것이라 비교를 거부한다.
#
# data_fingerprint 는 test gold 의 정체성(내용·순서)이라 data_path(경로 문자열)
# 를 대체한다 — 경로가 같아도 내용이 바뀌면 다른 자, 경로만 바뀌어도(rename)
# 내용이 같으면 같은 자다. seed·stratify 는 fold 멤버십을 결정하므로, paired
# 비교(fold 별 Δ 짝짓기)가 성립하려면 자에 포함돼야 한다.
RULER_FIELDS = ("lang", "data_fingerprint", "kfold", "group_key",
                "seed", "stratify")


def run_config(run_dir: Path) -> dict:
    """실험의 split 구성을 첫 fold metrics.json 에서 읽는다.

    RULER_FIELDS(lang·data_fingerprint·kfold·group_key·seed·stratify)는 run
    전체에서 상수라 첫 fold 로 대표한다 — split seed 는 fold 마다 바뀌지 않는다.
    fold_index(fold 축)·train_seed(학습 재현 축)만 fold 마다 달라 자에서 제외한다.
    """
    folds = _fold_dirs(Path(run_dir))
    if not folds:
        raise FileNotFoundError(
            f"no fold*/metrics.json found under {run_dir}")
    return _load_json(folds[0] / "metrics.json")


def check_comparable(cfg_a: dict, cfg_b: dict) -> Tuple[bool, List[str]]:
    """두 실험이 같은 자(split 구성)로 측정됐는지 판정한다.

    RULER_FIELDS 중 하나라도 다르면 비교 불가로 보고 사유를 반환한다.
    키 자체가 없으면(스키마 드리프트) 같은 자임을 확인할 수 없으므로
    조용히 통과시키지 않고 비교 불가 사유로 올린다(fail-loud). group_key
    값이 null 인 것과 키가 아예 없는 것은 구분한다 — null 은 유효한 값이다.
    """
    issues = []
    for field in RULER_FIELDS:
        if field not in cfg_a or field not in cfg_b:
            issues.append(f"{field}: missing (cannot verify same ruler)")
        elif cfg_a[field] != cfg_b[field]:
            issues.append(f"{field}: {cfg_a[field]!r} != {cfg_b[field]!r}")
    return (not issues, issues)
