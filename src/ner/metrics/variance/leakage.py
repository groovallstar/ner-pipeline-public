"""누출 축 — pooled 산출물에서 cross-fold 누출 카운터와 그 판정 근거를 읽는다.

카운터 0 이 '이상 없음'인지 '볼 수단이 없었음'인지는 판정 근거로 갈린다.
신뢰 근거(group·orig)로 센 0 만 통과이고, text·none·unknown 의 0 은 미검증이다.
"""
from pathlib import Path
from typing import Optional, Tuple

from ner.metrics.variance._common import load_pooled_metrics

#: 누출 카운터를 신뢰할 수 있는 판정 근거. 'text'(문장 전체 비교)는 문장을
#: 재작성하는 증강 코퍼스에서 형제 행을 못 알아보고, 'none'은 행 단위 분할을
#: 명시적으로 택한 경우라 애초에 측정된 적이 없다. 둘 다 0 을 "이상 없음"으로
#: 읽어선 안 된다.
TRUSTED_LEAK_BASIS = ("group", "orig")


def leakage(run_dir: Path) -> Tuple[Optional[int], str]:
    """pooled_metrics.json 에서 (누출 카운터, 판정 근거)를 읽는다.

    카운터가 None 이면 미측정이다. 근거 키가 없는 옛 산출물은 'unknown' 이다 —
    그때 무엇으로 셌는지 알 수 없기 때문이다. 'orig' 로 간주하면 실제로는
    text 기준이던 옛 run 을 신뢰하게 되므로, 모르는 것은 신뢰하지 않는다.

    카운터 키 자체가 없으면 누출을 검증할 수 없으므로 0(이상 없음)을 가정하지
    않고 예외를 던진다(fail-loud) — 스키마 드리프트가 게이트를 조용히
    통과시키는 것을 막는다.
    """
    pooled = load_pooled_metrics(run_dir)
    if "cross_fold_group_dups" in pooled:
        raw = pooled["cross_fold_group_dups"]
    elif "cross_fold_orig_dups" in pooled:
        raw = pooled["cross_fold_orig_dups"]
    else:
        raise ValueError(
            f"{run_dir}/pooled_metrics.json missing leak counter "
            "('cross_fold_group_dups'); cannot verify cross-fold leakage")
    basis = pooled.get("leak_check_basis", "unknown")
    return (None if raw is None else int(raw)), basis


def leakage_dups(run_dir: Path) -> Optional[int]:
    """누출 카운터만 반환한다. None 이면 미측정."""
    return leakage(run_dir)[0]
