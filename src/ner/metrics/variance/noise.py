"""노이즈 축 — fold 간 분산(σ_fold)·시드 재현 분산(σ_repro)·paired fold Δ·
방향 일관성.

밴드 재료를 계산한다: σ_fold(fold 간 표준편차)는 넓은 기본 프록시, σ_repro
(시드 반복 pooled 헤드라인의 흔들림)는 수요기반 정밀 밴드. paired fold Δ 와
방향 일관성은 magnitude gain 을 조이는 데 쓴다(조이기 전용).
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ner.metrics.variance._common import (
    _fold_overall,
    _fold_per_entity,
    _load_json,
    _pooled_overall,
    _pooled_per_entity,
    load_fold_metrics,
    load_pooled_metrics,
)

#: 방향 일관성 게이트의 임계 — 후보가 양쪽에 엔티티가 있는 fold 중 이 비율
#: 이상에서 이겨야 magnitude gain 을 확정한다. 미만이면 한 fold 가 pooled 를
#: 끌어올린 것으로 보고 INCONCLUSIVE 로 내린다. 게이트는 조이기 전용이다 —
#: gain 을 within_noise 로만 내리고, 절대 올리지 않으며 regression 은 안 건드린다.
MIN_CONSISTENCY_FRAC = 2.0 / 3.0


def fold_std(run_dir: Path, matching: str = "strict") -> Dict[str, dict]:
    """per-entity·overall F1 의 fold 간 평균·표준편차(σ_fold)를 계산한다.

    Args:
        run_dir: fold{N}/metrics.json 을 담은 실험 디렉토리.
        matching: 'strict' 또는 'relaxed'.

    Returns:
        {entity: {"mean", "std", "n", "values"}} — 'overall' 키 포함.
        std 는 표본 표준편차(n>=2)이며, n<2 이면 None.
    """
    folds = load_fold_metrics(run_dir)
    if not folds:
        raise FileNotFoundError(
            f"no fold*/metrics.json found under {run_dir}")
    series: Dict[str, List[float]] = {}
    for fold in folds:
        for entity, f1 in _fold_per_entity(fold, matching).items():
            series.setdefault(entity, []).append(f1)
        overall = _fold_overall(fold, matching)
        if overall is not None:
            series.setdefault("overall", []).append(overall)
    out: Dict[str, dict] = {}
    for entity, vals in series.items():
        out[entity] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) >= 2 else None,
            "n": len(vals),
            "values": vals,
        }
    return out


def fold_paired_deltas(baseline_dir: Path, candidate_dir: Path,
                       matching: str = "strict") -> Dict[str, List[float]]:
    """fold 별 per-entity Δᵢ = F1ᵢ(candidate) − F1ᵢ(baseline) 를 모은다.

    두 run 이 같은 fold 멤버십(=comparable: 같은 지문·seed·kfold·group_key·
    stratify)일 때만 유효하다. fold 는 fold 번호 순으로 정렬돼 인덱스로 짝짓고,
    각 fold 에서 두 run 모두 엔티티가 존재하는 경우만 Δᵢ 를 낸다.

    Returns:
        {entity: [Δ_i, ...]} — 엔티티가 양쪽에 있는 fold 만, fold 순서 보존.
    """
    base_folds = load_fold_metrics(baseline_dir)
    cand_folds = load_fold_metrics(candidate_dir)
    n = min(len(base_folds), len(cand_folds))
    series: Dict[str, List[float]] = {}
    for i in range(n):
        bpe = _fold_per_entity(base_folds[i], matching)
        cpe = _fold_per_entity(cand_folds[i], matching)
        for entity in set(bpe) & set(cpe):
            series.setdefault(entity, []).append(cpe[entity] - bpe[entity])
    return series


def _fold_consistency(deltas: List[float]) -> Tuple[int, int, int, Optional[bool]]:
    """fold 별 Δᵢ 목록에서 (승, 패, n, 일관적?) 를 낸다.

    승 = Δ>0 fold 수, 패 = Δ<0 fold 수. n<2 면 판정 불가라 일관성은 None.
    일관적 = 이긴 fold 비율 >= MIN_CONSISTENCY_FRAC. gain 확정용 — regression
    쪽은 호출자가 이 값으로 다운그레이드하지 않는다(조이기 전용 불변식).
    """
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    if n < 2:
        return wins, losses, n, None
    return wins, losses, n, (wins / n) >= MIN_CONSISTENCY_FRAC


def repro_std(run_dirs: List[Path],
              matching: str = "strict") -> Dict[str, dict]:
    """시드-반복 CV run 들의 pooled 헤드라인 F1 재현 분산(σ_repro)을 계산한다.

    각 run_dir 는 서로 다른 train seed 로 돌린 완전한 K-fold CV 결과
    (pooled_metrics.json)여야 한다. per-entity·overall pooled F1 이 시드에
    따라 얼마나 재현되는지의 표준편차를 반환한다 — σ_fold(fold 간 편차)와
    달리, 게이팅에 실제로 쓰는 헤드라인 그 자체의 흔들림이다.

    Args:
        run_dirs: 시드만 다른 완전한 CV run 디렉토리 목록(>=2).
        matching: 'strict' 또는 'relaxed'.

    Returns:
        {entity: {"mean", "std", "n", "values"}} — 'overall' 키 포함.
    """
    run_dirs = list(run_dirs)
    if len(run_dirs) < 2:
        raise ValueError(
            "repro_std needs >= 2 seed-repeat runs to estimate sigma_repro")
    series: Dict[str, List[float]] = {}
    for run_dir in run_dirs:
        pooled = load_pooled_metrics(run_dir)
        for entity, f1 in _pooled_per_entity(pooled, matching).items():
            series.setdefault(entity, []).append(f1)
        overall = _pooled_overall(pooled, matching)
        if overall is not None:
            series.setdefault("overall", []).append(overall)
    out: Dict[str, dict] = {}
    for entity, vals in series.items():
        out[entity] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) >= 2 else None,
            "n": len(vals),
            "values": vals,
        }
    return out


def write_sigma_repro(run_dirs: List[Path], out_path: Path,
                      matching: str = "strict") -> dict:
    """σ_repro 를 계산해 캐시 JSON 으로 저장한다.

    σ_repro 는 (데이터×아키텍처×학습설정) setup 의 성질이라 setup 당 한 번만
    재서 이후 실험에 재사용한다. 저장 형식:
    {"matching", "n_runs", "run_dirs", "sigma": {entity: std}}.
    """
    run_dirs = list(run_dirs)
    repro = repro_std(run_dirs, matching)
    payload = {
        "matching": matching,
        "n_runs": len(run_dirs),
        "run_dirs": [str(d) for d in run_dirs],
        "sigma": {e: v["std"] for e, v in repro.items()},
    }
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_sigma_map(path: Path) -> Dict[str, Optional[float]]:
    """캐시된 σ_repro JSON 에서 {entity: std} 밴드 맵을 읽는다."""
    return _load_json(Path(path)).get("sigma", {})
