"""판정 축 — 세 축(comparability·leakage·variance)을 통합해 최종 verdict 를 낸다.

compare() 는 비교 가능성·누출·노이즈 밴드·방향 일관성을 한 판정 dict 로 묶는다.
세 축이 만나는 유일한 통합점이라 축 모듈 위에 얹힌다.
"""
from pathlib import Path
from typing import Dict, Optional

from ner.validity._common import (
    _pooled_per_entity,
    load_pooled_metrics,
)
from ner.validity.comparability import check_comparable, run_config
from ner.validity.leakage import TRUSTED_LEAK_BASIS, leakage
from ner.validity.variance import (
    MIN_CONSISTENCY_FRAC,
    _fold_consistency,
    fold_paired_deltas,
    fold_std,
)


def compare(baseline_dir: Path, candidate_dir: Path, target_entity: str,
            matching: str = "strict", band_k: float = 2.0,
            sigma_override: Optional[Dict[str, Optional[float]]] = None
            ) -> dict:
    """candidate 가 baseline 대비 타깃 엔티티를 노이즈 밴드 밖에서 개선했는지,
    다른 엔티티를 회귀시키지 않았는지 판정한다.

    Δ 는 pooled per-entity F1(candidate - baseline)이고, 밴드는 σ × band_k 다.
    엔티티마다 sigma_override(σ_repro 캐시)에 값이 있으면 그것을, 없으면
    baseline 의 σ_fold 를 밴드로 쓰고, 어느 쪽을 썼는지 band_source 로 남긴다.
    σ_fold 는 재현 분산(σ_repro)의 더 거친·대체로 넓은 프록시다.

    Args:
        sigma_override: {entity: std} — σ_repro 등 측정된 밴드. None 이면
            전부 σ_fold 로 폴백한다.

    Returns:
        최상위 "verdict" 가 PASS|FAIL|INVALID|INCONCLUSIVE 인 판정 dict.
    """
    comparable, issues = check_comparable(
        run_config(baseline_dir), run_config(candidate_dir))
    dups_a, basis_a = leakage(baseline_dir)
    dups_b, basis_b = leakage(candidate_dir)
    # 누출을 '검증했고 0' 인 경우에만 통과다. 미측정(None)이나 신뢰할 수 없는
    # 근거(text·none·unknown)의 0 은 검증된 0 이 아니다 — 정직하게 opt-out 한
    # run 이 크래시하고 거짓 0 을 낸 run 이 통과하면, 규칙이 편법을 보상한다.
    #
    # 다만 카운터가 0 보다 크면 근거가 약해도 **본 것**이다. 약한 근거는 누출을
    # 놓칠 뿐 없는 누출을 만들어내지 않으므로, 이 경우는 확인된 누출(FAIL)이다.
    verified = []
    observed_leak = False
    for name, dups, basis in (("baseline", dups_a, basis_a),
                              ("candidate", dups_b, basis_b)):
        if dups is not None and dups > 0:
            observed_leak = True
            verified.append(dups)
        elif dups is None:
            issues = issues + [f"{name}: leakage unmeasured (basis={basis})"]
        elif basis not in TRUSTED_LEAK_BASIS:
            issues = issues + [
                f"{name}: leakage counter untrustworthy (basis={basis}); "
                f"0 here means 'not observable', not 'no leak'"]
        else:
            verified.append(dups)
    leakage_verified = observed_leak or len(verified) == 2
    leakage_ok = leakage_verified and not observed_leak

    base_f1 = _pooled_per_entity(load_pooled_metrics(baseline_dir), matching)
    cand_f1 = _pooled_per_entity(load_pooled_metrics(candidate_dir), matching)
    sigma_fold = fold_std(baseline_dir, matching)
    paired = fold_paired_deltas(baseline_dir, candidate_dir, matching)
    override = sigma_override or {}

    all_deltas: Dict[str, dict] = {}
    for entity in sorted(set(base_f1) & set(cand_f1)):
        delta = cand_f1[entity] - base_f1[entity]
        if override.get(entity) is not None:
            std, band_source = override[entity], "sigma_repro"
        else:
            std = sigma_fold.get(entity, {}).get("std")
            band_source = "sigma_fold"
        band = None if std is None else band_k * std
        if band is None:
            status = "unknown_sigma"
        elif delta > band:
            status = "real_gain"
        elif delta < -band:
            status = "real_regression"
        else:
            status = "within_noise"
        # 방향 일관성 게이트 — 조이기 전용. magnitude 가 real_gain 인데 후보가
        # fold 승률 <2/3 이면 한 fold 가 pooled 를 끌어올린 것으로 보고
        # within_noise 로 내린다. regression 은 건드리지 않는다(회귀는 소수
        # fold 에서 나타나도 계속 막는 게 보수적).
        wins, losses, n_fold, consistent = _fold_consistency(
            paired.get(entity, []))
        downgraded = False
        if status == "real_gain" and consistent is False:
            status = "within_noise"
            downgraded = True
        all_deltas[entity] = {
            "delta": delta, "sigma": std, "band": band,
            "band_source": band_source, "status": status,
            "fold_wins": wins, "fold_losses": losses, "fold_n": n_fold,
            "consistent": consistent, "consistency_downgraded": downgraded,
        }

    target = None
    if target_entity in all_deltas:
        target = {
            "entity": target_entity,
            "baseline_f1": base_f1[target_entity],
            "candidate_f1": cand_f1[target_entity],
            **all_deltas[target_entity],
        }
    regressions = [
        {"entity": e, **v}
        for e, v in all_deltas.items()
        if e != target_entity and v["status"] == "real_regression"
    ]

    if not comparable:
        verdict = "INVALID"
    elif not leakage_verified:
        # 검증되지 않은 누출은 FAIL(누출 확인)과 다르다 — 잴 수 없었을 뿐이다.
        verdict = "INVALID"
    elif not leakage_ok:
        verdict = "FAIL"
    elif target is None:
        verdict = "INVALID"
        issues = issues + [f"target entity {target_entity!r} not in pooled"]
    elif target["status"] == "real_regression":
        verdict = "FAIL"
    elif regressions:
        verdict = "FAIL"
    elif target["status"] == "real_gain":
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "comparable": comparable,
        "comparability_issues": issues,
        "leakage_ok": leakage_ok,
        "leakage_verified": leakage_verified,
        "leakage": {"baseline_dups": dups_a, "candidate_dups": dups_b,
                    "baseline_basis": basis_a, "candidate_basis": basis_b},
        "matching": matching,
        "band_k": band_k,
        "target": target,
        "regressions": regressions,
        "all_deltas": all_deltas,
        "notes": [
            "point estimate is pooled per-entity F1 delta (matches the "
            "reported headline). Band = band_k * sigma: sigma_repro "
            "(seed-repeat reproducibility via sigma_override) when available, "
            "else sigma_fold (fold-to-fold std) — a rougher, typically wider "
            "proxy that is safe because a noise gate should err wide. Measure "
            "sigma_repro only when an INCONCLUSIVE verdict blocks a decision "
            "you will adopt.",
            "consistency gate is tighten-only: a magnitude real_gain is "
            "downgraded to within_noise (INCONCLUSIVE) when the candidate wins "
            f"in fewer than {MIN_CONSISTENCY_FRAC:.2f} of the folds where both "
            "runs have the entity. It never upgrades and never suppresses a "
            "regression, so it cannot create a false PASS.",
        ],
    }
