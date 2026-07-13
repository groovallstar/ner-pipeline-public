"""K-fold 실험 결과의 fold 간 분산·비교 유효성 게이트.

학습 재실행 없이 기존 K-fold 산출물(`fold{N}/metrics.json`,
`pooled_metrics.json`)만 읽어, per-entity F1 의 fold 간 표준편차(σ_fold)를
계산하고 두 실험을 비교해도 되는지(같은 자로 잰 값인지)를 판정한다.

핵심 원칙:
- **같은 자만 비교한다** — split 구성(lang·data_path·kfold·group_key)이
  다른 두 실험은 비교를 거부한다(INVALID). group_key=null(누출 보호 없음)과
  group_key=orig 를 섞어 비교하는 것을 구조적으로 막는다.
- **누출 트립와이어** — pooled 의 cross_fold_orig_dups != 0 이면 FAIL.
- **타깃 단독 최적화 금지** — 타깃 엔티티가 올라도 다른 엔티티가 노이즈
  밴드를 넘어 회귀하면 FAIL.

σ_fold 는 fold 간 표준편차로, 학습 재현 분산(σ_repro, 시드 반복 필요)의
프록시다. 더 엄격한 게이트는 시드 반복 측정(별도 증분)이 필요하다.
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 비교 가능성(같은 자) 판정에 쓰는 split 구성 필드 — 이 값이 하나라도
# 다르면 두 실험은 서로 다른 자로 잰 것이라 비교를 거부한다.
RULER_FIELDS = ("lang", "data_path", "kfold", "group_key")


def _load_json(path: Path) -> dict:
    """JSON 파일을 읽어 dict 로 반환한다."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fold_dirs(run_dir: Path) -> List[Path]:
    """run_dir 아래 fold{N}/metrics.json 을 가진 fold 디렉토리를 fold 번호
    순으로 반환한다."""
    indexed = []
    for p in run_dir.glob("fold*/metrics.json"):
        try:
            idx = int(p.parent.name[len("fold"):])
        except ValueError:
            continue
        indexed.append((idx, p.parent))
    return [d for _, d in sorted(indexed)]


def load_fold_metrics(run_dir: Path) -> List[dict]:
    """run_dir 의 모든 fold metrics.json 을 fold 순서로 로드한다."""
    return [_load_json(d / "metrics.json") for d in _fold_dirs(Path(run_dir))]


def load_pooled_metrics(run_dir: Path) -> dict:
    """run_dir 의 pooled_metrics.json 을 로드한다."""
    return _load_json(Path(run_dir) / "pooled_metrics.json")


def _fold_per_entity(fold: dict, matching: str) -> Dict[str, float]:
    """fold metrics.json 에서 per-entity F1 을 추출한다."""
    per = fold.get(f"per_entity_{matching}") or fold.get("per_entity", {})
    return {e: v["f1"] for e, v in per.items()}


def _fold_overall(fold: dict, matching: str) -> Optional[float]:
    """fold metrics.json 에서 overall F1 을 추출한다."""
    ov = fold.get(f"overall_{matching}") or fold.get("overall")
    return ov["f1"] if ov else None


def _pooled_per_entity(pooled: dict, matching: str) -> Dict[str, float]:
    """pooled_metrics.json 에서 per-entity F1 을 추출한다."""
    per = pooled.get(matching, {}).get("per_entity", {})
    return {e: v["f1"] for e, v in per.items()}


def _pooled_overall(pooled: dict, matching: str) -> Optional[float]:
    """pooled_metrics.json 에서 overall F1 을 추출한다."""
    ov = pooled.get(matching, {}).get("overall")
    return ov["f1"] if ov else None


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


def run_config(run_dir: Path) -> dict:
    """실험의 split 구성을 첫 fold metrics.json 에서 읽는다.

    fold 마다 동일한 필드(lang·data_path·kfold·group_key 등)를 담고 있어
    첫 fold 로 대표한다. seed·fold_index·train_seed 는 fold 마다 다를 수
    있어 비교 가능성 판정에는 쓰지 않는다.
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
        all_deltas[entity] = {
            "delta": delta, "sigma": std, "band": band,
            "band_source": band_source, "status": status,
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
            "band_source per entity: sigma_repro (seed-repeat "
            "reproducibility via sigma_override) when available, else "
            "sigma_fold (fold-to-fold std) — a rougher, typically wider "
            "proxy. Measure sigma_repro only when an INCONCLUSIVE verdict "
            "blocks a decision you will adopt.",
        ],
    }


def _build_parser():
    """CLI 파서를 구성한다 (std / repro / compare 서브커맨드)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="K-fold variance and comparison-validity gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_std = sub.add_parser("std", help="per-entity fold std of one run")
    p_std.add_argument("--run", required=True, help="run directory")
    p_std.add_argument("--matching", default="strict",
                       choices=["strict", "relaxed"])

    p_repro = sub.add_parser(
        "repro", help="seed-repeat reproducibility sigma (sigma_repro)")
    p_repro.add_argument("--runs", required=True, nargs="+",
                         help="seed-repeat CV run dirs (>=2)")
    p_repro.add_argument("--matching", default="strict",
                         choices=["strict", "relaxed"])
    p_repro.add_argument("--out", default=None,
                         help="write sigma_repro cache JSON here")

    p_cmp = sub.add_parser("compare", help="gate candidate vs baseline")
    p_cmp.add_argument("--baseline", required=True, help="baseline run dir")
    p_cmp.add_argument("--candidate", required=True, help="candidate run dir")
    p_cmp.add_argument("--target", required=True, help="target entity type")
    p_cmp.add_argument("--matching", default="strict",
                       choices=["strict", "relaxed"])
    p_cmp.add_argument("--band-k", type=float, default=2.0,
                       help="noise band width in sigma units")
    p_cmp.add_argument("--sigma-repro", default=None,
                       help="cached sigma_repro JSON to use as band")
    p_cmp.add_argument("--out", default=None, help="write verdict JSON here")
    return parser


def main(argv=None) -> int:
    """CLI 진입점. std / repro / compare 결과를 JSON 으로 출력한다."""
    args = _build_parser().parse_args(argv)
    if args.cmd == "std":
        result = fold_std(args.run, args.matching)
    elif args.cmd == "repro":
        if args.out:
            result = write_sigma_repro(args.runs, args.out, args.matching)
        else:
            result = repro_std(args.runs, args.matching)
    else:
        override = (load_sigma_map(args.sigma_repro)
                    if args.sigma_repro else None)
        result = compare(args.baseline, args.candidate, args.target,
                         args.matching, args.band_k, override)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
