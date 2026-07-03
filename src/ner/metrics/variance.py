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


def leakage_dups(run_dir: Path) -> int:
    """pooled_metrics.json 의 cross_fold_orig_dups 를 반환한다(누출 카운터).

    키가 없으면 누출을 검증할 수 없으므로 0(이상 없음)을 가정하지 않고
    예외를 던진다(fail-loud) — 스키마 드리프트가 게이트를 조용히 통과시키는
    것을 막는다.
    """
    pooled = load_pooled_metrics(run_dir)
    if "cross_fold_orig_dups" not in pooled:
        raise ValueError(
            f"{run_dir}/pooled_metrics.json missing 'cross_fold_orig_dups'; "
            "cannot verify cross-fold leakage")
    return int(pooled["cross_fold_orig_dups"])


def compare(baseline_dir: Path, candidate_dir: Path, target_entity: str,
            matching: str = "strict", band_k: float = 2.0) -> dict:
    """candidate 가 baseline 대비 타깃 엔티티를 노이즈 밴드 밖에서 개선했는지,
    다른 엔티티를 회귀시키지 않았는지 판정한다.

    Δ 는 pooled per-entity F1(candidate - baseline)이고, 밴드는 baseline 의
    σ_fold × band_k 다. σ_fold 는 재현 분산(σ_repro)의 프록시임에 유의한다.

    Returns:
        최상위 "verdict" 가 PASS|FAIL|INVALID|INCONCLUSIVE 인 판정 dict.
    """
    comparable, issues = check_comparable(
        run_config(baseline_dir), run_config(candidate_dir))
    dups_a, dups_b = leakage_dups(baseline_dir), leakage_dups(candidate_dir)
    leakage_ok = (dups_a == 0 and dups_b == 0)

    base_f1 = _pooled_per_entity(load_pooled_metrics(baseline_dir), matching)
    cand_f1 = _pooled_per_entity(load_pooled_metrics(candidate_dir), matching)
    sigma = fold_std(baseline_dir, matching)

    all_deltas: Dict[str, dict] = {}
    for entity in sorted(set(base_f1) & set(cand_f1)):
        delta = cand_f1[entity] - base_f1[entity]
        std = sigma.get(entity, {}).get("std")
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
            "delta": delta, "sigma_fold": std, "band": band,
            "status": status,
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
        "leakage": {"baseline_dups": dups_a, "candidate_dups": dups_b},
        "matching": matching,
        "band_k": band_k,
        "target": target,
        "regressions": regressions,
        "all_deltas": all_deltas,
        "notes": [
            "sigma_fold is fold-to-fold std, a proxy for reproducibility "
            "sigma (sigma_repro); measure sigma_repro via seed repeats for "
            "a stricter gate.",
        ],
    }


def _build_parser():
    """CLI 파서를 구성한다 (std / compare 서브커맨드)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="K-fold variance and comparison-validity gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_std = sub.add_parser("std", help="per-entity fold std of one run")
    p_std.add_argument("--run", required=True, help="run directory")
    p_std.add_argument("--matching", default="strict",
                       choices=["strict", "relaxed"])

    p_cmp = sub.add_parser("compare", help="gate candidate vs baseline")
    p_cmp.add_argument("--baseline", required=True, help="baseline run dir")
    p_cmp.add_argument("--candidate", required=True, help="candidate run dir")
    p_cmp.add_argument("--target", required=True, help="target entity type")
    p_cmp.add_argument("--matching", default="strict",
                       choices=["strict", "relaxed"])
    p_cmp.add_argument("--band-k", type=float, default=2.0,
                       help="noise band width in sigma_fold units")
    p_cmp.add_argument("--out", default=None, help="write verdict JSON here")
    return parser


def main(argv=None) -> int:
    """CLI 진입점. std 또는 compare 결과를 JSON 으로 출력한다."""
    args = _build_parser().parse_args(argv)
    if args.cmd == "std":
        result = fold_std(args.run, args.matching)
    else:
        result = compare(args.baseline, args.candidate, args.target,
                         args.matching, args.band_k)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
