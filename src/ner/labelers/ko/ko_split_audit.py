"""옛 실험 두 팔의 fold 분할이 서로 대응하는지 사후 감사한다.

**왜 필요한가.** fold 별 Δ 를 짝지어 내는 paired 통계는 두 팔이 같은 fold
멤버십을 가질 때만 성립한다. 그런데 층화 분할은 라벨을 보고 층을 나누므로,
gold 를 고치면 분할이 통째로 재배열될 수 있다 — 그렇게 어긋난 뒤에도 fold 를
번호로 짝지으면 값은 그럴듯하게 나온다. 서로 다른 문장 집합을 비교한 수인데
겉으로는 구별되지 않는다.

**무엇이 감사를 가능하게 하나.** 분할은 (행 순서·라벨·seed·fold 수·group_key·
층화 여부) 의 순수 함수라 학습 없이 재생성된다. 판정 원장이 회수 좌표와 그때의
gold 지문을 함께 담고 있어 과거 시점의 gold 도 되돌릴 수 있고, 되돌린 것이
맞는지는 커밋된 지문이 판정한다.

**한계 — 설정이 안 남았으면 대개 감사도 못 한다.** 층화 여부는 그 run 의
`fold*/metrics.json` 에만 적히는데 그 폴더는 휘발이다. 지워졌으면 두 시나리오를
모두 계산해 병기하고, **그 둘이 다른 답을 주면** 어느 쪽인지 고를 수 없어 판정을
`undecidable` 로 남긴다 — 모르는 것을 아는 것처럼 적지 않기 위해서다. 반대로 두
시나리오가 **같은 답이면 설정과 무관하게 답이 정해지므로** 근거가 없어도 판정한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import tempfile
from typing import Dict, List, Optional, Sequence, Set

from ner.classifier.data_utils import split_kfold_stratified
from ner.labelers.ko.ko_evt_r2_audit import dump_gold, file_sha256, load_gold

logger = logging.getLogger(__name__)

EVT = "EVT"
STRAT_LABELS = ("PROD", "EVT")

# 판정 어휘. `undecidable` 은 결과가 없다는 뜻이 아니라 **고를 근거가 없다**는
# 뜻이다 — 두 시나리오 값은 그대로 남고 어느 쪽인지만 미정이다.
VERDICT_VALID = "valid"
VERDICT_INVALID = "invalid"
VERDICT_UNDECIDABLE = "undecidable"

# 겹침이 이 아래면 분할이 어긋난 것으로, 이 위면 대응하는 것으로 읽는다. 문턱은
# 실측 전에 고정한다 — 값을 본 뒤 고르면 자를 결과에 맞추는 것이 된다. 10-fold
# 에서 무작위 겹침의 기대값이 0.10 이므로 0.20 은 우연의 두 배, 0.90 은 거의
# 완전 일치만 통과한다.
OVERLAP_BROKEN_MAX = 0.20
OVERLAP_MATCHED_MIN = 0.90


def strip_recovered(rows: Sequence[dict], ledger: Sequence[dict],
                    start_field: str, end_field: str) -> tuple:
    """원장이 EVT 로 판정한 좌표의 span 을 지워 회수 이전 gold 로 되돌린다.

    좌표로 지우므로 같은 행에 다른 EVT 가 있어도, 같은 좌표에 다른 타입이 있어도
    건드리지 않는다. 되돌린 결과가 옳은지는 개수가 아니라 지문이 판정한다 — 원장
    EVT 판정 수만큼 지우는 것이 정상 경로라 개수 일치는 검사가 되지 못한다. 어긋날
    때는 **덜** 지워지는 쪽이고(좌표가 안 맞거나 그 자리가 EVT 가 아니거나), 그건
    지문이 더 확실히 잡는다.
    """
    targets: Dict[int, Set[tuple]] = {}
    for rec in ledger:
        if str(rec.get("verdict", "")).upper() != EVT:
            continue
        targets.setdefault(int(rec["row_index"]), set()).add(
            (int(rec[start_field]), int(rec[end_field])))

    out: List[dict] = []
    removed = 0
    for i, row in enumerate(rows):
        want = targets.get(i)
        if not want:
            out.append(row)
            continue
        kept = []
        for ent in row["entities"]:
            if (ent["label"] == EVT
                    and (ent["start_char"], ent["end_char"]) in want):
                removed += 1
                continue
            kept.append(ent)
        out.append({**row, "entities": kept})
    return out, removed


def gold_sha256(rows: Sequence[dict]) -> str:
    """gold 를 정본 직렬화로 써서 지문을 낸다.

    커밋된 provenance 의 지문과 같은 함수로 재야 대조가 성립하므로, 직렬화는
    회수를 적용한 쪽과 같은 `dump_gold` 를 쓴다.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fp:
        tmp = fp.name
    try:
        dump_gold(rows, tmp)
        return file_sha256(tmp)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def regenerate_folds(rows: Sequence[dict], stratify: bool,
                     group_key: Optional[str] = "id", seed: int = 42,
                     n_folds: int = 10) -> List[Set[str]]:
    """fold 별 test 행 id 집합을 재생성한다 — 학습은 필요 없다."""
    strat = STRAT_LABELS if stratify else ()
    folds: List[Set[str]] = []
    for fold_index in range(n_folds):
        _, _, test = split_kfold_stratified(
            list(rows), n_folds=n_folds, fold_index=fold_index, seed=seed,
            strat_labels=strat, group_key=group_key)
        folds.append({str(r["id"]) for r in test})
    return folds


def load_dumped_folds(run_dir: str, n_folds: int = 10) -> List[Set[str]]:
    """실행 덤프에서 fold 별 test 행 id 를 읽는다 — `fold{N}/pred_spans.json`."""
    base = pathlib.Path(run_dir)
    folds: List[Set[str]] = []
    for i in range(n_folds):
        path = base / f"fold{i}" / "pred_spans.json"
        with open(path, encoding="utf-8") as fp:
            folds.append(set(json.load(fp).keys()))
    return folds


def fold_overlap(base: Sequence[Set[str]],
                 head: Sequence[Set[str]]) -> dict:
    """두 팔의 fold 대응을 잰다 — 기존 재채점 하네스와 같은 자.

    교집합을 base fold 크기로 나눈다(Jaccard 아님). 자가 다르면 같은 사건을 잰
    수가 문서마다 달라져, 이 감사가 없애려는 부채를 새로 만든다.
    """
    ratios = [len(b & h) / max(len(b), 1) for b, h in zip(base, head)]
    return {
        "exact_folds": sum(1 for b, h in zip(base, head) if b == h),
        "n_folds": len(ratios),
        "mean_fold_overlap": round(sum(ratios) / max(len(ratios), 1), 4),
        "per_fold_overlap": [round(r, 4) for r in ratios],
    }


def classify_overlap(mean_overlap: float) -> str:
    """겹침 하나를 판정 어휘로 옮긴다 — 문턱은 모듈 상수로 고정돼 있다."""
    if mean_overlap >= OVERLAP_MATCHED_MIN:
        return VERDICT_VALID
    if mean_overlap < OVERLAP_BROKEN_MAX:
        return VERDICT_INVALID
    return VERDICT_UNDECIDABLE


def decide(scenarios: Dict[str, dict], head_basis: Optional[str]) -> tuple:
    """시나리오별 판정에서 최종 판정을 고른다.

    **근거 없이는 고르지 않는다.** head 팔 설정의 출처가 없으면 시나리오가
    서로 다른 답을 주더라도 어느 쪽인지 알 수 없고, 그때 하나를 고르는 것은
    판정이 아니라 추측이다. 두 시나리오가 우연히 같은 답이면 설정과 무관하게
    답이 정해지므로 그때는 근거가 없어도 판정할 수 있다.
    """
    verdicts = {s["verdict"] for s in scenarios.values()}
    if len(verdicts) == 1:
        only = verdicts.pop()
        return only, "both scenarios agree — the setting does not change it"
    if head_basis is None:
        return (VERDICT_UNDECIDABLE,
                "scenarios disagree and the head arm split setting is not "
                "recorded in any document, commit, or surviving artifact")
    return (scenarios[head_basis]["verdict"],
            f"head arm setting is documented: {head_basis}")


def audit_report(base_gold: Sequence[dict], head_gold: Sequence[dict],
                 base_stratify: bool, base_basis: Optional[str],
                 head_basis: Optional[str],
                 expected_sha: Optional[Dict[str, str]] = None,
                 positive_control: Optional[Dict[str, str]] = None,
                 group_key: Optional[str] = "id", seed: int = 42,
                 n_folds: int = 10) -> dict:
    """두 팔의 fold 대응을 감사한다 — 재현 파라미터를 값과 함께 남긴다.

    head 팔 설정이 미지라 두 시나리오를 모두 계산한다. `head_basis` 가 주어지면
    그 시나리오가 판정이 되고, 없으면 `decide()` 가 정한다 — 두 시나리오가 같은
    답이면 설정과 무관하게 답이 정해지므로 그 답을, 갈리면 `undecidable` 을 낸다.
    """
    report: dict = {
        "params": {
            "seed": seed,
            "n_folds": n_folds,
            "group_key": group_key,
            "strat_labels": list(STRAT_LABELS),
            "overlap_metric": "intersection / |base fold|",
            "thresholds": {"broken_below": OVERLAP_BROKEN_MAX,
                           "matched_at_or_above": OVERLAP_MATCHED_MIN},
        },
        "arms": {
            "base": {
                "gold_sha256": gold_sha256(base_gold),
                "evt_spans": sum(1 for r in base_gold
                                 for e in r["entities"] if e["label"] == EVT),
                "stratify": base_stratify,
                "basis": base_basis,
            },
            "head": {
                "gold_sha256": gold_sha256(head_gold),
                "evt_spans": sum(1 for r in head_gold
                                 for e in r["entities"] if e["label"] == EVT),
                "stratify": None,
                "basis": head_basis,
            },
        },
    }

    if expected_sha:
        got = {"base": report["arms"]["base"]["gold_sha256"],
               "head": report["arms"]["head"]["gold_sha256"]}
        mismatch = {k: {"expected": v, "got": got[k]}
                    for k, v in expected_sha.items() if got.get(k) != v}
        report["gold_reconstruction"] = {
            "expected": expected_sha, "match": not mismatch,
            "mismatch": mismatch,
        }

    # 양성 대조 — 재생성 파이프라인이 보존된 실행의 fold 를 그대로 되살리나.
    # 이게 없으면 "겹침이 낮다" 가 "도구가 고장" 과 구별되지 않는다. 낮은 겹침은
    # 가설이 예상한 방향이라 조용히 지나간다.
    if positive_control:
        control = {}
        for run_dir, gold_path in positive_control.items():
            rows = load_gold(gold_path)
            got_folds = load_dumped_folds(run_dir, n_folds)
            regen = regenerate_folds(rows, True, group_key, seed, n_folds)
            control[run_dir] = {
                "stratify": True,
                "exact_folds": sum(1 for a, b in zip(got_folds, regen)
                                   if a == b),
                "n_folds": n_folds,
            }
        report["positive_control"] = control

    base_folds = regenerate_folds(base_gold, base_stratify, group_key, seed,
                                  n_folds)
    scenarios = {}
    for name, stratify in (("head_no_stratify", False),
                           ("head_stratify", True)):
        head_folds = regenerate_folds(head_gold, stratify, group_key, seed,
                                      n_folds)
        entry = fold_overlap(base_folds, head_folds)
        entry["stratify"] = stratify
        entry["verdict"] = classify_overlap(entry["mean_fold_overlap"])
        scenarios[name] = entry
    report["scenarios"] = scenarios

    # head 팔 설정은 근거가 있을 때만 채운다 — 근거 없이 적으면 두 시나리오 중
    # 하나를 고른 것이 되고, 그게 바로 이 감사가 안 하기로 한 일이다.
    if head_basis:
        report["arms"]["head"]["stratify"] = scenarios[head_basis]["stratify"]

    verdict, reason = decide(scenarios, head_basis)
    report["verdict"] = verdict
    report["verdict_reason"] = reason
    return report


def check_report(report: dict) -> List[str]:
    """산출물이 스스로 모순되지 않는지 본다 — 위반은 비영 종료 사유다.

    넷을 본다 — gold 역산이 지문으로 검증됐나 · 양성 대조가 돌았고 통과했나 ·
    **base 팔** 설정에 근거가 있나 · 근거 없이 단일 판정을 싣지 않았나.

    **head 팔 근거 부재는 위반이 아니다** — 그게 이 감사의 전제이고 실제 산출물의
    상태다. 넷째가 그것을 조건부로 본다: head 근거가 없고 **두 시나리오가 갈리는데도**
    단일 판정이 실렸을 때만 막는다. base 를 무조건 요구하는 것은 그쪽이 문서로
    확정되는 팔이기 때문이다.

    **마지막 것이 가장 중요하다.** 근거 없이 판정을 실으면 추측이 판정으로 굳고,
    그 뒤로는 아무도 그것이 추측이었음을 알 수 없다. 앞의 **둘**은 "검사를 안 돌린
    것" 과 "돌았는데 실패한 것" 을 각각 잡는다 — 안 돈 검사를 통과로 읽으면 감사가
    근거를 잃는다. 셋째는 값이 없는 것이라 그 구분이 없다.
    """
    problems = []
    recon = report.get("gold_reconstruction")
    if recon is None:
        # 지문 대조를 건너뛴 리포트는 "복원이 맞았다" 를 주장할 수 없다. 없는
        # 검사를 통과로 읽으면 감사 전체가 근거를 잃는다.
        problems.append(
            "gold reconstruction was never verified — rerun with "
            "--expect-base-sha/--expect-head-sha")
    elif not recon.get("match"):
        problems.append(f"gold reconstruction mismatch: {recon['mismatch']}")
    control = report.get("positive_control")
    if not control:
        # 대조를 안 돌리면 낮은 겹침이 "분할이 어긋났다" 인지 "재생성이 고장" 인지
        # 갈리지 않는다. 그리고 낮은 겹침은 가설이 예상한 답이라 조용히 지나간다.
        problems.append(
            "positive control was never run — rerun with --positive-control")
    for slug, ctrl in (control or {}).items():
        if ctrl["exact_folds"] != ctrl["n_folds"]:
            problems.append(
                f"positive control failed for {slug}: "
                f"{ctrl['exact_folds']}/{ctrl['n_folds']} folds reproduced")
    base = report["arms"]["base"]
    if base.get("basis") is None:
        # head 팔에 건 가드와 같은 이유다 — 근거 없이 적은 설정은 기록이 아니라
        # 가정이고, 산출물에 들어가는 순간 둘이 구별되지 않는다.
        problems.append(
            "base arm split setting has no documented basis — pass "
            "--base-basis with the document path:line that records it")
    head = report["arms"]["head"]
    verdicts = {s["verdict"] for s in report["scenarios"].values()}
    if (head.get("basis") is None and len(verdicts) > 1
            and report["verdict"] != VERDICT_UNDECIDABLE):
        problems.append(
            "head arm has no documented basis and the scenarios disagree, "
            f"but the verdict is {report['verdict']!r} — a guess is being "
            "recorded as a decision")
    return problems


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp if ln.strip()]


def cmd_audit(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    head_gold, n_head = strip_recovered(
        rows, _load_jsonl(args.head_ledger),
        args.head_ledger_start, args.head_ledger_end)
    base_gold, n_base = strip_recovered(
        head_gold, _load_jsonl(args.base_ledger),
        args.base_ledger_start, args.base_ledger_end)
    logger.info("reversed %d head-arm spans and %d base-arm spans",
                n_head, n_base)

    expected = None
    if args.expect_base_sha and args.expect_head_sha:
        expected = {"base": args.expect_base_sha, "head": args.expect_head_sha}

    control = None
    if args.positive_control:
        if not (args.positive_control_base_run
                and args.positive_control_head_run):
            raise SystemExit(
                "--positive-control needs --positive-control-base-run and "
                "--positive-control-head-run: the run dirs it once defaulted "
                "to are no longer committed")
        # 대조할 두 실행 중 base 는 head 팔 gold 로, head 는 현 gold 로 돌았다.
        with tempfile.NamedTemporaryFile(suffix=".jsonl",
                                         delete=False) as fp:
            head_path = fp.name
        dump_gold(head_gold, head_path)
        control = {args.positive_control_base_run: head_path,
                   args.positive_control_head_run: args.gold}

    try:
        report = audit_report(
            base_gold, head_gold, base_stratify=args.base_stratify,
            base_basis=args.base_basis, head_basis=args.head_basis,
            expected_sha=expected, positive_control=control,
            group_key=args.group_key, seed=args.seed, n_folds=args.n_folds)
    finally:
        if control:
            pathlib.Path(head_path).unlink(missing_ok=True)

    problems = check_report(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)
    for name, s in report["scenarios"].items():
        logger.info("%s: exact %d/%d, mean overlap %.4f -> %s",
                    name, s["exact_folds"], s["n_folds"],
                    s["mean_fold_overlap"], s["verdict"])
    logger.info("verdict: %s — %s", report["verdict"],
                report["verdict_reason"])
    if problems:
        raise SystemExit("FAIL: " + "; ".join(problems))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="audit whether two arms of a past k-fold run share "
                    "fold membership")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "audit",
        help="reverse gold recoveries and compare regenerated fold splits")
    a.add_argument("--gold", required=True, help="current gold JSONL")
    a.add_argument("--head-ledger", required=True,
                   help="judgement ledger applied after the head arm ran")
    a.add_argument("--head-ledger-start", default="insert_start")
    a.add_argument("--head-ledger-end", default="insert_end")
    a.add_argument("--base-ledger", required=True,
                   help="judgement ledger that produced the head arm gold")
    a.add_argument("--base-ledger-start", default="start")
    a.add_argument("--base-ledger-end", default="end")
    a.add_argument("--expect-base-sha", default=None,
                   help="committed gold sha256 for the base arm")
    a.add_argument("--expect-head-sha", default=None,
                   help="committed gold sha256 for the head arm")
    a.add_argument("--base-stratify", action="store_true",
                   help="the base arm used stratified folds")
    a.add_argument("--base-basis", default=None,
                   help="document path:line recording the base arm setting")
    a.add_argument("--head-basis", default=None,
                   choices=["head_no_stratify", "head_stratify"],
                   help="scenario key the head arm setting is documented as; "
                        "omit when no record survives")
    a.add_argument("--positive-control", action="store_true",
                   help="check regenerated folds against a run whose "
                        "fold*/pred_spans.json dumps are on disk")
    a.add_argument("--positive-control-base-run",
                   help="run dir holding the base arm fold*/pred_spans.json")
    a.add_argument("--positive-control-head-run",
                   help="run dir holding the head arm fold*/pred_spans.json")
    a.add_argument("--group-key", default="id")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--n-folds", type=int, default=10)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_audit)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
