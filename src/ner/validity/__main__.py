"""CLI 진입점 — `python -m ner.validity {std,repro,compare}`."""
import argparse
import json

from ner.validity.gate import compare
from ner.validity.variance import (
    fold_std,
    load_sigma_map,
    repro_std,
    write_sigma_repro,
)


def _build_parser():
    """CLI 파서를 구성한다 (std / repro / compare 서브커맨드)."""
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
