#!/usr/bin/env python3
"""반복 지적을 모으는 도구 (lessons-digest 스킬의 부품).

리퓨터 판정(.omc/state/refuter/*.json)의 결함성 지적을 워크트리 전체에서
긁어 추적 렛저(docs/specs/lessons-ledger.jsonl)에 누적한다. id 로 중복을
막아 재실행해도 안전하다(idempotent). 소스는 SOURCES 레지스트리에 어댑터
함수 하나를 추가하면 확장된다 — 지금은 refuter 만 등록.

서브커맨드:
  ingest         소스에서 결함성 지적을 긁어 렛저에 append (새것만)
  list           렛저 항목 출력 (--status open, --json)
  status         항목 상태 변경 (--id ... --set promoted|declined)
  pending-count  아직 렛저에 없는 결함성 지적 수 출력 (넛지 훅용)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

# 렛저 경로 (프로젝트 루트 기준)
LEDGER_REL = "docs/specs/lessons-ledger.jsonl"

# 긍정(통과) 지적 — 결함이 아니므로 수집에서 제외.
# refuter 축 라인은 "(a) <라벨> PASS:" / "[OK] ..." / "... 해소 확인" 형태로
# PASS·OK 가 문장 중간에 오므로, 축 프리픽스와 통과 어휘를 함께 잡는다.
_POS = re.compile(
    r"^\s*(\([a-z0-9]\)\s*)?(PASS|CONFIRMED)\b"
    r"|^\s*\([a-z0-9]\)[^:]*\b(PASS|OK)\b"
    r"|^\s*\[(OK|RESOLVED)"
    r"|[—-]\s*PASS\b|:\s*PASS\b|\bPASS\)"
    r"|충족|없음 확인|해소 확인|사유 없음"
    r"|정합.{0,4}(확인|일치)|무결성.{0,8}PASS",
    re.IGNORECASE,
)
# 주의(결함) 마커 — 하나라도 있으면 수집
_NEG = re.compile(
    r"\b(HIGH|MEDIUM|LOW|MAJOR|MINOR|FAIL|BLOCKING|NIT|WARN(?:ING)?)\b"
    r"|\[nit\]|\(nit\)"
    r"|권장|과장|약함|약화|미배선|불일치|누락|버그|위험|오류|stale",
    re.IGNORECASE,
)


def _severity(text):
    # 매칭된 마커로 심각도 정규화 (NON-BLOCKING NIT 오분류 방지 위해 순서 고정)
    t = text.upper()
    if "NON-BLOCKING" in t or "[NIT]" in t or "(NIT)" in t \
            or re.search(r"\bNIT\b", t):
        return "nit"
    if re.search(r"\b(FAIL|HIGH|MAJOR|BLOCKING)\b", t):
        return "high"
    if re.search(r"\bMEDIUM\b", t):
        return "medium"
    if re.search(r"\b(LOW|MINOR)\b", t):
        return "low"
    return "medium"


def _classify(text):
    # (수집여부, 심각도) — 긍정이면 버리고, 결함 마커가 있으면 수집한다
    if _POS.search(text):
        return False, None
    if _NEG.search(text):
        return True, _severity(text)
    return False, None


def _root(args):
    # 렛저가 놓일 리포 루트: --root > CLAUDE_PROJECT_DIR > 파일 위치 3단계 상위
    if getattr(args, "root", None):
        return os.path.abspath(args.root)
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(d, "..", "..", ".."))


def _worktrees(root):
    # 같은 리포의 모든 워크트리 경로 (.omc 는 워크트리마다 물리적으로 분리)
    try:
        out = subprocess.run(
            ["git", "-C", root, "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return [root]
    paths = [
        ln[len("worktree "):]
        for ln in out.splitlines()
        if ln.startswith("worktree ")
    ]
    return paths or [root]


def _refuter_findings(worktrees):
    # refuter 어댑터: (source, ref, model, text) 튜플 산출
    seen = set()
    for wt in worktrees:
        pat = os.path.join(wt, ".omc/state/refuter/*.json")
        for fp in sorted(glob.glob(pat)):
            real = os.path.realpath(fp)
            if real in seen:
                continue
            seen.add(real)
            try:
                with open(fp, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            ref = d.get("diff_hash") or os.path.basename(fp)[:12]
            model = d.get("model", "")
            for f in d.get("findings", []):
                yield ("refuter", ref, model, str(f))


# 소스 어댑터 레지스트리 — 새 소스는 여기에 함수 하나를 추가한다
SOURCES = {"refuter": _refuter_findings}


def _entry_id(source, ref, text):
    raw = f"{source}|{ref}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _load_ledger(path):
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        entries.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    return entries


def _write_ledger(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _iter_findings(worktrees):
    for source, fn in SOURCES.items():
        for tup in fn(worktrees):
            yield tup


def cmd_ingest(args):
    root = _root(args)
    ledger = os.path.join(root, LEDGER_REL)
    existing = {e["id"] for e in _load_ledger(ledger)}
    wts = _worktrees(root)
    new, scanned = [], 0
    for src, ref, model, text in _iter_findings(wts):
        scanned += 1
        keep, sev = _classify(text)
        if not keep:
            continue
        eid = _entry_id(src, ref, text)
        if eid in existing:
            continue
        existing.add(eid)
        new.append({
            "id": eid, "source": src, "ref": ref, "model": model,
            "severity": sev, "text": text, "status": "open",
        })
    if new:
        with open(ledger, "a", encoding="utf-8") as fh:
            for e in new:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(
        f"[lessons] scanned {scanned} findings across {len(wts)} "
        f"worktree(s); added {len(new)} new -> {LEDGER_REL}"
    )
    if new:
        c = Counter(e["severity"] for e in new)
        print("  new by severity: "
              + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


def cmd_list(args):
    root = _root(args)
    ledger = os.path.join(root, LEDGER_REL)
    rows = _load_ledger(ledger)
    if args.status:
        rows = [e for e in rows if e.get("status") == args.status]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for e in rows:
        print(
            f"- [{e['id']}] ({e['severity']}) {e['source']}:{e['ref']}  "
            f"{e['text'][:140]}"
        )
    print(f"\n{len(rows)} entries"
          + (f" (status={args.status})" if args.status else ""))


def cmd_status(args):
    root = _root(args)
    ledger = os.path.join(root, LEDGER_REL)
    entries = _load_ledger(ledger)
    ids = set(args.id)
    n = 0
    for e in entries:
        if e["id"] in ids:
            e["status"] = args.set
            if args.target:
                e["target"] = args.target
            n += 1
    _write_ledger(ledger, entries)
    print(f"[lessons] set status={args.set} on {n} entry(ies)")


def cmd_pending(args):
    # 아직 렛저에 없는 결함성 지적 수 (넛지 훅이 읽는 유일 출력)
    root = _root(args)
    ledger = os.path.join(root, LEDGER_REL)
    existing = {e["id"] for e in _load_ledger(ledger)}
    wts = _worktrees(root)
    pend = 0
    for src, ref, model, text in _iter_findings(wts):
        keep, _ = _classify(text)
        if keep and _entry_id(src, ref, text) not in existing:
            pend += 1
    print(pend)


def main():
    p = argparse.ArgumentParser(
        description="Collect recurring review findings."
    )
    p.add_argument("--root", help="repo root (default: CLAUDE_PROJECT_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="scrape sources into the ledger")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("list", help="print ledger entries")
    sp.add_argument("--status", choices=["open", "promoted", "declined"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("status", help="update entry status")
    sp.add_argument("--id", nargs="+", required=True)
    sp.add_argument("--set", required=True,
                    choices=["open", "promoted", "declined"])
    sp.add_argument("--target",
                    help="promotion target note (e.g. conventions)")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("pending-count",
                        help="count findings not yet in the ledger")
    sp.set_defaults(func=cmd_pending)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
