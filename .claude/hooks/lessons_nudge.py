#!/usr/bin/env python3
"""Stop 훅: 미수집 반복-지적 넛지 (비차단).

렛저에 아직 안 들어간 결함성 지적(현재는 refuter 소스) 수를 세어 임계
이상이면 stderr 로 'lessons-digest 실행 권장' 알림만 낸다. 절대 완료를
막지 않는다 — 규칙 승격은 항상 사람이 손으로 하므로 훅은 알림 역할만.

카운트는 collect.py 에 위임(로직 이중화 방지). 같은 pending 값에서는
재알림하지 않아 매 턴 스팸을 피한다.
"""
import json
import os
import subprocess
import sys

THRESHOLD = int(os.environ.get("LESSONS_DIGEST_THRESHOLD", "3"))


def main():
    # OMC 킬스위치 존중 (다른 훅과 동일 컨벤션)
    skip = [
        s.strip()
        for s in (os.environ.get("OMC_SKIP_HOOKS") or "").split(",")
    ]
    if os.environ.get("DISABLE_OMC") == "1" or "lessons-nudge" in skip:
        sys.exit(0)

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    # 재진입 Stop 에서는 조용히 통과
    if data.get("stop_hook_active") is True:
        sys.exit(0)

    proj = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
    script = os.path.join(
        proj, ".claude/skills/lessons-digest/collect.py"
    )
    if not os.path.exists(script):
        sys.exit(0)

    try:
        out = subprocess.run(
            ["/usr/bin/env", "python3", script, "pending-count"],
            cwd=proj, capture_output=True, text=True, timeout=30,
        )
        pending = int((out.stdout or "0").strip() or "0")
    except Exception:
        sys.exit(0)

    if pending < THRESHOLD:
        sys.exit(0)

    # 같은 pending 값에서 재알림 억제 (매 턴 스팸 방지)
    state = os.path.join(proj, ".omc/state/refuter/.lessons-nudge")
    last = None
    try:
        last = int(open(state).read().strip())
    except Exception:
        pass
    if pending == last:
        sys.exit(0)
    try:
        os.makedirs(os.path.dirname(state), exist_ok=True)
        with open(state, "w") as fh:
            fh.write(str(pending))
    except Exception:
        pass

    print(
        f"[lessons] {pending} un-digested review findings accumulated "
        f"(>= {THRESHOLD}). Consider running the `lessons-digest` skill to "
        "review recurring ones and promote them to conventions.",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
