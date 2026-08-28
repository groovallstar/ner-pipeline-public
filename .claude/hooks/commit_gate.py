#!/usr/bin/env python3
"""PreToolUse 훅: 커밋을 만드는 git 명령 직전에 기계 검사를 건다.

Stop 훅만 있으면 커밋을 마친 턴에는 미커밋 diff 가 남지 않아 검사 대상
자체가 사라진다 — 한 턴에서 수정하고 커밋까지 하면 아무 검사도 받지
않는다. 그래서 커밋 명령을 가로채, 커밋되기 전 상태에서 같은 검사를
돌린다. 이 시점에는 `git add` 된 새 파일도 `git diff HEAD` 에 잡히므로
Stop 훅이 못 보던 신규 파일까지 검사 범위에 들어온다.

반박자 판정은 여기서 새로 띄우지 않는다 — 커밋 명령 중간에 서브에이전트를
부를 수 없다. 반박자를 부르는 쪽은 Stop 훅이다.

차단은 누적 상한 없이 매번 건다. Stop 훅과 달리 재진입 루프를 만들지
않고, 상한을 두면 "여러 번 시도하면 통과"가 회피 경로가 되기 때문이다.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_core as core  # noqa: E402

# 커밋을 만드는 git 하위명령 감지. `commit` 만 잡으면 충돌을 해소하고
# 이어가는 경로(`rebase --continue` 등)와 다른 커밋을 옮겨오는 경로
# (`cherry-pick`·`revert`·`am`)가 통째로 빠진다. 경로 있는 git, 전역
# 플래그(-C · -c · --git-dir 등), `&&`·`;` 로 이어붙인 형태를 모두 잡는다.
# 셸 인용을 파싱하지 않으므로 문자열 안의 우연한 일치는 오탐이 되지만,
# 보수적 오탐이라 허용한다.
GIT_COMMIT_RE = re.compile(
    r'(?:^|[\s;&|(])'
    r'(?:[^\s;&|]*/)?git'
    r'(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir[=\s]\S+|--work-tree[=\s]\S+'
    r'|--namespace[=\s]\S+|--exec-path[=\s]\S+|-[^\s]+))*'
    r'\s+(?:commit|cherry-pick|revert|am|rebase|merge)\b'
)

# 진행 중인 작업을 접거나 건너뛰는 형태는 커밋을 만들지 않는다. 막으면
# 충돌 상태에서 빠져나올 길이 사라진다.
GIT_ABORT_RE = re.compile(r'--(?:abort|quit|skip)\b')


def _passthrough(notice=None):
    # exit 0 = 훅이 판단하지 않음 (기존 권한 규칙에 맡긴다). 검사가 대상 없이
    # 지나갔다면 그 사실만 화면에 알리고 통과시킨다 — 게이트 고장이 커밋을
    # 영구 봉쇄해선 안 되지만, 조용히 통과하면 통과의 의미가 달라진다.
    if notice:
        print(json.dumps({'systemMessage': notice}, ensure_ascii=False))
    sys.exit(0)


def _deny(reason):
    # PreToolUse deny = 도구 호출 자체가 실행되지 않는다
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        }
    }, ensure_ascii=False))
    sys.exit(0)


def creates_a_commit(command):
    command = command or ''
    if GIT_ABORT_RE.search(command):
        return False
    return bool(GIT_COMMIT_RE.search(command))


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        _passthrough()

    if core.skip_requested():
        _passthrough()
    if data.get('tool_name') != 'Bash':
        _passthrough()
    command = (data.get('tool_input') or {}).get('command', '')
    if not creates_a_commit(command):
        _passthrough()

    proj = core.project_dir(data)

    # 커밋할 내용이 없으면(빈 커밋·메시지만 고치는 amend) 검사 대상도 없다
    text = core.diff_text(proj)
    if not text.strip():
        _passthrough()

    dhash = core.anchor_hash(proj, text)
    sdir = core.state_dir(proj)

    result = core.run_deterministic(proj, sdir, dhash)
    if result is None:
        _passthrough(core.pass_notice(sdir, dhash))

    check, reason, findings = result
    core.log(sdir, {
        'diff_hash': dhash, 'entry': 'commit', 'layer': 'machine',
        'check': check, 'result': 'BLOCK', 'findings': findings,
    })
    _deny(reason)


if __name__ == '__main__':
    main()
