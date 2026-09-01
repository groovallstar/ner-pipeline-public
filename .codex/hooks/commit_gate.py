#!/usr/bin/env python3
"""Codex PreToolUse 훅에서 커밋 생성 명령을 검사한다."""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_core as core  # noqa: E402

GIT_COMMIT_RE = re.compile(
    r'(?:^|[\s;&|(])'
    r'(?:[^\s;&|]*/)?git'
    r'(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir[=\s]\S+|--work-tree[=\s]\S+'
    r'|--namespace[=\s]\S+|--exec-path[=\s]\S+|-[^\s]+))*'
    r'\s+(?:commit|cherry-pick|revert|am|rebase|merge)\b'
)
GIT_ABORT_RE = re.compile(r'--(?:abort|quit|skip)\b')


def creates_a_commit(command: str | None) -> bool:
    """명령이 커밋을 만들거나 커밋 작업을 계속하는지 판정한다."""
    command = command or ''
    if GIT_ABORT_RE.search(command):
        return False
    return bool(GIT_COMMIT_RE.search(command))


def deny(result):
    """Codex PreToolUse deny 응답을 출력한다."""
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': (
                f'[{result.check}] {result.reason}\n'
                + '\n'.join(f'- {item}' for item in result.findings)
            ),
        }
    }, ensure_ascii=False))


def _input_error(diagnostic):
    return core.GateResult(
        False,
        'input-error',
        'Codex hook input is invalid',
        (diagnostic,),
    )


def main() -> None:
    """표준 입력의 Codex 훅 JSON을 검증하고 필요한 커밋만 검사한다."""
    try:
        data = json.load(sys.stdin)
    except Exception as exc:
        deny(_input_error(f'JSON parsing failed: {type(exc).__name__}: {exc}'))
        return

    if not isinstance(data, dict):
        deny(_input_error('hook payload must be a JSON object'))
        return

    tool_name = data.get('tool_name')
    tool_input = data.get('tool_input')
    cwd = data.get('cwd')
    if not isinstance(tool_name, str):
        deny(_input_error('tool_name must be a string'))
        return
    if not isinstance(tool_input, dict):
        deny(_input_error('tool_input must be an object'))
        return
    if cwd is not None and not isinstance(cwd, str):
        deny(_input_error('cwd must be a string or null'))
        return
    if tool_name != 'Bash':
        return

    command = tool_input.get('command')
    if not isinstance(command, str):
        deny(_input_error('tool_input.command must be a string'))
        return
    if not creates_a_commit(command):
        return

    project_dir = cwd or os.getcwd()
    try:
        result = core.evaluate(project_dir)
    except Exception as exc:
        result = core.GateResult(
            False,
            'internal-error',
            'Codex gate checks could not complete',
            (f'{type(exc).__name__}: {exc}',),
        )
    if result.allowed is not True:
        deny(result)


if __name__ == '__main__':
    main()
