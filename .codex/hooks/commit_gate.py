#!/usr/bin/env python3
"""Codex PreToolUse 훅에서 커밋 생성 명령을 검사한다."""

import json
import os
import re
import shlex
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_core as core  # noqa: E402

HISTORY_EXITS = {'rebase', 'merge', 'cherry-pick', 'am'}
EXIT_OPTIONS = {'--abort', '--quit', '--skip'}
HISTORY_SUBCOMMANDS = {
    'am',
    'branch',
    'cherry-pick',
    'commit',
    'filter-branch',
    'merge',
    'notes',
    'pull',
    'push',
    'rebase',
    'replace',
    'reset',
    'revert',
    'stash',
    'tag',
    'update-ref',
}
PASSTHROUGH_SUBCOMMANDS = {
    'add',
    'blame',
    'cat-file',
    'checkout',
    'clean',
    'config',
    'diff',
    'grep',
    'help',
    'log',
    'ls-files',
    'ls-tree',
    'mv',
    'name-rev',
    'restore',
    'rev-parse',
    'rm',
    'shortlog',
    'show',
    'status',
    'switch',
    'version',
}
WRAPPERS = {'bash', 'command', 'dash', 'env', 'nice', 'nohup', 'sh', 'sudo', 'zsh'}
ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
DIRECT_GIT_RE = re.compile(r'^\s*git(?:[ \t]|$)')
NONLITERAL_PATH_CHARS = frozenset('*?[]{}~$`\\')
SHELL_PUNCTUATION = frozenset(';&|()<>')


class CommandPlan(NamedTuple):
    creates_commit: bool
    project_dir: str | None
    error: str | None


def _tokens(command):
    """Shell operator를 실행하지 않고 토큰 경계만 확인한다."""
    lexer = shlex.shlex(
        command,
        posix=True,
        punctuation_chars=';&|()<>',
    )
    lexer.whitespace_split = True
    lexer.commenters = ''
    return list(lexer)


def _is_shell_operator(token):
    return bool(token) and set(token) <= SHELL_PUNCTUATION


def _mentions_wrapped_git(tokens):
    if not tokens:
        return False
    executable = os.path.basename(tokens[0])
    if ASSIGNMENT_RE.match(tokens[0]):
        index = 0
        while index < len(tokens) and ASSIGNMENT_RE.match(tokens[index]):
            index += 1
        return index < len(tokens) and os.path.basename(tokens[index]) == 'git'
    if executable not in WRAPPERS and not executable.startswith('python'):
        return False
    return any(
        os.path.basename(token) == 'git' or re.search(r'\bgit\s+', token)
        for token in tokens[1:]
    )


def _literal_project_dir(cwd, path, command):
    if not path or any(char in NONLITERAL_PATH_CHARS for char in path):
        raise ValueError(f'-C requires a literal path without expansion: {path}')
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ValueError('-C requires a literal path without control characters')
    if '\\' in command:
        raise ValueError('-C requires a literal path without backslash escaping')
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(cwd, path))


def analyze_command(command: str, cwd: str) -> CommandPlan:
    """명령을 실행하지 않고 direct commit allowlist로 분류한다."""
    try:
        tokens = _tokens(command)
    except ValueError as exc:
        return CommandPlan(False, None, f'could not parse shell command: {exc}')
    if not tokens:
        return CommandPlan(False, None, None)

    if any(char in command for char in ('\0', '\r', '\n')):
        potential_git = any(
            os.path.basename(token) == 'git' or re.search(r'\bgit\s+', token)
            for token in tokens
        )
        if potential_git:
            return CommandPlan(
                True,
                None,
                'control characters are unsupported; use a direct git commit',
            )

    has_operator = any(_is_shell_operator(token) for token in tokens)
    potential_git = any(
        os.path.basename(token) == 'git' or re.search(r'\bgit\s+', token)
        for token in tokens
        if not _is_shell_operator(token)
    )
    if has_operator and potential_git:
        return CommandPlan(
            True,
            None,
            'compound shell commands are unsupported; use a direct git commit',
        )
    if _mentions_wrapped_git(tokens):
        return CommandPlan(
            True,
            None,
            'Git wrappers are unsupported; use a direct git commit',
        )
    if os.path.basename(tokens[0]) != 'git':
        return CommandPlan(False, None, None)
    if not DIRECT_GIT_RE.match(command):
        return CommandPlan(
            True,
            None,
            'ambiguous Git spelling is unsupported; use a direct git commit',
        )

    project_dir = os.path.abspath(cwd)
    index = 1
    used_git_c = False
    if index < len(tokens) and tokens[index] == '-C':
        if index + 1 >= len(tokens):
            return CommandPlan(True, None, '-C requires a literal path')
        try:
            project_dir = _literal_project_dir(cwd, tokens[index + 1], command)
        except ValueError as exc:
            return CommandPlan(True, None, str(exc))
        used_git_c = True
        index += 2
    elif index < len(tokens) and tokens[index].startswith('-'):
        return CommandPlan(
            True,
            None,
            f'Git global option is unsupported: {tokens[index]}; '
            'use a direct git commit',
        )

    if '\\' in command:
        return CommandPlan(
            True,
            None,
            'ambiguous Git spelling is unsupported; use a direct git commit',
        )

    if index >= len(tokens):
        return CommandPlan(False, None, None)
    subcommand = tokens[index]
    arguments = tokens[index + 1:]
    if (
        not used_git_c
        and subcommand in HISTORY_EXITS
        and len(arguments) == 1
        and arguments[0] in EXIT_OPTIONS
    ):
        return CommandPlan(False, None, None)
    if subcommand == 'commit':
        if '`' in command or '$(' in command:
            return CommandPlan(
                True,
                None,
                'shell expansion is unsupported; use a direct git commit',
            )
        return CommandPlan(True, project_dir, None)
    if subcommand in PASSTHROUGH_SUBCOMMANDS:
        return CommandPlan(False, None, None)
    if subcommand in HISTORY_SUBCOMMANDS:
        return CommandPlan(
            True,
            None,
            'history operations must be performed by the user outside this hook',
        )
    return CommandPlan(
        False,
        None,
        f'unsupported Git subcommand: {subcommand}',
    )


def creates_a_commit(command: str | None) -> bool:
    """명령이 커밋을 만들거나 커밋 작업을 계속하는지 판정한다."""
    if not command:
        return False
    return analyze_command(command, '.').creates_commit


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
    }, ensure_ascii=True))


def _input_error(diagnostic):
    return core.GateResult(
        False,
        'input-error',
        'Codex hook input is invalid',
        (diagnostic,),
    )


def _command_error(diagnostic):
    return core.GateResult(
        False,
        'command-error',
        'Commit command cannot be inspected safely; split the command',
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
    plan = analyze_command(command, cwd or os.getcwd())
    if plan.error is not None:
        deny(_command_error(plan.error))
        return
    if not plan.creates_commit:
        return

    try:
        result = core.evaluate(plan.project_dir)
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
