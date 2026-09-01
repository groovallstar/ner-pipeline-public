#!/usr/bin/env python3
"""Codex PreToolUse 훅에서 커밋 생성 명령을 검사한다."""

import json
import os
import shlex
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_core as core  # noqa: E402

COMMIT_SUBCOMMANDS = {'commit', 'cherry-pick', 'revert', 'am', 'rebase', 'merge'}
ABORT_OPTIONS = {'--abort', '--quit', '--skip'}
SHELL_OPERATORS = {'&&', '||', ';', '|', '&', '(', ')', '<', '>', '<<', '>>'}
SAFE_GIT_FLAGS = {
    '--no-pager',
    '--paginate',
    '--literal-pathspecs',
    '--glob-pathspecs',
    '--noglob-pathspecs',
    '--icase-pathspecs',
}


class CommandPlan(NamedTuple):
    creates_commit: bool
    project_dir: str | None
    error: str | None


def _tokens(command):
    """Shell line continuation을 접은 뒤 operator 경계를 보존해 토큰화한다."""
    continued = command.replace('\\\r\n', '').replace('\\\n', '')
    if '\n' in continued or '\r' in continued:
        raise ValueError('raw newlines require the command to be split')
    lexer = shlex.shlex(
        continued,
        posix=True,
        punctuation_chars=';&|()<>',
    )
    lexer.whitespace_split = True
    lexer.commenters = ''
    return list(lexer)


def _shell_units(tokens):
    units = []
    operators = []
    current = []
    for token in tokens:
        if token in SHELL_OPERATORS:
            if not current:
                raise ValueError(f'unsupported shell operator placement: {token}')
            units.append(current)
            operators.append(token)
            current = []
        else:
            current.append(token)
    if not current:
        if tokens:
            raise ValueError('command ends with a shell operator')
        return [], []
    units.append(current)
    return units, operators


def _resolve_path(base, value):
    if any(token in value for token in ('$', '`')):
        raise ValueError(f'dynamic path cannot be inspected safely: {value}')
    value = os.path.expanduser(value)
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(base, value))


def _git_unit(unit, cwd):
    if not unit or os.path.basename(unit[0]) != 'git':
        return False, cwd, None

    project_dir = cwd
    unsafe_option = None
    index = 1
    while index < len(unit):
        token = unit[index]
        if token == '-C':
            if index + 1 >= len(unit):
                return False, cwd, '-C requires a path'
            try:
                project_dir = _resolve_path(project_dir, unit[index + 1])
            except ValueError as exc:
                return False, cwd, str(exc)
            index += 2
        elif token.startswith('-C') and token != '-C':
            try:
                project_dir = _resolve_path(project_dir, token[2:])
            except ValueError as exc:
                return False, cwd, str(exc)
            index += 1
        elif token in ('--git-dir', '--work-tree'):
            unsafe_option = unsafe_option or token
            index += 2
        elif token.startswith(('--git-dir=', '--work-tree=')):
            unsafe_option = unsafe_option or token.split('=', 1)[0]
            index += 1
        elif token in ('-c', '--namespace', '--exec-path'):
            if index + 1 >= len(unit):
                return False, cwd, f'{token} requires a value'
            index += 2
        elif token.startswith(('-c', '--namespace=', '--exec-path=')):
            index += 1
        elif token in SAFE_GIT_FLAGS:
            index += 1
        elif token.startswith('-'):
            if any(name in unit[index + 1:] for name in COMMIT_SUBCOMMANDS):
                return False, cwd, f'Git option cannot be inspected safely: {token}'
            return False, cwd, None
        else:
            break

    if index >= len(unit):
        return False, project_dir, None
    subcommand = unit[index]
    if subcommand not in COMMIT_SUBCOMMANDS:
        return False, project_dir, None
    if unsafe_option is not None:
        return True, project_dir, (
            f'{unsafe_option} cannot be inspected safely; split the command '
            'and commit from the target working tree'
        )
    if subcommand != 'commit' and any(
        option in ABORT_OPTIONS for option in unit[index + 1:]
    ):
        return False, project_dir, None
    return True, project_dir, None


def analyze_command(command: str, cwd: str) -> CommandPlan:
    """하나의 안전한 commit unit과 검사할 저장소를 결정한다."""
    try:
        tokens = _tokens(command)
        units, operators = _shell_units(tokens)
    except ValueError as exc:
        return CommandPlan(False, None, f'could not parse shell command: {exc}')

    current_dir = os.path.abspath(cwd)
    commits = []
    non_cd_units = []
    for index, unit in enumerate(units):
        if unit[0] == 'cd':
            if unit not in ([unit[0], unit[-1]], [unit[0], '--', unit[-1]]):
                return CommandPlan(False, None, 'cd form cannot be inspected safely')
            if index >= len(operators) or operators[index] != '&&':
                return CommandPlan(
                    False,
                    None,
                    'cd before commit must use && so failure cannot change the target',
                )
            try:
                current_dir = _resolve_path(current_dir, unit[-1])
            except ValueError as exc:
                return CommandPlan(False, None, str(exc))
            continue

        creates, project_dir, error = _git_unit(unit, current_dir)
        if error:
            return CommandPlan(creates, project_dir, error)
        if creates:
            commits.append((index, project_dir))
        else:
            non_cd_units.append(index)

    if not commits:
        return CommandPlan(False, None, None)
    if len(commits) != 1 or non_cd_units or commits[0][0] != len(units) - 1:
        return CommandPlan(
            True,
            None,
            'split compound commands so the commit snapshot can be inspected',
        )
    return CommandPlan(True, commits[0][1], None)


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
    }, ensure_ascii=False))


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
