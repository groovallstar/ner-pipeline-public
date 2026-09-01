#!/usr/bin/env python3
"""Codex 커밋 게이트의 상태 없는 결정적 검사."""

import glob
import json
import os
import re
import subprocess
import sys
from typing import NamedTuple

NUM_RE = re.compile(r'\b[01]\.\d{2,4}\b')
CERT_SRC_RE = re.compile(r'<!--\s*certified:\s*([^\s>]+?)\s*-->')
WEAKEN_RE = re.compile(
    r'@pytest\.mark\.skip\b|@pytest\.mark\.xfail'
    r'|pytest\.skip\(|pytest\.xfail\('
)

CATALOG_EXCLUDE = ('test_predictions', 'checkpoint-')
CATALOG_MAX_BYTES = 20_000_000
CATALOG_EXT = ('json',)

RULER_PATHS = (
    'docs/manual/data/canonical-entity-schema.md',
    'src/ner/classifier/data_utils.py',
    'src/ner/classifier/kfold_pool.py',
)
RULER_DIRS = (
    'src/ner/metrics/',
    'src/ner/validity/',
)

SELF_DIRS = ('.codex/hooks/', 'tests/hooks/')
SELF_TESTS = 'tests/hooks'


class GateResult(NamedTuple):
    allowed: bool
    check: str
    reason: str
    findings: tuple[str, ...]


def _run_git(args, cwd, text):
    try:
        out = subprocess.run(
            ['git', *args],
            cwd=cwd,
            capture_output=True,
            text=text,
            timeout=20,
        )
    except Exception as exc:
        raise RuntimeError(
            f'git {args[0]} could not run: {type(exc).__name__}: {exc}'
        ) from exc
    if out.returncode != 0:
        diagnostic = (out.stderr or out.stdout).strip()
        if isinstance(diagnostic, bytes):
            diagnostic = os.fsdecode(diagnostic)
        diagnostic = diagnostic or 'no diagnostic'
        raise RuntimeError(
            f'git {args[0]} exited {out.returncode}: {diagnostic}'
        )
    return out.stdout


def git(args, cwd):
    """Git 명령을 실행하고 예상하지 않은 실패를 예외로 올린다."""
    return _run_git(args, cwd, text=True)


def _git_bytes(args, cwd):
    return _run_git(args, cwd, text=False)


def diff_text(proj):
    """인덱스를 포함한 HEAD 기준 미커밋 diff를 반환한다."""
    return git(['diff', 'HEAD'], proj)


def _changed_paths(proj):
    raw = _git_bytes(
        ['diff', '--no-renames', '--name-only', '-z', 'HEAD'],
        proj,
    )
    return [os.fsdecode(path) for path in raw.split(b'\0') if path]


def _collect_numbers(node, catalog):
    if isinstance(node, dict):
        for value in node.values():
            _collect_numbers(value, catalog)
    elif isinstance(node, list):
        for value in node:
            _collect_numbers(value, catalog)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        for digits in (2, 3, 4):
            text = f'{float(node):.{digits}f}'
            catalog.add(text)
            catalog.add(text.rstrip('0'))


def ledger_files(root):
    """인용 수치의 근거로 사용하는 원장 파일을 열거한다."""
    found = []
    for ext in CATALOG_EXT:
        found += glob.glob(
            os.path.join(root, '**', f'*.{ext}'),
            recursive=True,
        )
    return sorted(found)


def _numbers_from_files(paths):
    catalog = set()
    for path in paths:
        if any(token in path for token in CATALOG_EXCLUDE):
            continue
        try:
            if os.path.getsize(path) > CATALOG_MAX_BYTES:
                continue
            with open(path) as handle:
                _collect_numbers(json.load(handle), catalog)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'could not parse certified JSON {path}: {exc}') from exc
    return catalog


def _metric_catalog(proj):
    root = os.path.join(proj, 'certified')
    if not os.path.isdir(root):
        return set()
    return _numbers_from_files(ledger_files(root))


def _added_table_numbers(diff, known_path=None):
    entries = []
    path = known_path
    lineno = 0
    for line in diff.splitlines():
        if known_path is None and line.startswith('+++ b/'):
            path = line[6:]
        elif line.startswith('@@'):
            match = re.search(r'\+(\d+)', line)
            if match is None:
                raise ValueError(f'could not parse diff hunk: {line}')
            lineno = int(match.group(1))
        elif line.startswith('+') and not line.startswith('+++'):
            row = line[1:]
            if row.strip().startswith('|'):
                numbers = NUM_RE.findall(row)
                if numbers and path:
                    entries.append((path, lineno, numbers))
            lineno += 1
        elif line.startswith('-') and not line.startswith('---'):
            continue
        elif line.startswith(' '):
            lineno += 1
    return entries


def _declared_sources(proj, path, lineno):
    try:
        with open(os.path.join(proj, path)) as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise RuntimeError(f'could not read cited document {path}: {exc}') from exc
    nearest = None
    every = []
    for index, line in enumerate(lines, start=1):
        match = CERT_SRC_RE.search(line)
        if match is None:
            continue
        every.append(match.group(1))
        if index < lineno:
            nearest = match.group(1)
    return [nearest] if nearest else every


def _catalog_of(proj, source):
    root_path = os.path.abspath(os.path.join(proj, 'certified'))
    root = os.path.realpath(root_path)
    if os.path.isabs(source):
        raise ValueError(f'source resolves outside certified/: {source}')
    rel = source.removeprefix('certified/')
    full = os.path.realpath(os.path.join(root, rel))
    try:
        contained = os.path.commonpath((root, full)) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f'source resolves outside certified/: {source}')
    if os.path.isdir(full):
        paths = ledger_files(full)
    elif os.path.isfile(full):
        paths = [full]
    else:
        return None
    return _numbers_from_files(paths)


def _preexisting_numbers(proj, path):
    tracked = _git_bytes(
        ['ls-tree', '-r', '--name-only', '-z', 'HEAD', '--', path],
        proj,
    )
    tracked_paths = {os.fsdecode(item) for item in tracked.split(b'\0') if item}
    if path not in tracked_paths:
        return set()
    blob = git(['show', f'HEAD:{path}'], proj)
    return set(NUM_RE.findall(blob))


def check_cited_metrics(proj):
    """새 표 수치가 선언된 certified 원장에 있는지 검사한다."""
    entries = []
    for path in _changed_paths(proj):
        if not path.startswith(('docs/reports/', 'docs/issues/')):
            continue
        diff = git(['diff', '--no-renames', 'HEAD', '--', path], proj)
        entries.extend(_added_table_numbers(diff, path))
    if not entries:
        return []

    global_catalog = None
    prior = {}
    findings = {}
    for path, lineno, numbers in entries:
        if path not in prior:
            prior[path] = _preexisting_numbers(proj, path)
        numbers = [number for number in numbers if number not in prior[path]]
        if not numbers:
            continue

        sources = _declared_sources(proj, path, lineno)
        if sources:
            catalog = set()
            unknown = []
            invalid = []
            for source in sources:
                try:
                    one = _catalog_of(proj, source)
                except ValueError as exc:
                    invalid.append(str(exc))
                    continue
                if one is None:
                    unknown.append(source)
                else:
                    catalog |= one
            if invalid:
                findings.setdefault(
                    f'{path}: invalid certified source - '
                    + ', '.join(sorted(set(invalid))),
                    set(),
                )
                continue
            if unknown:
                findings.setdefault(
                    f'{path}: declared source not in certified/ - '
                    + ', '.join(sorted(set(unknown))),
                    set(),
                )
                continue
            origin = ', '.join(sorted(set(sources)))
        else:
            if global_catalog is None:
                global_catalog = _metric_catalog(proj)
            if not global_catalog:
                findings.setdefault(
                    f'{path}: cites metrics but certified/ holds no committed '
                    'metrics JSON',
                    set(),
                )
                continue
            catalog = global_catalog
            origin = 'certified/** (no source declared)'

        missing = {
            number
            for number in numbers
            if number not in catalog and number.rstrip('0') not in catalog
        }
        if missing:
            findings.setdefault(
                f'{path}: cited in a table but absent from {origin}',
                set(),
            ).update(missing)

    return [
        f'{key}: {", ".join(sorted(numbers))}' if numbers else key
        for key, numbers in findings.items()
    ]


def is_ruler(path):
    return path in RULER_PATHS or (
        path.endswith('.py') and path.startswith(RULER_DIRS)
    )


def check_ruler_touched(proj):
    """gold, metric, split 기준 파일 변경을 반환한다."""
    return [path for path in sorted(_changed_paths(proj)) if is_ruler(path)]


def _diff_changes(diff):
    added = []
    removed = []
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith('@@'):
            in_hunk = True
        elif not in_hunk:
            continue
        elif line.startswith('+'):
            added.append(line[1:])
        elif line.startswith('-'):
            removed.append(line[1:])
    return added, removed


def check_test_integrity(proj):
    """테스트 삭제와 무조건 skip, assert 순삭제를 파일별로 검사한다."""
    def count(lines, pattern):
        return sum(1 for line in lines if re.match(pattern, line.strip()))

    findings = []
    paths = sorted(path for path in _changed_paths(proj) if path.startswith('tests/'))
    for path in paths:
        diff = git(['diff', '--no-renames', 'HEAD', '--', path], proj)
        added, removed = _diff_changes(diff)
        dropped = count(removed, r'def test_') - count(added, r'def test_')
        if dropped > 0:
            findings.append(f'{path}: {dropped} test function(s) net removed')
        skips = sum(1 for line in added if WEAKEN_RE.search(line))
        if skips:
            findings.append(f'{path}: {skips} unconditional skip/xfail added')
        weakened = count(removed, r'assert\b') - count(added, r'assert\b')
        if weakened > 0:
            findings.append(
                f'{path}: {weakened} assert statement(s) net removed'
            )
    return findings


def check_protected_paths(proj):
    """커밋해서는 안 되는 certified JSON 변경을 반환한다."""
    return [
        path
        for path in sorted(_changed_paths(proj))
        if path.startswith('certified/') and path.endswith('.json')
    ]


def check_ruff(proj):
    """변경된 Python 파일을 Ruff로 검사한다."""
    changed = [
        path
        for path in _changed_paths(proj)
        if path.endswith('.py') and os.path.exists(os.path.join(proj, path))
    ]
    if not changed:
        return []
    local = os.path.join(proj, '.venv/bin/ruff')
    executable = local if os.path.exists(local) else 'ruff'
    try:
        out = subprocess.run(
            [executable, 'check', *changed],
            cwd=proj,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        raise RuntimeError(
            f'ruff could not run: {type(exc).__name__}: {exc}'
        ) from exc
    if out.returncode == 0:
        return []
    diagnostic = (out.stdout + out.stderr).strip() or 'no diagnostic'
    if out.returncode != 1:
        raise RuntimeError(f'ruff exited {out.returncode}: {diagnostic}')
    return [diagnostic]


def _pytest_cmd(proj):
    venv = os.path.join(proj, '.venv/bin/python')
    return [venv if os.path.exists(venv) else sys.executable, '-m', 'pytest']


def check_self_tests(proj):
    """Codex 훅 또는 훅 테스트 변경 시 훅 회귀 테스트를 실행한다."""
    changed = _changed_paths(proj)
    if not any(path.startswith(SELF_DIRS) for path in changed):
        return []
    if not os.path.isdir(os.path.join(proj, SELF_TESTS)):
        raise RuntimeError(f'self-test directory missing: {SELF_TESTS}')
    try:
        out = subprocess.run(
            [*_pytest_cmd(proj), SELF_TESTS, '-q'],
            cwd=proj,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as exc:
        raise RuntimeError(
            f'self-test could not run: {type(exc).__name__}: {exc}'
        ) from exc
    if out.returncode == 0:
        return []
    diagnostic = (out.stdout or out.stderr).strip() or 'no diagnostic'
    if out.returncode != 1:
        raise RuntimeError(
            f'self-test pytest exited {out.returncode}: {diagnostic[-1500:]}'
        )
    return [diagnostic[-1500:]]


def evaluate(project_dir: str) -> GateResult:
    """고정 순서로 모든 검사를 실행하고 명시적 허용 또는 차단을 반환한다."""
    try:
        self_test = tuple(check_self_tests(project_dir))
        if self_test:
            return GateResult(
                False,
                'self-test',
                'Codex gate self-tests failed',
                self_test,
            )

        protected = tuple(check_protected_paths(project_dir))
        if protected:
            return GateResult(
                False,
                'protected-path',
                'Protected certified JSON files changed; Codex must not '
                'commit these files.',
                protected,
            )

        ruff = tuple(check_ruff(project_dir))
        if ruff:
            return GateResult(
                False,
                'ruff',
                'Ruff failed for changed Python files',
                ruff,
            )

        ruler = tuple(check_ruler_touched(project_dir))
        if ruler:
            return GateResult(
                False,
                'ruler-lock',
                'Protected criteria changed; obtain an independent reviewer '
                'report and let the user perform the commit outside this hook.',
                ruler,
            )

        hard = tuple(
            check_test_integrity(project_dir) + check_cited_metrics(project_dir)
        )
        if hard:
            return GateResult(
                False,
                'hard',
                'Deterministic integrity checks failed',
                hard,
            )

        return GateResult(True, 'pass', '', ())
    except Exception as exc:
        return GateResult(
            False,
            'internal-error',
            'Codex gate checks could not complete',
            (f'{type(exc).__name__}: {exc}',),
        )
