"""Tests for the commit gate deterministic layer (.claude/hooks/gate_core.py).

게이트는 커밋을 막는 장치라, 깨져도 조용하다 — 아무것도 안 막히면 그냥
평소처럼 커밋이 될 뿐이라 사람이 알아채지 못한다. 그래서 "무엇을 막나"
보다 "막아야 할 것을 여전히 막나"를 고정해 두는 게 이 파일의 목적이다.

임시 git 저장소를 만들어 차단부터 해제까지 실제로 태운다 — 게이트가
`git` 을 직접 호출하므로 diff 를 흉내내면 검증이 되지 않는다.
"""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUFF = REPO_ROOT / '.venv/bin/ruff'


def _load_gate_core():
    """`.claude/hooks/` 는 설치 패키지가 아니라 훅 스크립트 디렉토리다 —
    import 경로에 없으므로 파일 경로로 직접 적재한다."""
    path = REPO_ROOT / '.claude/hooks/gate_core.py'
    spec = importlib.util.spec_from_file_location('gate_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_gate_core()

RULER_DOC = 'docs/manual/data/canonical-entity-schema.md'
SCHEMA_BEFORE = '# 스키마\n\n## 정의\n\nPER LOC ORG\n\n## 변경 이력\n\n- 최초\n'

# 게이트는 diff 의 글자를 그대로 읽을 뿐 코드와 테스트 데이터를 구별하지
# 못한다 — 진짜 skip 마커를 이 파일에 문자열로 두면 게이트가 자기 검증
# 테스트를 오탐해 커밋을 막는다. 그래서 마커는 실행 시점에 조립한다.
MARK = '@pytest.mark.{}'


def _sh(cwd, *args):
    subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def _write(proj, rel, text):
    path = Path(proj) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture(autouse=True)
def _clean_degraded():
    """훅은 매번 새 프로세스라 모듈 전역이 비어 있다 — 테스트에서도 같게."""
    core._DEGRADED.clear()
    yield
    core._DEGRADED.clear()


@pytest.fixture
def repo(tmp_path):
    """초기 커밋이 하나 있는 빈 git 저장소.

    `.omc/` 를 ignore 하는 것이 핵심이다 — 승인 파일과 반박자 판정이 거기
    쌓이는데, 추적되면 그것들이 diff 를 바꿔 자기 해시를 무효화한다.
    실제 저장소도 같은 이유로 `.omc/` 를 ignore 한다.
    """
    proj = tmp_path / 'repo'
    proj.mkdir()
    _sh(proj, 'git', 'init', '-q')
    _sh(proj, 'git', 'config', 'user.email', 'test@example.com')
    _sh(proj, 'git', 'config', 'user.name', 'test')
    _write(proj, '.gitignore', '.omc/\n.venv/\n')
    _write(proj, 'README.md', 'seed\n')
    _sh(proj, 'git', 'add', '-A')
    _sh(proj, 'git', 'commit', '-qm', 'init')
    return str(proj)


def _gate(proj):
    """게이트를 한 번 돌린다 → (diff_hash, state_dir, 결과).

    결과는 통과면 None, 막히면 (check, reason, findings) 다. `git add` 를
    먼저 하는 것은 실제 커밋 직전 상태를 재현하기 위함이다 — untracked
    파일은 `git diff HEAD` 에 안 잡힌다.
    """
    _sh(proj, 'git', 'add', '-A')
    diff = core.diff_text(proj)
    dhash = core.diff_hash(diff)
    sdir = core.state_dir(proj)
    return dhash, sdir, core.run_deterministic(proj, sdir, dhash)


def _allow(sdir, dhash):
    """사람이 예외를 승인했다고 가정하고 승인 파일을 만든다."""
    Path(core.allow_file(sdir, dhash)).touch()


def _seed_ruler(proj):
    _write(proj, RULER_DOC, SCHEMA_BEFORE)
    _sh(proj, 'git', 'add', '-A')
    _sh(proj, 'git', 'commit', '-qm', 'schema')


# ── 기준 파일(정답·채점규칙·분할) 잠금 ────────────────────────────────

def test_ruler_change_is_blocked(repo):
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('PER LOC ORG', 'PER LOC'))
    _, _, result = _gate(repo)
    assert result is not None
    assert result[0] == 'ruler-lock'
    assert RULER_DOC in result[2]


def test_block_message_gives_a_runnable_command(repo):
    """막힌 사람이 다음에 무엇을 칠지 메시지만 보고 알 수 있어야 한다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    dhash, sdir, result = _gate(repo)
    message = result[1]
    assert f'touch {core.allow_file(sdir, dhash)}' in message
    assert 'ack-' not in message  # 옛 이름이 남아 있으면 복붙이 안 통한다


def test_allow_file_is_named_after_the_diff(repo):
    _write(repo, 'README.md', 'changed\n')
    dhash, sdir, _ = _gate(repo)
    assert os.path.basename(core.allow_file(sdir, dhash)) == f'human-allow-{dhash}'
    assert not core.human_allowed(sdir, dhash)
    _allow(sdir, dhash)
    assert core.human_allowed(sdir, dhash)


def test_ruler_needs_both_human_allow_and_refuter_pass(repo):
    """기준 파일만은 사람 승인 하나로 안 풀린다 — 반박자 판정도 있어야 한다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    dhash, sdir, _ = _gate(repo)

    _allow(sdir, dhash)
    _, _, result = _gate(repo)
    assert result[0] == 'ruler-refuter'  # 승인만으론 부족

    Path(core.verdict_path(sdir, dhash)).write_text(json.dumps(
        {'verdict': 'PASS', 'diff_hash': dhash, 'defects': [],
         'checked': ['gold 재계산 일치'], 'model': 'sonnet', 'round': 1}
    ))
    _, _, result = _gate(repo)
    assert result is None


def test_refuter_fail_keeps_the_gate_shut(repo):
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    dhash, sdir, _ = _gate(repo)
    _allow(sdir, dhash)
    Path(core.verdict_path(sdir, dhash)).write_text(json.dumps(
        {'verdict': 'FAIL', 'diff_hash': dhash, 'defects': ['gold moved'],
         'checked': [], 'model': 'opus', 'round': 1}
    ))
    _, _, result = _gate(repo)
    assert result[0] == 'ruler-refuter'
    assert 'gold moved' in result[1]


def test_old_verdict_files_are_still_readable(repo):
    """옛 판정은 결함과 확인 기록을 `findings` 한 배열에 섞어 썼다.
    섞인 배열에서 결함만 골라낼 수는 없으니 통째로 사유로 읽는다."""
    sdir = core.state_dir(repo)
    Path(core.verdict_path(sdir, 'oldhash')).write_text(json.dumps(
        {'verdict': 'FAIL', 'diff_hash': 'oldhash',
         'findings': ['PASS: 테스트 무결성 확인', 'σ 표기 불일치'],
         'model': 'sonnet', 'round': 1}
    ))
    state, defects, meta = core.read_verdict(sdir, 'oldhash')
    assert state == 'FAIL'
    assert 'σ 표기 불일치' in defects
    assert meta['model'] == 'sonnet'


def test_verdict_without_defects_key_passes(repo):
    """새 형식에서 `defects` 가 비면 PASS 다."""
    sdir = core.state_dir(repo)
    Path(core.verdict_path(sdir, 'h2')).write_text(json.dumps(
        {'verdict': 'PASS', 'diff_hash': 'h2', 'defects': [],
         'checked': ['확인 기록 세 줄'], 'model': 'opus', 'round': 1}
    ))
    state, defects, _ = core.read_verdict(sdir, 'h2')
    assert state == 'PASS' and defects == []


def test_allow_dies_when_the_diff_changes(repo):
    """승인은 '그 변경분'에만 유효하다 — 한 번 풀고 계속 우회할 수 없다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    first, sdir, _ = _gate(repo)
    _allow(sdir, first)

    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD EVT'))
    second, _, result = _gate(repo)
    assert second != first
    assert result[0] == 'ruler-lock'  # 이전 승인이 따라오지 않는다


def test_exempt_sections_do_not_lock(repo):
    """정의가 안 움직이는 절(변경 이력)만 고친 커밋까지 잠그면 확인이
    허울이 된다 — 그래서 면제된다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE + '- 두 번째 항목\n')
    _, _, result = _gate(repo)
    assert result is None


# ── 테스트 무결성 ─────────────────────────────────────────────────────

def test_deleting_tests_is_blocked_then_allowed(repo):
    _write(repo, 'tests/test_a.py',
           'def test_x():\n    assert 1\n\n\ndef test_y():\n    assert 2\n')
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'tests')

    _write(repo, 'tests/test_a.py', 'def test_x():\n    assert 1\n')
    dhash, sdir, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('test function' in f for f in result[2])
    assert f'touch {core.allow_file(sdir, dhash)}' in result[1]

    _allow(sdir, dhash)
    _, _, result = _gate(repo)
    assert result is None  # 반박자 없이 승인 하나로 풀린다


def test_unconditional_skip_is_blocked_but_skipif_is_not(repo):
    _write(repo, 'tests/test_a.py', 'def test_x():\n    assert 1\n')
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'tests')

    _write(repo, 'tests/test_a.py',
           f'import pytest\n\n\n{MARK.format("skip")}\n'
           'def test_x():\n    assert 1\n')
    _, _, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('skip/xfail' in f for f in result[2])

    _write(repo, 'tests/test_a.py',
           f'import pytest\n\n\n{MARK.format("skipif(False, reason=\'env\')")}\n'
           'def test_x():\n    assert 1\n')
    _, _, result = _gate(repo)
    assert result is None  # 환경 조건부는 정당하다


def test_deletions_do_not_cancel_across_files(repo):
    """한 파일에서 지운 만큼 다른 파일에 채워 넣어 신호를 지울 수 없다."""
    _write(repo, 'tests/test_a.py',
           'def test_1():\n    assert 1\n\n\ndef test_2():\n    assert 1\n')
    _write(repo, 'tests/test_b.py', 'def test_z():\n    assert 1\n')
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'tests')

    _write(repo, 'tests/test_a.py', '')                       # 2개 삭제
    _write(repo, 'tests/test_b.py',                           # 2개 추가
           'def test_z():\n    assert 1\n\n\ndef test_p():\n    assert 1\n'
           '\n\ndef test_q():\n    assert 1\n')
    _, _, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('tests/test_a.py' in f and 'net removed' in f
               for f in result[2])


# ── 기준 파일 목록이 조용히 낡지 않는가 ──────────────────────────────

def test_new_file_in_a_ruler_package_is_locked(repo):
    """`validity`·`metrics` 는 패키지 전체가 자다 — 목록에 없던 새 파일도
    잠겨야 목록이 조용히 낡지 않는다."""
    _write(repo, 'src/ner/validity/brand_new.py', 'X = 1\n')
    _, _, result = _gate(repo)
    assert result[0] == 'ruler-lock'
    assert 'src/ner/validity/brand_new.py' in result[2]


def test_non_scoring_code_stays_unlocked(repo):
    """학습·분석 코드까지 잠그면 확인이 허울이 된다."""
    for rel in ('src/ner/classifier/train_eval.py',
                'src/ner/labelers/ko/thing.py'):
        _write(repo, rel, 'X = 1\n')
    _, _, result = _gate(repo)
    assert result is None


# ── 게이트가 스스로 못 돌면 조용히 넘어가지 않는가 ───────────────────

def test_gate_reports_when_it_could_not_run(tmp_path):
    """게이트는 고장 나면 열린 채 빠진다(커밋 영구 봉쇄 방지). 다만 그때의
    '통과' 는 검사 통과가 아니라 검사 부재이므로 화면에 알려야 한다."""
    notrepo = tmp_path / 'notrepo'
    notrepo.mkdir()
    _write(notrepo, 'tests/test_a.py', 'def test_x():\n    assert 1\n')
    sdir = core.state_dir(str(notrepo))
    assert core.run_deterministic(str(notrepo), sdir, 'deadbeef') is None
    assert core.degraded(), 'git 실패가 기록돼야 한다'
    notice = core.degraded_notice(sdir, 'deadbeef')
    assert notice and 'passed by default' in notice


def test_healthy_run_reports_nothing(repo):
    _write(repo, 'README.md', 'seed\nmore\n')
    _, sdir, result = _gate(repo)
    assert result is None
    assert core.degraded() == []
    assert core.degraded_notice(sdir, 'x') is None


# ── 인용 수치 ↔ certified/ 원장 대조 ──────────────────────────────────

def test_cited_table_number_must_exist_in_certified(repo):
    _write(repo, 'certified/run/pooled_metrics.json',
           json.dumps({'overall': {'f1': 0.9312, 'precision': 0.88}}))
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'ledger')

    _write(repo, 'docs/reports/r.md',
           '<!-- certified: run/pooled_metrics.json -->\n\n'
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.7700 |\n')
    _, _, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('0.7700' in f for f in result[2])

    _write(repo, 'docs/reports/r.md',
           '<!-- certified: run/pooled_metrics.json -->\n\n'
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.9312 |\n')
    _, _, result = _gate(repo)
    assert result is None


def test_reformatting_an_already_published_row_passes(repo):
    """옛 리포트의 표 행을 손보는 것은 새 주장이 아니다.

    막아 버리면 인용된 수치를 전부 원장에 채워야 풀리는데, 원장이 커질수록
    '존재 검사'인 대조가 아무 값이나 통과시켜 검사 자체가 무력해진다.
    그래서 원장이 비어 있어도 이 편집은 지나가야 한다.
    """
    _write(repo, 'docs/reports/r.md',
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.7700 |\n')
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'report')

    # 수치를 담은 행 자체를 고친다 — diff 에는 새 행으로 실린다
    _write(repo, 'docs/reports/r.md',
           '| 타입 | F1 | 비고 |\n|---|---|---|\n| ALL | 0.7700 | strict |\n')
    _, _, result = _gate(repo)
    assert result is None


def test_changing_an_already_published_number_is_blocked(repo):
    """값을 바꾸는 것은 새 주장이므로 원장과 대조한다."""
    _write(repo, 'certified/run/pooled_metrics.json', json.dumps({'f1': 0.9312}))
    _write(repo, 'docs/reports/r.md',
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.7700 |\n')
    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'report')

    _write(repo, 'docs/reports/r.md',
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.8899 |\n')
    _, _, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('0.8899' in f for f in result[2])


def test_declaring_an_unpromoted_source_is_blocked(repo):
    _write(repo, 'docs/reports/r.md',
           '<!-- certified: nope/pooled_metrics.json -->\n\n'
           '| 타입 | F1 |\n|---|---|\n| ALL | 0.9312 |\n')
    _, _, result = _gate(repo)
    assert result[0] == 'hard'
    assert any('not in certified/' in f for f in result[2])


# ── ruff — 유일하게 승인으로 못 푸는 검사 ─────────────────────────────

@pytest.mark.skipif(not RUFF.exists(), reason='ruff not installed in .venv')
def test_ruff_failure_cannot_be_waived(repo):
    """기계로 고칠 수 있는 결함이라 예외 승인 대상이 아니다."""
    venv_bin = Path(repo) / '.venv/bin'
    venv_bin.mkdir(parents=True)
    shutil.copy(RUFF, venv_bin / 'ruff')

    _write(repo, 'bad.py', 'import os\n')  # F401 미사용 import
    dhash, sdir, result = _gate(repo)
    assert result[0] == 'ruff'

    _allow(sdir, dhash)
    _, _, result = _gate(repo)
    assert result[0] == 'ruff'  # 승인이 있어도 여전히 막힌다


# ── 통과해야 할 것은 통과한다 ─────────────────────────────────────────

def test_harmless_change_passes(repo):
    _write(repo, 'README.md', 'seed\nmore prose\n')
    _, _, result = _gate(repo)
    assert result is None


def test_nothing_changed_passes(repo):
    _, _, result = _gate(repo)
    assert result is None
