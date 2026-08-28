"""Tests for the gate (.claude/hooks/gate_core.py · refuter_gate.py).

게이트는 커밋을 막는 장치라, 깨져도 조용하다 — 아무것도 안 막히면 그냥
평소처럼 커밋이 될 뿐이라 사람이 알아채지 못한다. 그래서 "무엇을 막나"
보다 "막아야 할 것을 여전히 막나"를 고정해 두는 게 이 파일의 목적이다.

임시 git 저장소를 만들어 차단부터 해제까지 실제로 태운다 — 게이트가
`git` 을 직접 호출하므로 diff 를 흉내내면 검증이 되지 않는다.

기계 검사(`gate_core`)는 함수를 직접 부르고, Stop 훅(`refuter_gate`)은 별
프로세스로 태운다 — 그쪽은 stdin·상태 파일까지가 계약이라 in-process 로
부르면 계약의 절반을 흉내내게 된다.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUFF = REPO_ROOT / '.venv/bin/ruff'


def _load_hook(name):
    """`.claude/hooks/` 는 설치 패키지가 아니라 훅 스크립트 디렉토리다 —
    import 경로에 없으므로 파일 경로로 직접 적재한다."""
    path = REPO_ROOT / f'.claude/hooks/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_hook('gate_core')
stop_hook = _load_hook('refuter_gate')
commit_gate = _load_hook('commit_gate')
STOP_HOOK_PATH = REPO_ROOT / '.claude/hooks/refuter_gate.py'

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
    core._UNENFORCED.clear()
    yield
    core._DEGRADED.clear()
    core._UNENFORCED.clear()


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
    dhash = core.anchor_hash(proj, diff)
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
         'checked': ['gold 재계산 일치'], 'model': 'default', 'round': 1}
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
         'checked': [], 'model': 'default', 'round': 1}
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
         'model': 'default', 'round': 1}
    ))
    state, defects, meta = core.read_verdict(sdir, 'oldhash')
    assert state == 'FAIL'
    assert 'σ 표기 불일치' in defects
    assert meta['model'] == 'default'


def test_verdict_without_defects_key_passes(repo):
    """새 형식에서 `defects` 가 비면 PASS 다."""
    sdir = core.state_dir(repo)
    Path(core.verdict_path(sdir, 'h2')).write_text(json.dumps(
        {'verdict': 'PASS', 'diff_hash': 'h2', 'defects': [],
         'checked': ['확인 기록 세 줄'], 'model': 'default', 'round': 1}
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


def test_no_section_is_exempt_from_the_ruler_lock(repo):
    """절 단위 면제는 없다 — 기준 문서는 어느 줄을 고쳐도 잠근다.

    면제 구간을 변경 *후* 파일에서 구하던 옛 규칙에는, 면제될 절을 새로
    만들면서 그 안에 정의를 바꿔 넣으면 추가한 줄이 통째로 자기가 만든
    구간에 들어가 게이트가 울리지 않는 구멍이 있었다. 그 경로가 실제로
    잠기는지를 기록으로 남긴다."""
    _seed_ruler(repo)

    # (1) 기록·맥락 절만 고쳐도 잠긴다
    _write(repo, RULER_DOC, SCHEMA_BEFORE + '- 두 번째 항목\n')
    _, _, result = _gate(repo)
    assert result[0] == 'ruler-lock'

    # (2) 옛 면제 구멍 — '## 변경 이력' 절을 새로 만들고 그 안에서 정의를
    #     바꾸는 경로. 추가 줄이 모두 새 절 안이라 옛 규칙은 통과시켰다
    _write(repo, RULER_DOC,
           '# 스키마\n\n## 정의\n\nPER LOC ORG\n\n'
           '## 변경 이력\n\n- 최초\n- PROD 를 추가하고 ORG 를 뺀다\n')
    _, _, result = _gate(repo)
    assert result[0] == 'ruler-lock'


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


# ── 원장 잠금이 '증거로 읽는 형식' 과 어긋나지 않는가 ────────────────

def _deny_patterns():
    with open(REPO_ROOT / '.claude/settings.json') as f:
        return set(json.load(f)['permissions']['deny'])


def test_every_ledger_format_is_write_denied():
    """인용 대조가 증거로 읽는 형식은 AI 가 손으로 못 쓰게 막혀 있어야 한다.

    카탈로그를 넓히면서(예: csv 추가) deny 를 안 늘리면 AI 가 증거 파일을
    직접 지어낼 수 있게 된다. 그 어긋남을 여기서 잡는다.
    """
    deny = _deny_patterns()
    for ext in core.CATALOG_EXT:
        for tool in ('Write', 'Edit'):
            # `**/*` 가 최상위를 안 잡는 경우가 있어 두 층 모두 필요하다
            assert f'{tool}(certified/**/*.{ext})' in deny
            assert f'{tool}(certified/*.{ext})' in deny


def test_ledger_docs_stay_editable():
    """원장의 설명 문서는 증거가 아니다 — 잠그면 보호되는 것 없이 갱신만
    막힌다(카탈로그가 안 읽는다)."""
    deny = _deny_patterns()
    assert not any(p.endswith('(certified/**)') for p in deny)
    assert not any('.md)' in p for p in deny if 'certified' in p)
    assert 'md' not in core.CATALOG_EXT


def test_ledger_files_reads_only_catalog_formats(tmp_path):
    ledger = tmp_path / 'certified'
    (ledger / 'run').mkdir(parents=True)
    (ledger / 'README.md').write_text('설명 문서\n')
    (ledger / 'top.json').write_text('{}')
    (ledger / 'run' / 'pooled_metrics.json').write_text('{}')
    (ledger / 'run' / 'notes.txt').write_text('메모')
    got = {os.path.basename(p) for p in core.ledger_files(str(ledger))}
    assert got == {'top.json', 'pooled_metrics.json'}


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


# ── Stop 훅 — 연속 차단 카운터의 수명 ─────────────────────────────────

def _stop_hook(proj, session_id='s1', reentrant=False):
    """Stop 훅을 실제 프로세스로 한 번 태운다 → (stdout JSON, 카운터 값).

    카운터 값은 파일이 없으면 None 이다. `git add` 를 먼저 하는 것은 실제
    턴 종료 상태를 재현하기 위함이다 — untracked 파일은 `git diff HEAD` 에
    안 잡혀 훅이 '변경 없음' 으로 읽는다.
    """
    _sh(proj, 'git', 'add', '-A')
    env = {k: v for k, v in os.environ.items()
           if k not in ('DISABLE_OMC', 'OMC_SKIP_HOOKS')}
    env['CLAUDE_PROJECT_DIR'] = proj
    payload = {'session_id': session_id, 'cwd': proj}
    if reentrant:
        payload['stop_hook_active'] = True
    out = subprocess.run(
        [sys.executable, str(STOP_HOOK_PATH)],
        input=json.dumps(payload),
        cwd=proj, capture_output=True, text=True, env=env, check=True,
    ).stdout
    counter = Path(core.state_dir(proj)) / f'.blocks-{session_id}'
    return (json.loads(out) if out.strip() else {},
            counter.read_text().strip() if counter.exists() else None)


def test_stop_gate_counts_consecutive_blocks(repo):
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))

    out, n = _stop_hook(repo)
    assert out.get('decision') == 'block'
    assert n == '1'

    out, n = _stop_hook(repo)
    assert out.get('decision') == 'block'
    assert n == '2'


def test_a_pass_clears_the_counter(repo):
    """상한이 '누적' 이 아니라 '연속' 실패만 세는지 — 정상 통과 경로."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    _stop_hook(repo)

    _sh(repo, 'git', 'add', '-A')
    _sh(repo, 'git', 'commit', '-qm', 'schema change')  # 미커밋 diff 가 없어진다
    out, n = _stop_hook(repo)
    assert 'decision' not in out
    assert n is None


def test_the_limit_opens_the_gate_for_one_turn_only(repo):
    """상한의 뜻은 '이 턴을 연다' 이지 '이 세션을 포기한다' 가 아니다.

    카운터를 안 지우면 상한에 얹힌 채 남아, 그 뒤로는 무엇을 바꾸든 이
    분기로만 빠져 그 세션의 Stop 게이트가 영구히 죽는다 — 실제로 한 세션이
    15 시간 동안 아무것도 검사하지 않았다. 게이트가 열린 채 빠지는 실패는
    화면이 평소와 같아 사람이 못 알아채므로 여기에 고정해 둔다.
    """
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    counter = Path(core.state_dir(repo)) / '.blocks-s1'
    counter.write_text(str(stop_hook.MAX_BLOCKS))

    out, n = _stop_hook(repo)
    assert 'consecutive block limit' in out.get('systemMessage', '')
    assert 'decision' not in out          # 이 턴은 열린다
    assert n is None, '상한 통과도 카운터를 지운다'

    # 같은 변경이 그대로 남아 있는데도 다음 사이클은 다시 검사받는다
    out, n = _stop_hook(repo)
    assert out.get('decision') == 'block'
    assert n == '1'


def test_a_reentrant_stop_neither_blocks_nor_clears(repo):
    """재진입 Stop 은 재-block 하지 않는다(연속 block 안전 오버라이드 방지).
    통과시키되 카운터는 건드리지 않아 다음 평가가 이어지게 둔다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    _stop_hook(repo)  # 카운터 1

    out, n = _stop_hook(repo, reentrant=True)
    assert 'decision' not in out
    assert n == '1'


# ── 앵커 지문 — 무엇이 판정을 무효로 만드는가 ─────────────────────────

ISSUE_DOC = 'docs/issues/issue-9-x.md'


def _anchor(proj):
    return core.anchor_hash(proj, core.diff_text(proj))


def test_issue_doc_edits_do_not_move_the_anchor(repo):
    """이슈 문서는 채점에 안 쓰이는 기록이라 앵커에서 뺀다."""
    _write(repo, 'src/a.py', 'X = 1\n')
    _write(repo, ISSUE_DOC, '# 계획\n')
    _sh(repo, 'git', 'add', '-A')
    first = _anchor(repo)

    _write(repo, ISSUE_DOC, '# 계획\n\n- 결정 로그 한 줄 더\n')
    _sh(repo, 'git', 'add', '-A')
    assert _anchor(repo) == first


def test_code_changes_still_move_the_anchor(repo):
    _write(repo, 'src/a.py', 'X = 1\n')
    _sh(repo, 'git', 'add', '-A')
    first = _anchor(repo)

    _write(repo, 'src/a.py', 'X = 2\n')
    _sh(repo, 'git', 'add', '-A')
    assert _anchor(repo) != first


def test_issue_doc_only_diff_falls_back_to_the_full_hash(repo):
    """제외하고 나면 빈 diff 가 되는 변경들이 한 해시를 공유하면 서로의
    판정을 물려받는다. 그때는 전체 diff 로 되돌아간다."""
    _write(repo, ISSUE_DOC, '# 이슈 아홉\n')
    _sh(repo, 'git', 'add', '-A')
    a = _anchor(repo)

    _write(repo, ISSUE_DOC, '# 전혀 다른 내용\n')
    _sh(repo, 'git', 'add', '-A')
    b = _anchor(repo)

    assert a != b
    assert b == core.diff_hash(core.diff_text(repo))


def test_a_pass_survives_an_issue_doc_append(repo):
    """이번 사건의 핵심 경로 — PASS 를 받은 뒤 이슈 문서에 기록을 더해도
    그 PASS 가 살아 있어야 한다. 죽으면 기록하는 행위가 재판정을 부르고,
    그 재판정이 다시 기록이 된다."""
    _seed_ruler(repo)
    _write(repo, RULER_DOC, SCHEMA_BEFORE.replace('ORG', 'ORG PROD'))
    _write(repo, ISSUE_DOC, '# 계획\n')
    dhash, sdir, _ = _gate(repo)

    _allow(sdir, dhash)
    Path(core.verdict_path(sdir, dhash)).write_text(json.dumps(
        {'verdict': 'PASS', 'diff_hash': dhash, 'defects': [],
         'checked': ['gold 재계산 일치'], 'model': 'default', 'round': 1}
    ))
    assert _gate(repo)[2] is None

    _write(repo, ISSUE_DOC, '# 계획\n\n- 반박자가 이걸 지적해 이렇게 바꿨다\n')
    again, _, result = _gate(repo)
    assert again == dhash, '이슈 문서 append 가 앵커를 옮기면 안 된다'
    assert result is None, '승인도 판정도 살아 있어야 한다'


# ── 게이트 자신을 고치는 diff — 자기 회귀 테스트 ──────────────────────

def _seed_self_tests(proj, body='def test_ok():\n    assert 1\n'):
    _write(proj, 'tests/hooks/test_x.py', body)
    _sh(proj, 'git', 'add', '-A')
    _sh(proj, 'git', 'commit', '-qm', 'seed self tests')


def test_a_gate_change_runs_its_own_tests(repo):
    """게이트는 목록으로 움직이는데 그 목록이 든 파일은 기준 파일이 아니다.
    목록을 좁히고 대응 assert 를 같이 고치면 다른 네 검사가 다 조용하다 —
    assert 는 개수만 세므로 내용 변경이 안 잡힌다. 그 경로를 여기서 막는다."""
    _seed_self_tests(repo)
    _write(repo, '.claude/hooks/gate_core.py', 'RULER = ()\n')
    _write(repo, 'tests/hooks/test_x.py', 'def test_ok():\n    assert 0\n')
    _, _, result = _gate(repo)
    assert result[0] == 'self-test'
    assert 'tests/hooks' in result[1]


def test_green_self_tests_do_not_block(repo):
    _seed_self_tests(repo)
    _write(repo, '.claude/hooks/gate_core.py', 'RULER = ()\n')
    _, _, result = _gate(repo)
    assert result is None


def test_a_failing_self_test_is_not_waivable(repo):
    """ruff 와 같은 등급이다 — 기계로 고칠 결함이라 승인 대상이 아니다."""
    _seed_self_tests(repo)
    _write(repo, '.claude/hooks/gate_core.py', 'RULER = ()\n')
    _write(repo, 'tests/hooks/test_x.py', 'def test_ok():\n    assert 0\n')
    dhash, sdir, _ = _gate(repo)
    _allow(sdir, dhash)
    _, _, result = _gate(repo)
    assert result[0] == 'self-test'


def test_the_self_test_is_skipped_when_the_gate_is_untouched(repo):
    """게이트를 안 건드린 diff 에서는 안 돈다 — 평소 무게가 0이어야 한다."""
    _seed_self_tests(repo, 'def test_bad():\n    assert 0\n')
    _write(repo, 'src/a.py', 'X = 1\n')
    _, _, result = _gate(repo)
    assert result is None


def test_missing_self_tests_degrade_instead_of_blocking(repo):
    """게이트 고장이 커밋을 영구 봉쇄해선 안 된다."""
    _write(repo, '.claude/hooks/gate_core.py', 'RULER = ()\n')
    _, _, result = _gate(repo)
    assert result is None
    assert any('self-test' in r for r in core.degraded())


# ── 커밋을 만드는 명령의 범위 ─────────────────────────────────────────

def test_every_commit_creating_command_is_gated():
    for cmd in ('git commit -m x', 'git cherry-pick abc123', 'git revert HEAD',
                'git am patch.mbox', 'git rebase --continue',
                'git merge --continue', 'cd /tmp && git cherry-pick abc'):
        assert commit_gate.creates_a_commit(cmd), cmd


def test_aborting_is_not_gated():
    """충돌 상태에서 빠져나올 길을 막으면 게이트가 사람을 가둔다."""
    for cmd in ('git rebase --abort', 'git merge --abort',
                'git cherry-pick --skip', 'git am --quit'):
        assert not commit_gate.creates_a_commit(cmd), cmd


def test_read_only_commands_are_not_gated():
    for cmd in ('git status', 'git log --oneline', 'git diff HEAD'):
        assert not commit_gate.creates_a_commit(cmd), cmd


# ── 확인마다 집행 주체가 있는가 ───────────────────────────────────────
#
# 반박자가 통과시킨 확인은 대부분 손으로 다시 센 것이라, 다음 라운드가 같은
# 계산을 처음부터 한다. 판정 파일이 확인마다 "다음번에 무엇이 이걸 잡나"를
# 담게 하고, 답이 없는 것을 사람에게 목록으로 넘겨 테스트로 옮길 수 있게
# 한다. 없다는 사실 자체는 판정을 바꾸지 않는다.

def _verdict(sdir, dhash, checked, verdict='PASS'):
    Path(core.verdict_path(sdir, dhash)).write_text(json.dumps(
        {'verdict': verdict, 'diff_hash': dhash, 'defects': [],
         'checked': checked, 'model': 'default', 'round': 1},
        ensure_ascii=False,
    ))


def test_a_checked_item_names_what_enforces_it(repo):
    """집행 주체가 적힌 확인은 사람에게 넘길 목록에 안 오른다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', [
        {'what': '테스트 무결성', 'enforced_by': 'tests/hooks/test_gate.py::t'},
    ])
    state, _, _ = core.read_verdict(sdir, 'h')
    assert state == 'PASS'
    assert core.unenforced() == []
    assert core.unenforced_notice(sdir, 'h') is None


def test_an_unowned_check_is_reported_but_does_not_block(repo):
    """`enforced_by` 가 null 이면 목록에 오르되 PASS 는 PASS 로 남는다 —
    집행 주체가 없다는 것은 결함이 아니라 아직 옮기지 않은 일이다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', [
        {'what': '표 아홉 칸 산술을 손으로 다시 셌다', 'enforced_by': None},
        {'what': '골든 상수 대조', 'enforced_by': 'pytest tests/test_g.py'},
    ])
    state, defects, _ = core.read_verdict(sdir, 'h')
    assert (state, defects) == ('PASS', [])
    assert core.unenforced() == ['표 아홉 칸 산술을 손으로 다시 셌다']
    notice = core.unenforced_notice(sdir, 'h')
    assert '표 아홉 칸 산술을 손으로 다시 셌다' in notice
    assert '골든 상수 대조' not in notice


def test_words_that_mean_none_count_as_none(repo):
    """모델이 null 대신 말로 없음을 적어도 집행 주체 없음으로 읽는다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', [
        {'what': 'a', 'enforced_by': '없음'},
        {'what': 'b', 'enforced_by': '  '},
        {'what': 'c', 'enforced_by': 'N/A'},
        {'what': 'd'},
    ])
    core.read_verdict(sdir, 'h')
    assert core.unenforced() == ['a', 'b', 'c', 'd']


def test_old_string_checked_lists_are_still_read(repo):
    """옛 판정은 `checked` 가 문자열 배열이라 집행 주체를 적을 자리가
    없었다 — 읽히되 전부 집행 주체 없음으로 센다. 실제로도 없었다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', ['확인 기록 한 줄', '또 한 줄'])
    state, defects, meta = core.read_verdict(sdir, 'h')
    assert (state, defects, meta['model']) == ('PASS', [], 'default')
    assert core.unenforced() == ['확인 기록 한 줄', '또 한 줄']


def test_a_verdict_without_checked_reports_nothing(repo):
    """`findings` 만 있던 더 옛 형식은 확인 기록 자체가 없다."""
    sdir = core.state_dir(repo)
    Path(core.verdict_path(sdir, 'h')).write_text(json.dumps(
        {'verdict': 'FAIL', 'diff_hash': 'h', 'findings': ['σ 불일치'],
         'model': 'default', 'round': 1}, ensure_ascii=False,
    ))
    state, defects, _ = core.read_verdict(sdir, 'h')
    assert state == 'FAIL' and 'σ 불일치' in defects
    assert core.unenforced() == []


def test_the_same_verdict_read_twice_lists_each_check_once(repo):
    """기준 파일 diff 는 한 훅 안에서 판정을 두 번 읽는다(기계 검사 ·
    Stop 7단계). 목록이 두 배로 부풀면 안 된다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', [{'what': '손 계산', 'enforced_by': None}])
    core.read_verdict(sdir, 'h')
    core.read_verdict(sdir, 'h')
    assert core.unenforced() == ['손 계산']


def test_a_long_check_is_cut_on_screen_but_whole_in_the_log(repo):
    """확인 문구는 한 항목이 문단 길이까지 자란다. 화면은 자르고 이력에는
    전문을 남긴다 — 자른 것만 남으면 나중에 무엇이었는지 복원이 안 된다."""
    sdir = core.state_dir(repo)
    long_what = '가' * 400
    _verdict(sdir, 'h', [{'what': long_what, 'enforced_by': None}])
    core.read_verdict(sdir, 'h')
    notice = core.unenforced_notice(sdir, 'h')
    assert long_what not in notice and '…' in notice
    logged = [json.loads(line) for line in
              open(os.path.join(sdir, 'log.jsonl'))]
    assert any(r.get('result') == 'UNENFORCED' and long_what in r['checked']
               for r in logged)


def test_pass_notice_carries_both_kinds(repo):
    """검사가 못 돈 사유와 집행 주체 없는 확인은 둘 다 '막지는 않지만
    사람이 봐야 하는 것' 이라, 한 통과에서 함께 나와야 한다."""
    sdir = core.state_dir(repo)
    _verdict(sdir, 'h', [{'what': '손 계산', 'enforced_by': None}])
    core.read_verdict(sdir, 'h')
    core._DEGRADED.append('git diff: exit 128')
    notice = core.pass_notice(sdir, 'h')
    assert 'could not run' in notice and '손 계산' in notice


def test_the_stop_gate_shows_unowned_checks_when_it_passes(repo):
    """계약의 끝까지 — 훅을 실제 프로세스로 태워 화면 문구까지 확인한다."""
    _write(repo, 'src/x.py', 'y = 1\n')
    dhash, sdir, _ = _gate(repo)
    Path(os.path.join(repo, '.omc/state/refuter-gate.on')).touch()
    _verdict(sdir, dhash, [{'what': '손으로 다시 센 표', 'enforced_by': None}])
    out, _ = _stop_hook(repo, 'sess-unowned')
    assert 'decision' not in out
    assert '손으로 다시 센 표' in out.get('systemMessage', '')
