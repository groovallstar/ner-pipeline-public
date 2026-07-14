#!/usr/bin/env python3
"""Stop 훅: 미커밋 diff 에 대한 두 층 완료 게이트.

- 결정적 층: 게이트가 직접 실행한다. 기준 파일(정답·채점규칙·분할)
  변경 감지, 변경된 `.py` 의 ruff, 테스트 무결성(삭제·무조건 skip·
  assert 약화), 인용 표 수치 ↔ `certified/` 대조. 모델이 개입할 수
  없고 토큰을 쓰지 않으므로 **루프 여부와 무관하게 항상** 돈다.
- 판단 층: 결정적 층 통과 후, 현재 diff 에 대한 반박자 판정 파일을
  요구한다. 반박자 서브에이전트를 부르는 비용이 있어 루프(ralph/
  ultrawork) 활성 시에만 건다. 이 파일은 반박자(모델)가 쓰므로 위조
  가능하며, 결정적 층을 대체하지 않는다.

결정적 층의 오탐은 사람이 `ack-<diff_hash>` 파일로 해제한다. 모든
판정·해제는 `log.jsonl` 에 append-only 로 남는다. 무한 루프는 세션
누적 block 카운터로 차단한다.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# 같은 게이트 세션에서 누적 block 상한 (무한루프 차단)
MAX_BLOCKS = 6

# 인용 수치로 볼 토큰 (0.93 · 0.9361 · 1.00 등)
NUM_RE = re.compile(r'\b[01]\.\d{2,4}\b')

# 테스트 약화 신호. `skipif` 는 환경 조건부라 정당 — `\b` 로 배제된다
WEAKEN_RE = re.compile(
    r'@pytest\.mark\.skip\b|@pytest\.mark\.xfail'
    r'|pytest\.skip\(|pytest\.xfail\('
)

# 카탈로그에서 제외할 산출물 (예측 덤프·체크포인트 내부 상태)
CATALOG_EXCLUDE = ('test_predictions', 'checkpoint-')
CATALOG_MAX_BYTES = 20_000_000

# 기준 파일 — 정답·채점규칙·분할의 정의. 건드리면 사람 확인(ack) 전까지
# 진행을 막는다 (CLAUDE.md 작업 흐름 ① 표와 동기화). 좁게 잡아 리팩터·
# 주석 파일까지 매번 잠그지 않는다.
RULER_PATHS = (
    'docs/manual/data/canonical-entity-schema.md',
    'src/ner/metrics/bio_metrics.py',
    'src/ner/metrics/span_metrics.py',
    'src/ner/validity/gate.py',
    'src/ner/validity/comparability.py',
    'src/ner/validity/leakage.py',
    'src/ner/validity/variance.py',
    'src/ner/classifier/data_utils.py',
    'src/ner/classifier/kfold_pool.py',
)


def _git(args, cwd):
    # git 호출 (실패해도 빈 문자열 — 게이트는 안전하게 빠진다)
    try:
        out = subprocess.run(
            ['git', *args], cwd=cwd, capture_output=True, text=True,
            timeout=20,
        )
        return out.stdout
    except Exception:
        return ''


def _read_input():
    # Stop 훅 stdin JSON 파싱 (실패해도 게이트는 조용히 빠진다)
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _proj_dir(data):
    # 프로젝트 루트: 훅 env > stdin cwd > git toplevel
    return (
        os.environ.get('CLAUDE_PROJECT_DIR')
        or data.get('cwd')
        or _git(['rev-parse', '--show-toplevel'], '.').strip()
        or '.'
    )


def _passthrough():
    # exit 0 = 완료 허용 (출력 없음)
    sys.exit(0)


def _block(reason):
    # Stop 차단 + 메인 모델에 reason 전달
    print(json.dumps({'decision': 'block', 'reason': reason}))
    sys.exit(0)


def _bump(counter, n):
    # 누적 block 카운터 증가
    try:
        with open(counter, 'w') as f:
            f.write(str(n + 1))
    except Exception:
        pass


def _loop_active(proj, session_id):
    # 수동 토글이 있으면 강제 on
    if os.path.exists(os.path.join(proj, '.omc/state/refuter-gate.on')):
        return True
    # ralph PRD: 미완료 story 가 있으면 루프 활성
    if not session_id:
        return False
    prd = os.path.join(
        proj, '.omc/state/sessions', session_id, 'prd.json'
    )
    if not os.path.exists(prd):
        return False
    try:
        with open(prd) as f:
            data = json.load(f)
    except Exception:
        return False
    stories = data.get('stories') or data.get('user_stories') or []
    return any(not s.get('passes', False) for s in stories)


def _collect_numbers(node, cat):
    # 중첩 JSON 을 훑어 수치를 2~4 자리 반올림 문자열로 적재한다
    if isinstance(node, dict):
        for value in node.values():
            _collect_numbers(value, cat)
    elif isinstance(node, list):
        for value in node:
            _collect_numbers(value, cat)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        for digits in (2, 3, 4):
            text = f'{float(node):.{digits}f}'
            cat.add(text)
            cat.add(text.rstrip('0'))


def _metric_catalog(proj):
    # certified/ 의 커밋된 metrics JSON = 인용 수치의 카탈로그 (git 원장)
    cat = set()
    root = os.path.join(proj, 'certified')
    if not os.path.isdir(root):
        return cat
    for path in glob.glob(os.path.join(root, '**', '*.json'), recursive=True):
        if any(token in path for token in CATALOG_EXCLUDE):
            continue
        try:
            if os.path.getsize(path) > CATALOG_MAX_BYTES:
                continue
            with open(path) as f:
                _collect_numbers(json.load(f), cat)
        except Exception:
            continue
    return cat


def _check_cited_metrics(proj):
    # 리포트·이슈 '표' 에 새로 실린 수치만 대조한다. 산문의 Δ·σ 는
    # 파생값이라 카탈로그에 없어 검증 범위 밖이다.
    #
    # 한계: 이것은 '존재 검사'(literal membership)이지 '출처 검사' 가
    # 아니다. 틀린 수치가 카탈로그의 무관한 값과 우연히 일치하면 통과한다.
    # 인용 자릿수가 낮을수록 검출력이 떨어진다(4자리 > 3자리 > 2자리).
    # 즉 이 검사는 거짓 인용을 *줄이되* 없애지 못한다.
    diff = _git(
        ['diff', 'HEAD', '--', 'docs/reports', 'docs/issues'], proj
    )
    rows = [
        ln[1:] for ln in diff.splitlines()
        if ln.startswith('+') and not ln.startswith('+++')
    ]
    cited = [
        n for row in rows if row.strip().startswith('|')
        for n in NUM_RE.findall(row)
    ]
    if not cited:
        return []
    cat = _metric_catalog(proj)
    if not cat:
        return [
            'cited metrics changed but certified/ holds no committed metrics '
            "JSON — copy the adopted run's metrics into certified/ and commit"
        ]
    missing = sorted({
        n for n in cited if n not in cat and n.rstrip('0') not in cat
    })
    if not missing:
        return []
    return [
        'cited in a table but absent from certified/*.json: '
        + ', '.join(missing)
    ]


def _check_ruler_touched(proj):
    # diff 가 기준 파일을 건드렸나. 건드렸으면 실험을 재는 정의(정답·채점
    # 규칙·분할) 자체가 움직인 것 — 사람이 새 정의를 못 박고 ack 로 풀기
    # 전까지 커밋을 막는다.
    changed = set(_git(['diff', '--name-only', 'HEAD'], proj).splitlines())
    return [p for p in RULER_PATHS if p in changed]


def _check_test_integrity(proj):
    # diff 에서 테스트 삭제·무조건 skip·assert 순삭제 신호를 뽑는다
    diff = _git(['diff', 'HEAD', '--', 'tests'], proj)
    if not diff.strip():
        return []
    added, removed = [], []
    for ln in diff.splitlines():
        if ln.startswith('+') and not ln.startswith('+++'):
            added.append(ln[1:])
        elif ln.startswith('-') and not ln.startswith('---'):
            removed.append(ln[1:])

    def count(lines, pattern):
        return sum(1 for ln in lines if re.match(pattern, ln.strip()))

    signals = []
    dropped = count(removed, r'def test_') - count(added, r'def test_')
    if dropped > 0:
        signals.append(f'{dropped} test function(s) net removed')
    skips = sum(1 for ln in added if WEAKEN_RE.search(ln))
    if skips:
        signals.append(f'{skips} unconditional skip/xfail added')
    weakened = count(removed, r'assert\b') - count(added, r'assert\b')
    if weakened > 0:
        signals.append(f'{weakened} assert statement(s) net removed')
    return signals


def _acked(state_dir, diff_hash):
    # 결정적 층의 오탐 해제 토글 — 사람이 만든다 (settings.json deny)
    return os.path.exists(os.path.join(state_dir, f'ack-{diff_hash}'))


def _log(state_dir, record):
    # append-only 판정 이력 — 게이트가 실제로 물었는지 사후 감사용
    record['ts'] = int(time.time())
    try:
        with open(os.path.join(state_dir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _spawn_instructions(diff_hash, verdict_file):
    # 반박자 미실행 시 메인 모델에 줄 지시문
    return (
        'Loop completion gate: spawn an ISOLATED-CONTEXT refuter before '
        'finishing.\n'
        'Invoke the `refuter` skill (or an Agent whose model is chosen by '
        'diff risk: Sonnet by default, Opus when refuting needs adversarial '
        'reasoning beyond number/JSON matching — see the skill\'s model '
        'selection rule) whose job is to REFUTE — not approve — the current '
        'diff against the original acceptance criteria (nearest '
        'docs/issues/issue-*.md plan section or the active PRD).\n\n'
        'The refuter MUST write its verdict to:\n'
        f'  {verdict_file}\n'
        'as JSON: {"verdict":"PASS"|"FAIL","diff_hash":"' + diff_hash +
        '","findings":[...],"model":"...","round":1}\n'
        'Keep it narrow: judge from the diff + criteria; read files only to '
        'verify a specific claim. Do not declare done until verdict is PASS.'
    )


def main():
    data = _read_input()
    proj = _proj_dir(data)
    session_id = data.get('session_id', '')

    # OMC 킬 스위치 존중 (OMC 훅 컨벤션과 동일)
    skip = [
        s.strip()
        for s in (os.environ.get('OMC_SKIP_HOOKS') or '').split(',')
    ]
    if os.environ.get('DISABLE_OMC') == '1' or 'refuter-gate' in skip:
        _passthrough()

    # 1) 재진입 Stop 에서는 재-block 금지 (Claude Code 의 연속 block 안전
    #    오버라이드 trip 방지 — OMC persistent-mode 와 동일 패턴). 다음
    #    변경 사이클에서 다시 평가된다.
    if data.get('stop_hook_active') is True:
        print(
            '[refuter-gate] re-entrant Stop — not re-blocking (safety); '
            'refuter enforced on the next change cycle.',
            file=sys.stderr,
        )
        _passthrough()

    # 2) 무한루프 차단: 누적 block 상한
    state_dir = os.path.join(proj, '.omc/state/refuter')
    os.makedirs(state_dir, exist_ok=True)
    counter = os.path.join(state_dir, f'.blocks-{session_id}')
    n_blocks = 0
    if os.path.exists(counter):
        try:
            n_blocks = int(open(counter).read().strip() or '0')
        except Exception:
            n_blocks = 0
    if n_blocks >= MAX_BLOCKS:
        print(
            f'[refuter-gate] block limit ({MAX_BLOCKS}) reached — '
            'passing through; review manually.',
            file=sys.stderr,
        )
        _passthrough()

    # 3) 미커밋 diff 없으면 통과 (잡담/조회 턴)
    diff = _git(['diff', 'HEAD'], proj)
    if not diff.strip():
        _passthrough()
    diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:12]

    # 4) 결정적 선통과: 변경된 .py 에 ruff (실패 시 차단)
    changed = [
        ln for ln in _git(
            ['diff', '--name-only', 'HEAD'], proj
        ).splitlines()
        if ln.endswith('.py') and os.path.exists(os.path.join(proj, ln))
    ]
    if changed:
        ruff = os.path.join(proj, '.venv/bin/ruff')
        ruff_bin = ruff if os.path.exists(ruff) else 'ruff'
        try:
            r = subprocess.run(
                [ruff_bin, 'check', *changed], cwd=proj,
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                _bump(counter, n_blocks)
                _log(state_dir, {
                    'diff_hash': diff_hash, 'layer': 'deterministic',
                    'check': 'ruff', 'result': 'BLOCK',
                })
                _block(
                    'Deterministic gate failed: ruff check did not pass on '
                    'changed files. Fix lint errors before completion.\n\n'
                    f'{r.stdout}\n{r.stderr}'
                )
        except FileNotFoundError:
            pass  # ruff 없으면 결정적 선통과 생략

    # 5) 결정적 층: 기준 파일 잠금 + 테스트 무결성 + 인용 수치 대조. 모델
    #    판단이 아니라 diff 와 certified/ 를 게이트가 직접 읽는다.
    if _acked(state_dir, diff_hash):
        _log(state_dir, {
            'diff_hash': diff_hash, 'layer': 'deterministic',
            'result': 'ACK', 'note': 'human-created ack file',
        })
    else:
        ruler = _check_ruler_touched(proj)
        if ruler:
            _bump(counter, n_blocks)
            _log(state_dir, {
                'diff_hash': diff_hash, 'layer': 'deterministic',
                'check': 'ruler-lock', 'result': 'BLOCK', 'ruler': ruler,
            })
            _block(
                'Ruler files changed — the definition used to score '
                'experiments (gold / metric / split) itself moved:\n'
                + '\n'.join(f'- {p}' for p in ruler)
                + '\n\nPin the new definition BEFORE re-scoring (CLAUDE.md '
                '작업 흐름 앞부분), then a HUMAN unlocks by creating:\n'
                f'  {os.path.join(state_dir, "ack-" + diff_hash)}\n'
                'The AI cannot create this file (settings.json deny).'
            )
        hard = _check_test_integrity(proj) + _check_cited_metrics(proj)
        if hard:
            _bump(counter, n_blocks)
            _log(state_dir, {
                'diff_hash': diff_hash, 'layer': 'deterministic',
                'result': 'BLOCK', 'findings': hard,
            })
            _block(
                'Deterministic gate failed (not model-judged):\n'
                + '\n'.join(f'- {x}' for x in hard)
                + '\n\nFix the diff. If this is a reviewed exception '
                '(justified test removal, derived figure), a HUMAN must '
                'create the ack file:\n'
                f'  {os.path.join(state_dir, "ack-" + diff_hash)}'
            )

    # 6) 여기까지가 결정적 층 — 루프 여부와 무관하게 항상 실행한다.
    #    모델을 부르지 않으므로 토큰 비용이 0 이다. 아래 판단 층은
    #    반박자 서브에이전트를 요구하므로 루프 모드에서만 건다.
    if not _loop_active(proj, session_id):
        _passthrough()

    # 7) 판단 층: 반박자 판정 파일. 파일명이 diff 해시라 diff 가 바뀌면
    #    이전 판정은 무효다. 다만 이 파일은 모델이 쓰므로 위조 가능하다 —
    #    신선도는 보증하되 진정성은 보증하지 않는다.
    verdict_file = os.path.join(state_dir, f'{diff_hash}.json')
    if not os.path.exists(verdict_file):
        _bump(counter, n_blocks)
        _block(_spawn_instructions(diff_hash, verdict_file))
    try:
        with open(verdict_file) as f:
            verdict = json.load(f)
    except Exception:
        _bump(counter, n_blocks)
        _block(
            f'Refuter verdict file {verdict_file} is unreadable. '
            'Re-run the refuter and write valid JSON.'
        )

    findings = verdict.get('findings', [])
    if str(verdict.get('verdict', '')).upper() == 'PASS':
        _log(state_dir, {
            'diff_hash': diff_hash, 'layer': 'judgment',
            'result': 'PASS', 'model': verdict.get('model'),
            'round': verdict.get('round'),
        })
        _passthrough()

    # FAIL: 결함 보고 + 수정 요구
    _bump(counter, n_blocks)
    _log(state_dir, {
        'diff_hash': diff_hash, 'layer': 'judgment', 'result': 'FAIL',
        'model': verdict.get('model'), 'round': verdict.get('round'),
        'findings': findings,
    })
    body = (
        '\n'.join(f'- {x}' for x in findings)
        if findings else '(no findings recorded)'
    )
    _block(
        'Refuter returned FAIL on the current diff. Address these, then the '
        'gate re-evaluates the new diff:\n' + body
    )


if __name__ == '__main__':
    main()
