#!/usr/bin/env python3
"""Stop 훅: 루프 모드에서 격리 컨텍스트 반박자 게이트.

루프(ralph/ultrawork) 활성 + 미커밋 diff 존재 시에만 작동한다.
변경된 `.py`에 ruff 를 먼저 돌리고(결정적 선통과), 그 다음 현재
diff 에 대한 반박자 판정 파일을 요구한다. 판정이 PASS 면 완료를
허용하고, FAIL 이면 막고, 없으면 반박자 실행을 지시한다. 무한
루프는 세션 누적 block 카운터로 차단한다.
"""
import hashlib
import json
import os
import subprocess
import sys

# 같은 게이트 세션에서 누적 block 상한 (무한루프 차단)
MAX_BLOCKS = 6


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

    # 1) 루프 비활성이면 게이트 자체를 끈다
    if not _loop_active(proj, session_id):
        _passthrough()

    # 2) 재진입 Stop 에서는 재-block 금지 (Claude Code 의 연속 block 안전
    #    오버라이드 trip 방지 — OMC persistent-mode 와 동일 패턴). 다음
    #    변경 사이클에서 다시 평가된다.
    if data.get('stop_hook_active') is True:
        print(
            '[refuter-gate] re-entrant Stop — not re-blocking (safety); '
            'refuter enforced on the next change cycle.',
            file=sys.stderr,
        )
        _passthrough()

    # 3) 무한루프 차단: 누적 block 상한
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
                _block(
                    'Deterministic gate failed: ruff check did not pass on '
                    'changed files. Fix lint errors before completion.\n\n'
                    f'{r.stdout}\n{r.stderr}'
                )
        except FileNotFoundError:
            pass  # ruff 없으면 결정적 선통과 생략

    # 5) 반박자 판정 파일 확인 (모델 말이 아니라 파일을 직접 읽는다)
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

    if str(verdict.get('verdict', '')).upper() == 'PASS':
        _passthrough()

    # FAIL: 결함 보고 + 수정 요구
    _bump(counter, n_blocks)
    findings = verdict.get('findings', [])
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
