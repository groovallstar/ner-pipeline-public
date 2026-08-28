#!/usr/bin/env python3
"""Stop 훅: 커밋하지 않고 턴을 끝낼 때 거는 완료 게이트.

검사 게이트는 두 진입점으로 걸린다.

- 커밋 직전(`commit_gate.py`, PreToolUse): 기계 검사. 커밋되는 내용을
  커밋 전에 검사한다 — 게이트의 주 진입점이다.
- 턴 종료(이 파일, Stop): 커밋 없이 끝나는 턴을 받는다. 기계 검사에
  더해 **반박자 판정**을 얹는다 — 서브에이전트를 요구하므로 커밋 명령
  중간에는 걸 수 없고 여기에만 있다.

기계 검사(기준 파일 변경 감지, ruff, 테스트 무결성, 인용 표 수치 ↔
`certified/` 대조)의 구현은 `gate_core.py` 에 있고 두 진입점이 공유한다.
반박자 판정 파일은 모델이 쓰므로 위조 가능하며, 기계 검사를 대체하지
않는다.

반박자를 자동으로 요구하는 구간은 둘뿐이다 — 기준 파일을 건드린
diff(`gate_core` 가 요구)와 자율 루프가 도는 중(`_loop_active`). 평범한
단발 작업에는 걸리지 않는다.

기계 검사의 오탐은 사람이 `human-allow-<diff_hash>` 파일로 해제한다.
해시가 같으므로 한 번 만든 승인은 두 진입점에서 함께 듣는다. 모든 판정·해제는
`log.jsonl` 에 append-only 로 남는다. 무한 루프는 연속 block 카운터로
차단하되 **통과할 때마다 리셋한다 — 상한에 걸려 여는 통과도 포함이다.**
상한으로 열리는 순간은 `log.jsonl` 과 사용자 화면 양쪽에 남긴다 — 조용히
꺼지지 않게.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_core as core  # noqa: E402

# 같은 게이트 세션에서 누적 block 상한 (무한루프 차단)
MAX_BLOCKS = 6


def _read_input():
    # Stop 훅 stdin JSON 파싱 (실패해도 게이트는 조용히 빠진다)
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _reset(counter):
    # 정상 통과 = 문제 해소. 카운터를 지워 상한이 '누적' 이 아니라 '연속'
    # 실패만 세게 한다 — 무관한 실패가 세션 내내 쌓여 게이트를 끄지 않도록.
    try:
        if os.path.exists(counter):
            os.remove(counter)
    except Exception:
        pass


def _passthrough(counter=None, notice=None):
    # exit 0 = 완료 허용. 검사가 대상 없이 지나갔으면 그 사실만 알린다.
    if counter:
        _reset(counter)
    if notice:
        print(json.dumps({'systemMessage': notice}, ensure_ascii=False))
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


def main():
    data = _read_input()
    proj = core.project_dir(data)
    session_id = data.get('session_id', '')

    if core.skip_requested():
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

    # 2) 상태 경로 · 연속 block 카운터
    state_dir = core.state_dir(proj)
    counter = os.path.join(state_dir, f'.blocks-{session_id}')
    n_blocks = 0
    if os.path.exists(counter):
        try:
            n_blocks = int(open(counter).read().strip() or '0')
        except Exception:
            n_blocks = 0

    # 3) 미커밋 diff 없으면 통과 (잡담·조회 턴, 또는 커밋을 마친 턴 —
    #    후자는 커밋 직전에 `commit_gate.py` 가 이미 검사했다)
    diff = core.diff_text(proj)
    dhash = core.anchor_hash(proj, diff)
    if not diff.strip():
        # git 호출이 실패해도 여기로 온다 — 그 경우 '변경 없음' 이 아니라
        # '못 봤음' 이므로 조용히 넘기지 않는다.
        _passthrough(counter, core.pass_notice(state_dir, dhash))

    # 4) 무한루프 차단: 연속 block 상한. 통과할 때마다 카운터가 지워지므로
    #    여기 걸리는 건 같은 문제를 못 고치고 도는 상황이다. 무력화는 조용히
    #    넘기지 않고 이력과 사용자 화면 양쪽에 남긴다.
    #
    #    상한으로 여는 이 통과도 카운터를 지운다. 상한의 뜻은 '이 턴을 연다'
    #    이지 '이 세션을 포기한다' 가 아니다 — 안 지우면 그 세션의 Stop
    #    게이트가 영구히 죽고, 화면이 평소와 같아 아무도 못 알아챈다.
    if n_blocks >= MAX_BLOCKS:
        core.log(state_dir, {
            'diff_hash': dhash, 'entry': 'stop', 'result': 'LIMIT',
            'note': f'consecutive block limit {MAX_BLOCKS} reached',
        })
        _passthrough(
            counter,
            f'[refuter-gate] consecutive block limit ({MAX_BLOCKS}) reached — '
            'this turn passes through UNCHECKED by the Stop gate. The counter '
            'resets, so the next change cycle is checked again. Commits are '
            'still gated separately. Review the diff manually.',
        )

    # 5) 기계 검사 — 루프 여부와 무관하게 항상 실행한다. 모델을 부르지
    #    않으므로 토큰 비용이 0 이다.
    result = core.run_deterministic(proj, state_dir, dhash)
    if result is not None:
        check, reason, findings = result
        _bump(counter, n_blocks)
        core.log(state_dir, {
            'diff_hash': dhash, 'entry': 'stop', 'layer': 'machine',
            'check': check, 'result': 'BLOCK', 'findings': findings,
        })
        _block(reason)

    # 6) 반박자는 서브에이전트를 요구해 비싸므로 자율 루프에서만 자동으로
    #    건다. 평범한 단발 작업은 여기서 끝난다.
    if not _loop_active(proj, session_id):
        _passthrough(counter, core.pass_notice(state_dir, dhash))

    # 7) 반박자 판정. 기준 파일을 건드린 diff 라면 기계 검사가 이미 같은
    #    판정을 요구했으므로 여기서는 추가 비용 없이 통과한다.
    state, defects, meta = core.read_verdict(state_dir, dhash)
    if state in ('missing', 'unreadable'):
        _bump(counter, n_blocks)
        _block(core.spawn_instructions(
            dhash, core.verdict_path(state_dir, dhash),
            f'Loop completion gate: refuter verdict is {state}.',
        ))
    if state == 'PASS':
        core.log(state_dir, {
            'diff_hash': dhash, 'entry': 'stop', 'layer': 'refuter',
            'result': 'PASS', **meta,
        })
        # 통과시키되 집행 주체 없는 확인은 화면에 남긴다 — 그걸 테스트로
        # 옮기는 것만이 다음 라운드를 없앤다.
        _passthrough(counter, core.pass_notice(state_dir, dhash))

    # FAIL: 결함 보고 + 수정 요구
    _bump(counter, n_blocks)
    core.log(state_dir, {
        'diff_hash': dhash, 'entry': 'stop', 'layer': 'refuter',
        'result': 'FAIL', 'defects': defects, **meta,
    })
    body = (
        '\n'.join(f'- {x}' for x in defects)
        if defects else '(no defects recorded)'
    )
    _block(
        'Refuter returned FAIL on the current diff. Address these, then the '
        'gate re-evaluates the new diff:\n' + body
    )


if __name__ == '__main__':
    main()
