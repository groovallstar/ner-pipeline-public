#!/usr/bin/env python3
"""검사 게이트 기계 검사의 공용 구현.

게이트가 diff 를 직접 읽어 판정한다 — 모델을 부르지 않으므로 토큰이 들지
않고 결과가 매번 같다. 두 진입점이 이 모듈을 공유한다.

- `commit_gate.py` (PreToolUse): `git commit` 이 실행되기 직전에 건다.
  커밋되는 내용이 아직 워킹트리·인덱스에 남아 있어 검사 대상이 있다.
- `refuter_gate.py` (Stop): 커밋하지 않고 턴을 끝낼 때 건다. 반박자
  서브에이전트를 요구하는 검사는 이쪽에만 있다 — 커밋 명령 중간에는
  에이전트를 부를 수 없다.

두 진입점이 같은 앵커 지문(`anchor_hash`)을 계산하므로, 사람이 만든 예외
승인 파일(`human-allow-<앵커>`) 하나로 양쪽이 함께 풀린다.

**검사 대상과 앵커는 다르다.** 검사는 `diff_text()` 전체를 본다. 앵커는
거기서 `ANCHOR_EXCLUDE` 를 뺀 것으로, 판정·승인이 무엇에 묶이는지만
정한다.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# 인용 수치로 볼 토큰 (0.93 · 0.9361 · 1.00 등)
NUM_RE = re.compile(r'\b[01]\.\d{2,4}\b')

# 표 위 출처 선언 — `<!-- certified: classifier/ko/<run>/pooled_metrics.json -->`.
# 경로는 `certified/` 기준 상대경로이며 디렉토리도 가리킬 수 있다.
CERT_SRC_RE = re.compile(r'<!--\s*certified:\s*([^\s>]+?)\s*-->')

# 테스트 약화 신호. `skipif` 는 환경 조건부라 정당 — `\b` 로 배제된다
WEAKEN_RE = re.compile(
    r'@pytest\.mark\.skip\b|@pytest\.mark\.xfail'
    r'|pytest\.skip\(|pytest\.xfail\('
)

# 카탈로그에서 제외할 산출물 (예측 덤프·체크포인트 내부 상태)
CATALOG_EXCLUDE = ('test_predictions', 'checkpoint-')
CATALOG_MAX_BYTES = 20_000_000

# 원장에서 '증거' 로 읽는 파일 형식. 인용 대조가 이 형식만 보므로, AI 의
# 손저작을 막는 `settings.json` deny 도 정확히 이 형식이어야 한다 — 좁으면
# 증거를 손댈 수 있고 넓으면 원장의 설명 문서(README)까지 잠겨 갱신이 막힌다.
# 둘의 어긋남은 tests/hooks 가 잡는다.
CATALOG_EXT = ('json',)

# 기준 파일 — 정답·채점규칙·분할의 정의. 건드리면 사람 승인 전까지
# 진행을 막는다 (CLAUDE.md 하네스 기준 파일 표와 동기화).
RULER_PATHS = (
    'docs/manual/data/canonical-entity-schema.md',
    'src/ner/classifier/data_utils.py',
    'src/ner/classifier/kfold_pool.py',
)

# 패키지 전체가 자인 곳 — 파일을 열거하면 새 파일이 생겨도 목록이 조용히
# 낡는다. 둘 다 변경이 드물어(각 3·9 커밋) 디렉토리로 잡아도 과잉 잠금이
# 되지 않는다. `classifier/` 는 학습·분석 코드가 섞여 있어 위에 파일로
# 적으며, 그래서 채점에 닿는 파일을 새로 만들면 여기도 갱신해야 한다.
RULER_DIRS = (
    'src/ner/metrics/',
    'src/ner/validity/',
)

# 이번 실행에서 검사가 '대상 없이' 지나간 사유. 게이트는 스스로 고장 나면
# 열린 채 빠지는데(커밋을 영구 봉쇄하면 안 되므로), 그게 조용하면 통과가
# 검사 통과인지 검사 부재인지 구별할 수 없다 — 그래서 기록해 훅이 알린다.
_DEGRADED = []


def degraded():
    """검사가 대상 없이 지나간 사유들. 비어 있지 않으면 이번 통과는
    '검사를 통과했다'가 아니라 '검사가 못 돌았다'는 뜻이다."""
    return sorted(set(_DEGRADED))


def git(args, cwd, allow_fail=False):
    # git 호출. 실패해도 빈 문자열을 돌려 게이트가 열린 채 빠진다.
    #
    # `allow_fail` 은 '실패가 곧 답' 인 호출용이다 — `git show HEAD:<새 파일>`
    # 은 파일이 아직 커밋 안 됐다는 뜻이라 정상이다. 그 밖의 호출이 실패하면
    # 검사가 대상을 못 본 것이므로 기록해 둔다(저장소가 아님·HEAD 없음 등은
    # 예외가 아니라 종료 코드로 오므로 둘 다 본다).
    try:
        out = subprocess.run(
            ['git', *args], cwd=cwd, capture_output=True, text=True,
            timeout=20,
        )
    except Exception as exc:
        _DEGRADED.append(f'git {args[0]}: {type(exc).__name__}')
        return ''
    if out.returncode != 0 and not allow_fail:
        _DEGRADED.append(f'git {args[0]}: exit {out.returncode}')
    return out.stdout


def project_dir(data):
    # 프로젝트 루트: 훅 env > stdin cwd > git toplevel
    return (
        os.environ.get('CLAUDE_PROJECT_DIR')
        or data.get('cwd')
        or git(['rev-parse', '--show-toplevel'], '.', allow_fail=True).strip()
        or '.'
    )


def skip_requested():
    # OMC 킬 스위치 존중 (OMC 훅 컨벤션과 동일)
    skip = [
        s.strip()
        for s in (os.environ.get('OMC_SKIP_HOOKS') or '').split(',')
    ]
    return os.environ.get('DISABLE_OMC') == '1' or 'refuter-gate' in skip


def diff_text(proj):
    # 인덱스 포함 미커밋 변경. `git add` 된 새 파일도 여기 잡힌다
    return git(['diff', 'HEAD'], proj)


# 앵커 지문에서 빼는 경로. 이슈 문서는 채점에 쓰이지 않는 기록이라, 여기
# 한 줄이 늘었다고 앞선 판정을 무효로 만들지 않는다.
ANCHOR_EXCLUDE = ('docs/issues',)


def anchor_hash(proj, full_diff):
    """판정·승인이 묶이는 지문.

    `ANCHOR_EXCLUDE` 를 뺀 diff 로 계산한다. 기계 검사와 반박자는
    `diff_text()` 전체를 그대로 보므로, 좁아지는 것은 "무엇이 판정을
    무효로 만드는가" 뿐이다.

    제외 후 diff 가 비면 전체 diff 로 되돌아간다 — 이슈 문서만 고친
    변경들이 빈 해시 하나를 공유하면 서로의 판정을 물려받는다.
    """
    args = ['diff', 'HEAD', '--', '.']
    args += [f':(exclude){path}' for path in ANCHOR_EXCLUDE]
    narrowed = git(args, proj)
    return diff_hash(narrowed if narrowed.strip() else full_diff)


def diff_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def state_dir(proj):
    path = os.path.join(proj, '.omc/state/refuter')
    os.makedirs(path, exist_ok=True)
    return path


def allow_file(sdir, dhash):
    # 이 diff 에 대한 예외 승인 파일 경로. 이름에 diff 해시가 박혀 있어
    # 코드가 한 글자라도 바뀌면 이전 승인은 자동으로 무효가 된다
    return os.path.join(sdir, f'human-allow-{dhash}')


def human_allowed(sdir, dhash):
    # 기계 검사의 오탐 해제 — 사람이 만든다. AI 의 Write/Edit 은
    # settings.json deny 지만 셸 경로는 안 막히므로 speed-bump 이지
    # 기계 보증은 아니다 (`certified/` 복사와 같은 성격)
    return os.path.exists(allow_file(sdir, dhash))


def log(sdir, record):
    # append-only 판정 이력 — 게이트가 실제로 물었는지 사후 감사용
    record['ts'] = int(time.time())
    try:
        with open(os.path.join(sdir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass


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


def ledger_files(root):
    # 원장 트리에서 증거로 읽을 파일들 (형식은 `CATALOG_EXT` 가 정본)
    found = []
    for ext in CATALOG_EXT:
        found += glob.glob(
            os.path.join(root, '**', f'*.{ext}'), recursive=True
        )
    return sorted(found)


def _metric_catalog(proj):
    # certified/ 의 커밋된 metrics JSON = 인용 수치의 카탈로그 (git 원장)
    cat = set()
    root = os.path.join(proj, 'certified')
    if not os.path.isdir(root):
        return cat
    for path in ledger_files(root):
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


def _added_table_numbers(diff):
    # diff 에서 새로 실린 표 행을 (문서경로, 새파일 줄번호, 수치들) 로 뽑는다.
    # hunk 헤더의 `+시작줄` 을 따라가며 세므로 원본 문서에서의 위치를 안다 —
    # 그 위치가 어느 출처 선언 아래인지 찾는 데 쓴다.
    entries = []
    path, lineno = None, 0
    for ln in diff.splitlines():
        if ln.startswith('+++ b/'):
            path = ln[6:]
        elif ln.startswith('@@'):
            m = re.search(r'\+(\d+)', ln)
            lineno = int(m.group(1)) if m else 0
        elif ln.startswith('+') and not ln.startswith('+++'):
            row = ln[1:]
            if row.strip().startswith('|'):
                nums = NUM_RE.findall(row)
                if nums and path:
                    entries.append((path, lineno, nums))
            lineno += 1
        elif ln.startswith('-') and not ln.startswith('---'):
            pass  # 삭제 행은 새 파일 줄번호를 늘리지 않는다
        elif ln.startswith(' '):
            lineno += 1
    return entries


def _declared_sources(proj, path, lineno):
    # 문서에서 표 위에 선언된 출처를 찾는다. 표 바로 앞의 선언이 우선이고,
    # 없으면 문서 전체 선언의 합집합으로 떨어진다(문서 머리에 한 번만 쓰는
    # 형태를 허용). 둘 다 없으면 빈 목록 — 호출부가 전체 원장으로 떨어진다.
    try:
        with open(os.path.join(proj, path)) as f:
            lines = f.read().splitlines()
    except Exception:
        return []
    nearest, every = None, []
    for i, ln in enumerate(lines, start=1):
        m = CERT_SRC_RE.search(ln)
        if not m:
            continue
        every.append(m.group(1))
        if i < lineno:
            nearest = m.group(1)
    return [nearest] if nearest else every


def _catalog_of(proj, src):
    # 선언된 출처 하나의 수치 집합. 파일도 디렉토리도 아니면 None (미승격).
    rel = src if src.startswith('certified/') else os.path.join('certified', src)
    full = os.path.join(proj, rel)
    if os.path.isdir(full):
        paths = ledger_files(full)
    elif os.path.isfile(full):
        paths = [full]
    else:
        return None
    cat = set()
    for p in paths:
        if any(token in p for token in CATALOG_EXCLUDE):
            continue
        try:
            if os.path.getsize(p) > CATALOG_MAX_BYTES:
                continue
            with open(p) as f:
                _collect_numbers(json.load(f), cat)
        except Exception:
            continue
    return cat


def _preexisting_numbers(proj, path):
    # 그 문서의 커밋된 판(HEAD)에 이미 있던 수치. 새 파일이면 `git show` 가
    # 실패해 빈 집합이 되고, 그러면 모든 수치가 새 주장으로 취급돼 전부
    # 검사된다 — 새로 쓰는 리포트가 정확히 그런 자리다.
    blob = git(['show', f'HEAD:{path}'], proj, allow_fail=True)
    return set(NUM_RE.findall(blob))


def check_cited_metrics(proj):
    # 리포트·이슈 '표' 에 **새로 주장된** 수치를 그 표가 선언한 출처와 대조한다.
    # 산문의 Δ·σ 는 파생값이라 대조 범위 밖이다.
    #
    # 이미 그 문서에 있던 수치는 건너뛴다. 표를 옮기거나 서식·오타를 고치면
    # diff 에는 그 행이 통째로 새로 실리는데, 그건 새 주장이 아니라서다.
    # 이 걸러내기가 없으면 옛 리포트를 손볼 때마다 막히고, 그렇다고 원장을
    # 채워 풀려 하면 원장이 커져 대조 자체가 무력해진다(아래 참조).
    # 값을 *바꾸면* 바뀐 값은 HEAD 에 없으므로 그대로 걸린다.
    #
    # 출처 선언(`<!-- certified: <경로> -->`)이 있으면 그 파일·디렉토리 안에서만
    # 찾으므로 원장이 커져도 검출력이 유지된다. 선언이 없으면 원장 전체를
    # 뒤지는 옛 동작으로 떨어지는데, 이건 '존재 검사'라 무관한 값과 우연히
    # 일치하면 통과한다 — 원장이 쌓일수록 약해지므로 선언을 붙이는 편이 낫다.
    entries = _added_table_numbers(git(
        ['diff', 'HEAD', '--', 'docs/reports', 'docs/issues'], proj
    ))
    if not entries:
        return []

    global_cat = None
    prior = {}
    findings = {}
    for path, lineno, nums in entries:
        if path not in prior:
            prior[path] = _preexisting_numbers(proj, path)
        nums = [n for n in nums if n not in prior[path]]
        if not nums:
            continue
        sources = _declared_sources(proj, path, lineno)
        if sources:
            cat, unknown = set(), []
            for src in sources:
                one = _catalog_of(proj, src)
                if one is None:
                    unknown.append(src)
                else:
                    cat |= one
            if unknown:
                findings.setdefault(
                    f'{path}: declared source not in certified/ — '
                    + ', '.join(sorted(set(unknown)))
                    + ' (promote the run before citing it)', set()
                )
                continue
            origin = ', '.join(sorted(set(sources)))
        else:
            if global_cat is None:
                global_cat = _metric_catalog(proj)
            if not global_cat:
                findings.setdefault(
                    f'{path}: cites metrics but certified/ holds no committed '
                    "metrics JSON — copy the adopted run's metrics into "
                    'certified/ and commit', set()
                )
                continue
            cat, origin = global_cat, 'certified/** (no source declared)'
        missing = {n for n in nums if n not in cat and n.rstrip('0') not in cat}
        if missing:
            findings.setdefault(
                f'{path}: cited in a table but absent from {origin}', set()
            ).update(missing)

    return [
        f'{key}: {", ".join(sorted(nums))}' if nums else key
        for key, nums in findings.items()
    ]


def is_ruler(path):
    # 그 경로가 실험을 재는 자인가 — 열거된 파일이거나 자 패키지 안의 `.py`
    return path in RULER_PATHS or (
        path.endswith('.py') and path.startswith(RULER_DIRS)
    )


def check_ruler_touched(proj):
    # diff 가 기준 파일을 건드렸나. 건드렸으면 실험을 재는 정의(정답·채점
    # 규칙·분할) 자체가 움직인 것 — 사람이 새 정의를 못 박고 예외 승인으로
    # 풀기 전까지 커밋을 막는다.
    #
    # 절 단위 면제는 두지 않는다. 면제 구간을 변경 *후* 파일에서 구하는
    # 이상, 면제될 절을 새로 만들면서 그 안에 내용을 넣으면 추가한 줄이
    # 통째로 자기가 만든 구간에 들어가 게이트가 한 번도 울리지 않는다 —
    # 문턱을 낮추는 방향으로 새는 구멍이라 오탐보다 비싸다. 대신 이 표에
    # 오른 문서에는 기록·맥락 절을 두지 않아 오탐 자체를 없앤다.
    changed = set(git(['diff', '--name-only', 'HEAD'], proj).splitlines())
    return [p for p in sorted(changed) if is_ruler(p)]


def _diff_by_file(diff):
    # diff 를 파일별 (추가 줄, 삭제 줄) 로 가른다. 파일이 통째로 지워지면
    # `+++ /dev/null` 이 나오므로 그때는 `--- a/` 쪽 경로로 귀속시킨다.
    per, old, path = {}, None, None
    for ln in diff.splitlines():
        if ln.startswith('--- '):
            rest = ln[4:]
            old = rest[2:] if rest.startswith('a/') else None
        elif ln.startswith('+++ '):
            rest = ln[4:]
            path = rest[2:] if rest.startswith('b/') else old
            if path:
                per.setdefault(path, ([], []))
        elif path is None:
            continue
        elif ln.startswith('+'):
            per[path][0].append(ln[1:])
        elif ln.startswith('-'):
            per[path][1].append(ln[1:])
    return per


def check_test_integrity(proj):
    # 테스트 삭제·무조건 skip·assert 순삭제 신호를 **파일별로** 센다.
    #
    # tests/ 전체를 합산하면 한 파일에서 지운 만큼 다른 파일에 채워 넣어
    # 신호를 상쇄시킬 수 있다. 파일 사이로 테스트를 옮기는 정당한 리팩터가
    # 걸릴 수 있지만, 이 저장소의 tests/ 커밋 96개를 되짚어 보면 그렇게
    # 판정이 갈리는 커밋은 1건이었고 그마저 한 파일에서 테스트 4개가 빠져
    # 사람이 봤어야 할 변경이었다.
    diff = git(['diff', 'HEAD', '--', 'tests'], proj)
    if not diff.strip():
        return []

    def count(lines, pattern):
        return sum(1 for ln in lines if re.match(pattern, ln.strip()))

    signals = []
    for path, (added, removed) in sorted(_diff_by_file(diff).items()):
        dropped = count(removed, r'def test_') - count(added, r'def test_')
        if dropped > 0:
            signals.append(f'{path}: {dropped} test function(s) net removed')
        skips = sum(1 for ln in added if WEAKEN_RE.search(ln))
        if skips:
            signals.append(f'{path}: {skips} unconditional skip/xfail added')
        weakened = count(removed, r'assert\b') - count(added, r'assert\b')
        if weakened > 0:
            signals.append(
                f'{path}: {weakened} assert statement(s) net removed'
            )
    return signals


def check_ruff(proj):
    # 변경된 .py 에 ruff. 통과·미설치면 None, 실패면 출력 문자열
    changed = [
        ln for ln in git(['diff', '--name-only', 'HEAD'], proj).splitlines()
        if ln.endswith('.py') and os.path.exists(os.path.join(proj, ln))
    ]
    if not changed:
        return None
    local = os.path.join(proj, '.venv/bin/ruff')
    ruff_bin = local if os.path.exists(local) else 'ruff'
    try:
        r = subprocess.run(
            [ruff_bin, 'check', *changed], cwd=proj,
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        _DEGRADED.append('ruff: not installed (lint check skipped)')
        return None
    except Exception as exc:
        _DEGRADED.append(f'ruff: {type(exc).__name__} (lint check skipped)')
        return None
    if r.returncode == 0:
        return None
    return f'{r.stdout}\n{r.stderr}'


def degraded_notice(sdir, dhash):
    """검사가 대상 없이 지나갔으면 사용자에게 보일 문구를 돌려준다 (없으면
    None). 통과를 막지는 않는다 — 게이트 고장이 커밋을 영구 봉쇄해선 안 되기
    때문이다. 다만 조용히 넘어가면 통과의 의미가 달라지므로 화면과 이력에
    남긴다."""
    reasons = degraded()
    if not reasons:
        return None
    log(sdir, {
        'diff_hash': dhash, 'result': 'DEGRADED', 'reasons': reasons,
    })
    return (
        '[gate] Some checks could not run and passed by default: '
        + '; '.join(reasons)
        + '. Review the diff manually.'
    )


def verdict_path(sdir, dhash):
    return os.path.join(sdir, f'{dhash}.json')


def read_verdict(sdir, dhash):
    """반박자 판정을 읽는다. (state, defects, meta) — state 는
    'missing' · 'unreadable' · 'PASS' · 'FAIL'. 파일명이 diff 해시라
    diff 가 바뀌면 이전 판정은 자동으로 무효다. 다만 이 파일은 모델이
    쓰므로 위조 가능하다 — 신선도는 보증하되 진정성은 보증하지 않는다.

    `defects`(막는 사유)와 `checked`(확인 기록)를 나눠 받는다. 옛 판정은
    둘을 `findings` 한 배열에 섞어 썼으므로 그 형식도 읽는다 — 다만 섞인
    배열에서 결함만 골라낼 수는 없어 통째로 사유로 취급한다.
    """
    path = verdict_path(sdir, dhash)
    if not os.path.exists(path):
        return 'missing', [], {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return 'unreadable', [], {}
    defects = data.get('defects')
    if defects is None:
        defects = data.get('findings') or []
    state = 'PASS' if str(data.get('verdict', '')).upper() == 'PASS' else 'FAIL'
    meta = {'model': data.get('model'), 'round': data.get('round')}
    return state, defects, meta


def spawn_instructions(dhash, path, why):
    # 반박자 미실행 시 메인 모델에 줄 지시문
    return (
        f'{why}\n'
        'Spawn an ISOLATED-CONTEXT refuter: invoke the `refuter` skill (or '
        'an Agent on the session default model — do not override it) whose '
        "job is to REFUTE — not approve — the current diff "
        'against the original acceptance criteria (nearest '
        'docs/issues/issue-*.md plan section or the active PRD).\n\n'
        'The refuter MUST write its verdict to:\n'
        f'  {path}\n'
        'as JSON: {"verdict":"PASS"|"FAIL","diff_hash":"' + dhash +
        '","defects":[...],"checked":[...],"model":"...","round":<N>}\n'
        'FAIL only when letting the diff through would record a wrong '
        'conclusion as true (numbers that disagree with the artifact, a '
        'silently moved gold/split/metric, a check that green-lights what it '
        'should block, a test that no longer verifies the same contract). '
        'Things that fail safe — over-blocking, not reachable with current '
        'data, a disclosed discrepancy — go in "checked" with a PASS. When '
        'genuinely unsure, FAIL — but a round is NOT cheap (~5M input tokens '
        'measured, 12M at the tail), so do not manufacture doubt to justify '
        'one.'
        '\nScope and budget live in the refuter skill (§범위와 예산) and must '
        'be pasted into the spawn prompt: read the diff, the criteria, files '
        'NAMED IN THE DIFF and prior verdicts, run existing tests — no '
        'corpus-wide scans, no repo-wide greps, no new analysis scripts, at '
        'most 20 Bash calls. Do not re-count a claim that only a full scan '
        'could check — record it as a defect instead.'
        '\n\nROUND CAP. Get <N> from `ls .omc/state/refuter/*.json` — read '
        'the count off disk, never from memory. Past round 3 on the same '
        'task, STOP: hand the open defects to the human instead of spawning '
        'another refuter. If the same place is flagged twice, the fix owed is '
        'a structural change, not another patch to that spot. Do not declare '
        'done until the verdict is PASS or you have handed it over at the cap.'
    )


# 게이트 자신이 바뀐 diff 에서만 도는 자기 검사. 게이트는 목록으로 움직이는데
# 그 목록이 든 파일은 기준 파일이 아니라, 목록을 좁혀도 네 검사가 다 조용하다.
SELF_DIRS = ('.claude/hooks/',)
SELF_TESTS = 'tests/hooks'


def _pytest_cmd(proj):
    # 훅은 시스템 python 으로 뜨므로 pytest 가 없을 수 있다. 저장소 venv 우선.
    venv = os.path.join(proj, '.venv/bin/python')
    return [venv if os.path.exists(venv) else sys.executable, '-m', 'pytest']


def check_self_tests(proj):
    """게이트를 고치는 diff 면 그 회귀 테스트를 돌린다.

    통과·대상 없음·실행 불가면 None, 실패면 pytest 출력 꼬리를 돌려준다.
    실행 불가는 차단하지 않고 기록만 한다 — 게이트 고장이 커밋을 영구
    봉쇄해선 안 된다.
    """
    changed = git(['diff', '--name-only', 'HEAD'], proj).splitlines()
    if not any(p.startswith(SELF_DIRS) for p in changed):
        return None
    if not os.path.isdir(os.path.join(proj, SELF_TESTS)):
        _DEGRADED.append(f'self-test: {SELF_TESTS} missing')
        return None
    try:
        out = subprocess.run(
            [*_pytest_cmd(proj), SELF_TESTS, '-q'],
            cwd=proj, capture_output=True, text=True, timeout=90,
        )
    except Exception as exc:
        _DEGRADED.append(f'self-test: {type(exc).__name__}')
        return None
    if out.returncode == 0:
        return None
    if out.returncode >= 4:  # pytest 사용 오류(수집 실패·인자 오류)
        _DEGRADED.append(f'self-test: pytest exit {out.returncode}')
        return None
    return (out.stdout or out.stderr)[-1500:]


def run_deterministic(proj, sdir, dhash):
    """기계 검사 전체. 통과면 None, 막히면 (check, reason, findings).

    모델 판단이 아니라 diff 와 `certified/` 를 게이트가 직접 읽는다.
    ruff 와 자기 검사는 사람 승인으로도 면제되지 않는다 — 기계적으로 고칠
    수 있는 결함이라 예외 승인 대상이 아니다.

    기준 파일을 건드렸을 때만은 예외로 반박자 판정까지 요구한다. 반박자가
    잡아야 할 위험(gold 변조·분할 누수·"올랐다=개선"의 순환)이 거기 몰려
    있고, 어차피 사람이 승인으로 멈춰 서는 지점이라 추가 마찰이 작기
    때문이다. 그 밖의 변경은 기계 검사만 지난다.
    """
    ruff_out = check_ruff(proj)
    if ruff_out is not None:
        return (
            'ruff',
            'Deterministic gate failed: ruff check did not pass on changed '
            f'files. Fix lint errors before committing.\n\n{ruff_out}',
            ['ruff check failed'],
        )

    self_fail = check_self_tests(proj)
    if self_fail is not None:
        return (
            'self-test',
            'The gate itself changed and its regression tests fail. The gate '
            'is defined by lists inside these files, and nothing else locks '
            'them — a narrowed list passes every other check silently. Make '
            f'{SELF_TESTS} green before committing.\n\n{self_fail}',
            ['gate self-tests failed'],
        )

    has_allow = human_allowed(sdir, dhash)
    allow_path = allow_file(sdir, dhash)
    ruler = check_ruler_touched(proj)

    if ruler:
        listed = '\n'.join(f'- {p}' for p in ruler)
        if not has_allow:
            return (
                'ruler-lock',
                'Ruler files changed — the definition used to score '
                f'experiments (gold / metric / split) itself moved:\n{listed}'
                '\n\nPin the new definition BEFORE re-scoring (CLAUDE.md '
                '하네스). Ruler changes need BOTH a human allow file and a '
                'refuter pass — run the refuter first, then a HUMAN grants '
                'the exception by running:\n'
                f'  touch {allow_path}\n'
                'The AI must not create it (Write/Edit are denied in '
                'settings.json); creating it by any other route defeats '
                'the gate.',
                ruler,
            )
        state, defects, meta = read_verdict(sdir, dhash)
        vpath = verdict_path(sdir, dhash)
        if state in ('missing', 'unreadable'):
            why = (
                'Ruler files changed and the human allow file is present, but '
                f'the refuter verdict is {state}:\n{listed}'
            )
            return ('ruler-refuter', spawn_instructions(dhash, vpath, why),
                    ruler)
        if state == 'FAIL':
            body = ('\n'.join(f'- {x}' for x in defects)
                    if defects else '(no defects recorded)')
            return (
                'ruler-refuter',
                'Refuter returned FAIL on this ruler-touching diff. Address '
                f'these, then the gate re-evaluates the new diff:\n{body}',
                defects,
            )
        log(sdir, {
            'diff_hash': dhash, 'layer': 'deterministic',
            'check': 'ruler-front', 'result': 'PASS',
            'note': 'human-allow + refuter verdict', 'ruler': ruler, **meta,
        })
        return None

    if has_allow:
        log(sdir, {
            'diff_hash': dhash, 'layer': 'deterministic',
            'result': 'ALLOW', 'note': 'human-created allow file',
        })
        return None

    hard = check_test_integrity(proj) + check_cited_metrics(proj)
    if hard:
        return (
            'hard',
            'Deterministic gate failed (not model-judged):\n'
            + '\n'.join(f'- {x}' for x in hard)
            + '\n\nFix the diff. If this is a reviewed exception (justified '
            'test removal, derived figure), a HUMAN grants it by running:\n'
            f'  touch {allow_path}',
            hard,
        )

    return None
