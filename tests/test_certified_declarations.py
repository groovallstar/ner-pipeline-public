"""문서의 출처 선언이 원장에 실제로 있는 경로를 가리키는가.

커밋 게이트의 인용 대조는 그 커밋의 diff 안에서만 돈다. 그래서 원장
디렉토리를 지우면서 문서를 하나도 안 건드린 커밋은 게이트가 통째로
지나가고, 가리킬 것이 없어진 `<!-- certified: ... -->` 선언만 문서에 남는다.
그 상태는 조용하다 — 그 표에 새 수치를 안 적는 한 아무도 안 밟는다. 여기서
저장소의 문서를 전수로 훑어 매달린 선언을 붙잡는다.

**선언 문법은 게이트가 정본이라 정규식을 그쪽에서 가져온다.** 여기에 같은
정규식을 한 벌 더 적으면, 게이트가 문법을 넓혔을 때 이 검사만 옛 문법에
남아 새 형태의 선언을 못 보게 된다.
"""
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / 'docs'
CERT_ROOT = REPO_ROOT / 'certified'


def _load_gate_core():
    """`.claude/hooks/` 는 설치 패키지가 아니라 훅 스크립트 디렉토리라
    import 경로에 없다 — 파일 경로로 직접 적재한다."""
    path = REPO_ROOT / '.claude/hooks/gate_core.py'
    spec = importlib.util.spec_from_file_location('gate_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CERT_SRC_RE = _load_gate_core().CERT_SRC_RE


def _declarations():
    """`docs/**` 의 모든 출처 선언을 (문서, 줄번호, 선언 경로)로 편다."""
    found = []
    for doc in sorted(DOC_ROOT.rglob('*.md')):
        for lineno, line in enumerate(
            doc.read_text(encoding='utf-8').splitlines(), start=1
        ):
            for m in CERT_SRC_RE.finditer(line):
                found.append((doc.relative_to(REPO_ROOT), lineno, m.group(1)))
    return found


def test_the_sweep_actually_reads_documents_and_declarations():
    """전수 검사가 빈 손으로 통과하지 않는지부터 본다.

    아래 검사는 "찾은 선언이 전부 유효하다"는 형태라, 하나도 못 찾으면
    가장 조용하게 통과한다. 문서 경로가 옮겨지거나 정규식이 어긋나는 순간
    이 파일은 아무것도 안 보면서 초록이 되므로, 커버리지 0 을 먼저 막는다.
    """
    assert DOC_ROOT.is_dir(), f'docs root missing: {DOC_ROOT}'
    assert list(DOC_ROOT.rglob('*.md')), 'no markdown found under docs/'
    assert _declarations(), (
        'no `<!-- certified: ... -->` declaration found under docs/ — the '
        'syntax moved or the sweep is looking in the wrong place'
    )


def test_every_declared_source_exists_in_the_ledger():
    """선언한 경로가 원장에 파일이나 디렉토리로 있어야 한다.

    없으면 그 표는 대조 상대가 사라진 수치를 싣고 있는 것이다. 게이트는
    그 표에 새 수치가 실릴 때만 이것을 말하므로, 사라진 시점에 여기서
    말한다.
    """
    dangling = [
        f'{doc}:{lineno} -> certified/{src}'
        for doc, lineno, src in _declarations()
        if not (CERT_ROOT / src).exists()
    ]
    assert not dangling, (
        'declared source missing from certified/:\n  '
        + '\n  '.join(dangling)
        + '\nEither promote the run, or repoint/remove the declaration in '
        'the same commit that removed the ledger entry.'
    )


def test_declared_sources_stay_inside_the_ledger():
    """선언이 원장 밖을 가리키지 못한다.

    `../` 나 절대경로를 적으면 게이트의 `_catalog_of` 가 원장 밖 파일을
    카탈로그로 읽어, 대조가 `certified/` 를 벗어난다.
    """
    escaping = []
    for doc, lineno, src in _declarations():
        rel = re.sub(r'^certified/', '', src)
        target = (CERT_ROOT / rel).resolve()
        if not target.is_relative_to(CERT_ROOT.resolve()):
            escaping.append(f'{doc}:{lineno} -> {src}')
    assert not escaping, (
        'declaration points outside certified/:\n  ' + '\n  '.join(escaping)
    )
