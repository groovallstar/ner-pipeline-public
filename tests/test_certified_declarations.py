"""문서의 결과 출처 선언이 보관된 원본을 가리키는지 독립적으로 검사한다.

선언 문법은 `certified/README.md`를 따르며 에이전트 훅이나 플러그인을 읽지 않는다.
이 검사는 경로 존재와 경계만 확인하며 metric 수치나 재현성을 보증하지 않는다.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / 'docs'
CERT_ROOT = REPO_ROOT / 'certified'


CERT_SRC_RE = re.compile(r'<!--\s*certified:\s*([^\s>]+?)\s*-->')


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

    없으면 그 표는 대조 상대가 사라진 수치를 싣고 있는 것이다. 이 검사는 문서의 모든 선언을 읽어 원본이 사라진 시점에 실패한다.
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

    `../` 나 절대경로를 적어 `certified/` 바깥 파일을 출처로 삼지 못한다.
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
