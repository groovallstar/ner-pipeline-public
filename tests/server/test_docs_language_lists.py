"""문서·주석의 언어 열거가 서버 상수와 어긋나면 실패시킨다.

서버가 지원하는 언어 목록은 코드 상수 하나인데, 그것을 서술하는 열거가
문서·docstring·주석·웹 UI 에 수십 벌 복제돼 있다. 복제본이 낡아도 증상이
에러가 아니라 "안내 문구가 거짓말" 이라 조용하다. 그래서 훑기 범위에서
발견되는 **모든 3 원소 이상 언어 열거**를 서버 상수에서 유도한 집합과
대조한다. 금지 목록이 아니라 유도 대조다 — 언어를 늘리면 상수만 고쳐도
검사가 따라 넓어진다.

**이 파일은 훑기 대상에서 스스로를 뺀다.** 아래 정규식의 원문(`ja|ko|vi|en`
교대)이 이 파일 안에 있어 자기 자신에 걸리기 때문이다.

**하한은 3 원소다.** `ja·ko 가 한자를 공유한다` 같은 영구히 참인 2 원소
서술이 여러 자리 있고 그것들은 어떤 유도 집합과도 같아질 일이 없다. 대가는
비대칭한 실패다 — 언어를 늘리면 3 원소 이상 자리는 시끄럽게 FAIL 하지만
2 원소 자리는 조용히 낡는다. 그래서 언어를 추가할 때 2 원소 열거 자리를
손으로 훑는 절차를 `src/server/CLAUDE.md` 에 못 박아 뒀다.

**걸린 자리를 푸는 방법은 셋뿐이다.** ① 열거를 완성한다 ② 2 단 감지 정책은
신호→언어 화살표 사슬로 다시 쓴다 ③ 지원 집합의 참인 진부분집합은 목록이
아니라 술어로 쓴다(`나머지 언어`). 아래 토큰 목록을 좁혀서 푸는 것은 자를
바꿔 통과시키는 일이라 금지다.
"""

import re
from pathlib import Path

import server
from server.config import SUPPORTED_LANGS
from server.translate import LANG_NAME, TRANSLATABLE_LANGS

_ROOT = Path(server.__file__).resolve().parent.parent.parent

# 한국어 이름표 — 키는 서버 상수에서 유도하고 값만 리터럴이다. ko 는
# 번역 대상이 아니라 `LANG_NAME` 에 없으므로 이름표 자체는 이 파일이 쥔다.
_LANG_KO_NAME = {'ja': '일본어', 'ko': '한국어', 'vi': '베트남어', 'en': '영어'}

_CODE = '(?:' + '|'.join(sorted(_LANG_KO_NAME)) + ')'
_NM = '(?:' + '|'.join(sorted(_LANG_KO_NAME.values())) + ')'
# 구분자에 공백을 넣지 않는다 — 넣으면 마크다운 표의 칸 구분(`| ja | ko |`)과
# 파이썬 튜플(`('ja', 'vi')`)이 열거로 잡힌다.
_SEP = r'(?:·|/|,|\\?\||\+)'

# 순서가 있다 — 이름+코드를 먼저 잡고 그 자리를 지운 뒤 나머지 셋을 본다.
_PATTERNS = (
    ('name+code', re.compile(rf'{_NM}\({_CODE}\)(?:·{_NM}\({_CODE}\)){{2,}}')),
    ('code+name', re.compile(rf'{_CODE}\({_NM}\)(?:·{_CODE}\({_NM}\)){{2,}}')),
    ('name', re.compile(rf'{_NM}(?:·{_NM}){{2,}}')),
    ('code', re.compile(
        rf'(?<![A-Za-z]){_CODE}(?:{_SEP}{_CODE}){{2,}}(?![A-Za-z])')),
)

_NAME_TO_CODE = {v: k for k, v in _LANG_KO_NAME.items()}

# 훑기가 조용히 비거나 좁아지는 것을 막는 sentinel. 수집 갈래가 넷이고
# 그중 명시 목록 갈래만 둘로 대표한다 — 소비자 계약 문서와 저장소 루트는
# 성격이 달라 한쪽만 빠지는 실수가 따로 난다.
#
# 재귀 대표를 하위 디렉토리 파일로 두는 것이 중요하다. `src/server/` 바로
# 아래 파일을 쓰면 `rglob` 을 `glob` 으로 바꿔도 sentinel 이 살아남아,
# `static/`·`scripts/` 가 통째로 빠진 채 검사가 초록이 된다.
_SENTINELS = (
    'src/server/static/index.html',      # 재귀 하위 디렉토리
    'docker/server/Dockerfile',          # 확장자 없는 파일
    'docs/manual/rest-api-spec.md',      # 명시 목록 — 소비자 계약 문서
    'README.md',                         # 명시 목록 — 저장소 루트
    'tests/server/test_api.py',          # 별도 glob
)


def _scan_files():
    """훑기 범위의 파일 경로를 모은다(저장소 루트 기준 상대경로 정렬)."""
    found = set()
    for path in (_ROOT / 'src' / 'server').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.html', '.md'):
            found.add(path)
    for path in (_ROOT / 'tests' / 'server').glob('*.py'):
        if path.name != Path(__file__).name:   # 자기 자신은 뺀다(모듈 docstring)
            found.add(path)
    for path in (_ROOT / 'docker' / 'server').iterdir():
        if path.is_file() and (path.suffix in ('.md', '.yml')
                               or path.name == 'Dockerfile'):
            found.add(path)
    for rel in ('docs/manual/rest-api-spec.md',
                'docs/manual/rest-api-integration-guide.md',
                'docs/manual/language-detection.md',
                'CLAUDE.md', 'README.md', 'docker/CLAUDE.md',
                'pyproject.toml'):
        path = _ROOT / rel
        if path.is_file():
            found.add(path)
    return sorted(found)


def _flatten(raw):
    """백틱을 지우고 구분자에 걸친 줄바꿈을 접되 원문 오프셋을 함께 남긴다.

    접기를 문자열 치환으로만 하면 줄바꿈이 사라져 매치 위치의 앞 줄바꿈을
    세는 방식이 행 번호를 틀리게 준다. 실패 메시지의 `파일:행` 이 이 검사의
    쓸모이므로, 남긴 글자마다 원문 인덱스를 기록해 그 인덱스로 행을 센다.
    백틱은 **제거**지 공백 치환이 아니다 — 공백으로 바꾸면 `ja`\\|`ko`\\|`vi`
    가 구분자 좌우에 공백을 얻어 검사에서 빠진다.
    """
    kept = [(ch, i) for i, ch in enumerate(raw) if ch != '`']
    out, n = [], len(kept)
    i = 0
    while i < n:
        ch, idx = kept[i]
        if ch == '\n':
            j = i + 1
            while j < n and kept[j][0] in ' \t':
                j += 1
            prev_is_sep = bool(out) and re.fullmatch(_SEP, out[-1][0])
            next_is_sep = j < n and re.fullmatch(_SEP, kept[j][0])
            if prev_is_sep or next_is_sep:
                i = j                      # 줄바꿈과 다음 줄 들여쓰기를 버린다
                continue
        out.append((ch, idx))
        i += 1
    return ''.join(c for c, _ in out), [i for _, i in out]


def _langs_of(match: str):
    """매치 문자열이 담은 언어 코드 집합."""
    codes = set(re.findall(_CODE, match))
    if codes:
        return codes
    return {_NAME_TO_CODE[n] for n in re.findall(_NM, match)}


def _enumerations(raw):
    """한 파일에서 3 원소 이상 언어 열거를 전수로 뽑는다."""
    text, offsets = _flatten(raw)
    hits = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            line = raw[:offsets[m.start()]].count('\n') + 1
            hits.append((line, kind, m.group(0), _langs_of(m.group(0))))
        # 잡은 자리를 공백으로 덮어 뒤 패턴이 같은 자리를 다시 세지 않게 한다.
        text = pattern.sub(lambda m: ' ' * len(m.group(0)), text)
    return hits


def test_every_language_enumeration_matches_a_server_list():
    """발견된 모든 언어 열거가 서버 상수에서 유도한 집합과 같은지.

    두 집합을 허용하는 것은 번역쌍이 정당한 열거이기 때문이고, 둘 다 서버
    상수에서 유도되므로 어느 쪽도 조용히 낡지 않는다.
    """
    assert set(_LANG_KO_NAME) == set(SUPPORTED_LANGS)
    assert all(_LANG_KO_NAME[k] == v for k, v in LANG_NAME.items())

    files = _scan_files()
    rel = {str(p.relative_to(_ROOT)) for p in files}
    for sentinel in _SENTINELS:
        assert sentinel in rel, f'훑기 범위에서 {sentinel} 이 빠졌다'

    allowed = (set(SUPPORTED_LANGS), set(TRANSLATABLE_LANGS))
    matches, bad = [], []
    for path in files:
        raw = path.read_text(encoding='utf-8')
        for line, kind, text, langs in _enumerations(raw):
            matches.append((path, line))
            if langs not in allowed:
                bad.append(
                    f'{path.relative_to(_ROOT)}:{line} [{kind}] {text!r} '
                    f'-> {sorted(langs)}')
    assert matches, '열거를 하나도 못 찾았다 — 훑기 범위가 비었을 수 있다'
    assert not bad, (
        '서버 상수와 어긋나는 언어 열거:\n  ' + '\n  '.join(bad)
        + f'\n허용 집합: {sorted(allowed[0])} 또는 {sorted(allowed[1])}')
