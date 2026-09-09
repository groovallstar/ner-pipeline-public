"""매핑표 전수성 게이트와 표면별 판정의 경계.

이 테스트가 지키는 것은 "18 종이 전부 선언돼 있다" 가 아니라 **"선언되지 않은
타입이 조용히 지나가지 못한다"** 이다. 앞엣것만 지키면 원천 버전이 올라
19 번째 타입이 생겼을 때 그 타입이 통째로 사라진 채 통과한다. `FAC` 는 표면별
판정이라 같은 요구가 한 겹 더 있다 — **표에 없는 표면도 기본값을 못 받는다.**

끝의 선언 자리 검사는 "코드가 정본, 문서가 사본" 을 기계가 보게 한다. 사본이
여섯 군데라 손으로 맞추면 반드시 한 곳이 낡고, 낡은 사본은 사람이 그걸 읽고
판단하는 만큼 위험하다.
"""
import re
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.mapping import (
    BY_SURFACE,
    DROP,
    FAC_VERDICTS,
    ONTONOTES_TO_CANONICAL,
    PRODUCED_LABELS,
    FacCoverageError,
    UndeclaredTypeError,
    UnlistedSurfaceError,
    assert_exhaustive,
    assert_fac_coverage,
    is_dropped,
    resolve,
)

_ROOT = Path(__file__).resolve().parents[4]

# 표면별 판정 타입의 표본 — 표가 아니라 이 파일이 소유한다. 표에서 읽어
# 오면 표를 뒤집어도 기대치가 함께 뒤집혀 검사가 자기참조가 된다.
SURFACE_PROBE = {
    'the Golden Gate Bridge': 'ORG',
    'East Third Ring Road': 'LOC',
}

# 원천 `tner/ontonotes5` 가 내놓는 엔티티 타입 전수. 이 목록은 매핑 모듈이
# 아니라 여기에 둔다 — 모듈에서 가져오면 모듈을 고칠 때 기대치도 함께
# 움직여 검사가 자기참조가 된다.
SOURCE_TYPES = frozenset({
    'PERSON', 'GPE', 'LOC', 'ORG', 'FAC', 'PRODUCT', 'WORK_OF_ART',
    'EVENT', 'DATE', 'LAW', 'NORP', 'LANGUAGE', 'TIME', 'QUANTITY',
    'MONEY', 'PERCENT', 'ORDINAL', 'CARDINAL',
})

# 타입만으로 canonical 라벨이 정해지는 매핑. 근거 절은 매핑 모듈 주석에 있다.
# `FAC` 는 여기 없다 — 표면이 정하므로 타입 축에서는 답이 없다.
EXPECTED_KEPT = {
    'PERSON': 'PER',
    'GPE': 'LOC',
    'LOC': 'LOC',
    'ORG': 'ORG',
    'PRODUCT': 'PROD',
    'WORK_OF_ART': 'PROD',
    'EVENT': 'EVT',
    'DATE': 'DAT',
}

EXPECTED_BY_SURFACE = frozenset({'FAC'})


def test_every_source_type_is_declared():
    assert set(ONTONOTES_TO_CANONICAL) == SOURCE_TYPES
    assert len(SOURCE_TYPES) == 18


def test_kept_types_map_as_expected():
    for src, canonical in EXPECTED_KEPT.items():
        # 표면은 무시된다 — 타입만으로 답이 나오는 자리다.
        assert resolve(src, 'irrelevant surface') == canonical


def test_dropped_types_resolve_to_none():
    for src in SOURCE_TYPES - set(EXPECTED_KEPT) - EXPECTED_BY_SURFACE:
        assert resolve(src, 'irrelevant surface') is None
        assert ONTONOTES_TO_CANONICAL[src] == DROP


def test_is_dropped_answers_without_a_surface():
    """타입 통째로 버리는 자리와 표면이 정하는 자리가 갈려 있다.

    `is_dropped` 가 `FAC` 에 `True` 를 주면 표를 안 보고 버리는 경로가
    생긴다 — 표가 `ORG`·`LOC` 로 살린 1,093 span 이 통째로 사라지고,
    살아남은 것끼리는 여전히 다 맞아 대조 검사가 초록이다.
    """
    for src in SOURCE_TYPES - set(EXPECTED_KEPT) - EXPECTED_BY_SURFACE:
        assert is_dropped(src) is True
    for src in set(EXPECTED_KEPT) | EXPECTED_BY_SURFACE:
        assert is_dropped(src) is False
    with pytest.raises(UndeclaredTypeError):
        is_dropped('NEW_TYPE_FROM_A_LATER_RELEASE')


def test_by_surface_types_are_declared_as_such():
    """`FAC` 는 라벨도 `DROP` 도 아닌 `BY_SURFACE` 로 선언돼 있다.

    `DROP` 과 같은 값을 쓰면 "타입 통째로 버렸다" 와 "타입만으로는 못
    정한다" 가 구별되지 않아, 표를 안 거치고 지나가는 경로가 생긴다.
    """
    declared = {
        t for t, v in ONTONOTES_TO_CANONICAL.items() if v == BY_SURFACE
    }
    assert declared == EXPECTED_BY_SURFACE
    assert BY_SURFACE != DROP


def test_surface_decides_the_label_for_by_surface_types():
    """같은 타입이 표면에 따라 다른 라벨을 받는다 — 표본은 이 파일이 쥔다."""
    for surface, want in SURFACE_PROBE.items():
        assert resolve('FAC', surface) == want
    assert len(set(SURFACE_PROBE.values())) == 2, (
        'the probe must show both sides of the split'
    )


def test_an_unlisted_surface_gets_no_default():
    """표에 없는 표면은 기본 라벨로 흐르지 않고 즉시 터진다.

    기본값을 두면 규칙이 못 가른 몫이 조용히 한쪽으로 몰린다 — 전수 사전을
    두는 이유 자체가 그것이다.
    """
    with pytest.raises(UnlistedSurfaceError):
        resolve('FAC', 'a surface no corpus ever produced')


def test_coverage_gate_reports_both_directions():
    """피복 검사가 미판정과 죽은 줄을 둘 다 잡는다."""
    listed = next(iter(FAC_VERDICTS))
    with pytest.raises(FacCoverageError) as excinfo:
        assert_fac_coverage(['a surface no corpus ever produced'],
                            ('train', 'valid', 'test'))
    message = str(excinfo.value)
    assert 'a surface no corpus ever produced' in message
    # 표에만 있는 줄도 같은 예외로 보고된다.
    assert 'in the table but not the corpus' in message
    assert listed in FAC_VERDICTS


def test_produced_labels_are_the_six_kept():
    """표의 판정값도 라벨을 내므로 `PRODUCED_LABELS` 가 그쪽까지 걷는다."""
    assert PRODUCED_LABELS == frozenset(
        {'PER', 'LOC', 'ORG', 'PROD', 'EVT', 'DAT'}
    )
    assert frozenset(EXPECTED_KEPT.values()) <= PRODUCED_LABELS
    assert {v for v in FAC_VERDICTS.values() if v != DROP} <= PRODUCED_LABELS


def test_undeclared_type_stops_conversion():
    """새 타입이 생기면 조용히 드롭되지 않고 즉시 터진다."""
    with pytest.raises(UndeclaredTypeError):
        resolve('NEW_TYPE_FROM_A_LATER_RELEASE', 'any surface')


def test_exhaustiveness_gate_reports_all_undeclared():
    with pytest.raises(UndeclaredTypeError) as excinfo:
        assert_exhaustive(SOURCE_TYPES | {'ALPHA', 'BETA'})
    message = str(excinfo.value)
    assert 'ALPHA' in message and 'BETA' in message


def test_exhaustiveness_gate_passes_on_known_types():
    assert_exhaustive(SOURCE_TYPES)


SCHEMA_PATH = _ROOT / 'docs' / 'manual' / 'data' / 'canonical-entity-schema.md'


def _section_43() -> str:
    schema = SCHEMA_PATH.read_text(encoding='utf-8')
    body = schema[schema.index('### 4.3 EN'):]
    # 절의 끝은 다음 `## ` 다 — 하위 heading(`#### `)을 끝으로 삼으면 그
    # heading 이 사라질 때 표가 아니라 파서가 먼저 깨진다.
    return body[:body.index('\n## ', 1)]


def test_schema_section_43_matches_this_table():
    """기준 파일 §4.3 의 매핑표가 코드와 갈리지 않는다.

    §4.3 은 이 표의 사본이다 — 사람이 "이 span 이 왜 ORG 인가" 를 되짚을 때
    읽는 자리이고, 기준 파일이라 바꾸려면 사람 승인과 반박자를 지나야 한다.
    사본이 조용히 낡으면 그 승인 절차가 낡은 표를 지키게 된다.

    지금까지 이 대조는 반박자가 라운드마다 **손으로** 했다. 손으로 하는 검사는
    누가 안 보면 그냥 지나간다.
    """
    body = _section_43()

    declared: dict[str, str] = {}
    for row in re.findall(r'^\|(.+?)\|(.+?)\|', body, re.M):
        raw, target = (c.strip() for c in row)
        if raw.startswith('---') or '원본 OntoNotes' in raw:
            continue
        types = re.findall(r'`([A-Z_]+)`', raw)
        if not types:
            continue  # "(OntoNotes 미커버)" 행 — PII 주입이 채우는 자리
        # 순서가 중요하다 — 표면별 행은 `ORG`·`LOC`·비-entity 를 한 칸에
        # 함께 적으므로, 라벨이나 "비-entity" 를 먼저 보면 그 행이 단일
        # 라벨 행으로 읽힌다.
        if '표면별' in target:
            canonical = BY_SURFACE
        elif '비-entity' in target:
            canonical = DROP
        else:
            label = re.findall(r'`([A-Z]+)`', target)
            canonical = label[0] if label else DROP
        for source_type in types:
            declared[source_type] = canonical

    assert declared == ONTONOTES_TO_CANONICAL, (
        '§4.3 과 mapping.py 가 갈렸다 — '
        f'문서만: {set(declared.items()) - set(ONTONOTES_TO_CANONICAL.items())} / '
        f'코드만: {set(ONTONOTES_TO_CANONICAL.items()) - set(declared.items())}'
    )


# ── 선언 자리 ─────────────────────────────────────────────────────────
#
# 인프라 경계를 선언하는 산문이 사본으로 흩어져 있다. 정본은 코드
# (`mapping.py` 의 표 + `fac_labels.json`)이고 나머지는 사람이 읽는 사본이라,
# 낡으면 사람이 낡은 규칙으로 판단한다. 줄번호가 아니라 **경로 + 문구**로
# 걸어 두 가지를 본다 — 분할 문장이 있는가, 낡은 문구가 남았는가.
#
# 프롬프트(`labelers/en/ner_prompts.py`)의 세 자리는
# `tests/ner/labelers/test_en_labeler.py` 가 같은 방식으로 본다. 라벨러 쪽
# 검사가 그 파일에 모여 있어서다.

SPLIT_SENTENCE = '개별 구조물은 `ORG`, 여러 지점을 잇는 경로는 `LOC`'

# 자리마다 그 자리에 실제로 있던 분할 이전 선언. **자리별로 묶는 것이
# 요점이다** — 한 배열로 두면 어느 자리에 감시가 붙어 있는지가 안 보이고,
# 백틱 한 쌍이 어긋난 유령 문자열이 섞여도 목록이 길어 보여 넘어간다.
# 실제로 그렇게 한 번 새어, `mapping.py` docstring 과 §4.3 두 자리가 감시
# 없이 지나갔다. 자리별로 적으면 빈 자리가 이 상수에서 바로 보인다.
#
# 문구는 **원문에서 한 줄 안에 이어지는 조각**으로 적는다. 아래 이력 검사가
# `git log -S` 로 원문을 훑는데, 줄바꿈을 건너뛴 문구는 이력에서 안 잡혀
# 유령과 구별되지 않는다.
STALE_PHRASES: dict[str, tuple[str, ...]] = {
    'src/ner/augmenters/ontonotes_en/mapping.py': (
        'EN 은 **JA·VI 관례를 따른다**',
        '(공항·역·경기장·다리·고속도로)를 ORG 로',
    ),
    'docs/manual/data/canonical-entity-schema.md': (
        '`FAC` 를 전량 ORG 로 태운다',
    ),
    'docs/manual/pipeline/README.md': (
        '`FAC`→`ORG` 경계는 고른 것이라',
    ),
}

# 삭제된 지침의 문구도 현행 선언에 다시 들어오지 않도록 계속 감시한다.
RETIRED_SITE_PHRASES = (
    '`FAC`(공항·역·경기장·다리·고속도로)를 `ORG` 로 흡수한다',
    'JA·VI 관례(`FAC`→`ORG`)를 따르고',
)

DECLARATION_SITES = tuple(STALE_PHRASES)


def _normalized(path: str) -> str:
    """줄바꿈을 지운 본문 — 문구가 어디서 접히든 같은 문자열로 읽힌다."""
    return ' '.join((_ROOT / path).read_text(encoding='utf-8').split())


def test_every_declaration_site_states_the_split():
    """현행 구현과 문서가 분할을 같은 문장으로 선언한다."""
    for path in DECLARATION_SITES:
        assert SPLIT_SENTENCE in _normalized(path), (
            f'{path} does not declare the infrastructure split'
        )


def test_no_declaration_site_keeps_the_pre_split_wording():
    """낡은 선언이 한 자리에도 남아 있지 않다.

    새 문장을 더하기만 하고 옛 문장을 안 지우면 한 파일이 두 규칙을 함께
    선언한다 — 읽는 사람이 어느 쪽을 따를지는 어디를 먼저 읽었는지가 정한다.

    금지 문구는 **자기 자리에서만** 검사하지 않는다. 같은 문장이 다른 자리로
    옮겨 붙는 것도 같은 결함이라 전 자리에 교차로 건다.
    """
    for path in DECLARATION_SITES:
        body = _normalized(path)
        for phrases in (*STALE_PHRASES.values(), RETIRED_SITE_PHRASES):
            for stale in phrases:
                assert stale not in body, f'{path} still says {stale!r}'


def test_every_site_has_at_least_one_stale_phrase_watching_it():
    """감시가 빈 자리가 없다.

    위 검사는 목록에 없는 자리를 조용히 지나간다 — 금지 목록이 무력해지는
    길은 문구를 지우는 것이 아니라 **자리를 안 적는 것**이다.
    """
    for path in DECLARATION_SITES:
        assert STALE_PHRASES[path], f'{path} has no pre-split wording watched'


def test_no_stale_phrase_states_the_split():
    """금지 문구가 새 선언을 담지 않는다 — 담으면 양성 검사와 충돌한다."""
    for phrases in (*STALE_PHRASES.values(), RETIRED_SITE_PHRASES):
        for stale in phrases:
            assert SPLIT_SENTENCE not in stale


def test_every_stale_phrase_really_stood_in_that_file():
    """금지 문구가 **그 파일에 실제로 있던 문장**인지 이력에서 확인한다.

    아무 데도 없던 문자열을 넣어 두면 금지 검사가 늘 통과한다 — 조용히
    무력해지는 흔한 길이고, 실제로 한 번 새었다. 백틱 한 쌍이 빠진 문구가
    §4.3 을 겨냥한 줄 알았지만 원문은 `` `FAC` 를 전량 ORG 로 태운다 `` 라
    부분문자열이 안 걸렸고, 그래서 그 자리는 감시 없이 지나갔다.

    현재 파일 내용으로는 확인할 수 없다 — 금지 문구는 지금 없어야 정상이다.
    확인할 수 있는 곳은 이력뿐이라 `git log -S` 로 그 문구의 등장 횟수를
    바꾼 커밋이 그 파일에 있는지 본다. 이력이 없는 환경(얕은 클론·아카이브)
    에서는 건너뛴다 — `tests/ner/AGENTS.md` 의 "로컬 자원 부재 → skip" 관례다.
    """
    import subprocess

    try:
        subprocess.run(
            ['git', 'rev-parse', '--verify', 'HEAD'],
            cwd=_ROOT, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('no git history in this checkout')

    for path, phrases in STALE_PHRASES.items():
        for stale in phrases:
            found = subprocess.run(
                ['git', 'log', '--oneline', '-1', f'-S{stale}', '--', path],
                cwd=_ROOT, capture_output=True, text=True,
            )
            assert found.stdout.strip(), (
                f'{path}: no commit ever added or removed {stale!r} — '
                f'the phrase is a ghost and watches nothing'
            )
