"""OntoNotes5 원본 18 타입 → canonical 라벨 매핑과 전수성 게이트.

이 표의 대부분은 새로 정한 것이 아니라 canonical 규칙이 이미 강제하던 것을
적은 것이고, 각 행에 근거 절 번호를 남긴다. **다만 한 가지는 고른 것이라
스키마 문서에도 함께 적었다** — 아래 LOC/ORG 경계이며, 그래서
`docs/manual/data/canonical-entity-schema.md` 에 §4.3 이 신설됐다.

**정본은 이 표이고 §4.3 은 근거를 적은 사본이다.** 둘이 갈리면
`tests/.../test_mapping.py` 의 동기화 검사가 잡는다 — 한쪽만 고치면 붉어진다.

**LOC/ORG 경계는 언어마다 다르므로 EN 이 어느 쪽인지 여기서 선언한다.**
10 종 평면 목록만 언어 공통이고 경계는 갈린다 — JA·VI 는 인공 시설을 모두
ORG 로 흡수하는 반면 KO 는 narrow-ORG 로 좁혀 인공 시설을 아예 버린다.
EN 은 canonical §3 인프라 규칙을 따라 원본 `FAC` 를 둘로 가른다 —
**개별 구조물은 `ORG`, 여러 지점을 잇는 경로는 `LOC`** 이며, 앞은
역·공항·경기장·교량·터널이고 뒤는 철도·지하철 노선·도로·고속도로다.
원본이 둘을 한 타입에 담고 있어 타입만 보고는 못 가르므로 판정이
**표면별**이다.

**`FAC` 판정의 정본은 표(`data/fac_labels.json`)이고 규칙이 아니다.**
규칙을 판정자로 쓰면 규칙이 못 가른 몫이 조용히 기본값으로 흐른다 —
실측으로 어휘 규칙은 634 표면 중 391 만 가르고 나머지에 번호 도로(`288`)·
접미 없는 철도 노선명(`the Beijing - Kowloon`)·지하철 시스템(`MRT`)이 전부
들어 있다. 그래서 표면을 전량 열거하고, 표에 없는 표면은 기본값 없이
`UnlistedSurfaceError` 로 변환을 세운다. 어휘 목록은 판정자가 아니라
**검사자**로 `tests/.../test_fac_labels.py` 에만 있다.

전수성 게이트 — 원본이 내놓는 **모든** 타입은 canonical 라벨로 가거나
`DROP`(타입 통째로) 또는 `BY_SURFACE`(표면별 판정)로 선언돼야 한다.
선언되지 않은 타입을 만나면 변환이 즉시 중단된다(`resolve`). 원천 버전이
올라 19 번째 타입이 생겨도 조용히 사라지지 않게 하려는 것이다 — 조용히
사라지면 그 타입이 통째로 빠진 것을 아무도 보지 못한다.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

# 명시적 드롭 — canonical 10 종 평면에 자리가 없거나 규칙이 비-entity 로
# 판정한 타입. 누락과 구별하기 위해 값으로 남긴다.
DROP = 'DROP'

# 표면별 판정 — 타입 하나가 canonical 라벨 여럿으로 갈린다. `DROP` 과
# 구별해 두는 것은 "타입 통째로 버렸다" 와 "타입만으로는 못 정한다" 가
# 다른 사실이기 때문이다. 값이 같으면 표 없이 지나가는 경로가 생긴다.
BY_SURFACE = 'BY_SURFACE'

# 원본 18 타입 전수. 값은 canonical 라벨이거나 DROP·BY_SURFACE 다.
ONTONOTES_TO_CANONICAL: dict[str, str] = {
    # ── canonical 로 살아남는 타입 ──────────────────────────────────
    'PERSON': 'PER',        # §1 인물
    'GPE': 'LOC',           # §2.2 순위 2 — 국가·행정구역
    'LOC': 'LOC',           # §2.2 순위 2 — 자연지명
    'ORG': 'ORG',           # §2.2 순위 1 — 조직·법인·공권력
    'PRODUCT': 'PROD',      # §1 — 시판 물품
    'WORK_OF_ART': 'PROD',  # §1 — 창작 작품. PROD 가 둘을 한 칸에 담는다
    'EVENT': 'EVT',         # §1 행사·사건
    'DATE': 'DAT',          # §1 — 모든 날짜
    # ── 표면별로 갈리는 타입 ────────────────────────────────────────
    'FAC': BY_SURFACE,  # §3 인프라 — 구조물 ORG · 경로 LOC. 표가 정한다
    # ── 드롭 ────────────────────────────────────────────────────────
    'LAW': DROP,        # §3.1 법령·법률·법안·규정·칙유 = 비-entity
    'NORP': DROP,       # §2.3 국적·언어·계통 복합어 = 비-LOC
    'LANGUAGE': DROP,   # §2.3 같은 근거 (언어명)
    'TIME': DROP,       # KO 가 KLUE TI 를 드롭한 것과 동형
    'QUANTITY': DROP,   # KO 가 KLUE QT 를 드롭한 것과 동형
    'MONEY': DROP,      # 수량 표현 — canonical 에 자리 없음
    'PERCENT': DROP,    # 〃
    'ORDINAL': DROP,    # 〃
    'CARDINAL': DROP,   # 〃
}

FAC_TABLE_PATH = Path(__file__).resolve().parent / 'data' / 'fac_labels.json'

_TABLE = json.loads(FAC_TABLE_PATH.read_text(encoding='utf-8'))

# 표면 → 판정(`ORG`·`LOC`·`DROP`). 표면 키는 **원본 토큰의 공백 조인**이라
# 자연문 복원 규칙이 바뀌어도 안 흔들린다.
FAC_VERDICTS: dict[str, str] = {
    entry['surface']: entry['verdict'] for entry in _TABLE['entries']
}

# 표면 → split 별 등장 수. 피복 검사가 `--splits` 부분집합에서도 정확하려면
# 표의 어느 줄이 그 split 에 걸려 있는지를 알아야 한다.
FAC_OCCURRENCES: dict[str, dict[str, int]] = {
    entry['surface']: dict(entry['occurrences']) for entry in _TABLE['entries']
}

# canonical 라벨 중 이 변환이 만들어낼 수 있는 것. PII 4 종
# (EMAIL/PHONE/ID_NUM/CREDIT_CARD)은 원본에 없고 주입 단계가 채운다.
# `FAC` 는 타입이 아니라 표의 판정값이 라벨을 내므로 그쪽에서도 걷는다.
PRODUCED_LABELS: frozenset[str] = frozenset(
    v for v in ONTONOTES_TO_CANONICAL.values()
    if v not in (DROP, BY_SURFACE)
) | frozenset(v for v in FAC_VERDICTS.values() if v != DROP)


class UndeclaredTypeError(ValueError):
    """매핑표에 선언되지 않은 원본 타입을 만났다."""


class UnlistedSurfaceError(ValueError):
    """표면별 판정 타입인데 그 표면이 판정 표에 없다."""


class FacCoverageError(ValueError):
    """판정 표와 원본 `FAC` 표면 집합이 어긋났다."""


def is_dropped(ontonotes_type: str) -> bool:
    """이 타입이 **표면과 무관하게** 통째로 드롭인가.

    `resolve` 와 갈라 두는 것은 표면 없이 답해야 하는 자리가 있기 때문이다 —
    `LAW`·`MONEY` 처럼 타입만으로 끝나는 span 까지 표면 문자열을 만들 이유가
    없고, 만들면 "표면이 필요하다" 와 "타입으로 끝난다" 가 호출부에서
    구별되지 않는다. 표면별 판정 타입(`FAC`)은 여기서 `False` 다 — 버릴지는
    표가 정하지 이 술어가 정하지 않는다.
    """
    return _declared(ontonotes_type) == DROP


def _declared(ontonotes_type: str) -> str:
    try:
        return ONTONOTES_TO_CANONICAL[ontonotes_type]
    except KeyError:
        raise UndeclaredTypeError(
            f'undeclared OntoNotes type: {ontonotes_type!r}. '
            f'Every source type must map to a canonical label, DROP or '
            f'BY_SURFACE in ONTONOTES_TO_CANONICAL.'
        ) from None


def resolve(ontonotes_type: str, surface: str) -> str | None:
    """원본 span 하나를 canonical 라벨로 옮긴다. 드롭이면 `None`.

    `surface` 는 **원본 토큰의 공백 조인**이며 표면별 판정 타입에서만
    쓰인다. 선택 인자로 두지 않는 것이 요점이다 — 기본값을 주면 표를 안
    거치는 호출이 조용히 생기고, 그 호출은 표가 무슨 판정을 했든 같은 답을
    받는다.

    두 자리에서 기본값 없이 중단한다. 선언되지 않은 타입은
    `UndeclaredTypeError`, 표에 없는 표면은 `UnlistedSurfaceError` 다.
    미판정을 기본 라벨로 흘리면 판정 표를 두는 이유 자체가 사라진다.
    """
    target = _declared(ontonotes_type)
    if target == DROP:
        return None
    if target != BY_SURFACE:
        return target
    try:
        verdict = FAC_VERDICTS[surface]
    except KeyError:
        raise UnlistedSurfaceError(
            f'{ontonotes_type} surface not in the verdict table: '
            f'{surface!r}. Add it to {FAC_TABLE_PATH.name} with a verdict, '
            f'reason and occurrence counts — there is no default.'
        ) from None
    return None if verdict == DROP else verdict


def listed_fac_surfaces(splits: Iterable[str]) -> set[str]:
    """주어진 split 에 등장한다고 표가 적은 표면 집합."""
    wanted = tuple(splits)
    return {
        surface for surface, occurrences in FAC_OCCURRENCES.items()
        if any(occurrences.get(split, 0) for split in wanted)
    }


def assert_fac_coverage(
    observed_surfaces: Iterable[str], splits: Iterable[str],
) -> None:
    """원본 `FAC` 표면 집합과 표의 표면 집합을 **양쪽 방향으로** 맞춘다.

    한쪽만 보면 절반을 놓친다 — 원본에만 있으면 미판정이고(그쪽은 `resolve`
    가 span 단위로 이미 세운다), 표에만 있으면 표가 코퍼스와 갈렸다는 뜻이라
    등장 수·잔여 수 같은 표의 다른 주장까지 함께 못 믿게 된다. 어느 쪽이든
    변환을 세운다.
    """
    splits = tuple(splits)
    observed = set(observed_surfaces)
    listed = listed_fac_surfaces(splits)
    unjudged = sorted(observed - listed)
    stale = sorted(listed - observed)
    if unjudged or stale:
        raise FacCoverageError(
            f'FAC verdict table does not match the corpus for '
            f'splits={list(splits)}: {len(unjudged)} surface(s) in the '
            f'corpus but not the table {unjudged[:10]}, {len(stale)} in the '
            f'table but not the corpus {stale[:10]}.'
        )


def assert_exhaustive(observed_types: Iterable[str]) -> None:
    """관측된 원본 타입이 전부 선언돼 있는지 확인한다.

    `resolve` 가 행 단위로 거는 것과 같은 게이트를 코퍼스 전체 라벨 목록에
    대해 미리 한 번 건다. 변환을 오래 돌린 뒤에 마지막 문장에서 터지는 것보다
    시작 전에 터지는 편이 싸다.
    """
    undeclared = sorted(set(observed_types) - set(ONTONOTES_TO_CANONICAL))
    if undeclared:
        raise UndeclaredTypeError(
            f'undeclared OntoNotes types: {undeclared}. '
            f'Add each one to ONTONOTES_TO_CANONICAL as a canonical '
            f'label, DROP or BY_SURFACE.'
        )
