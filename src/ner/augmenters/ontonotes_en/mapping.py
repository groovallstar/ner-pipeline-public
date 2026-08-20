"""OntoNotes5 원본 18 타입 → canonical 라벨 매핑과 전수성 게이트.

이 표의 대부분은 새로 정한 것이 아니라 canonical 규칙이 이미 강제하던 것을
적은 것이고, 각 행에 근거 절 번호를 남긴다. **다만 한 가지는 고른 것이라
스키마 문서에도 함께 적었다** — 아래 LOC/ORG 경계이며, 그래서
`docs/manual/data/canonical-entity-schema.md` 에 §4.5 가 신설됐다.

**정본은 이 표이고 §4.5 는 근거를 적은 사본이다.** 둘이 갈리면
`tests/.../test_mapping.py` 의 동기화 검사가 잡는다 — 한쪽만 고치면 붉어진다.

**LOC/ORG 경계는 언어마다 다르므로 EN 이 어느 쪽인지 여기서 선언한다.**
10 종 평면 목록만 언어 공통이고 경계는 갈린다 — JA·VI 는 인공 시설을 모두
ORG 로 흡수하는 반면 KO 는 narrow-ORG 로 좁혀 인공 시설을 아예 버린다.
EN 은 **JA·VI 관례를 따른다**: `FAC`(공항·역·경기장·다리·고속도로)를 ORG 로
흡수한다. 이건 물려받은 기본값이 아니라 선택이므로, 나중에 "이 span 이 왜
ORG 인가" 를 되짚을 근거가 이 문단이다.

전수성 게이트 — 원본이 내놓는 **모든** 타입은 canonical 라벨로 가거나
`DROP` 으로 선언돼야 한다. 선언되지 않은 타입을 만나면 변환이 즉시
중단된다(`resolve`). 원천 버전이 올라 19 번째 타입이 생겨도 조용히
사라지지 않게 하려는 것이다 — 조용히 사라지면 그 타입이 통째로 빠진 것을
아무도 보지 못한다.
"""
from __future__ import annotations

from collections.abc import Iterable

# 명시적 드롭 — canonical 10 종 평면에 자리가 없거나 규칙이 비-entity 로
# 판정한 타입. 누락과 구별하기 위해 값으로 남긴다.
DROP = 'DROP'

# 원본 18 타입 전수. 값은 canonical 라벨이거나 DROP 이다.
ONTONOTES_TO_CANONICAL: dict[str, str] = {
    # ── canonical 로 살아남는 타입 ──────────────────────────────────
    'PERSON': 'PER',        # §1 인물
    'GPE': 'LOC',           # §2.2 순위 2 — 국가·행정구역
    'LOC': 'LOC',           # §2.2 순위 2 — 자연지명
    'ORG': 'ORG',           # §2.2 순위 1 — 조직·법인·공권력
    'FAC': 'ORG',           # §2.3 — 인공 시설(공항·역·경기장·병원)은 ORG
    'PRODUCT': 'PROD',      # §1 — 시판 물품
    'WORK_OF_ART': 'PROD',  # §1 — 창작 작품. PROD 가 둘을 한 칸에 담는다
    'EVENT': 'EVT',         # §1 행사·사건
    'DATE': 'DAT',          # §1 — 모든 날짜
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

# canonical 라벨 중 이 변환이 만들어낼 수 있는 것. PII 4 종
# (EMAIL/PHONE/ID_NUM/CREDIT_CARD)은 원본에 없고 주입 단계가 채운다.
PRODUCED_LABELS: frozenset[str] = frozenset(
    v for v in ONTONOTES_TO_CANONICAL.values() if v != DROP
)


class UndeclaredTypeError(ValueError):
    """매핑표에 선언되지 않은 원본 타입을 만났다."""


def resolve(ontonotes_type: str) -> str | None:
    """원본 타입 하나를 canonical 라벨로 옮긴다. 드롭이면 `None`.

    선언되지 않은 타입은 `UndeclaredTypeError` 로 즉시 중단한다. 기본값을
    두지 않는 것이 요점이다 — 모르는 타입을 조용히 드롭하면 전수성 게이트가
    있으나 마나가 된다.
    """
    try:
        target = ONTONOTES_TO_CANONICAL[ontonotes_type]
    except KeyError:
        raise UndeclaredTypeError(
            f'undeclared OntoNotes type: {ontonotes_type!r}. '
            f'Every source type must map to a canonical label or DROP '
            f'in ONTONOTES_TO_CANONICAL.'
        ) from None
    return None if target == DROP else target


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
            f'label or DROP.'
        )
