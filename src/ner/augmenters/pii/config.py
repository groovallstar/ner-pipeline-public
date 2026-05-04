"""PII 주입 설정."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Lang = Literal['ja', 'vi']

DEFAULT_DENSITY: dict[int, float] = {0: 0.2, 1: 0.4, 2: 0.3, 3: 0.1}
# `NAME`·`ADDRESS`는 내부 생성 토큰이며 `DEFAULT_MERGE_RULES`로 canonical
# 라벨(`PER`·`LOC`)에 무조건 병합된다. `DAT`는 모든 날짜를 포괄하는
# canonical 라벨(생년월일 한정이 아님).
DEFAULT_PII_LABELS: list[str] = [
    'NAME', 'PHONE', 'ADDRESS', 'DAT',
    'ID_NUM', 'EMAIL', 'CREDIT_CARD',
]
DEFAULT_MERGE_RULES: dict[str, str] = {
    'NAME': 'PER',
    'ADDRESS': 'LOC',
}


@dataclass
class InjectionConfig:
    """PII 주입 파이프라인 설정."""
    lang: Lang = 'ja'
    density: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_DENSITY)
    )
    pii_labels: list[str] | None = None
    label_merge_rules: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MERGE_RULES)
    )
    seed: int = 42

    def validate(self) -> None:
        """density 분포 합이 1.0인지 검증한다."""
        if not self.density:
            raise ValueError('density must not be empty')
        total = sum(self.density.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f'density must sum to 1.0 (got {total:.6f})'
            )
        for k, v in self.density.items():
            if v < 0:
                raise ValueError(f'density[{k}] must be >= 0 (got {v})')
        if self.lang not in ('ja', 'vi'):
            raise ValueError(f'unsupported lang: {self.lang}')

    def effective_labels(self) -> list[str]:
        """사용할 PII 라벨 목록."""
        return list(self.pii_labels) if self.pii_labels else list(
            DEFAULT_PII_LABELS
        )
