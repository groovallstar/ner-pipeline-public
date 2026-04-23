"""WikiANN-vi → canonical 8종 스키마 재라벨 모듈.

canonical 라벨 정의는 `docs/manual/data/canonical-entity-schema.md`,
매핑 기준은 `docs/manual/data/vietnamese-ner-8types.md` 참조.
"""
from augmenters.wikiann_vi.prompts import (
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
    BATCH_PROMPT_TEMPLATE,
    format_entity_types,
)

__all__ = [
    'DEFAULT_ENTITY_TYPES',
    'SINGLE_PROMPT_TEMPLATE',
    'BATCH_PROMPT_TEMPLATE',
    'format_entity_types',
]
