"""WikiANN-vi → Stockmark 8종 스키마 재라벨 모듈.

이슈 #10 3~5단계 구현. 상세 스펙은
`docs/specs/entities/vietnamese-ner-8types.md` 참조.
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
