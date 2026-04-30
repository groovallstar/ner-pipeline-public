"""베트남어 OpenAI NER 라벨러 — BaseOpenAILabeler의 경량 서브클래스."""
from typing import List, Optional

from labelers.base_openai_labeler import BaseOpenAILabeler
from labelers.vi.ner_prompts import (
    DEFAULT_ENTITY_TYPES,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


class OpenAINERLabeler(BaseOpenAILabeler):
    def __init__(
        self,
        model: str = "gpt-5-mini",
        entity_types: Optional[List[str]] = None,
        max_tokens_per_batch: int = 1000,
        base_url: Optional[str] = None,
        concurrency: int = 4,
    ) -> None:
        super().__init__(
            model=model,
            entity_types=entity_types or DEFAULT_ENTITY_TYPES,
            max_tokens_per_batch=max_tokens_per_batch,
            base_url=base_url,
            concurrency=concurrency,
            lang="vi",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
        )
