"""Japanese Ollama NER labeler — thin subclass of BaseOllamaLabeler."""
from typing import List, Optional

from labelers.base_ollama_labeler import BaseOllamaLabeler
from labelers.ja.ner_prompts import (
    BATCH_PROMPT_TEMPLATE,
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
)


class OllamaNERLabeler(BaseOllamaLabeler):
    def __init__(
        self,
        model: str = "qwen3.5:27b",
        entity_types: Optional[List[str]] = None,
        base_url: str = "http://localhost:11434",
        num_ctx: Optional[int] = None,
        batch_size: int = 10,
    ) -> None:
        super().__init__(
            model=model,
            entity_types=entity_types or DEFAULT_ENTITY_TYPES,
            base_url=base_url,
            num_ctx=num_ctx,
            batch_size=batch_size,
            lang="ja",
            single_prompt_template=SINGLE_PROMPT_TEMPLATE,
            batch_prompt_template=BATCH_PROMPT_TEMPLATE,
        )
