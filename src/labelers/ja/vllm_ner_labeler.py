"""Japanese vLLM NER labeler — thin subclass of BaseVllmLabeler."""
from typing import List, Optional

from labelers.base_vllm_labeler import BaseVllmLabeler
from labelers.ja.ner_prompts import DEFAULT_ENTITY_TYPES, SINGLE_PROMPT_TEMPLATE


class VllmNERLabeler(BaseVllmLabeler):
    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 4096,
        concurrency: int = 32,
        thinking: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            entity_types=entity_types or DEFAULT_ENTITY_TYPES,
            max_tokens=max_tokens,
            concurrency=concurrency,
            thinking=thinking,
            lang="ja",
            single_prompt_template=SINGLE_PROMPT_TEMPLATE,
        )
