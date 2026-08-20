"""한국어 vLLM NER 라벨러 — BaseVllmLabeler의 경량 서브클래스.

한국어 엔티티 타입과 단일 문장 프롬프트 템플릿을 주입한다.
"""
from typing import List, Optional

from ner.labelers.base_vllm_labeler import BaseVllmLabeler
from ner.labelers.ko.ner_prompts import DEFAULT_ENTITY_TYPES, SINGLE_PROMPT_TEMPLATE


class VllmNERLabeler(BaseVllmLabeler):
    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 1024,
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
            lang="ko",
            single_prompt_template=SINGLE_PROMPT_TEMPLATE,
        )
