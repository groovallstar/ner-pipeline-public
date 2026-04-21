"""WikiANN-vi 문장을 Stockmark 8종 스키마로 재라벨하는 async 클라이언트.

vLLM OpenAI 호환 엔드포인트로 SINGLE 프롬프트를 호출해 8종 엔티티를 추출하고,
원본 텍스트에서 문자 오프셋을 매칭해 `gold_spans_8type` 필드로 반환한다.

주요 구성:
- `parse_spans`: LLM 원문 출력에서 JSON span 리스트 추출 (think 제거·정규식 폴백).
- `match_offsets`: LLM 표면형 span을 원본 텍스트 오프셋에 매핑 (중복 안전 소비).
- `Relabeler`: concurrency 제한 + usage 누적을 가진 async 재라벨러.

이슈 #10 3단계 구현. 모호 사례 기준은
`docs/specs/entities/vietnamese-ner-8types.md` 참조.
"""
import asyncio
import json
import logging
import re
from typing import List, Optional

from openai import AsyncOpenAI

from augmenters.wikiann_vi.prompts import (
    BATCH_PROMPT_TEMPLATE,
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
    format_entity_types,
)

logger = logging.getLogger(__name__)


def parse_spans(raw: str) -> List[dict]:
    """LLM 출력에서 JSON span 리스트를 추출한다.

    `<think>...</think>` 제거 후 `json.loads` 시도. list/dict 형태 모두 허용하고,
    실패 시 정규식으로 `[...]` 블록 재시도. 모두 실패하면 []를 반환한다.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if 'text' in data and 'type' in data:
                return [data]
            for val in data.values():
                if isinstance(val, list):
                    return val
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def match_offsets(text: str, spans: List[dict]) -> List[dict]:
    """LLM 표면형 span을 text 내 문자 오프셋과 함께 반환한다.

    중복 표면형 및 부분 문자열 충돌을 방지하기 위해:
    1. 길이 역순 정렬 (긴 엔티티 먼저 매칭)
    2. 이미 소비된 문자 인덱스 집합을 추적
    3. 매칭되지 않은 span은 drop

    반환 순서는 LLM 출력 원래 순서를 보존한다.
    """
    if not text or not spans:
        return []

    indexed = sorted(
        enumerate(spans),
        key=lambda p: -len((p[1].get('text') or '')),
    )

    consumed: set = set()
    results: List[tuple] = []

    for orig_idx, span in indexed:
        ent_text = (span.get('text') or '').strip()
        ent_type = (span.get('type') or '').strip()
        if not ent_text or not ent_type:
            continue
        start = 0
        while True:
            pos = text.find(ent_text, start)
            if pos < 0:
                break
            end = pos + len(ent_text)
            if not any(i in consumed for i in range(pos, end)):
                consumed.update(range(pos, end))
                results.append((orig_idx, {
                    'text': ent_text,
                    'type': ent_type,
                    'start': pos,
                    'end': end,
                }))
                break
            start = pos + 1

    results.sort(key=lambda p: p[0])
    return [s for _, s in results]


class Relabeler:
    """vLLM OpenAI 호환 엔드포인트로 WikiANN-vi 레코드를 8종으로 재라벨."""

    def __init__(
        self,
        base_url: str = 'http://localhost:8081/v1',
        model: str = 'cyankiwi/gemma-4-31B-it-AWQ-8bit',
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 2048,
        concurrency: int = 16,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(
            base_url=base_url, api_key='none', timeout=timeout,
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        logger.info(
            'Relabeler ready: %s @ %s (concurrency=%d)',
            model, base_url, concurrency,
        )

    def _build_single(self, sentence: str) -> str:
        return SINGLE_PROMPT_TEMPLATE.format(
            entity_types=format_entity_types(self.entity_types),
            sentence=sentence,
        )

    async def _chat(self, prompt: str) -> str:
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        if response.usage:
            self.total_prompt_tokens += (
                response.usage.prompt_tokens or 0
            )
            self.total_completion_tokens += (
                response.usage.completion_tokens or 0
            )
        return response.choices[0].message.content or ''

    async def _relabel_one(self, record: dict) -> dict:
        text = record.get('text', '') or ''
        if not text.strip():
            return {
                **record,
                'gold_spans_8type': [],
                'relabel_model': self.model,
            }
        prompt = self._build_single(text)
        try:
            raw = await self._chat(prompt)
        except Exception as exc:
            logger.warning(
                'Relabel failed for id=%s: %s', record.get('id'), exc,
            )
            return {
                **record,
                'gold_spans_8type': [],
                'relabel_model': self.model,
                'relabel_error': str(exc),
            }
        parsed = parse_spans(raw)
        offset_spans = match_offsets(text, parsed)
        return {
            **record,
            'gold_spans_8type': offset_spans,
            'relabel_model': self.model,
        }

    async def relabel(self, records: List[dict]) -> List[dict]:
        """레코드 리스트를 concurrency 제한 하에 병렬 재라벨한다."""
        tasks = [self._relabel_one(r) for r in records]
        return await asyncio.gather(*tasks)

    def relabel_sync(self, records: List[dict]) -> List[dict]:
        return asyncio.run(self.relabel(records))


# BATCH_PROMPT_TEMPLATE은 후속 단계(다중 문장 배치 호출)에서 사용 예정
__all__ = [
    'Relabeler',
    'parse_spans',
    'match_offsets',
    'BATCH_PROMPT_TEMPLATE',
]
