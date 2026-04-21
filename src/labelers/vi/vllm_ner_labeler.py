"""vLLM 컨테이너의 OpenAI 호환 API를 사용하는 베트남어 vLLM 기반 NER 라벨러.

사용 예::

    from labelers.vi.vllm_ner_labeler import VllmNERLabeler

    labeler = VllmNERLabeler(base_url="http://localhost:8081/v1", model="Qwen/Qwen3.5-27B")
    records = labeler.label("Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng.")
"""
import asyncio
import json
import logging
import re
from typing import List, Optional

from openai import AsyncOpenAI

from labelers.vi.ner_prompts import (
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = DEFAULT_ENTITY_TYPES
_PROMPT_TEMPLATE = SINGLE_PROMPT_TEMPLATE


def _split_sentences(text: str) -> List[str]:
    """구두점 또는 개행 문자 기준으로 텍스트를 문장으로 분리한다."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= 10:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences if sentences else [text.strip()]


def _parse_spans(raw: str) -> List[dict]:
    """LLM 출력에서 JSON span 리스트를 추출한다. 실패 시 []를 반환한다."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "text" in data and "type" in data:
                return [data]
            for val in data.values():
                if isinstance(val, list):
                    return val
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _spans_to_bio(tokens: List[str], spans: List[dict]) -> List[str]:
    """엔티티 span을 토큰 수준 BIO 태그로 변환한다 (공백 분리 토큰 기준)."""
    tags = ["O"] * len(tokens)
    for span in spans:
        entity_text = span.get("text", "").strip()
        entity_type = span.get("type", "").strip()
        if not entity_text or not entity_type:
            continue
        span_tokens = entity_text.split()
        n = len(span_tokens)
        matched = False

        # 1. 정확한 매칭
        for i in range(len(tokens) - n + 1):
            if tokens[i: i + n] == span_tokens:
                tags[i] = f"B-{entity_type}"
                for j in range(1, n):
                    tags[i + j] = f"I-{entity_type}"
                matched = True
                break

        # 2. 부분 문자열 매칭
        if not matched:
            for i in range(len(tokens) - n + 1):
                if all(
                    sp in tokens[i + j] or tokens[i + j] in sp
                    for j, sp in enumerate(span_tokens)
                ):
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, n):
                        tags[i + j] = f"I-{entity_type}"
                    matched = True
                    break

        if not matched:
            logger.warning("Span '%s' not found in tokens: %s", entity_text, tokens)
    return tags


class VllmNERLabeler:
    """vLLM 컨테이너의 OpenAI 호환 API를 사용하는 베트남어 배치 NER 라벨러."""

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 4096,
        concurrency: int = 32,
        thinking: bool = False,
    ) -> None:
        self.model = model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(base_url=base_url, api_key="none")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        logger.info("VllmNERLabeler ready: %s @ %s", model, base_url)

    def label(self, text: str) -> List[dict]:
        sentences = _split_sentences(text)
        return asyncio.run(self._label_sentences(sentences, id_offset=0))

    def label_spans(self, text: str) -> List[dict]:
        sentences = _split_sentences(text)
        non_empty = [s for s in sentences if s.strip()]

        async def _get_spans():
            tasks = [self._chat(s) for s in non_empty]
            raw_results = await asyncio.gather(*tasks)
            all_spans = []
            for raw in raw_results:
                all_spans.extend(_parse_spans(raw))
            return all_spans

        return asyncio.run(_get_spans())

    def label_records(self, records: List[dict]) -> List[dict]:
        all_sentences: List[str] = []
        meta: List[tuple] = []

        for r_idx, record in enumerate(records):
            text = record.get("text", "")
            if not text:
                continue
            for s_idx, sentence in enumerate(_split_sentences(text)):
                all_sentences.append(sentence)
                meta.append((r_idx, s_idx))

        if not all_sentences:
            return []

        return asyncio.run(self._run_batch(all_sentences, meta))

    def _build_user_content(self, sentence: str) -> str:
        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(self.entity_types),
            sentence=sentence,
        )

    async def _chat(self, sentence: str) -> str:
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_user_content(sentence)}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
            )
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content or ""

    async def _label_sentences(self, sentences: List[str], id_offset: int = 0) -> List[dict]:
        sentences = [s for s in sentences if s.split()]
        if not sentences:
            return []

        raw_outputs = await asyncio.gather(*[self._chat(s) for s in sentences])

        results = []
        for idx, (sentence, raw) in enumerate(zip(sentences, raw_outputs)):
            tokens = sentence.split()
            spans = _parse_spans(raw.strip())
            tags = _spans_to_bio(tokens, spans)
            results.append({
                "tokens": tokens,
                "ner_tags": tags,
                "id": str(id_offset + idx),
            })
        return results

    async def _run_batch(self, sentences: List[str], meta: List[tuple]) -> List[dict]:
        raw_outputs = await asyncio.gather(*[self._chat(s) for s in sentences])

        results = []
        for (r_idx, s_idx), sentence, raw in zip(meta, sentences, raw_outputs):
            tokens = sentence.split()
            if not tokens:
                continue
            spans = _parse_spans(raw.strip())
            tags = _spans_to_bio(tokens, spans)
            results.append({
                "tokens": tokens,
                "ner_tags": tags,
                "id": f"{r_idx}-{s_idx}",
            })
        return results
