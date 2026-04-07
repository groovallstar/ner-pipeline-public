"""vLLM-based NER labeler using the vLLM container's OpenAI-compatible API.

Unlike OllamaNERLabeler (one request at a time), this class sends all
sentences concurrently via async HTTP requests — much faster when labeling
large corpora.

Usage::

    from labelers.ko.vllm_ner_labeler import VllmNERLabeler

    labeler = VllmNERLabeler(base_url="http://localhost:8081/v1", model="Qwen/Qwen3.5-27B")
    records = labeler.label("서울시에서 김철수가 2024년 1월에 기자회견을 열었다.")
    # [{"tokens": [...], "ner_tags": [...], "id": "0"}]

    bulk = labeler.label_records([{"text": "..."}, {"text": "..."}])
"""
import asyncio
import json
import logging
import re
from typing import List, Optional

from openai import AsyncOpenAI

from labelers.ja.ner_prompts import (
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
    format_entity_types,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = DEFAULT_ENTITY_TYPES
_PROMPT_TEMPLATE = SINGLE_PROMPT_TEMPLATE


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on punctuation or newlines."""
    parts = re.split(r'(?<=[.!?。！？])\s*|\n+', text.strip())
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
    """Extract JSON span list from LLM output. Returns [] on failure."""
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
    """Convert entity spans to token-level BIO tags (whitespace-split tokens)."""
    tags = ["O"] * len(tokens)
    for span in spans:
        entity_text = span.get("text", "").strip()
        entity_type = span.get("type", "").strip()
        if not entity_text or not entity_type:
            continue
        span_tokens = entity_text.split()
        n = len(span_tokens)
        matched = False

        # 1. Exact match
        for i in range(len(tokens) - n + 1):
            if tokens[i: i + n] == span_tokens:
                tags[i] = f"B-{entity_type}"
                for j in range(1, n):
                    tags[i + j] = f"I-{entity_type}"
                matched = True
                break

        # 2. Substring match (handles particles attached to root)
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
    """Batch NER labeler using the vLLM container's OpenAI-compatible API.

    Args:
        base_url: vLLM server URL, e.g. "http://localhost:8081/v1".
        model: Model name served by the vLLM container.
        entity_types: NER tag set. Defaults to KLUE NER tags.
        max_tokens: Max new tokens per sample.
        concurrency: Max concurrent requests to the vLLM server.
    """

    def __init__(
        self,
        base_url: str = "http://vllm-server:8081/v1",
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
        # Token usage tracking (accumulated across calls, reset by consumer)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        logger.info("VllmNERLabeler ready: %s @ %s", model, base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, text: str) -> List[dict]:
        """Label a text string. Returns a list of NER records (one per sentence)."""
        sentences = _split_sentences(text)
        return asyncio.run(self._label_sentences(sentences, id_offset=0))

    def label_spans(self, text: str) -> List[dict]:
        """Return raw entity spans without BIO conversion."""
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
        """Label a list of raw text records concurrently.

        Input records must have a ``"text"`` field.
        Returns NER records with ``"tokens"``, ``"ner_tags"``, and ``"id"`` fields.
        """
        all_sentences: List[str] = []
        meta: List[tuple] = []  # (record_idx, sent_idx)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_content(self, sentence: str) -> str:
        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(self.entity_types),
            sentence=sentence,
        )

    async def _chat(self, sentence: str) -> str:
        """Send a single chat completion request with concurrency limiting."""
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


