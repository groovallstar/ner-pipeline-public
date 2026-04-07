"""OpenAI-based NER labeler with batch processing.

API key is read from the OPENAI_API_KEY environment variable.
Never hardcode or pass the key as an argument.

Usage (standalone):
    export OPENAI_API_KEY="sk-..."
    python3 -m labelers.ko.openai_ner_labeler \
        --urls "https://n.news.naver.com/article/138/0002222481" \
        --output /tmp/result.jsonl \
        --model gpt-4o-mini
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from typing import List, Optional

from openai import AsyncOpenAI, OpenAI

from labelers.ko.ner_prompts import (
    DEFAULT_ENTITY_TYPES,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = DEFAULT_ENTITY_TYPES
_SYSTEM_PROMPT = SYSTEM_PROMPT
_USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE


def _split_sentences(text: str) -> List[str]:
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


def _spans_to_bio(tokens: List[str], spans: List[dict]) -> List[str]:
    tags = ["O"] * len(tokens)
    for span in spans:
        entity_text = span.get("text", "").strip()
        entity_type = span.get("type", "").strip()
        if not entity_text or not entity_type:
            continue
        span_tokens = entity_text.split()
        n = len(span_tokens)

        # 1. Exact match
        matched = False
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


class OpenAINERLabeler:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        entity_types: Optional[List[str]] = None,
        max_tokens_per_batch: int = 1000,
        base_url: Optional[str] = None,
        concurrency: int = 4,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "none")
        if not base_url and api_key == "none":
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.model = model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.max_tokens_per_batch = max_tokens_per_batch
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
        self._semaphore = asyncio.Semaphore(concurrency)
        # Token usage tracking (accumulated across calls, reset by consumer)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def label(self, text: str) -> List[dict]:
        """Label a text string. Returns a list of NERRecords (one per sentence)."""
        sentences = _split_sentences(text)
        non_empty = [(i, s) for i, s in enumerate(sentences) if s.split()]

        # Token-based batching: group sentences until token budget is exhausted
        batches: List[List[tuple]] = []
        current_batch: List[tuple] = []
        current_tokens = 0
        for idx, sentence in non_empty:
            token_est = len(sentence.split())
            if current_batch and current_tokens + token_est > self.max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append((idx, sentence))
            current_tokens += token_est
        if current_batch:
            batches.append(current_batch)

        print(f"[TIMER]   sentences: {len(sentences)} (non-empty: {len(non_empty)}, batches: {len(batches)}, max_tokens/batch: {self.max_tokens_per_batch})")

        records: List[Optional[dict]] = [None] * len(sentences)
        t_label_start = time.time()

        for b_idx, batch in enumerate(batches):
            t_s = time.time()
            batch_spans = self._call_api_batch([s for _, s in batch])
            t_e = time.time()
            entity_counts = [len(spans) for spans in batch_spans]
            print(f"[TIMER]   batch [{b_idx+1:2d}/{len(batches)}] {t_e-t_s:.2f}s  sentences={len(batch)}  entities={entity_counts}")

            for (orig_idx, sentence), spans in zip(batch, batch_spans):
                tokens = sentence.split()
                tags = _spans_to_bio(tokens, spans)
                records[orig_idx] = {"tokens": tokens, "ner_tags": tags, "id": str(orig_idx)}

        # Fill skipped empty sentences
        for i, sentence in enumerate(sentences):
            if records[i] is None:
                records[i] = {"tokens": sentence.split(), "ner_tags": [], "id": str(i)}

        print(f"[TIMER]   label total: {time.time()-t_label_start:.2f}s")
        return [r for r in records if r and r["tokens"]]  # type: ignore[misc]

    def label_spans(self, text: str) -> List[dict]:
        """Return raw entity spans without BIO conversion (async concurrent)."""
        sentences = _split_sentences(text)
        non_empty = [s for s in sentences if s.split()]
        batches = self._make_batches(non_empty)

        async def _run():
            tasks = [self._call_api_batch_async(batch) for batch in batches]
            results = await asyncio.gather(*tasks)
            all_spans = []
            for batch_spans in results:
                for spans in batch_spans:
                    all_spans.extend(spans)
            return all_spans

        return asyncio.run(_run())

    def _make_batches(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into batches by estimated token count."""
        batches = []
        current_batch, current_tokens = [], 0
        for s in sentences:
            t = len(s.split())
            if current_batch and current_tokens + t > self.max_tokens_per_batch:
                batches.append(current_batch)
                current_batch, current_tokens = [], 0
            current_batch.append(s)
            current_tokens += t
        if current_batch:
            batches.append(current_batch)
        return batches

    def label_records(self, records: List[dict]) -> List[dict]:
        results = []
        for raw in records:
            text = raw.get("text", "")
            if not text:
                continue
            results.extend(self.label(text))
        return results

    def _call_api_batch(self, sentences: List[str]) -> List[List[dict]]:
        """Call OpenAI API for a batch of sentences (sync)."""
        return self._parse_batch_response(sentences, self._call_api_batch_sync(sentences))

    def _call_api_batch_sync(self, sentences: List[str]) -> str:
        """Sync API call, returns raw content string."""
        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = _USER_PROMPT_TEMPLATE.format(sentences=sentences_str)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=1,
            )
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens or 0
                self.total_completion_tokens += response.usage.completion_tokens or 0
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("API call failed (%s), returning empty", e)
            return ""

    async def _call_api_batch_async(self, sentences: List[str]) -> List[List[dict]]:
        """Async API call with concurrency limiting."""
        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = _USER_PROMPT_TEMPLATE.format(sentences=sentences_str)
        async with self._semaphore:
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=1,
                )
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                raw = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("Async API call failed (%s), returning empty spans", e)
                return [[] for _ in sentences]
        return self._parse_batch_response(sentences, raw)

    @staticmethod
    def _parse_batch_response(sentences: List[str], raw: str) -> List[List[dict]]:
        """Parse batch JSON response into per-sentence span lists."""
        if not raw:
            return [[] for _ in sentences]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [[] for _ in sentences]
        if not isinstance(data, dict):
            return [[] for _ in sentences]
        results = []
        for i in range(len(sentences)):
            spans = data.get(str(i), [])
            results.append(spans if isinstance(spans, list) else [])
        return results


