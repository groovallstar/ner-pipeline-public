"""OpenAI 호환 채팅 NER 라벨러의 추상 베이스 클래스.

서브클래스는 언어 팩(entity_types, system_prompt, user_prompt_template, lang)을 주입한다.
동기/비동기 클라이언트를 모두 지원한다(label() = 동기, label_spans() = 비동기).
"""
import asyncio
import json
import logging
import os
import time
from typing import List, Optional

from openai import AsyncOpenAI, OpenAI

from labelers.llm_helpers import spans_to_bio, split_sentences

logger = logging.getLogger(__name__)


class BaseOpenAILabeler:
    """OpenAI 호환 배치 NER 라벨러 베이스 클래스."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        entity_types: Optional[List[str]] = None,
        max_tokens_per_batch: int = 1000,
        base_url: Optional[str] = None,
        concurrency: int = 4,
        *,
        lang: str = "ko",
        system_prompt: str = "",
        user_prompt_template: str = "",
    ) -> None:
        if not system_prompt or not user_prompt_template:
            raise ValueError("system_prompt and user_prompt_template are required")

        api_key = os.environ.get("OPENAI_API_KEY", "none")
        if not base_url and api_key == "none":
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.model = model
        self.entity_types = entity_types or []
        self.max_tokens_per_batch = max_tokens_per_batch
        self.lang = lang
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._client = (
            OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        )
        self._async_client = (
            AsyncOpenAI(api_key=api_key, base_url=base_url)
            if base_url else AsyncOpenAI(api_key=api_key)
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def label(self, text: str) -> List[dict]:
        """텍스트를 라벨링한다. 문장당 하나의 NERRecord 리스트를 반환한다."""
        sentences = split_sentences(text, lang=self.lang)
        non_empty = [(i, s) for i, s in enumerate(sentences) if s.split()]

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

        logger.info(
            "[TIMER]   sentences: %d (non-empty: %d, batches: %d, max_tokens/batch: %d)",
            len(sentences), len(non_empty), len(batches), self.max_tokens_per_batch,
        )

        records: List[Optional[dict]] = [None] * len(sentences)
        t_label_start = time.time()

        for b_idx, batch in enumerate(batches):
            t_s = time.time()
            batch_spans = self._call_api_batch([s for _, s in batch])
            t_e = time.time()
            entity_counts = [len(spans) for spans in batch_spans]
            logger.info(
                "[TIMER]   batch [%2d/%d] %.2fs  sentences=%d  entities=%s",
                b_idx + 1, len(batches), t_e - t_s, len(batch), entity_counts,
            )

            for (orig_idx, sentence), spans in zip(batch, batch_spans):
                tokens = sentence.split()
                tags = spans_to_bio(tokens, spans)
                records[orig_idx] = {
                    "tokens": tokens, "ner_tags": tags, "id": str(orig_idx),
                }

        for i, sentence in enumerate(sentences):
            if records[i] is None:
                records[i] = {"tokens": sentence.split(), "ner_tags": [], "id": str(i)}

        logger.info("[TIMER]   label total: %.2fs", time.time() - t_label_start)
        return [r for r in records if r and r["tokens"]]  # type: ignore[misc]

    def label_spans(self, text: str) -> List[dict]:
        """BIO 변환 없이 원시 엔티티 span을 반환한다 (비동기 병렬 처리)."""
        sentences = split_sentences(text, lang=self.lang)
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
        """RawTextRecord 리스트를 라벨링하여 NERRecord 리스트를 반환한다.

        문장 ID는 레코드마다 "0"부터 재시작한다(리팩터 이전 동작).
        vLLM은 f"{r_idx}-{s_idx}" 형태의 전역 ID를 사용하며, 이 비대칭은 의도적이다.
        """
        results = []
        for raw in records:
            text = raw.get("text", "")
            if not text:
                continue
            results.extend(self.label(text))
        return results

    def _call_api_batch(self, sentences: List[str]) -> List[List[dict]]:
        return self._parse_batch_response(sentences, self._call_api_batch_sync(sentences))

    def _call_api_batch_sync(self, sentences: List[str]) -> str:
        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = self._user_prompt_template.format(sentences=sentences_str)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=1,
            )
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens or 0
                self.total_completion_tokens += response.usage.completion_tokens or 0
            return response.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("API call failed (%s), returning empty", e)
            return ""

    async def _call_api_batch_async(self, sentences: List[str]) -> List[List[dict]]:
        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = self._user_prompt_template.format(sentences=sentences_str)
        async with self._semaphore:
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=1,
                )
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                raw = response.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                logger.warning("Async API call failed (%s), returning empty spans", e)
                return [[] for _ in sentences]
        return self._parse_batch_response(sentences, raw)

    @staticmethod
    def _parse_batch_response(sentences: List[str], raw: str) -> List[List[dict]]:
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
