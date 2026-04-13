"""Abstract base for vLLM OpenAI-compatible NER labelers.

Subclasses inject a language pack (entity_types, single_prompt_template, lang).
"""
import asyncio
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from labelers.llm_helpers import parse_spans, spans_to_bio, split_sentences

logger = logging.getLogger(__name__)


class BaseVllmLabeler:
    """Batch NER labeler using the vLLM container's OpenAI-compatible API.

    Args:
        base_url: vLLM server URL, e.g. "http://localhost:8081/v1".
        model: Model name served by the vLLM container.
        entity_types: NER tag set. Defaults to the language pack's default.
        max_tokens: Max new tokens per sample.
        concurrency: Max concurrent requests to the vLLM server.
        thinking: Enable vLLM chat-template `enable_thinking` flag.
        lang: Language code ("ko"/"ja") — passed to sentence splitter.
        single_prompt_template: Prompt template (f-string with {entity_types}, {sentence}).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 4096,
        concurrency: int = 32,
        thinking: bool = False,
        *,
        lang: str = "ko",
        single_prompt_template: str = "",
    ) -> None:
        if not single_prompt_template:
            raise ValueError("single_prompt_template is required")
        self.model = model
        self.entity_types = entity_types or []
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.lang = lang
        self._single_prompt_template = single_prompt_template
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(base_url=base_url, api_key="none")
        # Token usage tracking (accumulated across calls, reset by consumer)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        logger.info("%s ready: %s @ %s", type(self).__name__, model, base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, text: str) -> List[dict]:
        """Label a text string. Returns a list of NER records (one per sentence)."""
        sentences = split_sentences(text, lang=self.lang)
        return asyncio.run(self._label_sentences(sentences, id_offset=0))

    def label_spans(self, text: str) -> List[dict]:
        """Return raw entity spans without BIO conversion."""
        sentences = split_sentences(text, lang=self.lang)
        non_empty = [s for s in sentences if s.strip()]

        async def _get_spans():
            tasks = [self._chat(s) for s in non_empty]
            raw_results = await asyncio.gather(*tasks)
            all_spans = []
            for raw in raw_results:
                all_spans.extend(parse_spans(raw))
            return all_spans

        return asyncio.run(_get_spans())

    def label_records(self, records: List[dict]) -> List[dict]:
        """Label a list of raw text records concurrently.

        Input records must have a ``"text"`` field.
        Returns NER records with ``"tokens"``, ``"ner_tags"``, and ``"id"`` fields.
        """
        all_sentences: List[str] = []
        meta: List[tuple] = []

        for r_idx, record in enumerate(records):
            text = record.get("text", "")
            if not text:
                continue
            for s_idx, sentence in enumerate(split_sentences(text, lang=self.lang)):
                all_sentences.append(sentence)
                meta.append((r_idx, s_idx))

        if not all_sentences:
            return []

        return asyncio.run(self._run_batch(all_sentences, meta))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_content(self, sentence: str) -> str:
        return self._single_prompt_template.format(
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
            spans = parse_spans(raw.strip())
            tags = spans_to_bio(tokens, spans)
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
            spans = parse_spans(raw.strip())
            tags = spans_to_bio(tokens, spans)
            results.append({
                "tokens": tokens,
                "ner_tags": tags,
                "id": f"{r_idx}-{s_idx}",
            })
        return results
