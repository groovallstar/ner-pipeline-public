"""Abstract base for Ollama-backed NER labelers.

Subclasses inject a language pack (entity_types, single+batch prompt templates, lang).
"""
import json
import logging
import time
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from labelers.llm_helpers import spans_to_bio, split_sentences

logger = logging.getLogger(__name__)


class BaseOllamaLabeler:
    """Ollama batch NER labeler base.

    Subclasses must pass ``single_prompt_template`` and ``batch_prompt_template``
    as keyword arguments.
    """

    def __init__(
        self,
        model: str = "qwen3.5:27b",
        entity_types: Optional[List[str]] = None,
        base_url: str = "http://localhost:11434",
        num_ctx: Optional[int] = None,
        batch_size: int = 10,
        *,
        lang: str = "ko",
        single_prompt_template: str = "",
        batch_prompt_template: str = "",
    ) -> None:
        if not single_prompt_template or not batch_prompt_template:
            raise ValueError(
                "single_prompt_template and batch_prompt_template are required"
            )
        self.model = model
        self.entity_types = entity_types or []
        self.batch_size = batch_size
        self.lang = lang
        self._single_prompt_template = single_prompt_template
        self._batch_prompt_template = batch_prompt_template
        # Ollama via langchain_ollama does not expose usage metadata reliably;
        # these stubs exist for a uniform interface with vllm/openai backends.
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            format="json",
            temperature=0,
            think=False,  # type: ignore[call-arg]
            num_ctx=num_ctx,
            reasoning=False,
            additional_kwargs={"think": False},  # type: ignore[call-arg]
        )

    def label(self, text: str) -> List[dict]:
        """Label a text string. Returns a list of NERRecords (one per sentence)."""
        sentences = split_sentences(text, lang=self.lang)
        non_empty = [(i, s) for i, s in enumerate(sentences) if s.split()]
        print(
            f"[TIMER]   sentences: {len(sentences)} (non-empty: {len(non_empty)}, "
            f"batch_size: {self.batch_size})"
        )

        records: List[dict] = [None] * len(sentences)  # type: ignore[list-item]
        t_label_start = time.time()
        n_batches = (len(non_empty) + self.batch_size - 1) // self.batch_size

        for b in range(n_batches):
            batch = non_empty[b * self.batch_size : (b + 1) * self.batch_size]
            t_s = time.time()
            batch_spans = self._call_llm_batch([s for _, s in batch])
            t_e = time.time()
            entity_counts = [len(spans) for spans in batch_spans]
            print(
                f"[TIMER]   batch [{b+1:2d}/{n_batches}] {t_e-t_s:.2f}s "
                f" sentences={len(batch)}  entities={entity_counts}"
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

        print(f"[TIMER]   label total: {time.time()-t_label_start:.2f}s")
        return [r for r in records if r["tokens"]]

    def label_spans(self, text: str) -> List[dict]:
        """Return raw entity spans without BIO conversion."""
        sentences = split_sentences(text, lang=self.lang)
        non_empty = [s for s in sentences if s.split()]
        all_spans = []
        for b_start in range(0, len(non_empty), self.batch_size):
            batch = non_empty[b_start : b_start + self.batch_size]
            batch_spans = self._call_llm_batch(batch)
            for spans in batch_spans:
                all_spans.extend(spans)
        return all_spans

    def label_records(self, records: List[dict]) -> List[dict]:
        """Label a list of RawTextRecords. Returns NERRecords.

        Note: sentence IDs are assigned within each record (str(orig_idx)), so IDs
        restart at "0" per record. This matches pre-refactor behavior for Ollama and
        OpenAI backends. vLLM uses f"{r_idx}-{s_idx}" (cross-record unique). The
        asymmetry is intentional to preserve existing downstream behavior.
        """
        results = []
        for raw in records:
            text = raw.get("text", "")
            if not text:
                continue
            labeled = self.label(text)
            results.extend(labeled)
        return results

    def _call_llm_batch(self, sentences: List[str]) -> List[List[dict]]:
        """Call Ollama LLM for a batch of sentences. Returns a list of span lists."""
        if len(sentences) == 1:
            return [self._call_llm(sentences[0])]

        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = self._batch_prompt_template.format(
            entity_types=", ".join(self.entity_types),
            sentences=sentences_str,
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = str(response.content).strip()
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("Batch response is not a dict, falling back: %s", raw[:200])
                return [self._call_llm(s) for s in sentences]

            results = []
            for i in range(len(sentences)):
                spans = data.get(str(i), [])
                if isinstance(spans, list):
                    results.append(spans)
                else:
                    logger.warning("Unexpected span type for sentence %d: %s", i, spans)
                    results.append([])
            return results
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            logger.warning("Batch LLM call failed (%s), falling back to single calls", e)
            return [self._call_llm(s) for s in sentences]

    def _call_llm(self, sentence: str) -> List[dict]:
        """Call Ollama LLM and parse entity spans. Returns [] on failure."""
        prompt = self._single_prompt_template.format(
            entity_types=", ".join(self.entity_types),
            sentence=sentence,
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = str(response.content).strip()
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "text" in data and "type" in data:
                    return [data]
                for val in data.values():
                    if isinstance(val, list):
                        return val
            logger.warning("Unexpected LLM response structure: %s", raw[:200])
            return []
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            logger.warning("LLM call failed for sentence '%s': %s", sentence[:50], e)
            return []
