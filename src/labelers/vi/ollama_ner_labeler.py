"""베트남어 Ollama 기반 NER 라벨러.

사용 예::

    from labelers.vi.ollama_ner_labeler import OllamaNERLabeler
    labeler = OllamaNERLabeler(model="qwen3.5:27b")
    records = labeler.label("Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng.")
"""

import json
import logging
import re
import time
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from labelers.vi.ner_prompts import (
    DEFAULT_ENTITY_TYPES,
    SINGLE_PROMPT_TEMPLATE,
    BATCH_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = DEFAULT_ENTITY_TYPES
_PROMPT_TEMPLATE = SINGLE_PROMPT_TEMPLATE
_BATCH_PROMPT_TEMPLATE = BATCH_PROMPT_TEMPLATE


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
            if tokens[i : i + n] == span_tokens:
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
            logger.warning(
                "Span '%s' not found in tokens: %s", entity_text, tokens
            )
    return tags


class OllamaNERLabeler:
    def __init__(
        self,
        model: str = "qwen3.5:27b",
        entity_types: Optional[List[str]] = None,
        base_url: str = "http://localhost:11434",
        num_ctx: Optional[int] = None,
        batch_size: int = 10,
    ) -> None:
        self.model = model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.batch_size = batch_size
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
        sentences = _split_sentences(text)
        non_empty = [(i, s) for i, s in enumerate(sentences) if s.split()]
        print(f"[TIMER]   sentences: {len(sentences)} (non-empty: {len(non_empty)}, batch_size: {self.batch_size})")

        records: List[dict] = [None] * len(sentences)  # type: ignore[list-item]
        t_label_start = time.time()
        n_batches = (len(non_empty) + self.batch_size - 1) // self.batch_size

        for b in range(n_batches):
            batch = non_empty[b * self.batch_size : (b + 1) * self.batch_size]
            t_s = time.time()
            batch_spans = self._call_llm_batch([s for _, s in batch])
            t_e = time.time()
            entity_counts = [len(spans) for spans in batch_spans]
            print(f"[TIMER]   batch [{b+1:2d}/{n_batches}] {t_e-t_s:.2f}s  sentences={len(batch)}  entities={entity_counts}")

            for (orig_idx, sentence), spans in zip(batch, batch_spans):
                tokens = sentence.split()
                tags = _spans_to_bio(tokens, spans)
                records[orig_idx] = {"tokens": tokens, "ner_tags": tags, "id": str(orig_idx)}

        for i, sentence in enumerate(sentences):
            if records[i] is None:
                records[i] = {"tokens": sentence.split(), "ner_tags": [], "id": str(i)}

        print(f"[TIMER]   label total: {time.time()-t_label_start:.2f}s")
        return [r for r in records if r["tokens"]]

    def label_spans(self, text: str) -> List[dict]:
        sentences = _split_sentences(text)
        non_empty = [s for s in sentences if s.split()]
        all_spans = []
        for b_start in range(0, len(non_empty), self.batch_size):
            batch = non_empty[b_start:b_start + self.batch_size]
            batch_spans = self._call_llm_batch(batch)
            for spans in batch_spans:
                all_spans.extend(spans)
        return all_spans

    def label_records(self, records: List[dict]) -> List[dict]:
        results = []
        for raw in records:
            text = raw.get("text", "")
            if not text:
                continue
            labeled = self.label(text)
            results.extend(labeled)
        return results

    def _call_llm_batch(self, sentences: List[str]) -> List[List[dict]]:
        if len(sentences) == 1:
            return [self._call_llm(sentences[0])]

        sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        prompt = _BATCH_PROMPT_TEMPLATE.format(
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
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Batch LLM call failed (%s), falling back to single calls", e)
            return [self._call_llm(s) for s in sentences]

    def _call_llm(self, sentence: str) -> List[dict]:
        prompt = _PROMPT_TEMPLATE.format(
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
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM call failed for sentence '%s': %s", sentence[:50], e)
            return []
