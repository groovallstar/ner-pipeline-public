"""Ollama 백엔드 NER 라벨러의 추상 베이스 클래스.

서브클래스는 언어 팩(entity_types, 단일/배치 프롬프트 템플릿, lang)을 주입한다.
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
    """Ollama 배치 NER 라벨러 베이스 클래스.

    서브클래스는 ``single_prompt_template``과 ``batch_prompt_template``을
    키워드 인자로 전달해야 한다.
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
        # langchain_ollama는 usage 메타데이터를 안정적으로 제공하지 않는다.
        # vllm/openai 백엔드와 인터페이스 통일을 위한 스텁이다.
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
        """텍스트를 라벨링한다. 문장당 하나의 NERRecord 리스트를 반환한다."""
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
        """BIO 변환 없이 원시 엔티티 span을 반환한다."""
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
        """RawTextRecord 리스트를 라벨링하여 NERRecord 리스트를 반환한다.

        문장 ID는 레코드 내에서 str(orig_idx)로 부여되므로 레코드마다 "0"부터
        재시작한다. 이는 Ollama/OpenAI 백엔드의 리팩터 이전 동작을 유지한다.
        vLLM은 f"{r_idx}-{s_idx}" 형태의 전역 고유 ID를 사용한다.
        이 비대칭은 기존 하위 동작 보존을 위해 의도적으로 유지한다.
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
        """문장 배치에 대해 Ollama LLM을 호출한다. span 리스트의 리스트를 반환한다."""
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
        """Ollama LLM을 호출하여 엔티티 span을 파싱한다. 실패 시 []를 반환한다."""
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
