"""vLLM-based NER labeler using the vLLM container's OpenAI-compatible API.

Unlike OllamaNERLabeler (one request at a time), this class sends all
sentences concurrently via async HTTP requests — much faster when labeling
large corpora.

Usage::

    from collectors.vllm_ner_labeler import VllmNERLabeler

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

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = ["PS", "LC", "OG", "DT", "TI", "QT"]

_PROMPT_TEMPLATE = """당신은 한국어 개체명 인식(NER) 전문가입니다. 텍스트에서 개체명을 찾아 JSON 배열로만 반환하세요.

## 개체명 유형 ({entity_types})
- PS (인명): 사람 이름 (예: 김철수, 이영희)
- LC (지명): 장소, 지역, 국가, 도시 (예: 서울, 미국, 한강)
- OG (기관명): 회사, 기관, 단체, 정당 (예: 삼성전자, 국회, 정부)
- DT (날짜): 연도, 월, 일, 기간 (예: 2024년, 1월 3일, 지난해)
- TI (시간): 시각, 시간대 (예: 오전 10시, 오후 3시 30분)
- QT (수량): 숫자, 금액, 비율, 단위 포함 (예: 100명, 50억원, 30%)

## 규칙
- 조사(은/는/이/가/을/를/에서/으로/의 등)는 제외하고 핵심 명사만 추출
- JSON 배열만 출력, 다른 설명 금지
- 개체명 없으면 [] 반환

## 예시
입력: 서울시에서 김철수가 2024년 1월에 기자회견을 열었다.
출력: [{{"text": "서울시", "type": "LC"}}, {{"text": "김철수", "type": "PS"}}, {{"text": "2024년 1월", "type": "DT"}}]

입력: 삼성전자가 지난해 매출 300조원을 달성했다고 밝혔다.
출력: [{{"text": "삼성전자", "type": "OG"}}, {{"text": "지난해", "type": "DT"}}, {{"text": "300조원", "type": "QT"}}]

입력: {sentence}
출력:"""


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on punctuation or newlines."""
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
                if all(entity_text in tokens[i] or tokens[i] in entity_text
                       for _ in [None]):
                    tags[i] = f"B-{entity_type}"
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
        think: Enable thinking mode (prepends /think to prompt).
        concurrency: Max concurrent requests to the vLLM server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 512,
        think: bool = False,
        concurrency: int = 32,
    ) -> None:
        self.model = model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.max_tokens = max_tokens
        self.think = think
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(base_url=base_url, api_key="none")

        logger.info("VllmNERLabeler ready: %s @ %s", model, base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, text: str) -> List[dict]:
        """Label a text string. Returns a list of NER records (one per sentence)."""
        sentences = _split_sentences(text)
        return asyncio.run(self._label_sentences(sentences, id_offset=0))

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
        prefix = "/no_think\n" if not self.think else "/think\n"
        return prefix + _PROMPT_TEMPLATE.format(
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
            )
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


if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="vLLM NER pipeline: crawl + label + save")
    parser.add_argument("--urls", nargs="+", required=True, help="URLs to crawl")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--base-url", default="http://localhost:8081/v1", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Model name served by vLLM")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    from collectors.web_crawler import WebCrawler
    from collectors.pipeline import NERPipeline

    pipeline = NERPipeline(
        crawler=WebCrawler(),
        labeler=VllmNERLabeler(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            think=args.think,
        ),
        output_path=args.output,
    )
    records = pipeline.run(args.urls)
    print(f"Done. {len(records)} NER records saved to {args.output}")
