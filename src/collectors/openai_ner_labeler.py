"""OpenAI-based NER labeler with batch processing.

API key is read from the OPENAI_API_KEY environment variable.
Never hardcode or pass the key as an argument.

Usage (standalone):
    export OPENAI_API_KEY="sk-..."
    python3 -m collectors.openai_ner_labeler \
        --urls "https://n.news.naver.com/article/138/0002222481" \
        --output /tmp/result.jsonl \
        --model gpt-4o-mini
"""

import argparse
import json
import logging
import os
import re
import time
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = ["PS", "LC", "OG", "DT", "TI", "QT"]

_SYSTEM_PROMPT = """당신은 한국어 개체명 인식(NER) 전문가입니다. 주어진 문장들에서 개체명을 추출하여 JSON으로만 반환하세요.

개체명 유형:
- PS (인명): 사람 이름 (예: 김철수, 이영희)
- LC (지명): 장소, 지역, 국가, 도시 (예: 서울, 미국, 한강)
- OG (기관명): 회사, 기관, 단체, 정당 (예: 삼성전자, 국회, 정부)
- DT (날짜): 연도, 월, 일, 기간 (예: 2024년, 1월 3일, 지난해)
- TI (시간): 시각, 시간대 (예: 오전 10시, 오후 3시 30분)
- QT (수량): 숫자, 금액, 비율, 단위 포함 (예: 100명, 50억원, 30%)

규칙:
- 조사(은/는/이/가/을/를/에서/으로/의 등)는 제외하고 핵심 명사만 추출
- 각 문장 인덱스를 키로 하는 JSON 객체 반환
- 개체명 없으면 빈 배열
- JSON 외 다른 출력 금지"""

_USER_PROMPT_TEMPLATE = """아래 문장들에서 개체명을 추출하세요.

입력:
{sentences}

출력 형식 예시 (문장이 3개일 때):
{{"0": [{{"text": "서울시", "type": "LC"}}, {{"text": "김철수", "type": "PS"}}], "1": [], "2": [{{"text": "삼성전자", "type": "OG"}}]}}

출력:"""


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

        # 2. Substring match
        if not matched:
            for i in range(len(tokens) - n + 1):
                if all(entity_text in tokens[i] or tokens[i] in entity_text for _ in [None]):
                    tags[i] = f"B-{entity_type}"
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
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "none")
        if not base_url and api_key == "none":
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.model = model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.max_tokens_per_batch = max_tokens_per_batch
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

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

    def label_records(self, records: List[dict]) -> List[dict]:
        results = []
        for raw in records:
            text = raw.get("text", "")
            if not text:
                continue
            results.extend(self.label(text))
        return results

    def _call_api_batch(self, sentences: List[str]) -> List[List[dict]]:
        """Call OpenAI API for a batch of sentences."""
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
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)

            if not isinstance(data, dict):
                logger.warning("Unexpected response structure, falling back: %s", raw[:200])
                return [[] for _ in sentences]

            results = []
            for i in range(len(sentences)):
                spans = data.get(str(i), [])
                results.append(spans if isinstance(spans, list) else [])
            return results

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("API call failed (%s), returning empty spans", e)
            return [[] for _ in sentences]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

    from collectors.web_crawler import WebCrawler
    from collectors.pipeline import NERPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="OpenAI NER pipeline: crawl + label + save")
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-tokens-per-batch", type=int, default=1000,
                        help="Max total tokens per batch (token-based batching)")
    args = parser.parse_args()

    pipeline = NERPipeline(
        crawler=WebCrawler(),
        labeler=OpenAINERLabeler(
            model=args.model,
            max_tokens_per_batch=args.max_tokens_per_batch,
        ),
        output_path=args.output,
    )
    records = pipeline.run(args.urls)
    print(f"Done. {len(records)} NER records saved to {args.output}")
