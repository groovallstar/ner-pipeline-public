"""LLM 기반 자연 PII 주입기.

기존 NER 레코드의 문장에 PII를 자연스럽게 삽입하도록 LLM에게 요청하고,
생성된 텍스트에서 span offset을 string match로 추출한다.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from augmenters.pii.config import (
    DEFAULT_DENSITY,
    DEFAULT_MERGE_RULES,
    DEFAULT_PII_LABELS,
)
from augmenters.pii.generators.base import generate_pii
from augmenters.pii.injector import merge_entities
from augmenters.pii.schema import Entity, Record

logger = logging.getLogger(__name__)

# ── 프롬프트 ─────────────────────────────────────────────────────────────

_INJECTION_PROMPT_JA = """\
あなたは日本語の文章編集の専門家です。

## タスク
以下の【原文】に、指定された【PII情報】を自然な日本語として文中に組み込んでください。

## ルール
1. PII の値は **1文字も変更しない**（そのままコピー）
2. 原文の固有表現（人名・地名・組織名など）はできるだけ保持する
3. 自然な文脈で挿入する（例: 「担当の○○は…」「連絡先は○○で…」「○○に住む…」）
4. 文末に羅列するのではなく、文の途中や適切な位置に埋め込む
5. 出力は **編集後の文章のみ**。説明や注釈は不要

## 原文
{original_text}

## PII情報
{pii_list}

## 出力"""

_INJECTION_PROMPT_VI = """\
Bạn là chuyên gia biên tập tiếng Việt.

## Nhiệm vụ
Chèn các thông tin PII được chỉ định vào [văn bản gốc] một cách tự nhiên
như tiếng Việt thường ngày.

## Quy tắc
1. Giá trị PII phải được sao chép **chính xác từng ký tự**
   (không sửa, không định dạng lại).
2. Giữ nguyên các thực thể trong văn bản gốc (tên người, địa danh, tổ
   chức, sản phẩm, sự kiện).
3. Chèn vào ngữ cảnh tự nhiên trong câu, không liệt kê ở cuối.
4. **KHÔNG** đặt giá trị PII ngay sau các từ-mào đầu cứng nhắc như:
   "Liên hệ:", "SĐT:", "Số điện thoại:", "Email:", "CCCD:",
   "Địa chỉ:", "Ngày sinh:", "Thẻ:", "ID:".
   Hãy lồng ghép vào câu (ví dụ: "Anh có thể liên hệ qua
   tran.linh@example.vn nếu cần thêm thông tin", thay vì
   "Liên hệ: tran.linh@example.vn").
5. **KHÔNG** xuất các tên nhãn tiếng Anh trong văn bản
   ("PHONE", "EMAIL", "ID_NUM", "ID_NUMBER", "CREDIT_CARD", "DAT",
   "ADDRESS", "NAME"). Chỉ dùng giá trị thực.
6. Đầu ra chỉ gồm **văn bản đã biên tập**, không kèm giải thích, ghi
   chú, hay bao bọc bằng ngoặc/markdown/tiền tố.

## Văn bản gốc
{original_text}

## Thông tin PII (chỉ tham khảo, không xuất nguyên dạng)
{pii_list}

## Văn bản đã biên tập"""

_INJECTION_PROMPTS: dict[str, str] = {
    'ja': _INJECTION_PROMPT_JA,
    'vi': _INJECTION_PROMPT_VI,
}

# 기존 호환용 별칭 (JA 기본 템플릿).
_INJECTION_PROMPT = _INJECTION_PROMPT_JA

_EMPTY_PII_LIST: dict[str, str] = {
    'ja': '（なし）',
    'vi': '(không có)',
}


def build_injection_prompt(
    original_text: str,
    pii_values: dict[str, str],
    lang: str = 'ja',
) -> str:
    """LLM 에 전달할 PII 주입 프롬프트를 언어별로 조립한다."""
    template = _INJECTION_PROMPTS.get(lang, _INJECTION_PROMPT_JA)
    if not pii_values:
        return template.format(
            original_text=original_text,
            pii_list=_EMPTY_PII_LIST.get(lang, _EMPTY_PII_LIST['ja']),
        )
    lines = [f'- {label}: {value}' for label, value in pii_values.items()]
    return template.format(
        original_text=original_text,
        pii_list='\n'.join(lines),
    )


# ── span 추출 ────────────────────────────────────────────────────────────

def extract_spans(
    text: str,
    *,
    pii_values: dict[str, str],
    original_entities: list[Entity],
) -> list[Entity]:
    """생성 텍스트에서 PII + 원본 엔티티의 span offset을 추출한다.

    PII 값이 텍스트에 없으면 ValueError.
    원본 엔티티가 텍스트에 없으면 해당 엔티티를 drop.
    """
    spans: list[Entity] = []
    used_ranges: list[tuple[int, int]] = []

    def _find_non_overlapping(needle: str) -> int | None:
        """used_ranges와 겹치지 않는 첫 번째 위치를 반환한다."""
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx == -1:
                return None
            end = idx + len(needle)
            if not any(
                s < end and idx < e for s, e in used_ranges
            ):
                return idx
            start = idx + 1

    # 1. PII 값 (필수 — 없으면 에러)
    for label, value in pii_values.items():
        idx = _find_non_overlapping(value)
        if idx is None:
            raise ValueError(
                f'PII value not found in generated text: '
                f'label={label} value={value!r}'
            )
        end = idx + len(value)
        spans.append(Entity(
            label=label,
            start_char=idx,
            end_char=end,
            text=value,
        ))
        used_ranges.append((idx, end))

    # 2. 원본 엔티티 (선택 — 없으면 drop)
    for ent in original_entities:
        idx = _find_non_overlapping(ent.text)
        if idx is None:
            logger.warning(
                'Original entity dropped (not found): '
                'label=%s text=%r', ent.label, ent.text,
            )
            continue
        end = idx + len(ent.text)
        spans.append(Entity(
            label=ent.label,
            start_char=idx,
            end_char=end,
            text=ent.text,
        ))
        used_ranges.append((idx, end))

    return sorted(spans, key=lambda e: e.start_char)


# ── LLM 클라이언트 프로토콜 ──────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """비동기 generate(prompt) -> str 을 구현하는 클라이언트."""

    async def generate(self, prompt: str) -> str: ...


# ── vLLM 클라이언트 ──────────────────────────────────────────────────────

class VllmClient:
    """vLLM OpenAI 호환 API 클라이언트."""

    def __init__(
        self,
        base_url: str = 'http://localhost:8081/v1',
        model: str = 'Qwen/Qwen3.5-27B',
        max_tokens: int = 2048,
        concurrency: int = 16,
    ) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=base_url, api_key='none',
        )
        self.model = model
        self.max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(concurrency)

    async def generate(self, prompt: str) -> str:
        async with self._semaphore:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
        return resp.choices[0].message.content or ''


# ── 메인 주입기 ──────────────────────────────────────────────────────────

class LLMInjector:
    """LLM 기반 자연 PII 주입기."""

    def __init__(
        self,
        client: LLMClient,
        lang: str = 'ja',
        seed: int = 42,
        pii_labels: list[str] | None = None,
        density: dict[int, float] | None = None,
        label_merge_rules: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self.lang = lang
        self.rng = random.Random(seed)
        self._labels = list(pii_labels or DEFAULT_PII_LABELS)
        self._density = dict(density or DEFAULT_DENSITY)
        self._merge_rules = dict(
            label_merge_rules
            if label_merge_rules is not None
            else DEFAULT_MERGE_RULES
        )

    def _sample_n(self) -> int:
        items = sorted(self._density.items())
        ks = [k for k, _ in items]
        ws = [v for _, v in items]
        return self.rng.choices(ks, weights=ws, k=1)[0]

    def _pick_labels(self, n: int) -> list[str]:
        if n <= 0:
            return []
        pool = list(self._labels)
        if n >= len(pool):
            self.rng.shuffle(pool)
            return pool
        return self.rng.sample(pool, n)

    def _generate_pii_values(
        self, labels: list[str],
    ) -> dict[str, str]:
        return {
            label: generate_pii(label, self.lang, self.rng)
            for label in labels
        }

    async def _inject_async(self, record: Record) -> Record:
        """단일 레코드 비동기 주입 본체."""
        n = self._sample_n()
        labels = self._pick_labels(n)

        if not labels:
            return Record(
                text=record.text,
                entities=list(record.entities),
                id=record.id,
            )

        pii_values = self._generate_pii_values(labels)
        prompt = build_injection_prompt(record.text, pii_values, self.lang)
        generated = await self._client.generate(prompt)
        generated = _clean_llm_output(generated)

        spans = extract_spans(
            generated,
            pii_values=pii_values,
            original_entities=record.entities,
        )
        # 라벨 병합 규칙 적용 (NAME → PER 등) — verifier와 label 일치를 위해.
        merged = merge_entities(spans, self._merge_rules)
        return Record(text=generated, entities=merged, id=record.id)

    def inject(self, record: Record) -> Record:
        """단일 레코드에 LLM으로 PII를 자연 삽입한다 (단발 호출 전용)."""
        return asyncio.run(self._inject_async(record))

    async def _gather_inject(
        self, records: list[Record],
    ) -> list[Record | None]:
        """전 레코드를 단일 event loop·asyncio.gather 로 병렬 처리한다.

        VllmClient 의 세마포어가 실제 동시 요청 수를 제어하므로 task 자체는
        모두 한 번에 생성해도 된다. event loop가 record마다 닫히지 않아
        AsyncOpenAI 의 connection pool이 재사용되며 'Event loop is closed'
        retry 폭주를 회피한다.
        """
        async def _safe(rec: Record) -> Record | None:
            try:
                return await self._inject_async(rec)
            except ValueError as e:
                logger.warning(
                    'Skipping record %s: %s', rec.id, e,
                )
                return None
        return await asyncio.gather(*(_safe(r) for r in records))

    def inject_dataset(
        self, records: Iterable[Record],
    ) -> Iterator[Record]:
        record_list = list(records)
        results = asyncio.run(self._gather_inject(record_list))
        for rec in results:
            if rec is not None:
                yield rec


def _clean_llm_output(text: str) -> str:
    """LLM 출력에서 불필요한 래핑을 제거한다."""
    text = text.strip()
    # <think>...</think> 블록 제거
    import re
    text = re.sub(
        r'<think>.*?</think>', '', text, flags=re.DOTALL,
    ).strip()
    # 마크다운 코드블록 제거
    if text.startswith('```') and text.endswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1]).strip()
    return text
