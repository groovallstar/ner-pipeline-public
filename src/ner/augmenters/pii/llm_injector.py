"""LLM 기반 자연 PII 주입기.

기존 NER 레코드의 문장에 PII를 자연스럽게 삽입하도록 LLM에게 요청하고,
생성된 텍스트에서 span offset을 string match로 추출한다.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from ner.augmenters.pii.config import (
    DEFAULT_DENSITY,
    DEFAULT_MERGE_RULES,
    DEFAULT_PII_LABELS,
)
from ner.augmenters.pii.generators.base import generate_pii
from ner.augmenters.pii.injector import merge_entities
from ner.augmenters.pii.schema import Entity, Record

logger = logging.getLogger(__name__)

# ── 프롬프트 ─────────────────────────────────────────────────────────────

_INJECTION_PROMPT_JA = """\
あなたは日本語の文章編集の専門家です。

## タスク
以下の【原文】に、指定された【PII情報】を自然な日本語として文中に組み込んでください。

## ルール
1. PII の値は **1文字も変更しない**（そのままコピー）
2. 原文の固有表現（人名・地名・組織名など）はできるだけ保持する
3. 自然な文脈で挿入する（文末に羅列せず、文の途中や適切な位置に埋め込む。例: 「担当の○○は…」「連絡先は○○で…」「○○に住む…」）
4. **禁止**: PII 値を以下のような硬直した前置語の直後に配置しない:
   「担当者：」「連絡先：」「電話：」「電話番号：」「メール：」「住所：」「ID番号：」「カード番号：」「クレカ番号：」「生年月日：」。
   自然な文に埋め込むこと（例: 「連絡先は taro@example.jp までお願いします」、× 「連絡先：taro@example.jp」）。
5. **禁止**: 英語のラベル名（"NAME", "PHONE", "EMAIL", "ID_NUM", "ID_NUMBER", "CREDIT_CARD", "DAT", "ADDRESS"）を本文中に出力しない。実際の値のみ使う。
6. 出力は **編集後の文章のみ**。説明・注釈・括弧/markdown による包み込み・前置語は不要

## 原文
{original_text}

## PII情報（参考のみ。ラベル名はそのまま出力しない）
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

_INJECTION_PROMPT_KO = """\
당신은 한국어 문장 편집 전문가입니다.

## 작업
아래 【원문】에 지정된 【PII 정보】를 자연스러운 한국어로 문장 속에 녹여 넣으세요.

## 규칙
1. PII 값은 **한 글자도 바꾸지 않고** 그대로 복사한다.
2. 원문의 고유표현(인명·지명·기관명·작품명·사건명 등)은 최대한 보존한다.
3. 자연스러운 문맥으로 삽입한다(문장 끝에 나열하지 말고 문장 중간·적절한 위치에 녹여 넣는다. 예: 「담당 ○○○이…」 「연락은 ○○○으로…」 「○○○에 사는…」).
4. **금지**: PII 값을 다음과 같은 경직된 안내어 바로 뒤에 두지 않는다:
   「담당자:」 「연락처:」 「전화:」 「전화번호:」 「이메일:」 「주소:」 「주민등록번호:」 「카드번호:」 「생년월일:」.
   자연스러운 문장에 녹여 넣는다(예: 「연락은 taro@example.com 으로 주세요」, × 「이메일: taro@example.com」).
5. **금지**: 영어 라벨명("NAME", "PHONE", "EMAIL", "ID_NUM", "ID_NUMBER", "CREDIT_CARD", "DAT", "ADDRESS")을 본문에 출력하지 않는다. 실제 값만 사용한다.
6. 출력은 **편집된 문장만**. 설명·주석·괄호/markdown 래핑·안내어는 불필요.

## 원문
{original_text}

## PII 정보(참고용. 라벨명은 그대로 출력하지 않음)
{pii_list}

## 출력"""

_INJECTION_PROMPTS: dict[str, str] = {
    'ja': _INJECTION_PROMPT_JA,
    'vi': _INJECTION_PROMPT_VI,
    'ko': _INJECTION_PROMPT_KO,
}

# 기존 호환용 별칭 (JA 기본 템플릿).
_INJECTION_PROMPT = _INJECTION_PROMPT_JA

_EMPTY_PII_LIST: dict[str, str] = {
    'ja': '（なし）',
    'vi': '(không có)',
    'ko': '(없음)',
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

# ── PII 포맷 충돌 하드닝 ──────────────────────────────────────────────────
# LLM 주입기가 자연 삽입 중 pii_values 에 없는 카드/마이넘버 포맷 digit 열을
# 환각 생성하면 string-match 라벨러가 놓쳐 무라벨(O)로 코퍼스에 박힌다. 같은
# 포맷 표면이 라벨/무라벨로 공존하면 학습 모순이 되어 정밀도 천장을 만든다
# (측정: CC P 0.888 / ID_NUM P 0.949 → 결정론 relabel 재학습 +6.1pp / +2.82pp
# 회복). 주입 후 무라벨 포맷 열을 해당 PII 타입으로 일관 relabel 한다.
# 카드: 13~19 연속 digit 또는 4-4-4-4 구분자 그룹 (구분자 1~2칸 —
# 생성기가 ' '/'-'/''/'  ' 사용, 더블스페이스 포함)
_CC_FORMAT = re.compile(
    r'(?<!\d)(?:\d{4}[ -]{1,2}\d{4}[ -]{1,2}\d{4}[ -]{1,2}\d{4}|\d{13,19})'
    r'(?!\d)'
)
# 마이넘버: 12 연속 digit 또는 4-4-4 (카드 16자리의 앞 12자리는 제외)
_ID_FORMAT = re.compile(
    r'(?<![\d-])(?:\d{4}[ -]{1,2}\d{4}[ -]{1,2}\d{4}|\d{12})'
    r'(?![\d]|[ -]{1,2}\d{4})'
)


def harden_pii_format_collisions(
    text: str, spans: list[Entity]
) -> list[Entity]:
    """무라벨 카드/마이넘버 포맷 digit 열을 PII 타입으로 일관 relabel 한다.

    LLM injector 환각으로 pii_values 밖에서 생성된 포맷 열이 무라벨로 남는
    것을 막는다. 카드(긴 포맷)를 먼저 잡아 마이넘버가 카드 앞 12자리를
    오인하지 않게 한다. 기존 span 과 겹치면 건너뛴다.
    """
    used = [(e.start_char, e.end_char) for e in spans]

    def _overlaps(s: int, e: int) -> bool:
        return any(s < ue and us < e for us, ue in used)

    for label, pattern in (
        ('CREDIT_CARD', _CC_FORMAT), ('ID_NUM', _ID_FORMAT),
    ):
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            spans.append(Entity(
                label=label, start_char=s, end_char=e, text=m.group(),
            ))
            used.append((s, e))
    return spans


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

    def _overlaps(s: int, e: int) -> bool:
        return any(s < ue and us < e for us, ue in used_ranges)

    def _find_non_overlapping(
        needle: str,
    ) -> tuple[int, int, str] | None:
        """needle 의 (start, end, 실제매칭문자열) 을 찾는다.

        exact 매칭 우선, 실패 시 공백 정규화 매칭(needle 내부 공백 run 을
        `\\s+` 로). LLM 이 PII/엔티티의 연속 공백(카드번호 이중 공백 등)을
        단일 공백으로 정규화하는 경우를 흡수한다. 반환 문자열은 텍스트에
        실제로 박힌 표면형이라 offset 이 정확하다.
        """
        # 1차: exact
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx == -1:
                break
            end = idx + len(needle)
            if not _overlaps(idx, end):
                return idx, end, needle
            start = idx + 1
        # 2차: 공백 정규화 (다중 토큰일 때만)
        parts = needle.split()
        if len(parts) > 1:
            pat = re.compile(r'\s+'.join(re.escape(p) for p in parts))
            for m in pat.finditer(text):
                if not _overlaps(m.start(), m.end()):
                    return m.start(), m.end(), m.group()
        return None

    # 1. PII 값 (필수 — 없으면 에러)
    for label, value in pii_values.items():
        found = _find_non_overlapping(value)
        if found is None:
            raise ValueError(
                f'PII value not found in generated text: '
                f'label={label} value={value!r}'
            )
        start_char, end_char, matched = found
        spans.append(Entity(
            label=label,
            start_char=start_char,
            end_char=end_char,
            text=matched,
        ))
        used_ranges.append((start_char, end_char))

    # 2. 원본 엔티티 (선택 — 없으면 drop)
    for ent in original_entities:
        found = _find_non_overlapping(ent.text)
        if found is None:
            logger.warning(
                'Original entity dropped (not found): '
                'label=%s text=%r', ent.label, ent.text,
            )
            continue
        start_char, end_char, matched = found
        spans.append(Entity(
            label=ent.label,
            start_char=start_char,
            end_char=end_char,
            text=matched,
        ))
        used_ranges.append((start_char, end_char))

    # 3. 무라벨 PII 포맷 충돌 하드닝 (LLM 환각 카드/마이넘버 relabel)
    spans = harden_pii_format_collisions(text, spans)
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
        temperature: float = 0.7,
    ) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=base_url, api_key='none',
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(concurrency)

    async def generate(self, prompt: str) -> str:
        async with self._semaphore:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
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
