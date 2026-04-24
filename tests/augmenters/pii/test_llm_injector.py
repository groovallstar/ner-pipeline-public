"""LLMInjector 단위 테스트 — TDD red phase."""
from __future__ import annotations

import pytest

from augmenters.pii.llm_injector import (
    LLMInjector,
    build_injection_prompt,
    extract_spans,
)
from augmenters.pii.schema import Entity, Record


# ── 헬퍼 ─────────────────────────────────────────────────────────────────

def _ent(label: str, start: int, end: int, text: str) -> Entity:
    return Entity(label=label, start_char=start, end_char=end, text=text)


def _rec(text: str, entities: list[Entity], rid: str = '0') -> Record:
    return Record(text=text, entities=entities, id=rid)


# ── 프롬프트 생성 ────────────────────────────────────────────────────────

class TestBuildInjectionPrompt:
    """build_injection_prompt가 올바른 프롬프트를 생성하는지."""

    def test_contains_original_text(self):
        prompt = build_injection_prompt(
            '創業にはミツカンも出資した。',
            {'NAME': '山田太郎', 'PHONE': '090-1234-5678'},
        )
        assert '創業にはミツカンも出資した。' in prompt

    def test_contains_pii_values(self):
        pii = {'NAME': '山田太郎', 'EMAIL': 'taro@example.com'}
        prompt = build_injection_prompt('テスト文。', pii)
        assert '山田太郎' in prompt
        assert 'taro@example.com' in prompt

    def test_contains_pii_labels(self):
        pii = {'PHONE': '090-1234-5678'}
        prompt = build_injection_prompt('テスト文。', pii)
        assert 'PHONE' in prompt

    def test_empty_pii_still_valid(self):
        prompt = build_injection_prompt('テスト文。', {})
        assert 'テスト文。' in prompt


# ── span 추출 ────────────────────────────────────────────────────────────

class TestExtractSpans:
    """생성 텍스트에서 PII + 원본 span offset을 추출."""

    def test_pii_exact_match(self):
        """PII 값이 생성 텍스트에 있으면 정확한 offset."""
        text = '担当の山田太郎（090-1234-5678）が窓口です。'
        pii = {'NAME': '山田太郎', 'PHONE': '090-1234-5678'}
        spans = extract_spans(text, pii_values=pii, original_entities=[])

        pii_spans = {s.label: s for s in spans}
        # NAME → PER 병합은 별도 단계이므로 여기서는 원래 라벨 유지
        assert pii_spans['NAME'].text == '山田太郎'
        assert text[pii_spans['NAME'].start_char:pii_spans['NAME'].end_char] == '山田太郎'
        assert pii_spans['PHONE'].text == '090-1234-5678'
        assert text[pii_spans['PHONE'].start_char:pii_spans['PHONE'].end_char] == '090-1234-5678'

    def test_original_entity_found(self):
        """원본 엔티티 텍스트가 생성 텍스트에 있으면 새 offset 추출."""
        text = '創業にはミツカンも出資した。担当は山田太郎です。'
        original = [_ent('ORG', 4, 8, 'ミツカン')]
        pii = {'NAME': '山田太郎'}
        spans = extract_spans(text, pii_values=pii, original_entities=original)

        labels = {s.label: s for s in spans}
        assert 'NAME' in labels
        assert 'ORG' in labels
        assert labels['ORG'].text == 'ミツカン'
        assert text[labels['ORG'].start_char:labels['ORG'].end_char] == 'ミツカン'

    def test_original_entity_missing_is_dropped(self):
        """원본 엔티티가 생성 텍스트에 없으면 drop."""
        text = '山田太郎が窓口です。'
        original = [_ent('ORG', 0, 4, 'ミツカン')]
        pii = {'NAME': '山田太郎'}
        spans = extract_spans(text, pii_values=pii, original_entities=original)

        labels = [s.label for s in spans]
        assert 'ORG' not in labels
        assert 'NAME' in labels

    def test_pii_value_missing_raises(self):
        """PII 값이 생성 텍스트에 없으면 ValueError."""
        text = '普通の文章です。'
        pii = {'PHONE': '090-1234-5678'}
        with pytest.raises(ValueError, match='PII.*not found'):
            extract_spans(text, pii_values=pii, original_entities=[])

    def test_duplicate_text_uses_distinct_offsets(self):
        """동일 텍스트가 여러 번 나타나면 각각 다른 offset."""
        text = '山田太郎と山田太郎が来た。'
        original = [_ent('PER', 0, 4, '山田太郎')]
        pii = {'NAME': '山田太郎'}
        spans = extract_spans(text, pii_values=pii, original_entities=original)

        offsets = [(s.start_char, s.end_char) for s in spans]
        # 두 span이 겹치지 않아야 한다
        assert len(set(offsets)) == len(offsets)


# ── LLMInjector 통합 (모의 LLM) ─────────────────────────────────────────

class FakeLLMClient:
    """프롬프트에서 PII 값을 파싱하여 자연스러운 문장을 조립하는 모의 클라이언트."""

    def __init__(self, base_text: str) -> None:
        self._base = base_text
        self.last_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        import re
        self.last_prompt = prompt
        # 프롬프트에서 PII 값 추출 (- LABEL: VALUE 형태)
        pairs = re.findall(r'^- (\w+): (.+)$', prompt, re.MULTILINE)
        if not pairs:
            return self._base
        # 원본 텍스트 끝에 PII를 자연스럽게 조합
        parts = [self._base.rstrip('。')]
        for label, value in pairs:
            if label == 'NAME':
                parts.append(f'担当の{value}が対応した')
            elif label == 'PHONE':
                parts.append(f'連絡先は{value}である')
            elif label == 'EMAIL':
                parts.append(f'メールは{value}宛')
            else:
                parts.append(f'{value}')
        return '、'.join(parts) + '。'


class TestLLMInjector:
    """LLMInjector.inject() 통합 테스트 (모의 LLM)."""

    def test_inject_returns_valid_record(self):
        original = _rec(
            '創業にはミツカンも出資した。',
            [_ent('ORG', 4, 8, 'ミツカン')],
        )
        fake = FakeLLMClient('創業にはミツカンも出資した。')
        injector = LLMInjector(
            client=fake, lang='ja', seed=42,
            pii_labels=['NAME', 'PHONE'],
            density={2: 1.0},
        )
        result = injector.inject(original)

        assert isinstance(result, Record)
        # offset 정합성
        for ent in result.entities:
            assert result.text[ent.start_char:ent.end_char] == ent.text
        # PII 2개 주입됨 (NAME은 인명으로 병합)
        pii_labels = {'PER', 'PHONE'}
        injected = [e for e in result.entities if e.label in pii_labels]
        assert len(injected) == 2

    def test_inject_preserves_original_entities(self):
        original = _rec(
            '東京都にある施設。',
            [_ent('LOC', 0, 3, '東京都')],
        )
        fake = FakeLLMClient('東京都にある施設。')
        injector = LLMInjector(
            client=fake, lang='ja', seed=42,
            pii_labels=['NAME'],
            density={1: 1.0},
        )
        result = injector.inject(original)

        labels = {e.label for e in result.entities}
        assert 'LOC' in labels
        # NAME → PER 병합 적용됨
        assert 'PER' in labels

    def test_inject_applies_name_merge_rule(self):
        """NAME 라벨은 inject() 결과에서 PER로 병합되어야 한다."""
        original = _rec('テスト文。', [])
        fake = FakeLLMClient('テスト文。')
        injector = LLMInjector(
            client=fake, lang='ja', seed=42,
            pii_labels=['NAME'],
            density={1: 1.0},
        )
        result = injector.inject(original)
        labels = {e.label for e in result.entities}
        assert 'PER' in labels
        assert 'NAME' not in labels

    def test_inject_with_zero_pii(self):
        """density에서 0이 선택되면 PII 없이 원본 반환."""
        original = _rec(
            '普通の文章。',
            [_ent('LOC', 0, 2, '普通')],
        )
        fake = FakeLLMClient('普通の文章。')
        injector = LLMInjector(
            client=fake, lang='ja', seed=42,
            density={0: 1.0},
        )
        result = injector.inject(original)

        assert result.text == '普通の文章。'
