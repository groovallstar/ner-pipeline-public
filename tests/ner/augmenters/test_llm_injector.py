"""LLMInjector 단위 테스트 — TDD red phase."""
from __future__ import annotations

import pytest

from ner.augmenters.llm_injector import (
    LLMInjector,
    build_injection_prompt,
    extract_spans,
    harden_pii_format_collisions,
)
from ner.augmenters.schema import Entity, Record


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

    def test_vi_prompt_contains_negative_examples(self):
        """VI 프롬프트는 단서어 직전 부착 금지 + 영문 라벨 leakage 금지."""
        pii = {'EMAIL': 'tran.linh@example.vn', 'PHONE': '0912345678'}
        prompt = build_injection_prompt(
            'Đại học Quốc gia Hà Nội là một cơ sở giáo dục.',
            pii,
            lang='vi',
        )
        assert 'Đại học Quốc gia Hà Nội' in prompt
        assert 'tran.linh@example.vn' in prompt
        # Rule 4: 단서어 부정 예시
        assert 'Liên hệ:' in prompt
        assert 'SĐT:' in prompt
        # Rule 5: 영문 라벨명 leakage 금지
        assert 'ID_NUMBER' in prompt
        assert 'CREDIT_CARD' in prompt
        # 일본어 템플릿 누설 없음
        assert '原文' not in prompt

    def test_vi_empty_pii(self):
        prompt = build_injection_prompt('Văn bản gốc.', {}, lang='vi')
        assert 'Văn bản gốc.' in prompt
        assert '(không có)' in prompt

    def test_ja_prompt_contains_negative_examples(self):
        """JA 프롬프트는 단서어 직전 부착 금지 + 영문 라벨 leakage 금지 (Rule 4·5)."""
        pii = {'EMAIL': 'taro@example.jp', 'PHONE': '090-1234-5678'}
        prompt = build_injection_prompt(
            '東京には多くの大学がある。',
            pii,
            lang='ja',
        )
        assert '東京には多くの大学がある。' in prompt
        assert 'taro@example.jp' in prompt
        # Rule 4: 단서어 부정 예시 (硬直した前置語)
        assert '担当者：' in prompt
        assert '連絡先：' in prompt
        assert '電話：' in prompt
        # Rule 5: 영문 라벨명 leakage 금지
        assert 'ID_NUMBER' in prompt
        assert 'CREDIT_CARD' in prompt
        # VI 템플릿 누설 없음
        assert 'KHÔNG' not in prompt
        assert 'tiếng Việt' not in prompt

    def test_ja_empty_pii(self):
        prompt = build_injection_prompt('原文テスト。', {}, lang='ja')
        assert '原文テスト。' in prompt
        assert '（なし）' in prompt


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

    def test_pii_whitespace_normalized_match(self):
        """LLM이 PII 연속 공백을 정규화해도 매칭(카드 이중 공백 등)."""
        # 생성기는 이중 공백이지만 LLM이 단일 공백으로 정규화한 경우
        text = '결제는 3791 1888 6964 4832 카드로 처리했습니다.'
        pii = {'CREDIT_CARD': '3791  1888  6964  4832'}
        spans = extract_spans(text, pii_values=pii, original_entities=[])

        cc = next(s for s in spans if s.label == 'CREDIT_CARD')
        # 실제 텍스트 표면형(단일 공백)으로 저장, offset 정확
        assert cc.text == '3791 1888 6964 4832'
        assert text[cc.start_char:cc.end_char] == cc.text


# ── PII 포맷 충돌 하드닝 ──────────────────────────────────────────────────

class TestHardenPiiFormatCollisions:
    """LLM 환각으로 무라벨된 카드/마이넘버 포맷 열을 일관 relabel."""

    def test_unlabeled_card_solid_relabeled(self):
        text = '決済番号は4065055130224539で処理した。'
        spans = harden_pii_format_collisions(text, [])
        cc = [s for s in spans if s.label == 'CREDIT_CARD']
        assert len(cc) == 1
        assert cc[0].text == '4065055130224539'

    def test_unlabeled_card_grouped_relabeled(self):
        text = 'カードは 4065 0551 3022 4539 を利用。'
        spans = harden_pii_format_collisions(text, [])
        cc = [s for s in spans if s.label == 'CREDIT_CARD']
        assert len(cc) == 1
        assert cc[0].text == '4065 0551 3022 4539'

    def test_unlabeled_card_double_space_relabeled(self):
        """생성기가 만드는 더블스페이스 구분자 카드도 relabel 한다."""
        text = 'カードは 3528  9893  7864  1469 を利用。'
        spans = harden_pii_format_collisions(text, [])
        cc = [s for s in spans if s.label == 'CREDIT_CARD']
        assert len(cc) == 1
        assert cc[0].text == '3528  9893  7864  1469'

    def test_unlabeled_mynumber_relabeled(self):
        text = '管理番号 1917-0792-7808 で登録されている。'
        spans = harden_pii_format_collisions(text, [])
        idn = [s for s in spans if s.label == 'ID_NUM']
        assert len(idn) == 1
        assert idn[0].text == '1917-0792-7808'

    def test_card_prefix_not_mislabeled_as_id(self):
        """16자리 카드의 앞 12자리를 ID_NUM 으로 오인하지 않는다."""
        text = '決済 5120-7030-4950-5800 を使用。'
        spans = harden_pii_format_collisions(text, [])
        labels = [s.label for s in spans]
        assert 'CREDIT_CARD' in labels
        assert 'ID_NUM' not in labels

    def test_existing_span_preserved(self):
        """이미 라벨된 카드 자리는 중복 라벨하지 않는다."""
        text = '決済 4065055130224539 を使用。'
        existing = [_ent('CREDIT_CARD', 3, 19, '4065055130224539')]
        spans = harden_pii_format_collisions(text, list(existing))
        cc = [s for s in spans if s.label == 'CREDIT_CARD']
        assert len(cc) == 1

    def test_short_number_not_relabeled(self):
        """연도·일반 짧은 숫자는 relabel 하지 않는다."""
        text = '2020年に設立され、約1500人が在籍した。'
        spans = harden_pii_format_collisions(text, [])
        assert spans == []

    def test_extract_spans_applies_hardening(self):
        """extract_spans 가 무라벨 환각 카드를 relabel 한다."""
        text = '担当の山田太郎が、5120-7030-4950-5800のカードで決済。'
        pii = {'NAME': '山田太郎'}
        spans = extract_spans(text, pii_values=pii, original_entities=[])
        labels = {s.label for s in spans}
        assert 'NAME' in labels
        assert 'CREDIT_CARD' in labels


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
