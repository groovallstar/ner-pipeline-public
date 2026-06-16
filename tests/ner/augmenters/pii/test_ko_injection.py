"""한국어 PII 생성기·주입기 동작 검증."""
from __future__ import annotations

import re

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.generators import ko
from ner.augmenters.pii.generators.base import generate_pii
from ner.augmenters.pii.injector import PIIInjector
from ner.augmenters.pii.schema import Entity, Record

_RRN_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]


def _make_record() -> Record:
    text = '봉준호 감독은 서울에서 기생충을 촬영했다'
    ents = [
        Entity(label='PER', start_char=0, end_char=3, text='봉준호'),
        Entity(label='LOC', start_char=8, end_char=10, text='서울'),
    ]
    return Record(text=text, entities=ents, id='r1')


def test_ko_name_from_known_pools():
    import random
    r = random.Random(1)
    for _ in range(50):
        name = ko.generate_name(r)
        assert name[0] in ko.KOREAN_FAMILY_NAMES
        assert name[1:] in ko.KOREAN_GIVEN_NAMES


def test_ko_phone_structure():
    import random
    r = random.Random(2)
    valid_starts = ('010', '070', '02', '+82') + tuple(
        ko.KOREAN_LANDLINE_AREAS
    )
    for _ in range(100):
        phone = ko.generate_phone(r)
        assert any(phone.startswith(s) for s in valid_starts), phone
        # 구분자 제거 후 숫자만 남는지(국제표기 + 제외)
        digits = re.sub(r'[ .\-]', '', phone).lstrip('+')
        assert digits.isdigit(), phone


def test_ko_rrn_format_and_invalid_checksum():
    """주민등록번호는 13자리 구조이되 검증 체크섬을 충족하지 않아야 한다."""
    import random
    r = random.Random(3)
    for _ in range(200):
        rrn = ko.generate_id_number(r)
        digits = rrn.replace('-', '')
        assert len(digits) == 13 and digits.isdigit(), rrn
        total = sum(int(c) * w for c, w in zip(digits[:12], _RRN_WEIGHTS))
        valid = (11 - (total % 11)) % 10
        # 실제 유효 번호 생성 방지: 마지막 자리가 정답 체크digit과 달라야 함
        assert int(digits[12]) != valid, rrn


def test_ko_generate_pii_dispatch():
    import random
    r = random.Random(4)
    email = generate_pii('EMAIL', 'ko', r)
    assert '@' in email and '.' in email.split('@')[1]
    cc = generate_pii('CREDIT_CARD', 'ko', r)
    assert sum(c.isdigit() for c in cc) in (15, 16)


def test_ko_injection_spans_consistent():
    cfg = InjectionConfig(lang='ko', seed=1)
    inj = PIIInjector(cfg)
    for i in range(50):
        out = inj.inject(_make_record())
        for ent in out.entities:
            assert out.text[ent.start_char:ent.end_char] == ent.text, (
                f'iter={i} label={ent.label}'
            )


def test_ko_pii_labels_restricted_to_four():
    """--pii-labels 4종 제한 시 PII 4종만 주입되고 원본 라벨은 불변."""
    four = ['EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD']
    cfg = InjectionConfig(
        lang='ko',
        pii_labels=four,
        density={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        seed=7,
    )
    inj = PIIInjector(cfg)
    injected_labels: set[str] = set()
    for _ in range(40):
        out = inj.inject(Record(text='기본 문장이다', entities=[]))
        for ent in out.entities:
            injected_labels.add(ent.label)
    assert injected_labels == set(four), injected_labels


def test_ko_seed_determinism():
    cfg_a = InjectionConfig(lang='ko', seed=42)
    cfg_b = InjectionConfig(lang='ko', seed=42)
    out_a = [
        PIIInjector(cfg_a).inject(_make_record()).text for _ in range(5)
    ]
    out_b = [
        PIIInjector(cfg_b).inject(_make_record()).text for _ in range(5)
    ]
    assert out_a == out_b


def test_ko_connectors_present():
    from ner.augmenters.pii.injector import _KO_CONNECTORS
    for label in ('EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'):
        assert label in _KO_CONNECTORS
        assert '{v}' in _KO_CONNECTORS[label]


def test_ko_llm_injection_prompt_natural():
    """llm 모드 ko 프롬프트: 값 포함 + 경직 접두 금지 규칙 명시."""
    from ner.augmenters.pii.llm_injector import (
        _EMPTY_PII_LIST, build_injection_prompt,
    )
    prompt = build_injection_prompt(
        '원문 문장이다',
        {'EMAIL': 'minjun@naver.com', 'PHONE': '010-1234-5678'},
        lang='ko',
    )
    assert '원문 문장이다' in prompt
    assert 'minjun@naver.com' in prompt and '010-1234-5678' in prompt
    # 경직된 안내어 뒤 배치 금지 규칙이 한국어로 들어있어야 함
    assert '금지' in prompt and '이메일:' in prompt
    assert _EMPTY_PII_LIST['ko'] == '(없음)'
