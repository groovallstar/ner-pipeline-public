"""베트남어 PII 생성기·주입기 동작 검증."""
from __future__ import annotations

import random
import re

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.generators import vi
from ner.augmenters.pii.generators.base import generate_pii
from ner.augmenters.pii.injector import PIIInjector
from ner.augmenters.pii.schema import Entity, Record

# VI canonical PII 라벨 집합 (NAME/ADDRESS는 병합 전 내부 토큰).
_VI_PII_CANONICAL = frozenset(
    {'PER', 'LOC', 'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'}
)


def _make_record() -> Record:
    # 멀티바이트 베트남어 문자로 인해 char offset을 직접 검증:
    # 'Nguyen Van An' = [0, 13), 'Hà Nội' = [21, 27)
    text = 'Nguyen Van An sống ở Hà Nội và làm việc tại đây'
    ents = [
        Entity(label='PER', start_char=0, end_char=13, text='Nguyen Van An'),
        Entity(label='LOC', start_char=21, end_char=27, text='Hà Nội'),
    ]
    return Record(text=text, entities=ents, id='r1')


def test_vi_name_three_parts():
    """generate_name은 공백으로 구분된 3부분(성·중간·이름)을 반환한다."""
    r = random.Random(1)
    for _ in range(50):
        name = vi.generate_name(r)
        parts = name.split(' ')
        assert len(parts) == 3, name
        assert all(p for p in parts), name


def test_vi_name_from_known_pools():
    """생성된 성(family)이 알려진 성 풀 중 하나에 속한다."""
    r = random.Random(2)
    all_families = set(vi.VIETNAMESE_FAMILY_NAMES) | set(
        vi.VIETNAMESE_FAMILY_NAMES_DIA
    )
    for _ in range(50):
        name = vi.generate_name(r)
        family = name.split(' ')[0]
        assert family in all_families, name


def test_vi_phone_starts_with_valid_prefix():
    """generate_phone 출력은 베트남 통신사 prefix(09/03/07/08/05) 또는
    국제 표기(+84·(+84))로 시작한다."""
    r = random.Random(3)
    valid_prefixes = ('09', '03', '07', '08', '05', '+84', '(+84)', '0')
    for _ in range(100):
        phone = vi.generate_phone(r)
        assert any(phone.startswith(p) for p in valid_prefixes), phone


def test_vi_phone_digits_only_after_strip():
    """구분자(공백·하이픈·괄호·+84 prefix)를 제거하면 숫자만 남는다."""
    r = random.Random(4)
    for _ in range(100):
        phone = vi.generate_phone(r)
        stripped = re.sub(r'[\s\-().+]', '', phone)
        assert stripped.isdigit(), phone


def test_vi_dat_known_formats():
    """generate_dat 출력은 dd/mm/yyyy·yyyy-mm-dd·dd-mm-yyyy 형식 중 하나다."""
    r = random.Random(5)
    pat_dmy_slash = re.compile(r'^\d{2}/\d{2}/\d{4}$')
    pat_ymd_dash = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    pat_dmy_dash = re.compile(r'^\d{2}-\d{2}-\d{4}$')
    for _ in range(100):
        dat = vi.generate_dat(r)
        matched = (
            pat_dmy_slash.match(dat)
            or pat_ymd_dash.match(dat)
            or pat_dmy_dash.match(dat)
        )
        assert matched, dat


def test_vi_id_number_12_digits():
    """generate_id_number는 구분자를 제거하면 정확히 12자리 숫자다(CCCD)."""
    r = random.Random(6)
    for _ in range(100):
        id_num = vi.generate_id_number(r)
        digits = re.sub(r'[-]', '', id_num)
        assert len(digits) == 12 and digits.isdigit(), id_num


def test_vi_address_nonempty_and_has_city():
    """generate_address 출력이 비어있지 않고 알려진 도시명을 포함한다."""
    r = random.Random(7)
    for _ in range(50):
        addr = vi.generate_address(r)
        assert addr, 'address must not be empty'
        assert any(city in addr for city in vi.VIETNAMESE_CITIES), addr


def test_vi_generate_pii_dispatch_email():
    """generate_pii('EMAIL', 'vi') 는 @ 와 도메인을 포함한다."""
    r = random.Random(8)
    for _ in range(20):
        email = generate_pii('EMAIL', 'vi', r)
        assert '@' in email, email
        domain_part = email.split('@')[1]
        assert '.' in domain_part, email


def test_vi_generate_pii_dispatch_credit_card():
    """generate_pii('CREDIT_CARD', 'vi') 는 15 또는 16자리 숫자를 가진다."""
    r = random.Random(9)
    for _ in range(20):
        cc = generate_pii('CREDIT_CARD', 'vi', r)
        digit_count = sum(c.isdigit() for c in cc)
        assert digit_count in (15, 16), cc


def test_vi_seed_determinism_generators():
    """동일 seed로 생성기를 반복 호출하면 동일한 결과를 반환한다."""
    for fn in (
        vi.generate_name,
        vi.generate_phone,
        vi.generate_address,
        vi.generate_dat,
        vi.generate_id_number,
    ):
        out_a = [fn(random.Random(42)) for _ in range(10)]
        out_b = [fn(random.Random(42)) for _ in range(10)]
        assert out_a == out_b, f'{fn.__name__} is not deterministic'


def test_vi_injection_spans_consistent():
    """주입 후 엔티티의 (start, end)가 실제 텍스트 슬라이스와 일치한다."""
    cfg = InjectionConfig(lang='vi', seed=1)
    inj = PIIInjector(cfg)
    for i in range(50):
        out = inj.inject(_make_record())
        for ent in out.entities:
            assert out.text[ent.start_char:ent.end_char] == ent.text, (
                f'iter={i} label={ent.label}'
            )


def test_vi_injected_labels_in_canonical_set():
    """주입된 PII 라벨은 canonical PII 집합 안에 있다."""
    cfg = InjectionConfig(
        lang='vi',
        density={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        seed=10,
    )
    inj = PIIInjector(cfg)
    for _ in range(30):
        out = inj.inject(Record(text='Đây là câu ví dụ.', entities=[]))
        for ent in out.entities:
            assert ent.label in _VI_PII_CANONICAL, ent.label


def test_vi_pii_labels_restricted():
    """pii_labels 제한 시 지정한 라벨(병합 후)만 주입된다."""
    four = ['EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD']
    cfg = InjectionConfig(
        lang='vi',
        pii_labels=four,
        density={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        seed=11,
    )
    inj = PIIInjector(cfg)
    injected: set[str] = set()
    for _ in range(40):
        out = inj.inject(Record(text='Câu kiểm tra.', entities=[]))
        for ent in out.entities:
            injected.add(ent.label)
    assert injected == set(four), injected


def test_vi_seed_determinism_injector():
    """동일 seed의 PIIInjector는 동일한 텍스트를 생성한다."""
    cfg_a = InjectionConfig(lang='vi', seed=42)
    cfg_b = InjectionConfig(lang='vi', seed=42)
    out_a = [
        PIIInjector(cfg_a).inject(_make_record()).text for _ in range(5)
    ]
    out_b = [
        PIIInjector(cfg_b).inject(_make_record()).text for _ in range(5)
    ]
    assert out_a == out_b


def test_vi_empty_input_no_crash():
    """빈 텍스트·빈 엔티티 레코드에서 주입이 정상 완료된다."""
    cfg = InjectionConfig(lang='vi', seed=13)
    inj = PIIInjector(cfg)
    out = inj.inject(Record(text='', entities=[]))
    for ent in out.entities:
        assert out.text[ent.start_char:ent.end_char] == ent.text


def test_vi_multiple_injection_no_span_overlap():
    """3개 PII 고정 주입 시 엔티티 span이 서로 겹치지 않는다."""
    cfg = InjectionConfig(
        lang='vi',
        density={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        seed=15,
    )
    inj = PIIInjector(cfg)
    for _ in range(30):
        out = inj.inject(_make_record())
        spans = [(e.start_char, e.end_char) for e in out.entities]
        spans.sort()
        for j in range(len(spans) - 1):
            assert spans[j][1] <= spans[j + 1][0], (
                f'overlap: {spans[j]} vs {spans[j + 1]}'
            )


def test_vi_connectors_present():
    """_VI_CONNECTORS에 canonical PII 키 전체가 등록되어 있다."""
    from ner.augmenters.pii.injector import _VI_CONNECTORS
    for label in ('NAME', 'PHONE', 'ADDRESS', 'DAT', 'ID_NUM', 'EMAIL',
                  'CREDIT_CARD'):
        assert label in _VI_CONNECTORS, label
        assert '{v}' in _VI_CONNECTORS[label], label
