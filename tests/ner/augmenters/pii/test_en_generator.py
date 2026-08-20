"""영어(미국) PII 생성기 — 무효 대역 강제가 핵심이다.

합성 PII 가 실제 유효 번호가 되면 학습 데이터가 누군가의 진짜 식별자를
담게 된다. `ko.py` 가 체크섬이 틀린 주민번호를 만드는 것과 같은 이유로,
EN 은 SSN 지역번호와 전화 국번을 미발급·예약 대역에 가둔다.
"""
import random
import re

import pytest

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.generators import en
from ner.augmenters.pii.generators.base import generate_pii

# 사회보장국이 한 번도 발급하지 않은 SSN 지역번호.
NEVER_ISSUED_AREAS = {'000', '666'} | {str(n) for n in range(900, 1000)}

SAMPLE = 3000


def _rng():
    return random.Random(20260818)


def test_ssn_area_is_always_never_issued():
    """수락: 생성된 SSN 의 지역번호가 미발급 대역에만 있다."""
    rng = _rng()
    for _ in range(SAMPLE):
        digits = re.sub(r'\D', '', en.generate_id_number(rng))
        assert len(digits) == 9
        assert digits[:3] in NEVER_ISSUED_AREAS


def test_ssn_group_and_serial_are_never_all_zero():
    """`00` 그룹과 `0000` 일련번호도 발급되지 않는 조합이다."""
    rng = _rng()
    for _ in range(SAMPLE):
        digits = re.sub(r'\D', '', en.generate_id_number(rng))
        assert digits[3:5] != '00'
        assert digits[5:] != '0000'


def test_phone_uses_the_reserved_fictional_block():
    """NANP 가 창작물용으로 예약한 `555-0100`~`555-0199` 만 쓴다."""
    rng = _rng()
    for _ in range(SAMPLE):
        digits = re.sub(r'\D', '', en.generate_phone(rng))
        # 국가번호 1 이 붙는 표기가 있어 뒤 7 자리로 본다.
        assert digits[-7:-4] == '555'
        line = int(digits[-4:])
        assert 100 <= line <= 199


def test_generated_values_are_deterministic_for_a_seed():
    """`--seed` 로 재현되어야 데이터셋을 다시 만들 수 있다."""
    fields = ('generate_name', 'generate_phone', 'generate_address',
              'generate_dat', 'generate_id_number')
    first = [getattr(en, f)(random.Random(7)) for f in fields]
    second = [getattr(en, f)(random.Random(7)) for f in fields]
    assert first == second


def test_names_look_like_us_names():
    rng = _rng()
    for _ in range(200):
        name = en.generate_name(rng)
        parts = name.split()
        assert len(parts) in (2, 3)
        assert parts[0] in en.US_GIVEN_NAMES
        assert parts[-1] in en.US_FAMILY_NAMES


def test_addresses_carry_a_us_city_or_street():
    rng = _rng()
    for _ in range(200):
        address = en.generate_address(rng)
        assert any(street in address for street in en.US_STREETS)


def test_email_uses_us_domains_not_shared_pool():
    """공용 목록은 ja·vi·ko 도메인이 섞여 있어 영문 문맥에 맞지 않는다."""
    rng = _rng()
    for _ in range(500):
        email = generate_pii('EMAIL', 'en', rng)
        assert email.rsplit('@', 1)[1] in en.EMAIL_DOMAINS


def test_dispatch_accepts_en_for_every_label():
    rng = _rng()
    for label in ('NAME', 'PHONE', 'ADDRESS', 'DAT',
                  'ID_NUM', 'EMAIL', 'CREDIT_CARD'):
        assert generate_pii(label, 'en', rng)


def test_injection_config_accepts_en():
    InjectionConfig(lang='en').validate()


def test_injection_config_still_rejects_unknown_lang():
    with pytest.raises(ValueError):
        InjectionConfig(lang='de').validate()
