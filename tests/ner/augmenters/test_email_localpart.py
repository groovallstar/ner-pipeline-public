"""EMAIL 로컬파트 표면형 분포 수락 검사.

주입 이메일의 `local@domain` 중 `local` 이 실제 이메일 분포를 따르는지 본다.
분포 자체가 학습·평가의 기준이라 값 하나가 아니라 비율을 검사한다.
"""
import random
import string

import pytest

from ner.augmenters.generators import base
from ner.augmenters.generators.base import generate_pii

LANGS = ('en', 'ko', 'ja', 'vi')
SAMPLE = 8000
# n=8000 이면 이항분포 표준편차가 0.6%p 미만이라 3%p 여유면 우연 실패가 없다.
TOLERANCE = 0.03


def locals_for(lang: str, seed: int = 20260910) -> list[str]:
    """한 언어의 로컬파트 표본."""
    rng = random.Random(seed)
    out = []
    for _ in range(SAMPLE):
        value = generate_pii('EMAIL', lang, rng)
        out.append(value.rsplit('@', 1)[0])
    return out


@pytest.fixture(scope='module')
def samples() -> dict[str, list[str]]:
    return {lang: locals_for(lang) for lang in LANGS}


@pytest.mark.parametrize('lang', LANGS)
def test_digit_prefix_stays_under_five_percent(samples, lang):
    """숫자로 시작하는 로컬파트는 드물다 — 목표 2%, 상한 5%."""
    values = samples[lang]
    rate = sum(v[0].isdigit() for v in values) / len(values)
    assert rate <= 0.05, f'{lang} digit-prefix rate {rate:.1%}'


@pytest.mark.parametrize('lang', LANGS)
def test_dotless_lowercase_local_parts_are_common(samples, lang):
    """`alice@` 형태가 학습 분포 안에 있다 — 점 없는 소문자 ≥ 10%."""
    values = samples[lang]
    rate = sum(
        '.' not in v and v == v.lower() for v in values
    ) / len(values)
    assert rate >= 0.10, f'{lang} dotless-lowercase rate {rate:.1%}'


@pytest.mark.parametrize('lang', LANGS)
def test_most_local_parts_carry_no_digits(samples, lang):
    """숫자 없는 로컬파트가 다수다 — 목표 72%, 하한 60%."""
    values = samples[lang]
    rate = sum(
        not any(c.isdigit() for c in v) for v in values
    ) / len(values)
    assert rate >= 0.60, f'{lang} no-digit rate {rate:.1%}'


@pytest.mark.parametrize('lang', LANGS)
def test_separator_axis_matches_target_weights(samples, lang):
    """형태 축 비율이 `LOCAL_PART_FORM_WEIGHTS` 목표와 맞는다."""
    values = samples[lang]
    weights = base.LOCAL_PART_FORM_WEIGHTS
    total = sum(weights.values())
    observed = {
        'dot': sum('.' in v for v in values),
        'underscore': sum('_' in v for v in values),
        'hyphen': sum('-' in v for v in values),
    }
    for key in observed:
        got = observed[key] / len(values)
        want = weights[key] / total
        assert abs(got - want) <= TOLERANCE, (
            f'{lang} {key} {got:.1%} vs target {want:.1%}'
        )


@pytest.mark.parametrize('lang', LANGS)
def test_case_axis_matches_target_weights(samples, lang):
    """대소문자 축 비율이 `LOCAL_PART_CAPITALIZED_RATE` 목표와 맞는다."""
    values = samples[lang]
    rate = sum(any(c.isupper() for c in v) for v in values) / len(values)
    want = base.LOCAL_PART_CAPITALIZED_RATE
    assert abs(rate - want) <= TOLERANCE, (
        f'{lang} capitalized {rate:.1%} vs target {want:.1%}'
    )


@pytest.mark.parametrize('lang', LANGS)
def test_local_part_is_a_valid_surface(samples, lang):
    """점이 처음·끝·연속으로 오지 않고 허용 문자만 쓴다."""
    allowed = set(string.ascii_letters + string.digits + '.-_')
    for v in samples[lang]:
        assert v, f'{lang} empty local part'
        assert set(v) <= allowed, f'{lang} bad chars in {v!r}'
        assert not v.startswith('.') and not v.endswith('.'), v
        assert '..' not in v, v
        assert len(v) <= 64, v


@pytest.mark.parametrize('lang', ('ko', 'ja'))
def test_cjk_locales_use_romanized_names(samples, lang):
    """ko·ja 로컬파트가 무작위 자모 나열이 아니라 로마자 이름이다."""
    from ner.augmenters.generators import ja, ko

    mod = {'ko': ko, 'ja': ja}[lang]
    # 붙여쓰기·이니셜 형태는 구분자가 없어 토큰으로 못 자른다. 그래서
    # 부분 문자열로 찾되, 무작위 글자에 우연히 걸리지 않도록 네 글자
    # 이상인 이름만 본다.
    pool = [
        name.lower()
        for name in mod.EMAIL_FAMILY_NAMES + mod.EMAIL_GIVEN_NAMES
        if len(name) >= 4
    ]
    hit = sum(
        any(name in v.lower() for name in pool) for v in samples[lang]
    )
    rate = hit / len(samples[lang])
    assert rate >= 0.80, f'{lang} romanized-name rate {rate:.1%}'


@pytest.mark.parametrize('lang', LANGS)
def test_email_generation_is_deterministic_for_a_seed(lang):
    """같은 seed 면 같은 코퍼스가 나와야 재생성이 재현된다."""
    first = [generate_pii('EMAIL', lang, random.Random(7)) for _ in range(5)]
    second = [generate_pii('EMAIL', lang, random.Random(7)) for _ in range(5)]
    assert first == second
