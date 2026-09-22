"""PII 생성기 공용 유틸 및 Protocol."""
from __future__ import annotations

import random
import re
import string

EMAIL_DOMAINS: list[str] = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'proton.me',
    'yahoo.co.jp', 'docomo.ne.jp', 'softbank.ne.jp', 'ezweb.ne.jp',
    'example.co.jp',
    'gmail.com.vn', 'yahoo.com.vn', 'example.com.vn', 'corp.vn',
    'mail.vn', 'vnn.vn', 'fpt.com.vn', 'viettel.com.vn', 'vnpt.vn',
    'icloud.com', 'gmx.com', 'zoho.com',
    'naver.com', 'daum.net', 'hanmail.net', 'kakao.com', 'nate.com',
]

# 로컬파트 형태별 목표 비율. 실제 이메일이 이름 두 토큰을 어떤 구분자로
# 잇는지의 분포를 따른다. 근거와 축 설계는
# `docs/manual/pipeline/2-augmentation.md` 의 EMAIL 로컬파트 절에 있다.
LOCAL_PART_FORM_WEIGHTS: dict[str, int] = {
    'dot': 30,        # emily.thompson
    'plain': 22,      # emilythompson
    'single': 20,     # emily
    'initial': 15,    # ethompson
    'underscore': 8,  # emily_thompson
    'hyphen': 5,      # emily-thompson
}

# 형태와 무관하게 걸리는 두 축. 대문자는 토큰 첫 글자에만 붙고, 숫자는
# 접두·접미 중 한쪽에만 붙는다. 접두를 0 으로 두지 않는 것은 숫자로
# 시작하는 이메일이 드물게 실재하기 때문이다.
LOCAL_PART_CAPITALIZED_RATE = 0.10
LOCAL_PART_DIGIT_PREFIX_RATE = 0.02
LOCAL_PART_DIGIT_SUFFIX_RATE = 0.26

_LOCAL_PART_SEPARATORS: dict[str, str] = {
    'dot': '.', 'underscore': '_', 'hyphen': '-',
}


def random_digits(n: int, rng: random.Random | None = None) -> str:
    """n 자리 숫자 문자열."""
    r = rng or random
    return ''.join(r.choices(string.digits, k=n))


def random_letters(n: int, rng: random.Random | None = None) -> str:
    """n 자리 소문자 알파벳 문자열."""
    r = rng or random
    return ''.join(r.choices(string.ascii_lowercase, k=n))


def random_credit_card_number(rng: random.Random | None = None) -> str:
    """16자리 카드번호(Luhn 미검증) + 구분자 무작위."""
    r = rng or random
    iin = r.choice(['4', '5', '34', '37', '3528', '3569'])
    remaining = 16 - len(iin)
    digits = iin + random_digits(remaining, r)
    blocks = [digits[i:i + 4] for i in range(0, 16, 4)]
    sep = r.choice([' ', '-', '', '  '])
    if sep == '':
        return ''.join(blocks)
    return sep.join(blocks)


def _local_tokens(name_hint: str, rng: random.Random) -> list[str]:
    """이름 힌트를 로컬파트에 쓸 ASCII 소문자 토큰으로 자른다.

    로컬파트에는 ASCII 만 들어가므로 한글·한자·성조 부호는 여기서 떨어진다.
    남는 토큰이 없으면 무작위 글자로 대체해 빈 로컬파트를 막는다.
    """
    tokens = [t.lower() for t in re.split(r'[^A-Za-z0-9]+', name_hint) if t]
    return tokens or [random_letters(6, rng)]


def _pick_local_part_form(rng: random.Random) -> str:
    """형태 축을 목표 비율대로 하나 뽑는다."""
    forms = list(LOCAL_PART_FORM_WEIGHTS)
    weights = [LOCAL_PART_FORM_WEIGHTS[f] for f in forms]
    return rng.choices(forms, weights=weights, k=1)[0]


def random_local_part(
    name_hint: str, rng: random.Random | None = None,
) -> str:
    """이름 힌트로 이메일 로컬파트를 만든다.

    형태·대소문자·숫자 세 축을 따로 추첨한다. 축이 서로 독립이라 형태마다
    대소문자 표를 다시 만들 필요가 없고, 한 축의 목표 비율을 조정해도 나머지
    두 축이 흔들리지 않는다.
    """
    r = rng or random
    tokens = _local_tokens(name_hint, r)
    given, family = tokens[0], tokens[-1]

    form = _pick_local_part_form(r)
    if len(tokens) == 1:
        # 이을 토큰이 하나뿐이면 구분자·이니셜 형태가 성립하지 않는다.
        form = 'single'
    if form == 'single':
        parts = [r.choice(tokens)]
    elif form == 'initial':
        parts = [given[0] + family]
    else:
        parts = [given, family]

    if r.random() < LOCAL_PART_CAPITALIZED_RATE:
        parts = [p.capitalize() for p in parts]
    local = _LOCAL_PART_SEPARATORS.get(form, '').join(parts)

    draw = r.random()
    if draw < LOCAL_PART_DIGIT_PREFIX_RATE:
        local = random_digits(r.randint(1, 2), r) + local
    elif draw < LOCAL_PART_DIGIT_PREFIX_RATE + LOCAL_PART_DIGIT_SUFFIX_RATE:
        local = local + random_digits(r.randint(1, 4), r)
    return local[:64].strip('.')


def random_email(
    name_hint: str, rng: random.Random | None = None,
    domains: list[str] | None = None,
) -> str:
    """이름 힌트 기반 이메일 생성.

    `domains` 를 주면 그 목록에서만 고른다. 기본값은 전 언어 공용
    `EMAIL_DOMAINS` 이라 기존 호출은 그대로 동작한다 — 로케일별 목록을
    선언한 생성기만 자기 목록을 쓴다(`generate_pii` 참조).
    """
    r = rng or random
    local = random_local_part(name_hint, r)
    domain = r.choice(domains or EMAIL_DOMAINS)
    return f'{local}@{domain}'


def locale_module(lang: str):
    """언어 코드에 해당하는 로케일 생성기 모듈."""
    if lang == 'ja':
        from ner.augmenters.generators import ja as mod
    elif lang == 'vi':
        from ner.augmenters.generators import vi as mod
    elif lang == 'ko':
        from ner.augmenters.generators import ko as mod
    elif lang == 'en':
        from ner.augmenters.generators import en as mod
    else:
        raise ValueError(f'Unsupported language: {lang}')
    return mod


def email_name_hint(lang: str, rng: random.Random | None = None) -> str:
    """로컬파트를 만들 이름 힌트를 로케일 목록에서 뽑는다.

    표기 이름(`generate_name`)을 쓰지 않는 것은 로컬파트가 ASCII 만 받기
    때문이다. ko·ja 는 한글·한자가 전량 탈락해 무작위 글자만 남고, en·vi 는
    공백이 점으로 바뀌어 점 100% 가 된다.
    """
    r = rng or random
    mod = locale_module(lang)
    return (f'{r.choice(mod.EMAIL_GIVEN_NAMES)} '
            f'{r.choice(mod.EMAIL_FAMILY_NAMES)}')


def generate_pii(
    label: str, lang: str, rng: random.Random | None = None,
) -> str:
    """언어/라벨에 해당하는 합성 PII 값을 생성한다."""
    mod = locale_module(lang)

    if label == 'NAME':
        return mod.generate_name(rng)
    if label == 'PHONE':
        return mod.generate_phone(rng)
    if label == 'ADDRESS':
        return mod.generate_address(rng)
    if label == 'DAT':
        return mod.generate_dat(rng)
    if label == 'ID_NUM':
        return mod.generate_id_number(rng)
    if label == 'EMAIL':
        r = rng or random
        hint = email_name_hint(lang, r)
        # 로케일 모듈이 자기 도메인 목록을 선언했으면 그것을 쓴다. 공용
        # 목록은 ja·vi·ko 도메인이 섞여 있어 영문 문장에 `docomo.ne.jp` 가
        # 붙는다. 선언하지 않은 로케일은 기존 동작 그대로다.
        return random_email(hint, r, getattr(mod, 'EMAIL_DOMAINS', None))
    if label == 'CREDIT_CARD':
        return random_credit_card_number(rng)
    raise ValueError(f'Unsupported PII label: {label}')
