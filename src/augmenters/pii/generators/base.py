"""PII 생성기 공용 유틸 및 Protocol."""
from __future__ import annotations

import random
import string

EMAIL_DOMAINS: list[str] = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'proton.me',
    'yahoo.co.jp', 'docomo.ne.jp', 'softbank.ne.jp', 'ezweb.ne.jp',
    'example.co.jp',
    'gmail.com.vn', 'yahoo.com.vn', 'example.com.vn', 'corp.vn',
    'mail.vn', 'vnn.vn', 'fpt.com.vn', 'viettel.com.vn', 'vnpt.vn',
    'icloud.com', 'gmx.com', 'zoho.com',
]

ALLOWED_EMAIL_SPECIALS = '.-_'


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


def random_email(
    name_hint: str, rng: random.Random | None = None
) -> str:
    """이름 힌트 기반 이메일 생성."""
    r = rng or random
    allowed = set(string.ascii_letters + string.digits + ALLOWED_EMAIL_SPECIALS)

    def normalize(s: str) -> str:
        s = s[:64]
        while '..' in s:
            s = s.replace('..', '.')
        s = s.strip('.')
        if not s:
            s = random_letters(8, r)
        return s[:64].strip('.') or random_letters(8, r)

    local = ''.join(ch for ch in name_hint if ch in allowed)
    local = normalize(local)
    if r.random() < 0.3 and len(local) >= 6:
        cut = r.randint(1, len(local) - 1)
        sep = r.choice(list(ALLOWED_EMAIL_SPECIALS))
        local = normalize(local[:cut] + sep + local[cut:])
    if r.random() < 0.5:
        local = normalize(local + random_digits(r.randint(1, 3), r))
    else:
        local = normalize(random_digits(r.randint(1, 2), r) + local)
    domain = r.choice(EMAIL_DOMAINS)
    return f'{local}@{domain}'


def generate_pii(
    label: str, lang: str, rng: random.Random | None = None,
) -> str:
    """언어/라벨에 해당하는 합성 PII 값을 생성한다."""
    if lang == 'ja':
        from augmenters.pii.generators import ja as mod
    elif lang == 'vi':
        from augmenters.pii.generators import vi as mod
    else:
        raise ValueError(f'Unsupported language: {lang}')

    if label == 'NAME':
        return mod.generate_name(rng)
    if label == 'PHONE':
        return mod.generate_phone(rng)
    if label == 'ADDRESS':
        return mod.generate_address(rng)
    if label == 'DOB':
        return mod.generate_dob(rng)
    if label == 'ID_NUMBER':
        return mod.generate_id_number(rng)
    if label == 'EMAIL':
        hint = mod.generate_name(rng)
        if lang == 'vi':
            hint = hint.replace(' ', '.')
        return random_email(hint, rng)
    if label == 'CREDIT_CARD':
        return random_credit_card_number(rng)
    raise ValueError(f'Unsupported PII label: {label}')
