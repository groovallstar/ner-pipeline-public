"""영어(미국) PII 생성기.

`en` 은 언어지 국가가 아니다 — ko·ja·vi 는 언어와 국가가 1:1 이라 전화·ID
체계가 자동으로 정해졌지만 영어는 US·UK·AU 가 전부 다르다. 그래서 **미국
단일 체계**로 못 박는다. 원천인 OntoNotes5 가 미국 뉴스·방송 중심이라 원문
도메인과도 맞는다.

## 합성값은 실제로 쓸 수 없는 대역에만 둔다

`ko.py` 의 `_invalid_rrn_check_digit()` 이 일부러 체크섬이 틀린 주민번호를
만드는 것과 같은 이유다 — 합성 PII 가 실제 유효 번호가 되면 학습 데이터가
누군가의 진짜 식별자를 담게 된다.

- **SSN**: 지역번호(앞 3 자리)를 `000`·`666`·`900`–`999` 로만 만든다. 사회보장국이
  한 번도 발급하지 않은 대역이다.
- **전화**: 국번을 `555`, 가입자번호를 `0100`–`0199` 로 고정한다. NANP 가 창작물
  용도로 예약해 둔 블록이라 실제로 연결되지 않는다.

`generate_dat` 은 인터페이스 대칭을 위해 둔다. **EN 파이프라인은 `DAT` 을 주입하지
않는다** — OntoNotes `DATE` 가 gold 로 주기 때문이며, KO 가 KLUE 에서 날짜를
받는 것과 같은 구조다.
"""
from __future__ import annotations

import random

from ner.augmenters.pii.generators.base import random_digits

US_GIVEN_NAMES = [
    'James', 'Robert', 'John', 'Michael', 'David', 'William', 'Richard',
    'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara',
    'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Margaret',
    'Anthony', 'Mark', 'Donald', 'Steven', 'Andrew', 'Kenneth', 'Brian',
    'Emily', 'Ashley', 'Michelle', 'Amanda', 'Melissa', 'Rebecca',
]

US_FAMILY_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
    'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
    'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green',
]

US_MIDDLE_INITIALS = list('ABCDEFGHJKLMNPRSTW')

US_STREETS = [
    'Main St', 'Oak Ave', 'Maple Dr', 'Cedar Ln', 'Elm St', 'Washington Ave',
    'Park Blvd', 'Lake Rd', 'Hill St', 'Sunset Blvd', 'Broadway',
    'Madison Ave', 'Lincoln Ave', 'Jefferson St', 'Franklin Rd',
    'Pine St', 'Church St', 'Market St', 'Highland Ave', 'River Rd',
]

# (도시, 주 약어) — 우편 표기가 도시·주를 함께 쓴다.
US_CITIES = [
    ('New York', 'NY'), ('Los Angeles', 'CA'), ('Chicago', 'IL'),
    ('Houston', 'TX'), ('Phoenix', 'AZ'), ('Philadelphia', 'PA'),
    ('San Antonio', 'TX'), ('San Diego', 'CA'), ('Dallas', 'TX'),
    ('Boston', 'MA'), ('Seattle', 'WA'), ('Denver', 'CO'),
    ('Atlanta', 'GA'), ('Miami', 'FL'), ('Portland', 'OR'),
    ('Nashville', 'TN'), ('Baltimore', 'MD'), ('Milwaukee', 'WI'),
]

# 영문 문장에 붙일 도메인. 공용 `base.EMAIL_DOMAINS` 는 ja·vi·ko 도메인이
# 섞여 있어 `docomo.ne.jp`·`kakao.com` 이 영문 문맥에 나타난다.
EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com',
    'icloud.com', 'proton.me', 'msn.com', 'live.com', 'comcast.net',
    'verizon.net', 'att.net', 'me.com', 'mac.com', 'example.com',
    'example.org', 'zoho.com', 'gmx.com', 'fastmail.com',
]

US_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]

# 실제로 존재하는 지역번호 표기 (NXX). 전화는 국번·가입자번호 쪽에서
# 예약 대역을 쓰므로 지역번호는 자연스러운 값을 쓴다.
US_AREA_CODES = [
    '212', '213', '312', '415', '512', '617', '702', '713', '718',
    '305', '404', '503', '602', '206', '303', '408', '469', '646',
]

# SSA 가 한 번도 발급하지 않은 지역번호 — 합성 SSN 을 여기에만 둔다.
_NEVER_ISSUED_SSN_AREAS = ['000', '666'] + [str(n) for n in range(900, 1000)]


def generate_name(rng: random.Random | None = None) -> str:
    """미국식 이름 (이름 + 중간 이니셜 선택 + 성)."""
    r = rng or random
    given = r.choice(US_GIVEN_NAMES)
    family = r.choice(US_FAMILY_NAMES)
    if r.random() < 0.25:
        return f'{given} {r.choice(US_MIDDLE_INITIALS)}. {family}'
    return f'{given} {family}'


def generate_phone(rng: random.Random | None = None) -> str:
    """NANP 전화번호 (창작물 예약 대역 `555-0100`~`555-0199`, 표기 변형 포함)."""
    r = rng or random
    area = r.choice(US_AREA_CODES)
    line = f'{r.randint(100, 199):04d}'
    pattern = r.choice([
        '({area}) 555-{line}',
        '{area}-555-{line}',
        '{area}.555.{line}',
        '+1 {area} 555 {line}',
        '+1-{area}-555-{line}',
        '1-{area}-555-{line}',
        '555-{line}',
    ])
    return pattern.format(area=area, line=line)


def generate_address(rng: random.Random | None = None) -> str:
    """미국식 주소 (번지 + 도로 + 도시, 주 약어 + ZIP, 표기 변형 포함)."""
    r = rng or random
    number = r.randint(1, 9999)
    street = r.choice(US_STREETS)
    city, state = r.choice(US_CITIES)
    zip_code = f'{r.randint(1000, 99999):05d}'
    pattern = r.choice([
        '{number} {street}, {city}, {state} {zip_code}',
        '{number} {street}, {city}, {state}',
        '{number} {street} Apt {apt}, {city}, {state} {zip_code}',
        '{number} {street} Suite {apt}, {city}, {state} {zip_code}',
        '{number} {street}',
    ])
    return pattern.format(
        number=number, street=street, city=city, state=state,
        zip_code=zip_code, apt=r.randint(1, 40),
    )


def generate_dat(rng: random.Random | None = None) -> str:
    """미국식 날짜 표기.

    EN 파이프라인은 `DAT` 을 주입하지 않는다(원본 `DATE` 가 gold). 인터페이스
    대칭과 다른 용도의 호출을 위해 둔다.
    """
    r = rng or random
    year = r.randint(1950, 2005)
    month = r.randint(1, 12)
    day = r.randint(1, 28)
    pattern = r.choice([
        '{month_name} {day}, {year}',
        '{month:02d}/{day:02d}/{year}',
        '{year}-{month:02d}-{day:02d}',
        '{day} {month_name} {year}',
    ])
    return pattern.format(
        year=year, month=month, day=day,
        month_name=US_MONTHS[month - 1],
    )


def generate_id_number(rng: random.Random | None = None) -> str:
    """미발급 대역 SSN (구분자 변형 포함).

    지역번호를 `000`·`666`·`900`–`999` 로만 만들어 실제 유효 번호가 되지
    않게 한다.
    """
    r = rng or random
    area = r.choice(_NEVER_ISSUED_SSN_AREAS)
    group = f'{r.randint(1, 99):02d}'
    serial = random_digits(4, r)
    if serial == '0000':
        serial = '0001'
    pattern = r.choice([
        '{area}-{group}-{serial}',
        '{area} {group} {serial}',
        '{area}{group}{serial}',
    ])
    return pattern.format(area=area, group=group, serial=serial)
