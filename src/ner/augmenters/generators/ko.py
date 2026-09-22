"""한국어 PII 생성기.

주민등록번호는 구조(`YYMMDD-Gxxxxxx`)만 생성하고 검증 체크섬을 **의도적으로
무효화**해 실제로 유효한 번호가 합성되지 않도록 한다(PII 유출 방지).
"""
from __future__ import annotations

import random

from ner.augmenters.generators.base import random_digits

KOREAN_FAMILY_NAMES = [
    '김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
    '한', '오', '서', '신', '권', '황', '안', '송', '류', '전',
    '홍', '고', '문', '양', '손', '배', '백', '허', '유', '남',
    '심', '노', '하', '곽', '성', '차', '주', '우', '구', '민',
]

KOREAN_GIVEN_NAMES = [
    '민준', '서준', '도윤', '예준', '시우', '주원', '하준', '지호',
    '준서', '현우', '도현', '건우', '우진', '선우', '서진', '연우',
    '서연', '서윤', '지우', '서현', '민서', '하은', '하윤', '윤서',
    '지유', '지민', '채원', '수아', '지아', '다은', '은서', '예은',
    '수빈', '지원', '예린', '시은', '유진', '채은', '가은', '소율',
]

# 이메일 로컬파트에 쓰는 로마자 이름. 로컬파트에는 ASCII 만 들어가므로 한글
# 이름을 그대로 넣으면 전량 탈락해 무작위 글자만 남는다. 표기 이름 목록과
# 같은 순서의 관용 표기다(`lee`·`park`) — 로마자 표기법(`i`·`bak`)이 아니라
# 실제 이메일·여권에 흔히 쓰는 철자를 따른다.
EMAIL_FAMILY_NAMES = [
    'kim', 'lee', 'park', 'choi', 'jung', 'kang', 'cho', 'yoon',
    'jang', 'lim', 'han', 'oh', 'seo', 'shin', 'kwon', 'hwang',
    'ahn', 'song', 'ryu', 'jeon', 'hong', 'ko', 'moon', 'yang',
    'son', 'bae', 'baek', 'heo', 'yoo', 'nam', 'sim', 'noh',
    'ha', 'kwak', 'sung', 'cha', 'joo', 'woo', 'koo', 'min',
]

EMAIL_GIVEN_NAMES = [
    'minjun', 'seojun', 'doyun', 'yejun', 'siwoo', 'juwon', 'hajun',
    'jiho', 'junseo', 'hyunwoo', 'dohyun', 'gunwoo', 'woojin', 'sunwoo',
    'seojin', 'yeonwoo', 'seoyeon', 'seoyun', 'jiwoo', 'seohyun',
    'minseo', 'haeun', 'hayun', 'yunseo', 'jiyu', 'jimin', 'chaewon',
    'sua', 'jia', 'daeun', 'eunseo', 'yeeun', 'subin', 'jiwon',
    'yerin', 'sieun', 'yujin', 'chaeeun', 'gaeun', 'soyul',
]

KOREAN_PROVINCES = [
    '서울특별시', '부산광역시', '대구광역시', '인천광역시',
    '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
    '경기도', '강원특별자치도', '충청북도', '충청남도',
    '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도',
]

KOREAN_CITY_DISTRICTS = [
    '강남구', '서초구', '송파구', '마포구', '종로구', '중구',
    '영등포구', '성동구', '광진구', '노원구',
    '해운대구', '수영구', '분당구', '일산동구', '권선구',
    '유성구', '남동구', '북구', '동구', '서구',
]

KOREAN_DONGS = [
    '역삼동', '삼성동', '정자동', '서교동', '청담동', '신사동',
    '여의도동', '잠실동', '구의동', '상계동', '우동', '광안동',
    '둔산동', '봉명동', '효자동', '대흥동', '연남동', '망원동',
]

KOREAN_ROADS = [
    '테헤란로', '강남대로', '세종대로', '올림픽로', '양재대로',
    '도산대로', '봉은사로', '여의대로', '종로', '을지로',
    '백제고분로', '동작대로', '한강대로', '청계천로',
]


def generate_name(rng: random.Random | None = None) -> str:
    """한국식 성+이름 (성 1자 + 이름 2자)."""
    r = rng or random
    family = r.choice(KOREAN_FAMILY_NAMES)
    given = r.choice(KOREAN_GIVEN_NAMES)
    return family + given


KOREAN_MOBILE_PREFIX = '010'
KOREAN_INTERNET_PREFIX = '070'
KOREAN_SEOUL_AREA = '02'
KOREAN_LANDLINE_AREAS = [
    '031', '032', '033', '041', '042', '043', '044',
    '051', '052', '053', '054', '055',
    '061', '062', '063', '064',
]


def _phone_separators(
    a: str, b: str, c: str, rng: random.Random,
) -> str:
    """국번 3블록을 다양한 구분자/국제표기로 조합한다."""
    pattern = rng.choice([
        '{a}-{b}-{c}',
        '{a} {b} {c}',
        '{a}.{b}.{c}',
        '{a}{b}{c}',
        '+82 {a2}-{b}-{c}',
    ])
    # +82 국제표기는 선행 0 제거
    return pattern.format(a=a, b=b, c=c, a2=a.lstrip('0'))


def generate_phone(rng: random.Random | None = None) -> str:
    """한국식 전화번호.

    휴대(010) + 인터넷전화(070) + 서울 시내(02) + 지역 시외국번(0XX)을
    가중치로 섞고 구분자/국제표기를 다양화한다.
    """
    r = rng or random
    kind = r.choices(
        ['mobile', 'internet', 'seoul', 'landline'],
        weights=[62, 8, 12, 18],
        k=1,
    )[0]

    if kind == 'mobile':
        return _phone_separators(
            KOREAN_MOBILE_PREFIX, random_digits(4, r), random_digits(4, r), r,
        )
    if kind == 'internet':
        return _phone_separators(
            KOREAN_INTERNET_PREFIX, random_digits(4, r),
            random_digits(4, r), r,
        )
    if kind == 'seoul':
        # 서울 02 + 가입자번호 3-4 또는 4-4
        b = random_digits(r.choice([3, 4]), r)
        return _phone_separators(
            KOREAN_SEOUL_AREA, b, random_digits(4, r), r,
        )
    # 지역 시외국번 + 가입자번호 3-4
    area = r.choice(KOREAN_LANDLINE_AREAS)
    return _phone_separators(
        area, random_digits(3, r), random_digits(4, r), r,
    )


def generate_address(rng: random.Random | None = None) -> str:
    """한국식 주소 (도로명 또는 지번)."""
    r = rng or random
    prov = r.choice(KOREAN_PROVINCES)
    district = r.choice(KOREAN_CITY_DISTRICTS)
    if r.random() < 0.5:
        # 도로명 주소
        road = r.choice(KOREAN_ROADS)
        num = r.randint(1, 500)
        pattern = r.choice([
            '{prov} {district} {road} {num}',
            '{prov} {district} {road} {num}길 {sub}',
            '{prov} {district} {road} {num} {bldg} {fl}층',
        ])
        return pattern.format(
            prov=prov, district=district, road=road, num=num,
            sub=r.randint(1, 40), fl=r.randint(1, 20),
            bldg=r.choice(['삼성빌딩', '센터빌딩', '타워', '오피스텔']),
        )
    # 지번 주소
    dong = r.choice(KOREAN_DONGS)
    pattern = r.choice([
        '{prov} {district} {dong} {num}',
        '{prov} {district} {dong} {num}-{sub}',
        '{prov} {district} {dong} {num}-{sub} {bldg} {ho}호',
    ])
    return pattern.format(
        prov=prov, district=district, dong=dong,
        num=r.randint(1, 900), sub=r.randint(1, 99),
        ho=r.randint(101, 2505),
        bldg=r.choice(['아파트', '빌라', '하이츠', '맨션']),
    )


def generate_dat(rng: random.Random | None = None) -> str:
    """한국식 날짜 (출생일·사건일·일반 표기 공통)."""
    r = rng or random
    y = r.randint(1950, 2005)
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    pattern = r.choice([
        '{y:04d}-{m:02d}-{d:02d}',
        '{y:04d}.{m:02d}.{d:02d}',
        '{y}년 {m}월 {d}일',
        '{y:04d}/{m:02d}/{d:02d}',
    ])
    return pattern.format(y=y, m=m, d=d)


_RRN_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]


def _invalid_rrn_check_digit(first12: str) -> str:
    """주민등록번호 검증 체크섬을 계산한 뒤 **틀린** 값을 반환한다.

    실제 유효한 번호 생성을 막기 위해 정답 체크digit에 +1(mod 10)을 적용한다.
    """
    total = sum(int(c) * w for c, w in zip(first12, _RRN_WEIGHTS))
    valid = (11 - (total % 11)) % 10
    return str((valid + 1) % 10)


def generate_id_number(rng: random.Random | None = None) -> str:
    """한국 주민등록번호 형식 (YYMMDD-Gxxxxxx, 체크섬 무효).

    생년월일 + 성별/세기 자리 + 임의 5자리 + 무효 체크digit. 검증 체크섬을
    의도적으로 어긋나게 해 실유효 번호 합성을 방지한다.
    """
    r = rng or random
    yy = random_digits(2, r)
    mm = f'{r.randint(1, 12):02d}'
    dd = f'{r.randint(1, 28):02d}'
    gender = str(r.choice([1, 2, 3, 4]))
    serial = random_digits(5, r)
    first12 = yy + mm + dd + gender + serial
    last = _invalid_rrn_check_digit(first12)
    back = gender + serial + last
    if r.random() < 0.2:
        # 하이픈 없는 13자리 변형
        return first12 + last
    return f'{yy}{mm}{dd}-{back}'
