"""베트남어 PII 생성기."""
from __future__ import annotations

import random

from ner.augmenters.pii.generators.base import random_digits

VIETNAMESE_FAMILY_NAMES = [
    'Nguyen', 'Tran', 'Le', 'Pham', 'Hoang', 'Phan', 'Vu', 'Vo',
    'Dang', 'Bui', 'Do', 'Ho', 'Ngo', 'Duong', 'Dinh',
    'Ly', 'Truong', 'Luong', 'Mai', 'Chau', 'Huynh', 'Phung', 'Ta',
]

VIETNAMESE_MIDDLE_NAMES = [
    'Van', 'Thi', 'Ngoc', 'Hoai', 'Quang', 'Duc',
    'Thanh', 'Thu', 'Huynh', 'Khanh', 'Phuong',
    'Gia', 'Bao', 'Minh', 'Anh', 'Hong', 'Huu', 'Tien', 'Mai',
]

VIETNAMESE_GIVEN_NAMES = [
    'Anh', 'Hoa', 'Minh', 'Tuan', 'Linh', 'Phong', 'Trang',
    'Hung', 'Huy', 'Lan', 'Huong', 'Son', 'Long', 'Dieu', 'Nhi',
    'Khoa', 'Nam', 'Quan', 'Thao', 'Yen', 'Nga', 'Dung', 'Hanh',
    'Phuc', 'Kiet', 'Nhan', 'Trinh', 'My', 'Thuy', 'Tung', 'Giang',
]

# 이메일 로컬파트에 쓰는 이름. 성조 부호 없는 목록이라 그대로 쓴다.
EMAIL_GIVEN_NAMES = VIETNAMESE_GIVEN_NAMES
EMAIL_FAMILY_NAMES = VIETNAMESE_FAMILY_NAMES

VIETNAMESE_FAMILY_NAMES_DIA = [
    'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Phan', 'Vũ', 'Võ',
    'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Đinh', 'Lý', 'Trương',
]

VIETNAMESE_MIDDLE_NAMES_DIA = [
    'Văn', 'Thị', 'Ngọc', 'Hoài', 'Quang', 'Đức',
    'Thanh', 'Thu', 'Khánh', 'Phương',
]

VIETNAMESE_GIVEN_NAMES_DIA = [
    'Anh', 'Hòa', 'Minh', 'Tuấn', 'Linh', 'Phong', 'Trang',
    'Hùng', 'Huy', 'Lan', 'Hương', 'Sơn', 'Long', 'Diệu', 'Nhi',
    'Khoa', 'Nam', 'Quân', 'Thảo', 'Yến', 'Nga', 'Dũng', 'Hạnh',
]

VIETNAMESE_STREETS = [
    'Nguyen Hue', 'Le Loi', 'Tran Hung Dao', 'Hai Ba Trung',
    'Pham Ngu Lao', 'Ly Thuong Kiet', 'Dien Bien Phu',
    'Nguyen Thi Minh Khai', 'Cach Mang Thang 8',
    'Vo Van Kiet', 'Nam Ky Khoi Nghia', 'Ton Duc Thang',
    'Pasteur', 'Nguyen Van Cu',
]

VIETNAMESE_DISTRICTS = [
    'District 1', 'District 3', 'District 5', 'District 7',
    'Binh Thanh', 'Tan Binh', 'Go Vap',
    'Thu Duc', 'Phu Nhuan', 'Tan Phu', 'Binh Tan',
]

VIETNAMESE_CITIES = [
    'Ho Chi Minh City', 'Hanoi', 'Da Nang', 'Hai Phong', 'Can Tho',
    'Nha Trang', 'Hue', 'Vung Tau', 'Bien Hoa',
]


def generate_name(rng: random.Random | None = None) -> str:
    """베트남식 성+중간이름+이름 (공백 구분, diacritic 변형 포함)."""
    r = rng or random
    if r.random() < 0.4:
        family = r.choice(VIETNAMESE_FAMILY_NAMES_DIA)
        middle = r.choice(VIETNAMESE_MIDDLE_NAMES_DIA)
        given = r.choice(VIETNAMESE_GIVEN_NAMES_DIA)
    else:
        family = r.choice(VIETNAMESE_FAMILY_NAMES)
        middle = r.choice(VIETNAMESE_MIDDLE_NAMES)
        given = r.choice(VIETNAMESE_GIVEN_NAMES)
    return f'{family} {middle} {given}'


def generate_phone(rng: random.Random | None = None) -> str:
    """베트남식 전화번호 (통신사 prefix + 국제표기/구분자 변형)."""
    r = rng or random
    prefix = r.choice(['09', '03', '07', '08', '05'])
    rest = random_digits(8, r)
    raw = prefix + rest
    pattern = r.choice([
        '{raw}',
        '{p}{d1} {d2} {d3}',
        '+84{p}{d_all}',
        '+84 {p}{d1} {d2} {d3}',
        '(+84) {p}{d1}-{d2}-{d3}',
        '0{d_all}',
    ])
    return pattern.format(
        raw=raw, p=raw[:2],
        d1=raw[3:6], d2=raw[6:8], d3=raw[8:],
        d_all=raw[2:],
    )


def generate_address(rng: random.Random | None = None) -> str:
    """베트남식 주소 (번지+거리+군구+도시, 표기 변형 포함)."""
    r = rng or random
    num = r.randint(1, 200)
    street = r.choice(VIETNAMESE_STREETS)
    district = r.choice(VIETNAMESE_DISTRICTS)
    city = r.choice(VIETNAMESE_CITIES)
    pattern = r.choice([
        '{num} {street}, {district}, {city}',
        'Số {num} {street}, {district}, {city}',
        '{num}/{sub} {street}, {district}, {city}',
        '{num} {street} - {district} - {city}',
        'P.{ward}, {num} {street}, {district}, {city}',
    ])
    return pattern.format(
        num=num, sub=r.randint(1, 50), ward=r.randint(1, 18),
        street=street, district=district, city=city,
    )


# 날짜 표면형과 목표 비율(숫자 60 : 서술형 40). 숫자 표기를 과반으로 남기는 것은
# 그것이 서식·증서·기록에 실제로 쓰이는 표기이고, 현 DAT metric 이 그 표기 위에서
# 나온 값이라 너무 깎으면 무회귀 기준이 흔들리기 때문이다.
VIETNAMESE_DAT_PATTERNS = (
    ('{d:02d}/{m:02d}/{y:04d}', 20),
    ('{y:04d}-{m:02d}-{d:02d}', 20),
    ('{d:02d}-{m:02d}-{y:04d}', 20),
    ('{d} tháng {m} năm {y}', 40),
)


def generate_dat(rng: random.Random | None = None) -> str:
    """베트남식 날짜 (출생일·사건일·일반 표기 공통).

    숫자 표기 세 가지와 서술형 `D tháng M năm Y` 를 낸다. 서술형은 베트남어의
    일상 날짜 표기인데 예전에는 숫자 표기만 내서 학습 커버리지가 0 이었고,
    그래서 모델이 `ngày 15 tháng 3 năm 2024` 를 조각내거나 통째로 놓쳤다.

    값 앞에 `ngày` 는 붙이지 않는다. 그 자리를 채우는 것은 LLM 주입기이고,
    gold span 도 `ngày` 를 밖에 둔다. 생성기가 달고 나가면 `ngày ngày 15 ...`
    가 된다.

    월-연 표기 `tháng M năm Y` 는 여기서 내지 않는다. 값 앞에 `ngày` 가 붙을지를
    생성기가 통제하지 못해 `ngày tháng 5 năm 1988` 같은 깨진 문맥이 나온다.
    앞 글자까지 함께 보는 코퍼스 치환 경로만 그 형식을 안전하게 만들 수 있다.
    """
    r = rng or random
    y = r.randint(1950, 2005)
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    pattern = r.choices(
        [p for p, _ in VIETNAMESE_DAT_PATTERNS],
        weights=[w for _, w in VIETNAMESE_DAT_PATTERNS],
        k=1,
    )[0]
    return pattern.format(y=y, m=m, d=d)


def generate_id_number(rng: random.Random | None = None) -> str:
    """베트남 시민식별번호(CCCD) 12자리 (구분자 변형 포함)."""
    r = rng or random
    base = random_digits(12, r)
    pattern = r.choice([
        '{base}',
        '{b1}{b2}{b3}',
        '{b1}-{b2}{b3}',
    ])
    return pattern.format(
        base=base, b1=base[0:4], b2=base[4:8], b3=base[8:12]
    )


