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


def generate_dat(rng: random.Random | None = None) -> str:
    """베트남식 날짜 (출생일·사건일·일반 표기 공통)."""
    r = rng or random
    y = r.randint(1950, 2005)
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    pattern = r.choice([
        '{d:02d}/{m:02d}/{y:04d}',
        '{y:04d}-{m:02d}-{d:02d}',
        '{d:02d}-{m:02d}-{y:04d}',
    ])
    return pattern.format(y=y, m=m, d=d)


def generate_id_number(rng: random.Random | None = None) -> str:
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


