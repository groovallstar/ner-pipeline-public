"""일본어 PII 생성기."""
from __future__ import annotations

import random

from ner.augmenters.pii.generators.base import random_digits

JAPANESE_FAMILY_NAMES = [
    '山田', '佐藤', '鈴木', '田中', '高橋', '伊藤', '渡辺', '中村',
    '小林', '加藤', '吉田', '山本', '井上', '木村', '林', '斎藤',
    '清水', '山崎', '森', '阿部',
    '池田', '橋本', '山下', '石井', '前田', '藤田', '後藤', '岡田',
    '長谷川', '石川', '村上', '近藤', '坂本', '遠藤', '青木', '藤井',
    '西村', '福田', '太田', '三浦', '藤原', '岡本', '松本', '中島',
]

JAPANESE_GIVEN_NAMES = [
    '太郎', '花子', '健一', '美咲', '翔太', '裕子', '大輔', '彩香',
    '直樹', '麻衣', '悠斗', '美穂', '拓也', '結衣', '陽介', '愛',
    '隆', '真由', '智也', '菜々子',
    '優奈', '大翔', '結菜', '蓮', '陽菜', '颯太', '美優', '心愛',
    '葵', '悠真', '凛', '蒼', '咲良', '陸', '朱里', '颯',
    '美月', '一輝', '美羽', '沙織', '杏奈', '玲奈', '梨花', '彩',
]

JAPANESE_PREFECTURES = [
    '東京都', '大阪府', '神奈川県', '愛知県', '北海道',
    '千葉県', '埼玉県', '福岡県', '兵庫県', '京都府',
    '静岡県', '広島県', '宮城県', '新潟県', '長野県', '岐阜県',
    '熊本県', '沖縄県', '岡山県', '鹿児島県',
]

JAPANESE_CITIES = [
    '新宿区', '渋谷区', '港区', '中央区', '千代田区', '豊島区',
    '横浜市西区', '横浜市中区', '名古屋市中村区', '大阪市北区',
    '札幌市中央区', '福岡市博多区',
    '仙台市青葉区', '広島市中区', '神戸市中央区', '京都市中京区',
    '那覇市', 'さいたま市大宮区', '川崎市中原区', '千葉市中央区',
]


def generate_name(rng: random.Random | None = None) -> str:
    """일본식 성+이름 (공백 포함 가능)."""
    r = rng or random
    family = r.choice(JAPANESE_FAMILY_NAMES)
    given = r.choice(JAPANESE_GIVEN_NAMES)
    if r.random() < 0.2:
        return family + ' ' + given
    return family + given


def generate_phone(rng: random.Random | None = None) -> str:
    """일본식 전화번호(하이픈/+81 등 다양한 포맷)."""
    r = rng or random
    carrier = r.choice(['80', '90', '70'])
    b2 = random_digits(4, r)
    b3 = random_digits(4, r)
    pattern = r.choice([
        '0{b1}-{b2}-{b3}',
        '0{b1}{b2}{b3}',
        '+81-{b1}-{b2}-{b3}',
        '+81(0){b1}-{b2}-{b3}',
        '+81 {b1} {b2} {b3}',
        '(0{b1}){b2}-{b3}',
    ])
    return pattern.format(b1=carrier, b2=b2, b3=b3)


def generate_address(rng: random.Random | None = None) -> str:
    """일본식 주소."""
    r = rng or random
    pref = r.choice(JAPANESE_PREFECTURES)
    city = r.choice(JAPANESE_CITIES)
    chome = f'{r.randint(1, 5)}丁目'
    banchi = f'{r.randint(1, 20)}-{r.randint(1, 30)}'
    pattern = r.choice([
        '{pref}{city}{chome}{banchi}',
        '{pref}{city}{banchi}',
        '{pref} {city} {chome}{banchi}',
        '{pref}{city}{chome}{banchi} ビル{num}F',
        '〒{zip} {pref}{city}{chome}{banchi}',
        '{pref}{city}{chome}{banchi} {bldg}{num}号室',
    ])
    return pattern.format(
        pref=pref, city=city, chome=chome, banchi=banchi,
        num=r.randint(2, 10),
        zip=random_digits(3, r) + '-' + random_digits(4, r),
        bldg=r.choice(['コーポ', 'ハイツ', 'メゾン', 'レジデンス']),
    )


def generate_dat(rng: random.Random | None = None) -> str:
    """일본식 날짜 (출생일·사건일·일반 표기 공통)."""
    r = rng or random
    y = r.randint(1950, 2005)
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    pattern = r.choice([
        '{y:04d}-{m:02d}-{d:02d}',
        '{y:04d}/{m:02d}/{d:02d}',
        '{y}年{m}月{d}日',
        '{y:04d}.{m:02d}.{d:02d}',
    ])
    return pattern.format(y=y, m=m, d=d)


def generate_id_number(rng: random.Random | None = None) -> str:
    """일본 마이넘버(12자리)."""
    r = rng or random
    base = random_digits(12, r)
    pattern = r.choice([
        '{base}',
        '{b1}-{b2}-{b3}',
        '{b1} {b2} {b3}',
    ])
    return pattern.format(
        base=base, b1=base[0:4], b2=base[4:8], b3=base[8:12]
    )


