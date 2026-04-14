import random
import string
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Literal
import re
from functools import lru_cache

Lang = Literal["ja", "vi"]

# =========================
# 1. 기본 데이터 사전 확장
# =========================

JAPANESE_FAMILY_NAMES = [
    "山田", "佐藤", "鈴木", "田中", "高橋", "伊藤", "渡辺", "中村",
    "小林", "加藤", "吉田", "山本", "井上", "木村", "林", "斎藤",
    "清水", "山崎", "森", "阿部",
    "池田", "橋本", "山下", "石井", "前田", "藤田", "後藤", "岡田",
    "長谷川", "石川", "村上", "近藤", "坂本", "遠藤", "青木", "藤井",
    "西村", "福田", "太田", "三浦", "藤原", "岡本", "松本", "中島"
]

JAPANESE_GIVEN_NAMES = [
    "太郎", "花子", "健一", "美咲", "翔太", "裕子", "大輔", "彩香",
    "直樹", "麻衣", "悠斗", "美穂", "拓也", "結衣", "陽介", "愛",
    "隆", "真由", "智也", "菜々子",
    "優奈", "大翔", "結菜", "蓮", "陽菜", "颯太", "美優", "心愛",
    "葵", "悠真", "凛", "蒼", "咲良", "陸", "朱里", "颯",
    "美月", "一輝", "美羽", "沙織", "杏奈", "玲奈", "梨花", "彩"
]

JAPANESE_PREFECTURES = [
    "東京都", "大阪府", "神奈川県", "愛知県", "北海道",
    "千葉県", "埼玉県", "福岡県", "兵庫県", "京都府",
    "静岡県", "広島県", "宮城県", "新潟県", "長野県", "岐阜県",
    "熊本県", "沖縄県", "岡山県", "鹿児島県"
]

JAPANESE_CITIES = [
    "新宿区", "渋谷区", "港区", "中央区", "千代田区", "豊島区",
    "横浜市西区", "横浜市中区", "名古屋市中村区", "大阪市北区",
    "札幌市中央区", "福岡市博多区",
    "仙台市青葉区", "広島市中区", "神戸市中央区", "京都市中京区",
    "那覇市", "さいたま市大宮区", "川崎市中原区", "千葉市中央区"
]

VIETNAMESE_FAMILY_NAMES = [
    "Nguyen", "Tran", "Le", "Pham", "Hoang", "Phan", "Vu", "Vo",
    "Dang", "Bui", "Do", "Ho", "Ngo", "Duong", "Dinh",
    "Ly", "Truong", "Luong", "Mai", "Chau", "Huynh", "Phung", "Ta"
]

VIETNAMESE_MIDDLE_NAMES = [
    "Van", "Thi", "Ngoc", "Hoai", "Quang", "Duc",
    "Thanh", "Thu", "Huynh", "Khanh", "Phuong",
    "Gia", "Bao", "Minh", "Anh", "Hong", "Huu", "Tien", "Mai"
]

VIETNAMESE_GIVEN_NAMES = [
    "Anh", "Hoa", "Minh", "Tuan", "Linh", "Phong", "Trang",
    "Hung", "Huy", "Lan", "Huong", "Son", "Long", "Dieu", "Nhi",
    "Khoa", "Nam", "Quan", "Thao", "Yen", "Nga", "Dung", "Hanh",
    "Phuc", "Kiet", "Nhan", "Trinh", "My", "Thuy", "Tung", "Giang"
]

VIETNAMESE_FAMILY_NAMES_DIA = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Đinh", "Lý", "Trương"
]

VIETNAMESE_MIDDLE_NAMES_DIA = [
    "Văn", "Thị", "Ngọc", "Hoài", "Quang", "Đức", "Thanh", "Thu", "Khánh", "Phương"
]

VIETNAMESE_GIVEN_NAMES_DIA = [
    "Anh", "Hòa", "Minh", "Tuấn", "Linh", "Phong", "Trang",
    "Hùng", "Huy", "Lan", "Hương", "Sơn", "Long", "Diệu", "Nhi",
    "Khoa", "Nam", "Quân", "Thảo", "Yến", "Nga", "Dũng", "Hạnh"
]

VIETNAMESE_STREETS = [
    "Nguyen Hue", "Le Loi", "Tran Hung Dao", "Hai Ba Trung", "Pham Ngu Lao",
    "Ly Thuong Kiet", "Dien Bien Phu", "Nguyen Thi Minh Khai", "Cach Mang Thang 8",
    "Vo Van Kiet", "Nam Ky Khoi Nghia", "Ton Duc Thang", "Pasteur", "Nguyen Van Cu"
]

VIETNAMESE_DISTRICTS = [
    "District 1", "District 3", "District 5", "District 7", "Binh Thanh",
    "Tan Binh", "Go Vap",
    "Thu Duc", "Phu Nhuan", "Tan Phu", "Binh Tan"
]

VIETNAMESE_CITIES = [
    "Ho Chi Minh City", "Hanoi", "Da Nang", "Hai Phong", "Can Tho",
    "Nha Trang", "Hue", "Vung Tau", "Bien Hoa"
]

EMAIL_DOMAINS = [
    # generic
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me",
    # Japan-ish
    "yahoo.co.jp", "docomo.ne.jp", "softbank.ne.jp", "ezweb.ne.jp", "example.co.jp",
    # Vietnam-ish
    "gmail.com.vn", "yahoo.com.vn", "example.com.vn", "corp.vn",
    "mail.vn", "vnn.vn", "fpt.com.vn", "viettel.com.vn", "vnpt.vn",
    "icloud.com", "gmx.com", "zoho.com"
]


# =========================
# 2. 유틸 함수
# =========================

def random_digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))

def random_letters(n: int) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))

def random_alnum(n: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=n))

def random_credit_card_number() -> str:
    """
    매우 단순한 16자리 카드번호 생성 (Luhn 체크는 생략).
    다양한 포맷: "1234 5678 9012 3456", "1234-5678-9012-3456"
    """
    iin = random.choice(["4", "5", "34", "37", "3528", "3569"])
    remaining = 16 - len(iin)
    digits = iin + random_digits(remaining)
    blocks = [digits[i:i+4] for i in range(0, 16, 4)]
    sep = random.choice([" ", "-", "", "  "])  # 빈 문자열이면 16자리 붙어서 감
    if sep == "":
        return "".join(blocks)
    return sep.join(blocks)

def random_email(name_hint: str) -> str:
    def normalize_local_part(s: str) -> str:
        s = s[:64]
        while ".." in s:
            s = s.replace("..", ".")
        s = s.strip(".")
        if not s:
            s = random_letters(8)
        s = s[:64]
        s = s.strip(".")
        if not s:
            s = random_letters(8)
        return s

    allowed_specials = ".-_"
    allowed = set(string.ascii_letters + string.digits + allowed_specials)

    local = "".join(ch for ch in name_hint if ch in allowed)
    local = normalize_local_part(local)

    if random.random() < 0.3 and len(local) >= 6:
        cut = random.randint(1, len(local) - 1)
        sep = random.choice(list(allowed_specials))
        local = normalize_local_part(local[:cut] + sep + local[cut:])

    if random.random() < 0.5:
        local = normalize_local_part(local + random_digits(random.randint(1, 3)))
    else:
        local = normalize_local_part(random_digits(random.randint(1, 2)) + local)

    domain = random.choice(EMAIL_DOMAINS)
    return f"{local}@{domain}"

# =========================
# 3. 언어별 PII 생성 함수 (포맷 다양화)
# =========================

def generate_name(lang: Lang) -> str:
    if lang == "ja":
        family = random.choice(JAPANESE_FAMILY_NAMES)
        given = random.choice(JAPANESE_GIVEN_NAMES)
        # 일본식: 성+이름 / 간혹 공백 포함
        if random.random() < 0.2:
            return family + " " + given
        return family + given
    elif lang == "vi":
        if random.random() < 0.4:
            family = random.choice(VIETNAMESE_FAMILY_NAMES_DIA)
            middle = random.choice(VIETNAMESE_MIDDLE_NAMES_DIA)
            given = random.choice(VIETNAMESE_GIVEN_NAMES_DIA)
        else:
            family = random.choice(VIETNAMESE_FAMILY_NAMES)
            middle = random.choice(VIETNAMESE_MIDDLE_NAMES)
            given = random.choice(VIETNAMESE_GIVEN_NAMES)
        return f"{family} {middle} {given}"
    else:
        raise ValueError("Unsupported language")

def generate_phone(lang: Lang) -> str:
    """
    언어별로 다양한 전화번호 포맷 생성
    - 일본: 090-1234-5678, 09012345678, +81-90-1234-5678 등
    - 베트남: 09xxxxxxxx, 09xx xxx xxx, +84 9x xxx xxxx 등
    """
    if lang == "ja":
        carrier = random.choice(["80", "90", "70"])
        block1 = carrier
        block2 = random_digits(4)
        block3 = random_digits(4)
        pattern = random.choice([
            "0{b1}-{b2}-{b3}",
            "0{b1}{b2}{b3}",
            "+81-{b1}-{b2}-{b3}",
            "+81(0){b1}-{b2}-{b3}",
            "+81 {b1} {b2} {b3}",
            "(0{b1}){b2}-{b3}",
        ])
        return pattern.format(b1=block1, b2=block2, b3=block3)

    elif lang == "vi":
        prefix = random.choice(["09", "03", "07", "08", "05"])
        rest = random_digits(8)  # 총 10자리
        # 예: 0987654321
        raw = prefix + rest
        pattern = random.choice([
            "{raw}",                                   # 0987654321
            "{p}{d1} {d2} {d3}",                       # 09x xxx xxxx
            "+84{p}{d_all}",                           # +8498xxxxxxx
            "+84 {p}{d1} {d2} {d3}",                   # +84 9x xxx xxxx
            "(+84) {p}{d1}-{d2}-{d3}",
            "0{d_all}",
        ])
        d1 = raw[3:6]
        d2 = raw[6:8]
        d3 = raw[8:]
        return pattern.format(
            raw=raw,
            p=raw[:2],
            d1=d1,
            d2=d2,
            d3=d3,
            d_all=raw[2:]
        )
    else:
        raise ValueError("Unsupported language")

def generate_address(lang: Lang) -> str:
    if lang == "ja":
        pref = random.choice(JAPANESE_PREFECTURES)
        city = random.choice(JAPANESE_CITIES)
        chome = f"{random.randint(1,5)}丁目"
        banchi = f"{random.randint(1,20)}-{random.randint(1,30)}"
        pattern = random.choice([
            "{pref}{city}{chome}{banchi}",
            "{pref}{city}{banchi}",
            "{pref} {city} {chome}{banchi}",
            "{pref}{city}{chome}{banchi} ビル{num}F",
            "〒{zip} {pref}{city}{chome}{banchi}",
            "{pref}{city}{chome}{banchi} {bldg}{num}号室",
        ])
        return pattern.format(
            pref=pref,
            city=city,
            chome=chome,
            banchi=banchi,
            num=random.randint(2, 10),
            zip=random_digits(3) + "-" + random_digits(4),
            bldg=random.choice(["コーポ", "ハイツ", "メゾン", "レジデンス"]),
        )

    elif lang == "vi":
        number = random.randint(1, 200)
        street = random.choice(VIETNAMESE_STREETS)
        district = random.choice(VIETNAMESE_DISTRICTS)
        city = random.choice(VIETNAMESE_CITIES)
        pattern = random.choice([
            "{num} {street}, {district}, {city}",
            "Số {num} {street}, {district}, {city}",
            "{num}/{sub} {street}, {district}, {city}",
            "{num} {street} - {district} - {city}",
            "P.{ward}, {num} {street}, {district}, {city}",
        ])
        return pattern.format(
            num=number,
            sub=random.randint(1, 50),
            ward=random.randint(1, 18),
            street=street,
            district=district,
            city=city
        )
    else:
        raise ValueError("Unsupported language")

def generate_id_number(lang: Lang) -> str:
    """
    일본/베트남 모두 12자리 기본, 포맷 약간 다양화 (하이픈 여부 등)
    """
    base = random_digits(12)
    if lang == "ja":
        # 마이넘버: 대체로 12자리 그대로, 가끔 공백/하이픈
        pattern = random.choice([
            "{base}",
            "{b1}-{b2}-{b3}",
            "{b1} {b2} {b3}"
        ])
    else:  # vi
        # CCCD도 마찬가지로 12자리
        pattern = random.choice([
            "{base}",
            "{b1}{b2}{b3}",
            "{b1}-{b2}{b3}"
        ])
    return pattern.format(
        base=base,
        b1=base[0:4],
        b2=base[4:8],
        b3=base[8:12]
    )

def generate_dob(lang: Lang) -> str:
    """
    다양한 생년월일 포맷 생성
    - 일본: 1990-01-02, 1990/01/02, 1990年1月2日, 1990.01.02 등
    - 베트남: 02/01/1990, 1990-01-02, 02-01-1990 등
    """
    year = random.randint(1950, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    if lang == "ja":
        patterns = [
            "{y:04d}-{m:02d}-{d:02d}",
            "{y:04d}/{m:02d}/{d:02d}",
            "{y}年{m}月{d}日",
            "{y:04d}.{m:02d}.{d:02d}",
        ]
    else:  # vi
        patterns = [
            "{d:02d}/{m:02d}/{y:04d}",   # 02/01/1990
            "{y:04d}-{m:02d}-{d:02d}",   # 1990-01-02
            "{d:02d}-{m:02d}-{y:04d}",   # 02-01-1990
        ]

    pattern = random.choice(patterns)
    return pattern.format(y=year, m=month, d=day)

# =========================
# 4. Synthetic 샘플 구조
# =========================

@dataclass
class Entity:
    label: str
    start_char: int
    end_char: int
    text: str

@dataclass
class SyntheticSample:
    lang: Lang
    text: str
    entities: List[Entity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "text": self.text,
            "entities": [asdict(e) for e in self.entities]
        }

# =========================
# 5. 템플릿 (format 기반)
# =========================

TEMPLATES_JA = [
    # 기본 정보 나열형
    "{name}様の生年月日は{dob}で、電話番号は{phone}、住所は{addr}です。メールアドレスは{email}、ID番号は{pid}、クレジットカード番号は{card}です。",
    "顧客名：{name}、生年月日：{dob}、連絡先：{phone}、メール：{email}、住所：{addr}、会員ID：{pid}、カード：{card}",
    "{name}さんのご登録情報は、住所が{addr}、電話番号が{phone}、メールが{email}、会員番号が{pid}です。生年月日は{dob}となっております。",
    "申込者：{name}、生年月日：{dob}、連絡先：{phone}、住所：{addr}、メール：{email}、本人確認番号：{pid}、カード情報：{card}",
    "{name}様は{dob}生まれで、現在の登録住所は{addr}です。連絡先は{phone}で、メールアドレスは{email}です。ID：{pid}、カード番号：{card}",
    "会員{pid}の{name}様、電話番号{phone}、住所{addr}、メール{email}が確認されました。生年月日は{dob}です。",
    # 대화체 / 콜센터
    "すみません、{name}様のお誕生日{dob}と連絡先{phone}、そして住所{addr}を確認させていただけますか？メールは{email}でよろしいでしょうか。",
    "{name}さん、登録されているメールアドレス{email}でよろしいでしょうか？ID番号は{pid}でしたよね？",
    "こちらに記載の生年月日{dob}で間違いありませんか、{name}様？電話番号は{phone}と記録されています。",
    "配送先は{addr}で、お支払いカードは{card}。これで手続き進めても大丈夫ですか、{name}さん？",
    # 노이즈 / 리스트형
    "{name} / {dob} / {phone} / {addr} / {email} / {pid} / {card}",
    "氏名 {name} 電話 {phone} メール {email} 住所 {addr} 生年月 {dob} ID {pid} CARD {card}",
]

TEMPLATES_VI = [
    # 기본 나열형
    "Khách hàng {name} sinh ngày {dob}, số điện thoại {phone}, địa chỉ {addr}. Email là {email}, số CCCD {pid}, số thẻ {card}.",
    "Tên: {name}, Ngày sinh: {dob}, Điện thoại: {phone}, Email: {email}, Địa chỉ: {addr}, CCCD: {pid}, Thẻ: {card}.",
    "Thông tin của {name}: sinh ngày {dob}, địa chỉ {addr}, điện thoại {phone}, email {email}, mã định danh {pid}, thẻ {card}.",
    "Khách hàng {name} có ngày sinh {dob}, liên hệ qua {phone} hoặc email {email}. Địa chỉ: {addr}. Số CCCD: {pid}, thẻ: {card}.",
    "Hệ thống ghi nhận {name} (CCCD {pid}), sinh ngày {dob}, địa chỉ {addr}, số điện thoại {phone}, email {email}, số thẻ {card}.",
    # 콜센터 스타일
    "Anh/Chị {name} vui lòng xác nhận ngày sinh {dob} và số điện thoại {phone} giúp em ạ. Địa chỉ hiện tại là {addr} đúng không ạ?",
    "{name}, địa chỉ của anh/chị hiện tại có phải {addr} không? Email đang dùng là {email} và số điện thoại là {phone}, CCCD {pid} đúng không ạ?",
    "Em kiểm tra thì CCCD của anh/chị là {pid}, ngày sinh {dob}, số thẻ {card}. SĐT: {phone}, email: {email}. Có đúng không ạ?",
    # 리스트/노이즈형
    "{name} | {dob} | {phone} | {addr} | {email} | {pid} | {card}",
    "Họ tên:{name} SĐT:{phone} Email:{email} ĐC:{addr} DOB:{dob} CCCD:{pid} Thẻ:{card}",
]

# =========================
# 6. Synthetic 문장 하나 생성
# =========================

def generate_synthetic_sentence(lang: Lang) -> SyntheticSample:
    """
    하나의 문장 안에 여러 종류의 PII를 섞어서 생성.
    """
    name = generate_name(lang)
    phone = generate_phone(lang)
    addr = generate_address(lang)
    dob = generate_dob(lang)
    pid = generate_id_number(lang)
    email = random_email(name)
    card = random_credit_card_number()

    if lang == "ja":
        template = random.choice(TEMPLATES_JA)
    else:
        template = random.choice(TEMPLATES_VI)

    text = template.format(
        name=name,
        phone=phone,
        addr=addr,
        dob=dob,
        pid=pid,
        email=email,
        card=card,
    )

    entities: List[Entity] = []

    def add_entity(label: str, value: str):
        start = text.find(value)
        if start == -1:
            return
        end = start + len(value)
        entities.append(Entity(label=label, start_char=start, end_char=end, text=value))

    add_entity("NAME", name)
    add_entity("PHONE", phone)
    add_entity("ADDRESS", addr)
    add_entity("DOB", dob)
    add_entity("ID_NUMBER", pid)
    add_entity("EMAIL", email)
    add_entity("CREDIT_CARD", card)

    return SyntheticSample(lang=lang, text=text, entities=entities)

# =========================
# 7. 여러 개 생성
# =========================

def generate_dataset(
    lang: Lang = "ja",
    n_samples: int = 100
) -> List[Dict[str, Any]]:
    dataset = []
    for _ in range(n_samples):
        sample = generate_synthetic_sentence(lang)
        dataset.append(sample.to_dict())
    return dataset


def _iter_tokens_with_spans(text: str):
    token_re = re.compile(
        r"\d+|[A-Za-z]+|[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+|[\uAC00-\uD7AF]+|\S"
    )
    for m in token_re.finditer(text):
        yield m.group(0), m.start(), m.end()


@lru_cache(maxsize=4)
def _get_spacy_nlp(lang: Lang):
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    model_candidates: List[str]
    if lang == "ja":
        model_candidates = [
            "ja_core_news_sm",
            "ja_core_news_md",
            "ja_core_news_lg",
            "ja_core_news_trf",
        ]
    else:  # vi
        model_candidates = [
            "vi_core_news_sm",
            "vi_core_news_md",
            "vi_core_news_lg",
            "xx_sent_ud_sm",
            "xx_ent_wiki_sm",
        ]

    for name in model_candidates:
        try:
            return spacy.load(name)
        except Exception:
            continue
    return None


def _iter_spacy_tokens_with_spans(text: str, lang: Lang):
    nlp = _get_spacy_nlp(lang)
    if nlp is None:
        return None
    doc = nlp(text)
    rows = []
    for t in doc:
        if t.is_space:
            continue
        pos = t.pos_ if t.pos_ else (t.tag_ if t.tag_ else "X")
        rows.append((t.text, t.idx, t.idx + len(t.text), pos))
    return rows


def _heuristic_pos(token: str) -> str:
    if re.fullmatch(r"\d+", token):
        return "NUM"
    if re.fullmatch(r"[A-Za-z]+", token):
        return "ALPHA"
    if re.fullmatch(r"[\u3040-\u309F]+", token):
        return "HIRA"
    if re.fullmatch(r"[\u30A0-\u30FF]+", token):
        return "KATA"
    if re.fullmatch(r"[\u4E00-\u9FFF]+", token):
        return "CJK"
    if re.fullmatch(r"[\uAC00-\uD7AF]+", token):
        return "HANGUL"
    if re.fullmatch(r"\s+", token):
        return "SPACE"
    if re.fullmatch(r"[\.,;:!?。．，、？！]", token):
        return "PUNCT"
    if re.fullmatch(r"[-–—_/\\]", token):
        return "PUNCT"
    return "SYM"


def _bio_tags_for_text(text: str, entities: List[Entity]) -> List[List[str]]:
    entities_sorted = sorted(entities, key=lambda e: (e.start_char, e.end_char))
    tok_rows: List[List[str]] = []

    for tok, s, e in _iter_tokens_with_spans(text):
        tag = "O"
        for ent in entities_sorted:
            if s >= ent.start_char and e <= ent.end_char:
                prefix = "B-" if s == ent.start_char else "I-"
                tag = prefix + ent.label
                break
        tok_rows.append([tok, tag])
    return tok_rows


def _dict_to_sample(d: Dict[str, Any]) -> SyntheticSample:
    ents = [
        Entity(
            label=e["label"],
            start_char=e["start_char"],
            end_char=e["end_char"],
            text=e["text"],
        )
        for e in d.get("entities", [])
    ]
    lang = d.get("lang")
    if lang not in ("ja", "vi"):
        lang = "ja"
    return SyntheticSample(lang=lang, text=d["text"], entities=ents)


def _token_pos_bio_rows(sample: SyntheticSample) -> List[List[str]]:
    spacy_rows = _iter_spacy_tokens_with_spans(sample.text, sample.lang)
    entities_sorted = sorted(sample.entities, key=lambda e: (e.start_char, e.end_char))
    out: List[List[str]] = []

    if spacy_rows is not None:
        for tok, s, e, pos in spacy_rows:
            tag = "O"
            for ent in entities_sorted:
                if s >= ent.start_char and e <= ent.end_char:
                    prefix = "B-" if s == ent.start_char else "I-"
                    tag = prefix + ent.label
                    break
            out.append([tok, pos, tag])
        return out

    for tok, tag in _bio_tags_for_text(sample.text, entities_sorted):
        pos = _heuristic_pos(tok)
        out.append([tok, pos, tag])
    return out


def write_bio_dataset(samples: List[SyntheticSample], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            rows = _token_pos_bio_rows(sample)
            for tok, pos, tag in rows:
                f.write(f"{tok}\t{pos}\t{tag}\n")
            f.write("\n")

def print_bio_dataset(samples: List[SyntheticSample]) -> None:
    for sample in samples:
        rows = _token_pos_bio_rows(sample)
        for tok, pos, tag in rows:
            print(f"{tok}\t{pos}\t{tag}")
        print("")

if __name__ == "__main__":
    random.seed(42)

    ja_data = generate_dataset(lang="ja", n_samples=1500)
    vi_data = generate_dataset(lang="vi", n_samples=1500)

    samples: List[SyntheticSample] = []
    samples.extend([_dict_to_sample(d) for d in ja_data])
    samples.extend([_dict_to_sample(d) for d in vi_data])

    write_bio_dataset(samples, "train_dataset.txt")
    #print_bio_dataset(samples)
