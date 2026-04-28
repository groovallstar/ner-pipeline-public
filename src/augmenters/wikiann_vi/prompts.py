"""베트남어 WikiANN을 canonical 5종 스키마로 재라벨하는 프롬프트.

지시문은 베트남어(LLM의 베트남어 문맥 이해도 극대화), 엔티티 태그는 canonical
OntoNotes 스타일 영문 축약으로 통일. 매핑·모호 사례 기준은
`docs/manual/data/canonical-entity-schema.md`·
`docs/manual/data/vietnamese-ner-8types.md` 참조.

5종 (이슈 #21 축소 결과 · 이슈 #27 LOC/ORG 경계 재정의):
  PER, LOC, ORG, PROD, EVT

축소·재배치 매핑 이력:
  CORP / POL → ORG (회사·대학 법인·정부·軍·정당·팀·협회) — #21
  FAC        → ORG (건물·역·공항·병원·학교·박물관·종교시설·캠퍼스) — #27
  地名/주소  → LOC (지명·주소만 보유) — #27
"""
from typing import List

DEFAULT_ENTITY_TYPES = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo hệ thống 5 loại chuẩn (nhãn dùng ký hiệu tiếng Anh). Tìm các thực thể trong câu và trả về mảng JSON.

## Các loại thực thể ({entity_types})
- PER: Tên người (họ và tên đầy đủ, họ, tên, biệt hiệu, nghệ danh). Loại trừ chức danh: "Ông", "Bà", "Chủ tịch", "Thủ tướng", "Tướng", "GS"
- LOC: **Chỉ vị trí địa lý** — quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh, hồ, địa chỉ (số nhà/tòa nhà/tầng). **Cơ sở nhân tạo (ga, sân bay, bệnh viện, trường học, bảo tàng, chùa, v.v.) thuộc ORG, không phải LOC**
- ORG: Tổ chức và mọi cơ sở nhân tạo — doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, **trường đại học (pháp nhân, khuôn viên, cơ sở phụ thuộc — toàn bộ)**, đảng phái, bộ/cơ quan chính phủ, quân đội, tòa án, quốc hội, tổ chức quốc tế, câu lạc bộ thể thao, hiệp hội, liên đoàn, dàn nhạc giao hưởng + **ga/nhà ga, sân bay, cảng, bệnh viện, trường tiểu học/THCS/THPT, bảo tàng, thư viện, chùa, nhà thờ, đền, sân vận động, tháp**
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình (không bao gồm tên công ty, tên người, tên cơ sở)
- EVT: Sự kiện một lần — giải đấu lớn, chiến tranh, hiệp ước, đại hội, cuộc cách mạng (không bao gồm giải đấu thường niên/câu lạc bộ)

## Quy tắc phân loại (áp dụng khi phân vân)
- Tổ chức/pháp nhân/cơ sở nhân tạo (công ty, trường đại học, chính phủ, quân đội, đảng, CLB + ga/sân bay/bệnh viện/trường PT/bảo tàng/chùa/sân vận động) → **ORG**
- Vị trí địa lý đơn thuần (quốc gia, tỉnh, thành phố, sông, núi, đảo, địa chỉ) → **LOC**
- Trường đại học (pháp nhân, khuôn viên, ký túc xá, viện nghiên cứu trực thuộc) → **ORG** (toàn bộ)
- Trường tiểu học/THCS/THPT/Bệnh viện → **ORG** (cơ sở nhân tạo)
- Sản phẩm/tác phẩm/chương trình → PROD; Sự kiện/chiến tranh/hiệp ước → EVT

## Quy tắc
1. Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
2. Loại trừ chức danh đứng trước tên ("Ông", "Bà", "Chủ tịch", "Thủ tướng")
3. Thực thể nhiều từ giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" = 1 LOC)
4. Chỉ trả về mảng JSON, không giải thích
5. Không có thực thể → trả về []

## Ví dụ
Đầu vào: Chủ tịch Hồ Chí Minh đã thăm Chùa Một Cột tại Hà Nội.
Đầu ra: [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "ORG"}}, {{"text": "Hà Nội", "type": "LOC"}}]

Đầu vào: Đảng Cộng sản Việt Nam và Bộ Giáo dục vừa ký kết hợp tác với Đại học Quốc gia Hà Nội.
Đầu ra: [{{"text": "Đảng Cộng sản Việt Nam", "type": "ORG"}}, {{"text": "Bộ Giáo dục", "type": "ORG"}}, {{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}]

Đầu vào: Vietnam Airlines vận hành chuyến bay từ Sân bay Nội Bài đến Bệnh viện Bạch Mai chuyển bệnh nhân khẩn cấp.
Đầu ra: [{{"text": "Vietnam Airlines", "type": "ORG"}}, {{"text": "Sân bay Nội Bài", "type": "ORG"}}, {{"text": "Bệnh viện Bạch Mai", "type": "ORG"}}]

Đầu vào: Hà Nội FC giành chức vô địch V.League mùa giải vừa qua.
Đầu ra: [{{"text": "Hà Nội FC", "type": "ORG"}}, {{"text": "V.League", "type": "ORG"}}]

Đầu vào: Trong Chiến tranh Việt Nam, Hiệp định Paris được ký kết tại Pháp.
Đầu ra: [{{"text": "Chiến tranh Việt Nam", "type": "EVT"}}, {{"text": "Hiệp định Paris", "type": "EVT"}}, {{"text": "Pháp", "type": "LOC"}}]

Đầu vào: Samsung giới thiệu Galaxy S24 cùng với phần mềm Windows tại sự kiện công nghệ.
Đầu ra: [{{"text": "Samsung", "type": "ORG"}}, {{"text": "Galaxy S24", "type": "PROD"}}, {{"text": "Windows", "type": "PROD"}}]

Đầu vào: Vịnh Hạ Long là một kỳ quan thiên nhiên nổi tiếng ở tỉnh Quảng Ninh.
Đầu ra: [{{"text": "Vịnh Hạ Long", "type": "LOC"}}, {{"text": "tỉnh Quảng Ninh", "type": "LOC"}}]

Đầu vào: {sentence}
Đầu ra:"""

# ── Batch prompt ──────────────────────────────────────────────────────

BATCH_PROMPT_TEMPLATE = """Bạn là chuyên gia NER tiếng Việt theo hệ thống 5 loại chuẩn (nhãn tiếng Anh). Trích xuất thực thể từ nhiều câu, trả về JSON object với chỉ số câu làm khóa.

## Các loại thực thể ({entity_types})
- PER: Tên người (không bao gồm chức danh "Ông/Bà/Chủ tịch/Thủ tướng")
- LOC: **Chỉ vị trí địa lý** (quốc gia/thành phố/sông/núi/đảo/địa chỉ). Cơ sở nhân tạo thuộc ORG
- ORG: Tổ chức và mọi cơ sở nhân tạo (công ty/tập đoàn/ngân hàng/hãng/đài/trường đại học toàn bộ/đảng/bộ/quân đội/CLB/hiệp hội/tổ chức quốc tế + ga/sân bay/bệnh viện/trường PT/bảo tàng/thư viện/chùa/nhà thờ/sân vận động)
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình
- EVT: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn

## Quy tắc phân loại (áp dụng khi phân vân)
- Tổ chức/cơ sở nhân tạo → ORG; vị trí địa lý đơn thuần → LOC
- Trường đại học (toàn bộ) → ORG; ga/sân bay/bệnh viện/trường PT → ORG
- Sản phẩm/tác phẩm → PROD; Sự kiện/chiến tranh/hiệp ước → EVT

## Quy tắc
1. Giữ nguyên dấu tiếng Việt
2. Loại trừ chức danh đứng trước tên
3. Thực thể nhiều từ giữ nguyên thành một thực thể
4. Trả về JSON object với chỉ số câu là key. Không có thực thể → mảng rỗng
5. Chỉ JSON, không giải thích

## Ví dụ
Đầu vào:
0: Chủ tịch Hồ Chí Minh đã thăm Chùa Một Cột.
1: Vietnam Airlines khai thác chuyến bay từ Sân bay Nội Bài.
2: Samsung giới thiệu Galaxy S24.

Đầu ra: {{"0": [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "ORG"}}], "1": [{{"text": "Vietnam Airlines", "type": "ORG"}}, {{"text": "Sân bay Nội Bài", "type": "ORG"}}], "2": [{{"text": "Samsung", "type": "ORG"}}, {{"text": "Galaxy S24", "type": "PROD"}}]}}

## Các câu đầu vào
{sentences}

Đầu ra:"""


def format_entity_types(entity_types: List[str]) -> str:
    return ', '.join(entity_types)
