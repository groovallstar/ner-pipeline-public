"""베트남어 WikiANN을 canonical 5종 스키마로 재라벨하는 프롬프트.

지시문은 베트남어(LLM의 베트남어 문맥 이해도 극대화), 엔티티 태그는 canonical
OntoNotes 스타일 영문 축약으로 통일. 매핑·모호 사례 기준은
`docs/manual/data/canonical-entity-schema.md`·
`docs/manual/data/vietnamese-ner-8types.md` 참조.

5종 (이슈 #21 축소 결과): PER, LOC, ORG, PROD, EVT

축소 매핑 (참고):
  CORP / POL → ORG   (회사·대학 법인·정부·軍·정당·팀·협회)
  FAC        → LOC   (건물·역·공항·병원·학교 캠퍼스 포함 지명)
"""
from typing import List

DEFAULT_ENTITY_TYPES = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo hệ thống 5 loại chuẩn (nhãn dùng ký hiệu tiếng Anh). Tìm các thực thể trong câu và trả về mảng JSON.

## Các loại thực thể ({entity_types})
- PER: Tên người (họ và tên đầy đủ, họ, tên, biệt hiệu, nghệ danh). Loại trừ chức danh: "Ông", "Bà", "Chủ tịch", "Thủ tướng", "Tướng", "GS"
- LOC: Địa danh và công trình vật lý cụ thể — quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh, ga, sân bay, cảng, bệnh viện, trường học, bảo tàng, thư viện, chùa, nhà thờ, ký túc xá/khu giảng đường đại học
- ORG: Tổ chức — doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, trường đại học (pháp nhân), đảng phái, bộ/cơ quan chính phủ, quân đội, tòa án, quốc hội, tổ chức quốc tế, câu lạc bộ thể thao, hiệp hội, liên đoàn, dàn nhạc giao hưởng, tổ chức trực thuộc đại học
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình (không bao gồm tên công ty, tên người, tên cơ sở)
- EVT: Sự kiện một lần — giải đấu lớn, chiến tranh, hiệp ước, đại hội, cuộc cách mạng (không bao gồm giải đấu thường niên/câu lạc bộ)

## Quy tắc phân loại (áp dụng khi phân vân)
- Thực thể thuộc loại "tổ chức/pháp nhân" (công ty, trường đại học bản thể, chính phủ, quân đội, đảng, CLB) → ORG
- Thực thể thuộc loại "địa điểm vật lý" (tòa nhà, ga, sân bay, bệnh viện, trường PT, cơ sở phụ thuộc đại học) → LOC
- Trường đại học: pháp nhân bản thể → ORG, cơ sở phụ thuộc (ký túc xá/khu giảng đường) → LOC, tổ chức trực thuộc (câu lạc bộ/viện nghiên cứu) → ORG
- Trường tiểu học/THCS/THPT → LOC (công trình); Bệnh viện → LOC (công trình)
- Sản phẩm/tác phẩm/chương trình → PROD; Sự kiện/chiến tranh/hiệp ước → EVT

## Quy tắc
1. Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
2. Loại trừ chức danh đứng trước tên ("Ông", "Bà", "Chủ tịch", "Thủ tướng")
3. Thực thể nhiều từ giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" = 1 LOC)
4. Chỉ trả về mảng JSON, không giải thích
5. Không có thực thể → trả về []

## Ví dụ
Đầu vào: Chủ tịch Hồ Chí Minh đã thăm Chùa Một Cột tại Hà Nội.
Đầu ra: [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "LOC"}}, {{"text": "Hà Nội", "type": "LOC"}}]

Đầu vào: Đảng Cộng sản Việt Nam và Bộ Giáo dục vừa ký kết hợp tác với Đại học Quốc gia Hà Nội.
Đầu ra: [{{"text": "Đảng Cộng sản Việt Nam", "type": "ORG"}}, {{"text": "Bộ Giáo dục", "type": "ORG"}}, {{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}]

Đầu vào: Vietnam Airlines vận hành chuyến bay từ Sân bay Nội Bài đến Bệnh viện Bạch Mai chuyển bệnh nhân khẩn cấp.
Đầu ra: [{{"text": "Vietnam Airlines", "type": "ORG"}}, {{"text": "Sân bay Nội Bài", "type": "LOC"}}, {{"text": "Bệnh viện Bạch Mai", "type": "LOC"}}]

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
- LOC: Địa danh và công trình vật lý (quốc gia/thành phố/sông/núi/đảo + ga/sân bay/bệnh viện/trường PT/chùa/ký túc xá đại học)
- ORG: Tổ chức (công ty/tập đoàn/ngân hàng/hãng/đài/trường đại học pháp nhân/đảng/bộ/quân đội/CLB/hiệp hội/tổ chức quốc tế)
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình
- EVT: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn

## Quy tắc phân loại (áp dụng khi phân vân)
- Pháp nhân/tổ chức → ORG; địa điểm vật lý → LOC
- Trường đại học pháp nhân → ORG; cơ sở phụ thuộc đại học → LOC
- Trường PT/Bệnh viện → LOC (công trình)
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

Đầu ra: {{"0": [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "LOC"}}], "1": [{{"text": "Vietnam Airlines", "type": "ORG"}}, {{"text": "Sân bay Nội Bài", "type": "LOC"}}], "2": [{{"text": "Samsung", "type": "ORG"}}, {{"text": "Galaxy S24", "type": "PROD"}}]}}

## Các câu đầu vào
{sentences}

Đầu ra:"""


def format_entity_types(entity_types: List[str]) -> str:
    return ', '.join(entity_types)
