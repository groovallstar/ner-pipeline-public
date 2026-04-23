"""베트남어 WikiANN을 canonical 8종 스키마로 재라벨하는 프롬프트.

지시문은 베트남어(LLM의 베트남어 문맥 이해도 극대화), 엔티티 태그는 canonical
OntoNotes 스타일 영문 축약으로 통일. 매핑·모호 사례 기준은
`docs/manual/canonical-entity-schema.md`·
`docs/specs/entities/vietnamese-ner-8types.md` 참조.

8종: PER, CORP, LOC, FAC, PROD, EVT, POL, ORG
"""
from typing import List

DEFAULT_ENTITY_TYPES = [
    'PER', 'CORP', 'LOC', 'FAC',
    'PROD', 'EVT', 'POL', 'ORG',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo hệ thống 8 loại chuẩn (nhãn dùng ký hiệu tiếng Anh). Tìm các thực thể trong câu và trả về mảng JSON.

## Các loại thực thể ({entity_types})
- PER: Tên người (họ và tên đầy đủ, họ, tên, biệt hiệu, nghệ danh). Loại trừ chức danh: "Ông", "Bà", "Chủ tịch", "Thủ tướng", "Tướng", "GS"
- CORP: Doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, công ty đường sắt — tổ chức hoạt động thương mại
- LOC: Quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh — địa danh tự nhiên hoặc hành chính
- FAC: Công trình/tòa nhà cụ thể — chùa, nhà thờ, đền, lăng, ga, sân bay, cảng, bảo tàng, thư viện, nhà hát, công viên, cửa hàng, bệnh viện, trường tiểu học/THCS/THPT
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình (không bao gồm tên công ty, tên người, tên cơ sở)
- EVT: Sự kiện một lần — giải đấu lớn, chiến tranh, hiệp ước, đại hội, cuộc cách mạng (không bao gồm giải đấu thường niên/câu lạc bộ)
- POL: Đảng phái, bộ/cơ quan chính phủ ("Bộ/Cục/Sở/Ủy ban"), quân đội, tòa án, quốc hội, tổ chức quốc tế (Liên Hợp Quốc, ASEAN)
- ORG: Trường đại học, giải đấu thể thao định kỳ, câu lạc bộ thể thao, hội, hiệp hội, liên đoàn — tổ chức không thuộc các loại trên

## Thứ tự ưu tiên phân loại (áp dụng từ trên xuống khi phân vân)
1. Chính phủ/Đảng/Quân đội/Tòa án/Quốc hội/Tổ chức quốc tế → POL
2. Đại học/Giải đấu/CLB thể thao/Hiệp hội/Liên đoàn → ORG
3. Chủ thể thương mại ("Công ty/Tập đoàn/Ngân hàng/Hãng/Đài truyền hình") → CORP
4. Công trình vật lý cụ thể (chùa/nhà thờ/ga/sân bay/bảo tàng/bệnh viện/trường PT) → FAC
5. Sản phẩm/tác phẩm/chương trình → PROD
6. Sự kiện/chiến tranh/hiệp ước → EVT
- Phán định "địa điểm vật lý vs hoạt động tổ chức" để phân biệt FAC và CORP/ORG
- Trường đại học → ORG, nhưng trường tiểu học/THCS/THPT → FAC
- Bệnh viện là công trình → FAC (pháp nhân vận hành bệnh viện là CORP riêng)

## Quy tắc
1. Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
2. Loại trừ chức danh đứng trước tên ("Ông", "Bà", "Chủ tịch", "Thủ tướng")
3. Thực thể nhiều từ giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" = 1 LOC)
4. Chỉ trả về mảng JSON, không giải thích
5. Không có thực thể → trả về []

## Ví dụ
Đầu vào: Chủ tịch Hồ Chí Minh đã thăm Chùa Một Cột tại Hà Nội.
Đầu ra: [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "FAC"}}, {{"text": "Hà Nội", "type": "LOC"}}]

Đầu vào: Đảng Cộng sản Việt Nam và Bộ Giáo dục vừa ký kết hợp tác với Đại học Quốc gia Hà Nội.
Đầu ra: [{{"text": "Đảng Cộng sản Việt Nam", "type": "POL"}}, {{"text": "Bộ Giáo dục", "type": "POL"}}, {{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}]

Đầu vào: Vietnam Airlines vận hành chuyến bay từ Sân bay Nội Bài đến Bệnh viện Bạch Mai chuyển bệnh nhân khẩn cấp.
Đầu ra: [{{"text": "Vietnam Airlines", "type": "CORP"}}, {{"text": "Sân bay Nội Bài", "type": "FAC"}}, {{"text": "Bệnh viện Bạch Mai", "type": "FAC"}}]

Đầu vào: Hà Nội FC giành chức vô địch V.League mùa giải vừa qua.
Đầu ra: [{{"text": "Hà Nội FC", "type": "ORG"}}, {{"text": "V.League", "type": "ORG"}}]

Đầu vào: Trong Chiến tranh Việt Nam, Hiệp định Paris được ký kết tại Pháp.
Đầu ra: [{{"text": "Chiến tranh Việt Nam", "type": "EVT"}}, {{"text": "Hiệp định Paris", "type": "EVT"}}, {{"text": "Pháp", "type": "LOC"}}]

Đầu vào: Samsung giới thiệu Galaxy S24 cùng với phần mềm Windows tại sự kiện công nghệ.
Đầu ra: [{{"text": "Samsung", "type": "CORP"}}, {{"text": "Galaxy S24", "type": "PROD"}}, {{"text": "Windows", "type": "PROD"}}]

Đầu vào: Vịnh Hạ Long là một kỳ quan thiên nhiên nổi tiếng ở tỉnh Quảng Ninh.
Đầu ra: [{{"text": "Vịnh Hạ Long", "type": "LOC"}}, {{"text": "tỉnh Quảng Ninh", "type": "LOC"}}]

Đầu vào: {sentence}
Đầu ra:"""

# ── Batch prompt ──────────────────────────────────────────────────────

BATCH_PROMPT_TEMPLATE = """Bạn là chuyên gia NER tiếng Việt theo hệ thống 8 loại chuẩn (nhãn tiếng Anh). Trích xuất thực thể từ nhiều câu, trả về JSON object với chỉ số câu làm khóa.

## Các loại thực thể ({entity_types})
- PER: Tên người (không bao gồm chức danh "Ông/Bà/Chủ tịch/Thủ tướng")
- CORP: Doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, công ty đường sắt
- LOC: Quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh
- FAC: Công trình vật lý — chùa/nhà thờ/ga/sân bay/bảo tàng/bệnh viện/trường PT/cửa hàng
- PROD: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình
- EVT: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn
- POL: Đảng, bộ/cục chính phủ, quân đội, tòa án, quốc hội, tổ chức quốc tế
- ORG: Đại học, giải đấu thể thao định kỳ, CLB thể thao, hiệp hội, liên đoàn

## Thứ tự ưu tiên (áp dụng từ trên xuống)
1. Chính phủ/Đảng/Quân đội/Tòa án/Quốc hội → POL
2. Đại học/CLB thể thao/Giải đấu định kỳ → ORG
3. Chủ thể thương mại → CORP
4. Công trình vật lý cụ thể (bao gồm bệnh viện, trường PT) → FAC
5. Sản phẩm/tác phẩm → PROD
6. Sự kiện/chiến tranh → EVT

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

Đầu ra: {{"0": [{{"text": "Hồ Chí Minh", "type": "PER"}}, {{"text": "Chùa Một Cột", "type": "FAC"}}], "1": [{{"text": "Vietnam Airlines", "type": "CORP"}}, {{"text": "Sân bay Nội Bài", "type": "FAC"}}], "2": [{{"text": "Samsung", "type": "CORP"}}, {{"text": "Galaxy S24", "type": "PROD"}}]}}

## Các câu đầu vào
{sentences}

Đầu ra:"""


def format_entity_types(entity_types: List[str]) -> str:
    return ', '.join(entity_types)
