"""베트남어 WikiANN을 Stockmark 8종 스키마로 재라벨하는 프롬프트.

지시문은 베트남어(LLM의 베트남어 문맥 이해도 극대화), 엔티티 태그는
Stockmark 원문(일본어)을 canonical 값으로 그대로 사용한다. 매핑·모호 사례
기준은 `docs/specs/entities/vietnamese-ner-8types.md` 참조.

8종: 人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名
"""
from typing import List

DEFAULT_ENTITY_TYPES = [
    '人名', '法人名', '地名', '施設名',
    '製品名', 'イベント名', '政治的組織名', 'その他の組織名',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo chuẩn Stockmark 8 loại (thẻ giữ nguyên ký tự gốc tiếng Nhật). Tìm các thực thể trong câu và trả về mảng JSON.

## Các loại thực thể ({entity_types})
- 人名: Tên người (họ và tên đầy đủ, họ, tên, biệt hiệu, nghệ danh). Loại trừ chức danh: "Ông", "Bà", "Chủ tịch", "Thủ tướng", "Tướng", "GS"
- 法人名: Doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, công ty đường sắt — tổ chức hoạt động thương mại
- 地名: Quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh — địa danh tự nhiên hoặc hành chính
- 施設名: Công trình/tòa nhà cụ thể — chùa, nhà thờ, đền, lăng, ga, sân bay, cảng, bảo tàng, thư viện, nhà hát, công viên, cửa hàng, bệnh viện, trường tiểu học/THCS/THPT
- 製品名: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình (không bao gồm tên công ty, tên người, tên cơ sở)
- イベント名: Sự kiện một lần — giải đấu lớn, chiến tranh, hiệp ước, đại hội, cuộc cách mạng (không bao gồm giải đấu thường niên/câu lạc bộ)
- 政治的組織名: Đảng phái, bộ/cơ quan chính phủ ("Bộ/Cục/Sở/Ủy ban"), quân đội, tòa án, quốc hội, tổ chức quốc tế (Liên Hợp Quốc, ASEAN)
- その他の組織名: Trường đại học, giải đấu thể thao định kỳ, câu lạc bộ thể thao, hội, hiệp hội, liên đoàn — tổ chức không thuộc các loại trên

## Thứ tự ưu tiên phân loại (áp dụng từ trên xuống khi phân vân)
1. Chính phủ/Đảng/Quân đội/Tòa án/Quốc hội/Tổ chức quốc tế → 政治的組織名
2. Đại học/Giải đấu/CLB thể thao/Hiệp hội/Liên đoàn → その他の組織名
3. Chủ thể thương mại ("Công ty/Tập đoàn/Ngân hàng/Hãng/Đài truyền hình") → 法人名
4. Công trình vật lý cụ thể (chùa/nhà thờ/ga/sân bay/bảo tàng/bệnh viện/trường PT) → 施設名
5. Sản phẩm/tác phẩm/chương trình → 製品名
6. Sự kiện/chiến tranh/hiệp ước → イベント名
- Phán định "địa điểm vật lý vs hoạt động tổ chức" để phân biệt 施設名 và 法人名/その他の組織名
- Trường đại học → その他の組織名, nhưng trường tiểu học/THCS/THPT → 施設名
- Bệnh viện là công trình → 施設名 (pháp nhân vận hành bệnh viện là 法人名 riêng)

## Quy tắc
1. Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
2. Loại trừ chức danh đứng trước tên ("Ông", "Bà", "Chủ tịch", "Thủ tướng")
3. Thực thể nhiều từ giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" = 1 地名)
4. Chỉ trả về mảng JSON, không giải thích
5. Không có thực thể → trả về []

## Ví dụ
Đầu vào: Chủ tịch Hồ Chí Minh đã thăm Chùa Một Cột tại Hà Nội.
Đầu ra: [{{"text": "Hồ Chí Minh", "type": "人名"}}, {{"text": "Chùa Một Cột", "type": "施設名"}}, {{"text": "Hà Nội", "type": "地名"}}]

Đầu vào: Đảng Cộng sản Việt Nam và Bộ Giáo dục vừa ký kết hợp tác với Đại học Quốc gia Hà Nội.
Đầu ra: [{{"text": "Đảng Cộng sản Việt Nam", "type": "政治的組織名"}}, {{"text": "Bộ Giáo dục", "type": "政治的組織名"}}, {{"text": "Đại học Quốc gia Hà Nội", "type": "その他の組織名"}}]

Đầu vào: Vietnam Airlines vận hành chuyến bay từ Sân bay Nội Bài đến Bệnh viện Bạch Mai chuyển bệnh nhân khẩn cấp.
Đầu ra: [{{"text": "Vietnam Airlines", "type": "法人名"}}, {{"text": "Sân bay Nội Bài", "type": "施設名"}}, {{"text": "Bệnh viện Bạch Mai", "type": "施設名"}}]

Đầu vào: Hà Nội FC giành chức vô địch V.League mùa giải vừa qua.
Đầu ra: [{{"text": "Hà Nội FC", "type": "その他の組織名"}}, {{"text": "V.League", "type": "その他の組織名"}}]

Đầu vào: Trong Chiến tranh Việt Nam, Hiệp định Paris được ký kết tại Pháp.
Đầu ra: [{{"text": "Chiến tranh Việt Nam", "type": "イベント名"}}, {{"text": "Hiệp định Paris", "type": "イベント名"}}, {{"text": "Pháp", "type": "地名"}}]

Đầu vào: Samsung giới thiệu Galaxy S24 cùng với phần mềm Windows tại sự kiện công nghệ.
Đầu ra: [{{"text": "Samsung", "type": "法人名"}}, {{"text": "Galaxy S24", "type": "製品名"}}, {{"text": "Windows", "type": "製品名"}}]

Đầu vào: Vịnh Hạ Long là một kỳ quan thiên nhiên nổi tiếng ở tỉnh Quảng Ninh.
Đầu ra: [{{"text": "Vịnh Hạ Long", "type": "地名"}}, {{"text": "tỉnh Quảng Ninh", "type": "地名"}}]

Đầu vào: {sentence}
Đầu ra:"""

# ── Batch prompt ──────────────────────────────────────────────────────

BATCH_PROMPT_TEMPLATE = """Bạn là chuyên gia NER tiếng Việt theo chuẩn Stockmark 8 loại (thẻ giữ nguyên tiếng Nhật gốc). Trích xuất thực thể từ nhiều câu, trả về JSON object với chỉ số câu làm khóa.

## Các loại thực thể ({entity_types})
- 人名: Tên người (không bao gồm chức danh "Ông/Bà/Chủ tịch/Thủ tướng")
- 法人名: Doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, công ty đường sắt
- 地名: Quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh
- 施設名: Công trình vật lý — chùa/nhà thờ/ga/sân bay/bảo tàng/bệnh viện/trường PT/cửa hàng
- 製品名: Sản phẩm, dịch vụ, phần mềm, tác phẩm, chương trình
- イベント名: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn
- 政治的組織名: Đảng, bộ/cục chính phủ, quân đội, tòa án, quốc hội, tổ chức quốc tế
- その他の組織名: Đại học, giải đấu thể thao định kỳ, CLB thể thao, hiệp hội, liên đoàn

## Thứ tự ưu tiên (áp dụng từ trên xuống)
1. Chính phủ/Đảng/Quân đội/Tòa án/Quốc hội → 政治的組織名
2. Đại học/CLB thể thao/Giải đấu định kỳ → その他の組織名
3. Chủ thể thương mại → 法人名
4. Công trình vật lý cụ thể (bao gồm bệnh viện, trường PT) → 施設名
5. Sản phẩm/tác phẩm → 製品名
6. Sự kiện/chiến tranh → イベント名

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

Đầu ra: {{"0": [{{"text": "Hồ Chí Minh", "type": "人名"}}, {{"text": "Chùa Một Cột", "type": "施設名"}}], "1": [{{"text": "Vietnam Airlines", "type": "法人名"}}, {{"text": "Sân bay Nội Bài", "type": "施設名"}}], "2": [{{"text": "Samsung", "type": "法人名"}}, {{"text": "Galaxy S24", "type": "製品名"}}]}}

## Các câu đầu vào
{sentences}

Đầu ra:"""


def format_entity_types(entity_types: List[str]) -> str:
    return ', '.join(entity_types)
