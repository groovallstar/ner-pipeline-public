"""베트남어 LLM 라벨러용 공통 NER 프롬프트 템플릿.

WikiANN NER 어노테이션 가이드라인에 맞게 조정되었다:
- 3개 엔티티 타입: PER, LOC, ORG
- 베트남어 성조 부호 및 다중 어절 이름에 대한 언어별 규칙
"""

from typing import List

DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG"]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt. Tìm các thực thể trong văn bản và trả về dưới dạng mảng JSON.

## Loại thực thể ({entity_types})
- PER (Người): Tên người, bao gồm họ và tên đầy đủ hoặc một phần. Bí danh, biệt hiệu, nghệ danh cũng là PER
  Ví dụ: "Nguyễn Văn A" → PER, "Hồ Chí Minh" → PER, "Bác Hồ" → PER
  Lưu ý: Tên nhóm nhạc, ban nhạc không phải PER mà là ORG
- LOC (Địa điểm): Địa danh, quốc gia, thành phố, tỉnh, huyện, xã, đường, sông, núi, biển, đảo, công trình kiến trúc
  Ví dụ: "Hà Nội", "Việt Nam", "Sông Hồng", "Phú Quốc", "Đà Nẵng", "Chùa Một Cột"
  Lưu ý: Địa danh phức hợp giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" → LOC 1 thực thể)
- ORG (Tổ chức): Công ty, cơ quan, tổ chức, đảng phái, đội bóng, trường học, bệnh viện
  Ví dụ: "Đảng Cộng sản Việt Nam", "Samsung", "Đại học Quốc gia Hà Nội", "Công an", "Bộ Giáo dục"
  Lưu ý: Tên tổ chức phức hợp giữ nguyên thành một thực thể

## Quy tắc chính
1. Trích xuất chính xác văn bản gốc, giữ nguyên dấu tiếng Việt
2. Thực thể nhiều từ phải giữ nguyên thành một thực thể (không tách)
3. Loại trừ các từ chức danh đứng trước tên ("Ông", "Bà", "Chủ tịch", "Thủ tướng") — chỉ trích xuất tên
4. Chỉ trả về mảng JSON, không giải thích thêm
5. Nếu không có thực thể, trả về []

## Ví dụ
Đầu vào: Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng và gặp đại diện Tập đoàn Vingroup.
Đầu ra: [{{"text": "Nguyễn Xuân Phúc", "type": "PER"}}, {{"text": "Đà Nẵng", "type": "LOC"}}, {{"text": "Tập đoàn Vingroup", "type": "ORG"}}]

Đầu vào: Sông Mekong chảy qua Campuchia và Việt Nam trước khi đổ ra Biển Đông.
Đầu ra: [{{"text": "Sông Mekong", "type": "LOC"}}, {{"text": "Campuchia", "type": "LOC"}}, {{"text": "Việt Nam", "type": "LOC"}}, {{"text": "Biển Đông", "type": "LOC"}}]

Đầu vào: Đại học Quốc gia Hà Nội vừa ký kết hợp tác với Samsung Electronics tại Hà Nội.
Đầu ra: [{{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}, {{"text": "Samsung Electronics", "type": "ORG"}}, {{"text": "Hà Nội", "type": "LOC"}}]

Đầu vào: {sentence}
Đầu ra:"""

# ── Batch prompt ──────────────────────────────────────────────────────

BATCH_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt. Trích xuất thực thể từ nhiều câu và trả về dưới dạng JSON.

## Loại thực thể ({entity_types})
- PER (Người): Tên người đầy đủ hoặc một phần, bí danh, nghệ danh. Không bao gồm chức danh ("Ông", "Bà", "Chủ tịch")
- LOC (Địa điểm): Địa danh, quốc gia, thành phố, tỉnh, sông, núi, biển, đảo, công trình. Địa danh phức hợp giữ nguyên thành một thực thể
- ORG (Tổ chức): Công ty, cơ quan, tổ chức, đảng phái, đội bóng, trường học, bệnh viện. Tên phức hợp giữ nguyên thành một thực thể

## Quy tắc chính
1. Trích xuất chính xác văn bản gốc, giữ nguyên dấu tiếng Việt
2. Thực thể nhiều từ phải giữ nguyên thành một thực thể
3. Loại trừ chức danh đứng trước tên
4. Trả về JSON object với chỉ số câu là key. Nếu không có thực thể, trả về mảng rỗng
5. Không giải thích thêm, chỉ JSON

## Ví dụ
Đầu vào:
0: Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng.
1: Đại học Quốc gia Hà Nội ký kết hợp tác với Samsung.

Đầu ra: {{"0": [{{"text": "Nguyễn Xuân Phúc", "type": "PER"}}, {{"text": "Đà Nẵng", "type": "LOC"}}], "1": [{{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}, {{"text": "Samsung", "type": "ORG"}}]}}

## Các câu đầu vào
{sentences}

Đầu ra:"""

# ── System prompt for OpenAI (chat format) ────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt. Trích xuất thực thể từ các câu và trả về dưới dạng JSON.

Loại thực thể:
- PER (Người): Tên người đầy đủ hoặc một phần, bí danh, nghệ danh. Không bao gồm chức danh ("Ông", "Bà", "Chủ tịch")
- LOC (Địa điểm): Địa danh, quốc gia, thành phố, tỉnh, sông, núi, biển, đảo, công trình. Địa danh phức hợp giữ nguyên thành một thực thể
- ORG (Tổ chức): Công ty, cơ quan, tổ chức, đảng phái, đội bóng, trường học, bệnh viện. Tên phức hợp giữ nguyên thành một thực thể

Quy tắc chính:
- Trích xuất chính xác văn bản gốc, giữ nguyên dấu tiếng Việt
- Thực thể nhiều từ phải giữ nguyên thành một thực thể
- Loại trừ chức danh đứng trước tên
- Trả về JSON object với chỉ số câu là key. Nếu không có thực thể, trả về mảng rỗng
- Không giải thích thêm, chỉ JSON"""

USER_PROMPT_TEMPLATE = """Trích xuất thực thể từ các câu dưới đây.

Đầu vào:
{sentences}

Định dạng đầu ra ví dụ:
{{"0": [{{"text": "Nguyễn Văn A", "type": "PER"}}, {{"text": "Hà Nội", "type": "LOC"}}], "1": [{{"text": "Samsung", "type": "ORG"}}], "2": []}}

Đầu ra:"""


def format_entity_types(entity_types: List[str]) -> str:
    return ", ".join(entity_types)
