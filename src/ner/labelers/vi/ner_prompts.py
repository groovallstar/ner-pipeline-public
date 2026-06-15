"""베트남어 LLM 라벨러용 공통 NER 프롬프트 템플릿.

canonical 10종 평면 목록(LOC = 지명·주소만, ORG = 인공 시설·조직 일체):
  PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD

LOC = 지명·주소만 (국가·행정구역·자연지명·주소).
ORG = 모든 인공 시설·조직 (역·공항·병원·학교·대학 일체·점포·박물관·
      도서관·종교시설 + 기업·정당·정부·국제기관·스포츠팀·협회 등).

WikiANN-vi 원본은 3종(PER/LOC/ORG)이지만, 본 라벨러는 canonical 전체를
출력해 다국어 벤치마크(JA Stockmark 5종·PII 주입 세트)와 동일한 라벨
공간에서 평가할 수 있도록 한다. WikiANN 3종 gold로 평가 시 PROD/EVT/
PII 예측은 FP로 잡혀 precision이 하락하며, 또한 WikiANN 원본은 시설을
LOC로 라벨링하므로 본 canonical(시설=ORG)과 시설 엔티티가 LOC↔ORG 간
어긋나는 미스매치가 추가로 발생한다 — 이는 설계 의도다. 5종·10종 라벨이
포함된 gold(`data/wikiann_vi/*.jsonl` 축소 후 5종)로 평가하면 정상 비교가
가능하다.
"""


DEFAULT_ENTITY_TYPES = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
    'EMAIL', 'PHONE', 'DAT', 'ID_NUM', 'CREDIT_CARD',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo hệ thống 10 loại chuẩn (nhãn dùng ký hiệu tiếng Anh). Tìm các thực thể trong văn bản và trả về dưới dạng mảng JSON.

## Loại thực thể ({entity_types})
- PER: Tên người (họ tên đầy đủ, họ, tên, biệt hiệu, nghệ danh). Loại trừ chức danh "Ông/Bà/Chủ tịch/Thủ tướng/Tướng"
- LOC: **Chỉ vị trí địa lý** — quốc gia, tỉnh, thành phố, huyện, xã, sông, núi, biển, đảo, vịnh, hồ, địa chỉ (số nhà/tòa nhà/tầng). **Cơ sở nhân tạo (ga, sân bay, bệnh viện, trường học, bảo tàng, chùa, v.v.) thuộc ORG, không phải LOC**
- ORG: Tổ chức và mọi cơ sở nhân tạo — doanh nghiệp, tập đoàn, ngân hàng, hãng hàng không, đài truyền hình, **trường đại học (pháp nhân, khuôn viên, cơ sở phụ thuộc — tất cả)**, đảng phái, bộ/cơ quan chính phủ, quân đội, tòa án, quốc hội, tổ chức quốc tế, câu lạc bộ/đội thể thao, giải đấu định kỳ, hiệp hội, liên đoàn, dàn nhạc giao hưởng + **ga/nhà ga, sân bay, cảng, bệnh viện, trường tiểu học/THCS/THPT, bảo tàng, thư viện, chùa, nhà thờ, đền, sân vận động, tháp**
- PROD: Sản phẩm hữu hình, tác phẩm sáng tạo (âm nhạc/phim/sách/tiểu thuyết/truyện tranh/anime/game/chương trình TV), phần mềm đóng gói, phương tiện/vũ khí/tàu/máy bay có tên model. **KHÔNG gồm: dịch vụ, SaaS, viễn thông, game vận hành trực tuyến, tiêu chuẩn/định dạng/giao thức kỹ thuật, giải thưởng/huân chương** (→ phi-thực-thể); tên công ty/người/cơ sở (→ ORG/PER)
- EVT: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn, cuộc cách mạng (không bao gồm giải đấu thường niên — đó là ORG)
- EMAIL: Địa chỉ email đầy đủ dạng `local@domain.TLD` (TLD bắt buộc: .com/.vn/.net/.org/.edu/.gov.vn…)
- PHONE: Số điện thoại (định dạng Việt Nam hoặc quốc tế: 090-1234-567, +84 90 1234 567, 0901234567)
- DAT: Ngày tháng tổng quát — năm, tháng, ngày, khoảng thời gian, thời đại. Ngày sinh, ngày sự kiện, ngày thành lập đều thuộc DAT không phân biệt ngữ cảnh. Địa chỉ vật lý (số nhà) thuộc LOC, không phải DAT
- ID_NUM: Số định danh cá nhân (CCCD, CMND, mã số thuế…), chuỗi số có thể có dấu gạch
- CREDIT_CARD: Số thẻ tín dụng (13~19 chữ số, cho phép dấu cách/gạch ngang)

## Quy tắc phân loại (áp dụng khi phân vân)
1. Tổ chức/pháp nhân/công quyền/cơ sở nhân tạo (công ty/trường đại học/chính phủ/quân đội/đảng/câu lạc bộ thể thao/hiệp hội + ga/sân bay/bệnh viện/trường PT/bảo tàng/chùa/nhà thờ/sân vận động) → **ORG**
2. Vị trí địa lý đơn thuần (quốc gia/tỉnh/thành phố/sông/núi/đảo/địa chỉ) → **LOC**
3. Sản phẩm/tác phẩm/chương trình → **PROD**
4. Sự kiện/chiến tranh/hiệp ước một lần → **EVT**
- Trường đại học (pháp nhân, khuôn viên, ký túc xá, viện nghiên cứu trực thuộc) → **ORG** (toàn bộ)
- Bệnh viện/Trường tiểu học/THCS/THPT → **ORG** (cơ sở nhân tạo)
- Phức hợp địa danh hành chính ("Thành phố Hồ Chí Minh") → **LOC** đơn nhất

## Ranh giới PROD/EVT (§3 — áp dụng nghiêm)
- Tác phẩm âm nhạc/phim/sách/anime/game/chương trình TV → **PROD** (cả tiêu đề đứng một mình trong dấu ngoặc kép)
- Giải định kỳ/CLB → **ORG**; phiên bản theo năm ("World Cup 2022") → **EVT**
- Mã số mẫu đơn ("RV522") → KHÔNG PHẢI PROD; chỉ thương hiệu+mã ("Galaxy S24") mới là PROD
- Luật/pháp lệnh/nghị định/quy định → **KHÔNG PHẢI thực thể** (chỉ hiệp ước = EVT)
- Triển lãm/tour/ra mắt, dự án/kế hoạch/chiến lược có tên → **EVT**; thảm họa/khủng hoảng có tên/bầu cử/phong trào có mốc → **EVT**
- Quá trình kéo dài nhiều năm (Chiến tranh Lạnh, Cách mạng Công nghiệp, Minh Trị Duy tân)/thời đại/chủ đề trừu tượng ("vấn đề ~") → **KHÔNG PHẢI thực thể**

## Quy tắc chung
1. Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
2. Thực thể nhiều từ giữ nguyên thành một thực thể ("Thành phố Hồ Chí Minh" → 1 LOC, "Đại học Quốc gia Hà Nội" → 1 ORG)
3. Loại trừ chức danh đứng trước tên ("Ông/Bà/Chủ tịch/Thủ tướng")
4. Chỉ trả về mảng JSON, không giải thích
5. Không có thực thể → trả về []
6. Nhãn dùng ký hiệu tiếng Anh (PER/LOC/ORG/PROD/EVT/EMAIL/PHONE/DAT/ID_NUM/CREDIT_CARD)

## Ví dụ
Đầu vào: Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng và gặp đại diện Tập đoàn Vingroup.
Đầu ra: [{{"text": "Nguyễn Xuân Phúc", "type": "PER"}}, {{"text": "Đà Nẵng", "type": "LOC"}}, {{"text": "Tập đoàn Vingroup", "type": "ORG"}}]

Đầu vào: Đảng Cộng sản Việt Nam và Bộ Giáo dục vừa ký kết hợp tác với Đại học Quốc gia Hà Nội.
Đầu ra: [{{"text": "Đảng Cộng sản Việt Nam", "type": "ORG"}}, {{"text": "Bộ Giáo dục", "type": "ORG"}}, {{"text": "Đại học Quốc gia Hà Nội", "type": "ORG"}}]

Đầu vào: Vietnam Airlines vận hành chuyến bay từ Sân bay Nội Bài đến Bệnh viện Bạch Mai.
Đầu ra: [{{"text": "Vietnam Airlines", "type": "ORG"}}, {{"text": "Sân bay Nội Bài", "type": "ORG"}}, {{"text": "Bệnh viện Bạch Mai", "type": "ORG"}}]

Đầu vào: Hà Nội FC giành chức vô địch V.League mùa giải vừa qua.
Đầu ra: [{{"text": "Hà Nội FC", "type": "ORG"}}, {{"text": "V.League", "type": "ORG"}}]

Đầu vào: Trong Chiến tranh Việt Nam, Hiệp định Paris được ký kết tại Pháp.
Đầu ra: [{{"text": "Chiến tranh Việt Nam", "type": "EVT"}}, {{"text": "Hiệp định Paris", "type": "EVT"}}, {{"text": "Pháp", "type": "LOC"}}]

Đầu vào: Samsung giới thiệu Galaxy S24 cùng phần mềm Windows tại sự kiện công nghệ.
Đầu ra: [{{"text": "Samsung", "type": "ORG"}}, {{"text": "Galaxy S24", "type": "PROD"}}, {{"text": "Windows", "type": "PROD"}}]

Đầu vào: Phụ trách là Trần Minh (sinh ngày 03/04/1985). Liên hệ: 090-1234-567, email: minh@example.com. Địa chỉ: 123 Lê Lợi, Quận 1, TP.HCM. CCCD: 079123456789.
Đầu ra: [{{"text": "Trần Minh", "type": "PER"}}, {{"text": "03/04/1985", "type": "DAT"}}, {{"text": "090-1234-567", "type": "PHONE"}}, {{"text": "minh@example.com", "type": "EMAIL"}}, {{"text": "123 Lê Lợi, Quận 1, TP.HCM", "type": "LOC"}}, {{"text": "079123456789", "type": "ID_NUM"}}]

Đầu vào: Vịnh Hạ Long là một kỳ quan thiên nhiên ở tỉnh Quảng Ninh.
Đầu ra: [{{"text": "Vịnh Hạ Long", "type": "LOC"}}, {{"text": "tỉnh Quảng Ninh", "type": "LOC"}}]

Đầu vào: Chùa Một Cột nằm ở quận Ba Đình, Hà Nội.
Đầu ra: [{{"text": "Chùa Một Cột", "type": "ORG"}}, {{"text": "quận Ba Đình", "type": "LOC"}}, {{"text": "Hà Nội", "type": "LOC"}}]

Đầu vào: {sentence}
Đầu ra:"""

# ── System prompt for OpenAI (chat format) ────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia nhận dạng thực thể có tên (NER) tiếng Việt theo hệ thống 10 loại chuẩn (nhãn tiếng Anh). Trích xuất thực thể từ các câu và trả về dưới dạng JSON.

Loại thực thể:
- PER: Tên người (không bao gồm chức danh)
- LOC: **Chỉ vị trí địa lý** (quốc gia/tỉnh/sông/núi/đảo/địa chỉ). Cơ sở nhân tạo thuộc ORG
- ORG: Tổ chức và mọi cơ sở nhân tạo (công ty/tập đoàn/ngân hàng/hãng/đài/trường đại học (toàn bộ)/đảng/bộ/quân đội/CLB thể thao/hiệp hội/tổ chức quốc tế + ga/sân bay/bệnh viện/trường PT/bảo tàng/thư viện/chùa/nhà thờ/sân vận động)
- PROD: Sản phẩm hữu hình, tác phẩm sáng tạo (âm nhạc/phim/sách/anime/game/TV), phần mềm đóng gói, phương tiện/vũ khí có tên model. **KHÔNG gồm dịch vụ/SaaS/tiêu chuẩn kỹ thuật/giải thưởng** (→ phi-thực-thể)
- EVT: Sự kiện một lần — chiến tranh, hiệp ước, đại hội, giải đấu lớn. Luật/pháp lệnh/nghị định và quá trình nhiều năm (Chiến tranh Lạnh, Minh Trị Duy tân)/thời đại → KHÔNG PHẢI thực thể
- EMAIL: Địa chỉ email đầy đủ (local@domain.TLD)
- PHONE: Số điện thoại (định dạng Việt Nam hoặc quốc tế)
- DAT: Ngày tháng tổng quát (năm/tháng/ngày/khoảng thời gian/thời đại)
- ID_NUM: Số định danh cá nhân (CCCD/CMND/mã số thuế)
- CREDIT_CARD: Số thẻ tín dụng (13~19 chữ số)

Quy tắc phân loại (áp dụng khi phân vân):
1. Tổ chức/pháp nhân/công quyền/cơ sở nhân tạo → ORG
2. Vị trí địa lý đơn thuần → LOC
3. Sản phẩm/tác phẩm/chương trình → PROD
4. Sự kiện/chiến tranh/hiệp ước → EVT

Quy tắc chung:
- Giữ nguyên dấu tiếng Việt, trích xuất chính xác văn bản gốc
- Thực thể nhiều từ giữ nguyên thành một thực thể
- Loại trừ chức danh đứng trước tên
- Trả về JSON object với chỉ số câu là key. Không có thực thể → mảng rỗng
- Chỉ JSON, không giải thích thêm
- Nhãn dùng ký hiệu tiếng Anh"""

USER_PROMPT_TEMPLATE = """Trích xuất thực thể từ các câu dưới đây. Nhãn dùng ký hiệu tiếng Anh.

Đầu vào:
{sentences}

Định dạng đầu ra ví dụ:
{{"0": [{{"text": "Nguyễn Văn A", "type": "PER"}}, {{"text": "Hà Nội", "type": "LOC"}}], "1": [{{"text": "Samsung", "type": "ORG"}}, {{"text": "Galaxy S24", "type": "PROD"}}], "2": []}}

Đầu ra:"""
