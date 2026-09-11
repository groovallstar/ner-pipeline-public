# NER REST API 연동 가이드

외부 시스템에서 NER(개체명 인식) REST API를 호출해 연동하기 위한
문서입니다. 텍스트를 보내면 인물·장소·조직 등 개체명과 그 위치(원문에서
몇 번째 글자인지)를 돌려줍니다. 일본어(`ja`)·한국어(`ko`)·베트남어(`vi`)·
영어(`en`)를 지원합니다.

이 문서는 연동에 필요한 계약(엔드포인트·요청/응답·에러·한도)만 다룹니다.
서버가 제공하는 대화형 API 문서(`GET /docs`, Swagger UI)와 기계용 스키마
(`GET /openapi.json`)를 함께 참고하시면 됩니다.

## 1. 접속 정보

| 항목 | 값 |
|---|---|
| 프로토콜 | HTTP/1.1, JSON |
| Base URL | `http://{host}:{port}` — **실제 주소는 운영팀이 별도 전달** |
| 인증 | **없음** — 별도 인증 헤더 없이 호출 |
| Content-Type | `application/json` (요청·응답 공통, UTF-8) |

## 2. 엔드포인트

| 메서드·경로 | 인증 | 설명 |
|---|---|---|
| `POST /v1/ner` | 불필요 | 텍스트에서 개체명 추출(단일·배치) |

## 3. `POST /v1/ner` — 개체명 추출

### 3.1 요청

본문에 `text`(단일) 또는 `texts`(배치) **둘 중 하나**를 넣습니다. 둘 다
넣거나 둘 다 비우면 400입니다.

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `text` | string | 택일 | 단일 텍스트 |
| `texts` | string[] | 택일 | 배치 텍스트(여러 문장을 한 번에) |
| `lang` | string | 선택 | `ja`·`ko`·`vi`·`en` 중 하나. **생략하면 자동 감지**. 그 외 값 → 400 |

### 3.2 응답 — 단일(`text` 요청)

`200 OK`. `lang`은 지정했거나 자동 감지된 언어를 리턴합니다.

```json
{
  "lang": "ja",
  "entities": [
    {"label": "PER", "start_char": 0, "end_char": 4, "text": "織田信長"},
    {"label": "LOC", "start_char": 5, "end_char": 12, "text": "東京都千代田区"}
  ]
}
```

### 3.3 응답 — 배치(`texts` 요청)

`200 OK`. `results`는 **입력 순서와 1:1**이며, 각 항목은 단일 응답과 같은
`{lang, entities}` 구조입니다.

```json
{
  "results": [
    {"lang": "ja", "entities": [{"label": "ORG", "start_char": 0, "end_char": 3, "text": "トヨタ"}]},
    {"lang": "ko", "entities": [{"label": "PER", "start_char": 0, "end_char": 3, "text": "이재용"}]},
    {"lang": "vi", "entities": [{"label": "LOC", "start_char": 0, "end_char": 6, "text": "Hà Nội"}]},
    {"lang": "en", "entities": [{"label": "PER", "start_char": 0, "end_char": 12, "text": "Barack Obama"}]}
  ]
}
```

### 3.4 결과 항목 필드

응답의 `entities`(배치는 각 `results[i].entities`)에 담기는 항목 하나는
원문에서 찾아낸 개체명(인물·장소·조직 등) 하나이며, 아래 필드로
이루어집니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `label` | string | 개체명 종류(§4의 10종 중 하나) |
| `start_char` | int | 개체가 시작하는 글자 위치(0부터, 포함) |
| `end_char` | int | 개체가 끝나는 글자 위치(제외) |
| `text` | string | 그 위치에 해당하는 원문 글자 |

원문을 `start_char`부터 `end_char` 앞까지 자르면 `text`가 됩니다
(예: `東京都千代田区` → `start_char` 5, `end_char` 12).

> **⚠️ 글자(코드포인트) 단위**: 위치는 바이트·UTF-16이 아니라 유니코드 글자
> 기준입니다. UTF-16을 쓰는 언어(자바스크립트·자바)에서 이 값으로 직접 자르면
> 이모지 등 일부 문자에서 어긋날 수 있으니, 응답의 `text`를 그대로 쓰는 것을
> 권합니다.

### 3.5 언어 처리 규칙

`lang`을 명시했는지에 따라 미지원 입력의 처리가 다릅니다.

| 상황 | 결과 |
|---|---|
| `lang`을 `ja`·`ko`·`vi`·`en`으로 명시 | 그 언어 모델로 추출 |
| `lang`을 그 외 값으로 명시 | **400** (`unsupported lang '...'`) |
| `lang` 생략, 판별 문자 있음 | 감지된 언어로 추출, 응답에 리턴 |
| `lang` 생략, 판별 문자 없음 | **200** + `{"lang": "unsupported", "entities": []}` (에러 아님) |

즉 언어를 생략하고 보낸 텍스트가 지원 언어가 아니면 **에러가 아니라
빈 결과**로 돌아옵니다. 배치에서는 지원 언어 항목만 추출하고 미지원 항목은
빈 결과로 두어 순서를 유지합니다(부분 성공).

**자동 감지 방식**은 두 단입니다. 먼저 일본어는 가나(히라가나·가타카나)로,
한국어는 한글(음절 또는 자모)로, 베트남어는 성조 부호(ơ·ư·ả·ạ 등)나 `đ`로
판별합니다. 세 신호는 서로 겹치지 않고, 한 문장에 둘 이상 있으면 일본어 →
한국어 → 베트남어 순으로 먼저 맞은 것이 이깁니다. 셋 중 어느 것도 아니면
둘째 단으로 넘어가, 라틴 글자가 있으면 영어로 보고 없으면 `unsupported`
입니다.

**앞 셋에 안 걸리고 라틴 글자가 있으면 영어로 판별됩니다.** 영어는 라틴 글자에 자기만의
문자가 없어 스크립트로는 가릴 수 없기 때문에, 이 판정은 "영어를 알아봤다" 가
아니라 "다른 셋이 아니고 라틴 글자는 있다" 는 뜻입니다. 그래서 **부호를 뗀
베트남어**(không dấu)와 **로마자로 적은 일본어**도 여기 걸려 영어 모델로
갑니다. 그런 입력은 `lang` 을 직접 지정해야 제 언어로 추출됩니다.

`unsupported` 로 남는 것은 라틴 글자마저 없는 문장입니다. **한자로만 된
문장**(인명·주소·헤드라인 등, 예: `東京都千代田区`)이 대표적인데, 일본어와
한국어가 한자를 함께 쓰므로 어느 쪽인지 가릴 수 없고 라틴 글자도 없기
때문입니다. `"lang": "ja"` 또는 `"lang": "ko"` 를 명시해야 추출됩니다.

**언어를 아는 경우 `lang`을 항상 명시하면** 이런 감지 한계를 겪지 않습니다.

**서버에 영어 모델이 배포돼 있어야 합니다.** 자동 감지가 라틴 텍스트를 영어로
보내므로, 영어 모델이 없는 서버에서는 그 요청이 200 이 아니라 503 이고 라틴
문장이 섞인 배치는 통째로 503 입니다.

## 4. 개체명 종류(label)

`label`은 아래 10종 중 하나입니다.

| 라벨 | 설명 | 예시 |
|---|---|---|
| `PER` | 인물 — 사람 이름(풀네임·성·이름·별명). 직함·호칭은 제외 | `織田信長` / `Hồ Chí Minh` |
| `LOC` | 지명·주소 — 국가·행정구역·자연지명(산·강·바다)·주소. | `東京`·`富士山` / `Hà Nội`·`Vịnh Hạ Long` |
| `ORG` | 조직·시설 — 기업·정부·학교·역·공항·병원·종교시설 등 조직과 모든 인공 시설 | `トヨタ自動車`·`東京駅` / `Vietnam Airlines`·`Sân bay Nội Bài` |
| `PROD` | 제품·작품 — 시판 물품·창작물(영화·음악·서적·게임)·소프트웨어·모델명 제조물 | `iPhone`·`NHKスペシャル` / `Galaxy S24`·`Doraemon` |
| `EVT` | 행사·사건 — 1회성 행사·전쟁·조약·대회·재해·선거 | `関ヶ原の戦い`·`東日本大震災` / `Chiến tranh Việt Nam` |
| `DAT` | 날짜 — 연·월·일·기간·상대날짜·시대·요일. 맥락 무관 **모든 날짜**를 추출 | `1985年4月3日`·`昨日` / `năm 1985` |
| `EMAIL` | 이메일 주소 — `local@domain.TLD` 완전 형식(TLD 포함) | `taro@example.com` |
| `PHONE` | 전화번호 — 국내·국제(하이픈·공백·국가번호 허용) | `090-1234-5678` / `+84 90 1234567` |
| `ID_NUM` | 개인 식별 번호 — 마이넘버·CCCD·주민등록번호·사원번호 등 숫자열 | `284257239645` |
| `CREDIT_CARD` | 신용카드 번호 — 13~19자리(공백·하이픈 구분자 허용) | `4065 0551 3022 4539` |

## 5. 에러 규격

서버가 내는 에러(400·404·405·413·422·429·500·503)는 모두 아래 구조로
반환됩니다.

```json
{"error": {"status": 400, "message": "provide exactly one of 'text' or 'texts'"}}
```

**예외 없이 이 한 가지 모양입니다.** `422`(JSON 스키마 검증 실패)·`500`
(미처리 서버 오류)은 물론, 잘못된 경로·메서드로 보내 라우팅 단계에서 나는
`404`·`405`도 같은 봉투입니다(프레임워크 기본형 `{"detail":[...]}`·평문이
아님). 따라서 클라이언트는 에러 파싱 경로를 하나만 두면 됩니다 — `error` 키를
읽고, 없으면 서버가 아니라 앞단 네트워크 장비의 응답으로 간주하면 됩니다.
`500`은 내부 예외 메시지·트레이스백을 응답에 노출하지 않습니다.

| 상태 | 발생 조건 | 대응 |
|---|---|---|
| **400** | `text`·`texts` 택일 위반, 또는 지원 외 `lang` 명시 | 요청 형식 수정 |
| **404** | 존재하지 않는 경로로 요청 | 엔드포인트 경로 확인(§2) |
| **405** | 경로는 맞으나 메서드 불일치(예: `/v1/ner`에 GET) | `Allow` 헤더의 메서드로 재요청 |
| **413** | 텍스트·배치 크기 한도 초과, 또는 요청 바디 2MB 초과(§6) | 입력을 나눠 재전송 |
| **422** | JSON 스키마 위반(타입 오류 등) | 본문 타입 확인 |
| **429** | 대기 큐가 가득 찼거나 추론 시작을 기다리는 시간이 초과됨 | `Retry-After`에 지정된 초만큼 대기 후 재시도 |
| **500** | 미처리 서버 오류(일시적일 수 있음) | 1회 백오프 후 지속 시 보고 |
| **503** | 요청 언어 모델 미준비 | 잠시 후 재시도(일시적) |

- **429**는 `Retry-After` 헤더에 지정된 초만큼 대기한 뒤 재시도합니다.
  서버는 `Retry-After: 1`을 반환합니다. 기다린 뒤에도 다시 429를 받을 수
  있으므로 재시도 횟수나 전체 대기 시간은 호출자의 정책에 따라 제한합니다.
- **503은 일시적**일 수 있으므로 지수 백오프로 재시도하기를 권장합니다
  (예: 0.5s → 1s → 2s, 상한 후 실패 처리). 503에는 `Retry-After` 헤더를
  제공하지 않습니다. 오류가 지속되면 운영팀에 모델 준비 상태를 확인합니다.
- **400·404·405·413·422는 요청 자체의 문제**이므로 그대로 재시도하지 말고
  요청을 고쳐야 합니다. `405`는 봉투와 함께 `Allow` 헤더가 오므로(예:
  `Allow: POST`) 허용 메서드를 응답에서 바로 확인할 수 있습니다.
- **500**은 서버 내부 오류로, 1회 백오프 재시도 후에도 지속되면 동일 입력을
  운영팀에 보고합니다(내부 메시지는 응답에 노출되지 않음).

### 429 발생 조건과 재시도

| 발생 조건 | 기본 설정 | `error.message` 예시 |
|---|---|---|
| 추론 슬롯이 모두 사용 중이고 대기 큐도 가득 찬 상태에서 요청이 도착함 | 동시 추론 8건, 대기 32건 | `queue full (>= 32 waiting)` |
| 대기 큐에 들어갔지만 제한 시간 안에 추론을 시작하지 못함 | 대기 시간 10초 | `acquire timed out (10.0s)` |

숫자는 배포 설정에 따라 달라질 수 있습니다. 단건과 배치는 모두 HTTP 요청
하나당 슬롯 하나를 사용합니다. 초당 요청 횟수를 제한하는 방식이 아니라 현재
처리·대기 상태에 따라 거절합니다. 자동 감지 결과가 모두 `unsupported`인
요청은 추론 슬롯을 사용하지 않습니다.

두 경우 모두 HTTP 429와 `error.status`·`error.message`를 담은 JSON 본문을
반환합니다. 응답에는 `Retry-After: 1` 헤더가 포함됩니다. 재시도하는
클라이언트는 헤더에 지정된 초만큼 대기합니다. 재시도 여부는 상태 코드로
판단하며, `message` 문구를 분기 조건으로 사용하지 않습니다.

`GET /openapi.json`과 Swagger UI(`GET /docs`)에도 429 응답의 오류 스키마와
`Retry-After` 헤더가 선언되어 있습니다.

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 1

{"error":{"status":429,"message":"queue full (>= 32 waiting)"}}
```

### 상태별 응답 예시

아래 `# →`는 현재 API 계약에 따른 응답 예시입니다. 한도와 오류 메시지의
숫자는 배포 설정에 따라 달라질 수 있습니다.

```bash
# 400 — text·texts 를 둘 다 안 보냄(택일 위반)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' -d '{}'
# → {"error":{"status":400,"message":"provide exactly one of 'text' or 'texts'"}}

# 400 — 지원하지 않는 lang 명시
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' -d '{"text":"a","lang":"th"}'
# → {"error":{"status":400,"message":"unsupported lang 'th'"}}

# 404 — 존재하지 않는 경로
curl -s -X POST 'http://{host}:{port}/v1/nonexistent' \
  -H 'Content-Type: application/json' -d '{"text":"a"}'
# → {"error":{"status":404,"message":"Not Found"}}

# 405 — 경로는 맞으나 메서드 불일치(허용 메서드는 Allow 헤더로 옴)
curl -si -X GET 'http://{host}:{port}/v1/ner' | grep -i '^allow\|error'
# → allow: POST
# → {"error":{"status":405,"message":"Method Not Allowed"}}

# 413 — 텍스트가 max_chars(20,000자) 초과
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' -d '{"text":"<20,001자 이상 텍스트>"}'
# → {"error":{"status":413,"message":"text exceeds max_chars (20000)"}}

# 413 — 요청 바디가 2MB(max_body_bytes) 초과 → 파싱·인증 전 거절
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' --data-binary @big-body.json
# → {"error":{"status":413,"message":"request body exceeds max_body_bytes (2097152)"}}

# 422 — 타입 오류(texts 가 리스트가 아님)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' -d '{"texts":5}'
# → {"error":{"status":422,"message":"request validation failed (1 error(s))"}}

# 429 — 대기 큐가 가득 찼을 때(부하 상황에 따라 발생)
# → Retry-After: 1
# → {"error":{"status":429,"message":"queue full (>= 32 waiting)"}}
# 429 — 제한 시간 안에 추론을 시작하지 못했을 때
# → Retry-After: 1
# → {"error":{"status":429,"message":"acquire timed out (10.0s)"}}

# 503 — 해당 언어 모델이 미로드일 때만 발생(정상 로드 상태에선 200)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' -d '{"text":"東京","lang":"ja"}'
# → {"error":{"status":503,"message":"model for lang 'ja' is not loaded"}}
```

## 6. 요청 크기 한도

과부하 방지를 위해 요청당 크기 한도가 있습니다.

| 한도 | 기본값 | 초과 시 |
|---|---|---|
| 요청 바디 크기 | 2MB | 413 (파싱·인증 전) |
| 텍스트 1건 길이 | 20,000자 | 413 |
| 배치 텍스트 개수(`texts`) | 64개 | 413 |
| 배치 전체 글자 합 | 100,000자 | 413 |

한도를 넘는 입력은 클라이언트에서 문장·문단 단위로 나눠 여러 요청으로
보내면 됩니다. 한 텍스트가 모델 토큰 한도를 넘으면 서버가 문장·공백 경계로
분할하고, 단어 자체도 한도를 넘으면 문자 중간에서 추가 분할합니다. 반환되는
글자 위치는 원문 기준으로 복원되지만, 강제 분할 경계에 걸친 개체의 추출은
보장하지 않습니다.

## 7. 호출 예시 (curl)

아래 `# →`는 실제 호출 결과입니다.

```bash
# 단일 — 언어 자동 감지(일본어)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' \
  -d '{"text":"織田信長は東京都千代田区に住んでいた。"}'
# → {"lang":"ja","entities":[
#      {"label":"PER","start_char":0,"end_char":4,"text":"織田信長"},
#      {"label":"LOC","start_char":5,"end_char":12,"text":"東京都千代田区"}]}

# 단일 — 언어 자동 감지(한국어)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' \
  -d '{"text":"이재용 삼성전자 회장은 지난달 부산에서 열린 국제가전박람회에 참석했다."}'
# → {"lang":"ko","entities":[
#      {"label":"PER","start_char":0,"end_char":3,"text":"이재용"},
#      {"label":"DAT","start_char":13,"end_char":16,"text":"지난달"},
#      {"label":"LOC","start_char":17,"end_char":19,"text":"부산"},
#      {"label":"EVT","start_char":25,"end_char":32,"text":"국제가전박람회"}]}

# 단일 — 언어 자동 감지(영어)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Barack Obama was born in Hawaii in 1961."}'
# → {"lang":"en","entities":[
#      {"label":"PER","start_char":0,"end_char":12,"text":"Barack Obama"},
#      {"label":"LOC","start_char":29,"end_char":35,"text":"Hawaii"}]}

# 배치 — 혼합 언어(텍스트별 감지)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' \
  -d '{"texts":["トヨタは日本の会社です。","이재용 회장은 부산에 갔다.","Hà Nội là thủ đô.","Barack Obama was born in Hawaii."]}'
# → {"results":[
#      {"lang":"ja","entities":[
#        {"label":"ORG","start_char":0,"end_char":3,"text":"トヨタ"},
#        {"label":"LOC","start_char":4,"end_char":6,"text":"日本"}]},
#      {"lang":"ko","entities":[
#        {"label":"PER","start_char":0,"end_char":3,"text":"이재용"},
#        {"label":"LOC","start_char":8,"end_char":10,"text":"부산"}]},
#      {"lang":"vi","entities":[
#        {"label":"LOC","start_char":0,"end_char":6,"text":"Hà Nội"}]},
#      {"lang":"en","entities":[
#        {"label":"PER","start_char":0,"end_char":12,"text":"Barack Obama"},
#        {"label":"LOC","start_char":26,"end_char":32,"text":"Hawaii"}]}]}

# 언어 명시(자동 감지 대신 직접 지정)
curl -s -X POST 'http://{host}:{port}/v1/ner' \
  -H 'Content-Type: application/json' \
  -d '{"text":"アップルは2007年にiPhoneを発売した。","lang":"ja"}'
# → {"lang":"ja","entities":[
#      {"label":"ORG","start_char":0,"end_char":4,"text":"アップル"},
#      {"label":"PROD","start_char":11,"end_char":17,"text":"iPhone"}]}
```
