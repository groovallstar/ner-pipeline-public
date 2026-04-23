# Canonical Entity Schema (augmenters)

`src/augmenters/` 하위에서 사용하는 통합 엔티티 라벨 공간. JA Stockmark 8종 +
합성 PII 6종 = 14종을 **OntoNotes 관용 영문 축약**으로 표기한다.

- 소스: `src/augmenters/{wikiann_vi,pii}/`
- 파생 데이터: `data/stockmark/`, `data/wikiann_vi/`, `data/pii/`
- 도입 이슈: #13 (2026-04-22)

## 배경

이전에는 JA Stockmark 데이터셋의 일본어 원문 라벨(`人名`, `法人名`, `地名` 등)을
그대로 사용했다. 이 접근은 프롬프트-골드 데이터 간 정규화를 생략할 수 있는 이점이
있었지만, 다음 문제가 있었다:

1. **다국어 라벨 공간 불균일** — JA는 일본어, KO는 영문 약어(`PS/LC/OG`), VI
   WikiANN은 영문(`PER/LOC/ORG`)으로 혼재
2. **리포트 가독성 저하** — 일본어 라벨이 표·본문에 혼입되어 영문 서사와 섞임
3. **라벨러 재사용성 제약** — 동일 타입을 다른 언어에서 재활용할 때 라벨 변환기
   필요

본 스키마는 `augmenters/` 범위에 한정해 canonical을 영문 축약으로 통일한다.
`labelers/`·`llm_eval/`·`classifier/`는 별도로 정리한다.

## 14종 canonical 정의

### 8종 (Stockmark 정렬)

| Label | 의미 | 정의 | 예시 |
|---|---|---|---|
| `PER` | 인물 | 사람 이름(풀네임·성·이름·별명·예명). 직함·호칭은 제외 | 織田信長, Nguyễn Xuân Phúc, Bác Hồ |
| `CORP` | 법인·영리기업 | 기업·회사·철도회사·방송사·공기업 (영리 법인격을 가진 조직) | トヨタ自動車, Samsung, Vietnam Airlines, VTV |
| `LOC` | 지명 | 국가·행정구역·자연지명(산·강·바다·섬 등). 물리적 건축물 제외 | 東京, Việt Nam, 富士山, Sông Hồng |
| `FAC` | 시설 | 개별 건축물·역·공항·점포·병원·학교 건물·종교시설 | 東京タワー, 東京駅, Chùa Một Cột, Sân bay Nội Bài |
| `PROD` | 제품·작품 | 상품·서비스·소프트웨어·작품명·번역판·방송 프로그램 | iPhone, Windows, プリウス, VinFast VF8 |
| `EVT` | 행사·사건 | 대회·전쟁·조약·일회성 공식 행사 (지속적 리그·팀은 ORG) | オリンピック, Chiến tranh Việt Nam, SEA Games |
| `POL` | 정치·행정 조직 | 정당·정부기관·국제기관·군대·재판소·의회 | 自民党, 国連, 財務省, Đảng Cộng sản Việt Nam |
| `ORG` | 기타 조직 | 대학·스포츠리그·팀·학회·협회·NGO 등 영리·정치 외 조직 | 早稲田大学, FCバルセロナ, V.League |

WikiANN 3종(`PER/LOC/ORG`) 축소 비교 시: `CORP ∪ POL ∪ ORG → ORG`.

### PII 6종

| Label | 의미 | 정의 | 예시 |
|---|---|---|---|
| `EMAIL` | 이메일 주소 | `local@domain.TLD` 완전 형식 (TLD 포함) | taro@example.com |
| `PHONE` | 전화번호 | 국내·국제 전화번호 (하이픈·공백·국가번호 허용) | 090-1234-5678, +81 80 1234 5678 |
| `ADDRESS` | 개인 PII 주소 | 번지·건물명·층수까지 포함된 물리적 주소. 도·시 단독 표기는 `LOC` | 東京都千代田区1丁目2-3 ビル7F |
| `DOB` | 생년월일 | 출생일 맥락(`生`, `出生`, `생년월일:` 등) 명시 시. 일반 연호·사건일은 제외 | 1985年4月3日, 1990-01-15 |
| `ID_NUM` | 개인 식별 번호 | 주민등록번호·마이넘버·사원번호 등 개인 식별 숫자열 | 284257239645, 123-45-6789 |
| `CREDIT_CARD` | 신용카드 번호 | 13~19자리 카드 번호 (공백·하이픈 구분자 허용) | 4065 0551 3022 4539 |

## 경계 규칙 (요약)

상세 경계 규칙과 모호 사례는 언어별 스펙을 참조:
- JA: `docs/manual/data/japanese-ner.md`
- VI: `docs/manual/data/vietnamese-ner-8types.md`

핵심 규칙만 재수록:

| 규칙 | 판정 | 예시 |
|---|---|---|
| 철도회사·방송사 | `CORP` (시설 아님) | `JR東日本`, `VTV` → `CORP` |
| 병원·학교 건물·우체국 | `FAC` | `東京駅`, `藤岡高校`, `神栖郵便局` → `FAC` |
| 대학 (조직체) | `ORG` | `早稲田大学` → `ORG` |
| 스포츠 리그·팀 | `ORG` (이벤트 아님) | `ラ・リーガ`, `FCバルセロナ` → `ORG` |
| 정부기관·군대 | `POL` | `財務省`, `米軍` → `POL` |
| 행정 지명 복합체 | `LOC` 단일체 | `東京都千代田区`, `Thành phố Hồ Chí Minh` → `LOC` |
| `ADDRESS` vs `LOC` | 번지·건물·층수 포함 시 `ADDRESS`, 아니면 `LOC` | `東京都千代田区` → `LOC`, `東京都千代田区1丁目2-3 ビル7F` → `ADDRESS` |
| `DOB` vs 일반 연도 | `生年月日:`·`生`·`出生` 맥락 필수 | `1985年4月3日生` → `DOB`, `2016年1月29日移籍` → 연도 무시 |

## 원본 → canonical 매핑 (1회 적용)

본 스키마 도입 시 기존 JSONL 파일의 `type`/`label` 필드를 결정론적 치환으로
마이그레이션한다 (LLM 재호출 없음). 매핑 테이블:

| 원본 JA | canonical | 원본 PII | canonical |
|---|---|---|---|
| `人名` | `PER` | `EMAIL` | `EMAIL` (그대로) |
| `法人名` | `CORP` | `PHONE` | `PHONE` (그대로) |
| `地名` | `LOC` | `ADDRESS` | `ADDRESS` (그대로) |
| `施設名` | `FAC` | `DOB` | `DOB` (그대로) |
| `製品名` | `PROD` | `ID_NUMBER` | **`ID_NUM`** |
| `イベント名` | `EVT` | `CREDIT_CARD` | `CREDIT_CARD` (그대로) |
| `政治的組織名` | `POL` | | |
| `その他の組織名` | `ORG` | | |

매핑은 단사(injective)이며 치환 전후 **span 수·오프셋이 완전 동일**하다. 멱등성을
위해 이미 영문인 필드는 skip + warning 로그.

## 적용 범위

- **데이터 파일**:
  - `data/stockmark/{train,test}.jsonl` (Stockmark HF → canonical 덤프)
  - `data/wikiann_vi/vi_wikiann_8type_recall_{train,validation,test}.jsonl`
    (`gold_spans_8type_merged[*].type`만 canonical; 기존 `gold_spans`의
    WikiANN 3종은 그대로 유지)
  - `data/pii/stockmark_pii_1000.jsonl`, `.stats.json`, `.verify.json`

- **코드**:
  - `augmenters/wikiann_vi/prompts.py` few-shot 예시가 canonical 출력
  - `augmenters/wikiann_vi/wikidata_anchor.py`의 `WIKIDATA_TO_CANONICAL`
    (134 Q-ID)
  - `augmenters/pii/` 프롬프트·verifier·generators

- **적용 제외** (후속 처리 예정):
  - `src/labelers/**`, `src/llm_eval/**`, `src/classifier/**`
  - `src/labelers/vi/ner_prompts.py`의 WikiANN 3종(`PER/LOC/ORG`) — 본
    canonical과 철자 겹치지만 스키마 context가 다르므로 별도 관리

## 변경 이력

- 2026-04-22 도입 (이슈 #13). 이전 JA 원문 라벨 → 영문 축약 14종으로 통일.
