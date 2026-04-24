# Canonical Entity Schema (augmenters · labelers/ja)

`src/augmenters/` 및 `src/labelers/ja/` 에서 사용하는 통합 엔티티 라벨
공간. NER/PII 구분 없이 **10종 평면 목록**을 **OntoNotes 관용 영문
축약**으로 표기한다.

- 소스: `src/augmenters/{wikiann_vi,pii}/`, `src/labelers/ja/`
- 파생 데이터: `data/stockmark/`, `data/wikiann_vi/`, `data/pii/`

## 변경 이력

- **2026-04-24 (이슈 #21)**: 축소 — 13종 → **10종 평면 목록**.
  - NER 8종 → 5종: `CORP/POL/ORG → ORG`, `FAC → LOC`.
  - NER/PII 구분 섹션 제거 — 모든 라벨을 단일 테이블에 평면 나열.
  - `src/labelers/ja/ner_prompts.py`와 `src/labelers/ja/dataset_loader.py`
    가 canonical 스키마를 직접 사용하도록 확장(이전에는 `augmenters/`
    한정이었음).
- **2026-04-23 (이슈 #17)**: PII 재설계 — `ADDRESS` → `LOC` 흡수,
  `DOB` → `DAT` 개명·의미 확장.
- **2026-04-23 (이슈 #13)**: 일본어 원문 라벨 → OntoNotes 영문 축약 통일
  (augmenters 범위).

## 배경

JA Stockmark HF 원본은 일본어 원문 라벨(`人名`, `法人名`, `地名` 등)을
사용하지만, 프로젝트 통합 라벨 공간은 다음 이유로 영문 canonical을 쓴다:

1. **다국어 라벨 공간 정합** — JA/KO/VI 모두 동일한 영문 3축 구조
   (PER/LOC/ORG) 상위에서 비교 가능
2. **리포트 가독성** — 일본어 라벨이 혼입된 영문 서사 방지
3. **라벨러 재사용성** — 동일 타입을 여러 언어에서 재활용할 때 변환기 불필요
4. **스키마 평면화** — 실제 라벨러·증강기·평가기는 `type` 문자열만 다루므로
   NER vs PII 메타 분류는 문서상 구분에 지나지 않음

## 10종 canonical 정의 (평면 목록)

| Label | 의미 | 정의 | 예시 |
|---|---|---|---|
| `PER` | 인물 | 사람 이름(풀네임·성·이름·별명·예명). 직함·호칭은 제외 | 織田信長, Nguyễn Xuân Phúc |
| `LOC` | 지명·시설 | 국가·행정구역·자연지명(산·강·바다·섬)·**물리적 건축물** — 역·공항·점포·병원·초중고·연구소·도서관·미술관·박물관·종교시설·대학 부속 시설(캠퍼스·植物園·校舎) | 東京, 富士山, 東京駅, 藤岡高校, 旭川キャンパス, Chùa Một Cột |
| `ORG` | 조직 | 기업·회사·철도회사·방송사·공사·공단·**대학 법인 본체**·정당·정부기관(`〜省`/`〜庁`/`〜政府`)·군대·부대·국제기관(`〜条約機構`)·재판소·의회·스포츠리그/팀·학회·협회·NGO·대학 부속 조직(동아리·研究会·학술 `〜学院`) | トヨタ自動車, NHK, 早稲田大学, 自民党, 米軍, FCバルセロナ, NHK交響楽団 |
| `PROD` | 제품·작품 | 상품·서비스·소프트웨어·작품명·번역판·방송 프로그램 | iPhone, Windows, プリウス, VinFast VF8 |
| `EVT` | 행사·사건 | 대회·전쟁·조약·일회성 공식 행사 (지속적 리그·팀은 `ORG`) | オリンピック, Chiến tranh Việt Nam, SEA Games |
| `DAT` | 날짜 | 연도·월·일·기간·상대날짜·시대·요일. 출생일·사건일·이적일·일반 표기 구분 없이 **모든 날짜** | 1985年4月3日, 2016年1月29日, 어제, ngày 15 tháng 3 |
| `EMAIL` | 이메일 주소 | `local@domain.TLD` 완전 형식 (TLD 포함) | taro@example.com |
| `PHONE` | 전화번호 | 국내·국제 전화번호 (하이픈·공백·국가번호 허용) | 090-1234-5678, +81 80 1234 5678 |
| `ID_NUM` | 개인 식별 번호 | 주민등록번호·마이넘버·사원번호 등 개인 식별 숫자열 | 284257239645, 123-45-6789 |
| `CREDIT_CARD` | 신용카드 번호 | 13~19자리 카드 번호 (공백·하이픈 구분자 허용) | 4065 0551 3022 4539 |

WikiANN 3종(`PER/LOC/ORG`)은 상위 3축이 그대로 일치하므로 본 canonical의
부분집합이다.

> `augmenters/pii`의 `generate_address()` 함수가 주소 문자열을 생성하지만,
> 출력 라벨은 `DEFAULT_MERGE_RULES`의 `'ADDRESS': 'LOC'` 무조건 병합
> 규칙에 의해 전부 `LOC`로 변환되어 학습 데이터에 반영된다.

## 경계 규칙

상세 경계 규칙과 모호 사례는 언어별 스펙을 참조:
- JA: `docs/manual/data/japanese-ner.md`
- VI: `docs/manual/data/vietnamese-ner-8types.md`

### 2단 판정 원칙 (모호 케이스)

5종 축소 이후 NER 경계 모호성은 두 축으로 단순화된다:

1. **ORG** — 조직·법인·공권력의 활동 주체 (기업·대학 법인·정당·정부·군대·
   국제기관·스포츠팀·협회·대학 부속 조직 등)
2. **LOC** — 물리적 지명 및 건축물 (국가·도시·자연지명·건물·역·공항·
   학교·병원·대학 부속 시설 등)

두 기준에 모두 해당하는 복합 사례(예: `東京都` = 행정구역+정부 조직)는
**표면형이 지칭하는 구체 대상**으로 결정한다. `東京都`가 행정구역을 지칭하면
`LOC`, 지방정부 조직을 지칭하면 `ORG`.

### 대학의 2단 규칙

축소 전 3단 규칙(CORP/FAC/ORG)에서 **2단**으로 단순화:

| 대상 | 라벨 | 예시 |
|---|---|---|
| 대학 법인 본체·부속 조직(동아리·研究会·학술 `〜学院`) | **ORG** | `早稲田大学`, `東京大学`, `北海道東海大学` |
| 대학 부속 **시설**(캠퍼스·植物園·校舎) | **LOC** | `旭川キャンパス`, `東京大学植物園`, `大学本部校舎` |

### 핵심 경계 케이스

| 규칙 | 판정 | 예시 |
|---|---|---|
| 철도회사·방송사·공사·공단 | `ORG` | `JR東日本`, `NHK`, `日本専売公社` → `ORG` |
| 철도 노선·역 | `LOC` | `JR京都線`, `東京駅` → `LOC` |
| 병원 (운영체와 무관하게 개별 시설) | `LOC` | `セントメアリー病院`, `東京大学病院` → `LOC` |
| 초·중·고등학교 | `LOC` | `藤岡高校`, `千代田小学校` → `LOC` |
| 대학 법인 본체 | `ORG` | `早稲田大学` → `ORG` |
| 정부·군대·국제기관 | `ORG` | `日本政府`, `米軍`, `北大西洋条約機構` → `ORG` |
| 스포츠 리그·팀·교향악단 | `ORG` | `ラ・リーガ`, `FCバルセロナ`, `NHK交響楽団` → `ORG` |
| 행정 지명 복합체 | `LOC` 단일체 | `東京都千代田区`, `Thành phố Hồ Chí Minh` → `LOC` |
| 주소 (번지·건물·층수 포함) | `LOC` | `東京都千代田区1丁目2-3 ビル7F` → `LOC` |
| 모든 날짜 | `DAT` | `1985年4月3日`, `2016年1月29日`, `어제` → `DAT` |

## 원본 → canonical 매핑

### HF JA (일본어) → canonical

| 원본 JA | canonical |
|---|---|
| `人名` | `PER` |
| `地名` | `LOC` |
| `施設名` | `LOC` |
| `法人名` | `ORG` |
| `政治的組織名` | `ORG` |
| `その他の組織名` | `ORG` |
| `製品名` | `PROD` |
| `イベント名` | `EVT` |

구현: `src/labelers/ja/dataset_loader.py`의 `JA_TO_CANONICAL`.

### 8종 canonical → 5종 canonical (이슈 #21 1회성 축소)

| 8종 | 5종 | 비고 |
|---|---|---|
| `PER` | `PER` | identity |
| `LOC` | `LOC` | identity |
| `FAC` | `LOC` | 병합 |
| `CORP` | `ORG` | 병합 |
| `POL` | `ORG` | 병합 |
| `ORG` | `ORG` | identity |
| `PROD` | `PROD` | identity |
| `EVT` | `EVT` | identity |

구현: `src/augmenters/migration/reduce_5type.py`(1회성, 적용 후 삭제).

### PII 라벨 (#17 정리)

| 과거 PII | canonical |
|---|---|
| `EMAIL` | `EMAIL` (identity) |
| `PHONE` | `PHONE` (identity) |
| `ADDRESS` | **`LOC`** (병합 규칙 흡수) |
| `DOB` | **`DAT`** (일반 날짜) |
| `ID_NUMBER` | **`ID_NUM`** |
| `CREDIT_CARD` | `CREDIT_CARD` (identity) |

## 적용 범위

- **데이터 파일** (모두 10종 canonical):
  - `data/stockmark/{train,test}.jsonl` (Stockmark HF → canonical 덤프)
  - `data/wikiann_vi/*.jsonl` (`gold_spans_8type*[*].type`만 canonical.
    원본 `gold_spans`의 WikiANN 3종은 원문 유지)
  - `data/pii/stockmark_pii_1000.jsonl`, `.stats.json`, `.verify.json`

- **코드**:
  - `src/augmenters/wikiann_vi/prompts.py` 5종 few-shot
  - `src/augmenters/wikiann_vi/wikidata_anchor.py`의 `WIKIDATA_TO_CANONICAL`
    (134 Q-ID → 5종 값)
  - `src/augmenters/pii/` 프롬프트·verifier·generators (10종 평면)
  - `src/labelers/ja/ner_prompts.py` 10종 평면 (영문 라벨 출력)
  - `src/labelers/ja/dataset_loader.py` HF 원본 JA → canonical 5종 매핑

- **적용 제외**:
  - `src/llm_eval/**`, `src/classifier/**` — 후속 정리 예정
  - `src/labelers/vi/ner_prompts.py`의 WikiANN 3종(`PER/LOC/ORG`) —
    canonical과 철자 겹치지만 스키마 context가 다르므로 별도 관리
  - `src/labelers/ko/**`, `docs/manual/data/korean-*.md` — 국문 데이터는
    별도 KLUE 스키마(`PS/LC/OG/DT/TI/QT`) 사용

## 평가 시 주의 (이슈 #21)

10종 평면 스키마로 **원본(PII 미주입) 데이터**를 평가할 때:

- 원본 Stockmark gold에는 `DAT` / 4 PII 라벨이 없다. 프롬프트는 모든
  날짜를 `DAT`로 추출하도록 지시하므로 Wikipedia 텍스트의 연도·기간은
  **FP(false positive)** 로 집계된다.
- 이는 설계 사항: `DAT`는 "맥락 무관 모든 날짜" 라벨이므로 gold가 비어도
  예측을 유지한다.
- 해석 시 주의: overall span F1은 **DAT 예측 건수만큼 precision 하락**한다.
  DAT 자체의 label-level precision은 gold가 0이므로 항상 0.
- 대책: PII-주입 평가셋(`data/pii/stockmark_pii_1000.jsonl`)에서는 gold에
  DAT/PII가 포함되므로 정상 평가가 가능하다. 원본 Stockmark에서 NER 5종
  성능만 보려면 per-entity 테이블에서 `DAT`·PII 4종 행을 제외하고 읽는다.
