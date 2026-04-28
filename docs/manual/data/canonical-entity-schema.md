# Canonical Entity Schema (augmenters · labelers/ja)

`src/augmenters/` 및 `src/labelers/ja/` 에서 사용하는 통합 엔티티 라벨
공간. NER/PII 구분 없이 **10종 평면 목록**을 **OntoNotes 관용 영문
축약**으로 표기한다.

- 소스: `src/augmenters/{wikiann_vi,pii}/`, `src/labelers/ja/`
- 파생 데이터: `data/stockmark/`, `data/wikiann_vi/`, `data/pii/`

## 변경 이력

- **2026-04-28 (이슈 #27)**: LOC/ORG 경계 재정의.
  - `LOC` = **지리적 위치만** — 국가·행정구역·자연지명(산·강·바다·
    섬·호수)·주소(번지·건물·층수 포함). **인공 시설은 모두 제외.**
  - `ORG` = **모든 인공 시설·조직 통합** — 기존 ORG(기업·정당·정부·
    군대·국제기관·스포츠팀·학회 등) + **역·공항·병원·초·중·
    고등학교·대학(본체·캠퍼스·부속 시설 모두)·점포·박물관·
    미술관·도서관·연구소·종교시설**.
  - `大学` 2단 규칙(법인 본체 ORG / 부속 시설 LOC) 폐기 — 본체·
    캠퍼스·시설 모두 ORG.
  - HF JA 매핑 변경: `施設名 LOC → ORG`.
  - WikiANN-vi Wikidata 매핑: 시설 Q-ID(역·공항·병원·학교 등)는
    LOC → ORG로 재배치.
- **2026-04-24 (이슈 #21)**: 축소 — 13종 → **10종 평면 목록**.
  - NER 8종 → 5종: `CORP/POL/ORG → ORG`, `FAC → LOC`
    (이후 #27에서 `FAC`가 차지하던 시설은 ORG로 재배치).
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
| `LOC` | 지명·주소 | 국가·행정구역·자연지명(산·강·바다·섬·호수)·주소(번지·건물·층수 포함). **인공 시설은 ORG로 분류** | 東京, 富士山, 北海道, 東京都千代田区1丁目2-3, Việt Nam, Sông Hồng |
| `ORG` | 조직·시설 | 기업·정당·정부기관(`〜省`/`〜庁`/`〜政府`)·군대·부대·국제기관·재판소·의회·스포츠리그/팀·학회·협회·NGO + **철도회사·방송사·공사·공단·역·공항·병원·초·중·고등학교·대학(본체·캠퍼스·부속 시설 모두)·점포·박물관·미술관·도서관·연구소·종교시설·교향악단** | トヨタ自動車, NHK, 早稲田大学, 旭川キャンパス, 自民党, 米軍, FCバルセロナ, NHK交響楽団, 東京駅, セントメアリー病院, 藤岡高校, 明治神宮 |
| `PROD` | 제품·작품 | 상품·서비스·소프트웨어·작품명·번역판·방송 프로그램 | iPhone, Windows, プリウス, VinFast VF8 |
| `EVT` | 행사·사건 | 대회·전쟁·조약·일회성 공식 행사 (지속적 리그·팀은 `ORG`) | オリンピック, Chiến tranh Việt Nam, SEA Games |
| `DAT` | 날짜 | 연도·월·일·기간·상대날짜·시대·요일. 출생일·사건일·이적일·일반 표기 구분 없이 **모든 날짜** | 1985年4月3日, 2016年1月29日, 어제, ngày 15 tháng 3 |
| `EMAIL` | 이메일 주소 | `local@domain.TLD` 완전 형식 (TLD 포함) | taro@example.com |
| `PHONE` | 전화번호 | 국내·국제 전화번호 (하이픈·공백·국가번호 허용) | 090-1234-5678, +81 80 1234 5678 |
| `ID_NUM` | 개인 식별 번호 | 주민등록번호·마이넘버·사원번호 등 개인 식별 숫자열 | 284257239645, 123-45-6789 |
| `CREDIT_CARD` | 신용카드 번호 | 13~19자리 카드 번호 (공백·하이픈 구분자 허용) | 4065 0551 3022 4539 |

WikiANN 3종(`PER/LOC/ORG`)은 상위 3축 표기는 일치하나, **WikiANN 원본은
시설을 LOC로 라벨링**하므로 본 canonical(시설=ORG)과 직접 비교 시 시설
엔티티가 LOC↔ORG 간 어긋난다. 비교 평가에서는 시설 매핑 차이를 별도
처리한다.

> `augmenters/pii`의 `generate_address()` 함수가 주소 문자열을 생성하지만,
> 출력 라벨은 `DEFAULT_MERGE_RULES`의 `'ADDRESS': 'LOC'` 무조건 병합
> 규칙에 의해 전부 `LOC`로 변환되어 학습 데이터에 반영된다.

## 경계 규칙

상세 경계 규칙과 모호 사례는 언어별 스펙을 참조:
- JA: `docs/manual/data/japanese-ner.md`
- VI: `docs/manual/data/vietnamese-ner-8types.md`

### 2단 판정 원칙

NER 5종 경계는 다음 두 축으로 단순화한다:

1. **LOC** — **지리적 위치 자체**: 국가·행정구역·자연지명·주소(번지·
   건물·층수 포함). 사람이 만든 시설(역·공항·건물·교육기관 등)은
   **모두 제외**.
2. **ORG** — **조직·시설 일체**: 기업·법인·정당·정부·군대·국제
   기관·스포츠팀·학회·협회 + 역·공항·병원·학교·대학·점포·
   박물관·도서관·연구소·종교시설 등 모든 인공 시설.

행정 지명 복합체(예: `東京都`)는 행정구역(LOC)과 지방정부 조직(ORG)을
모두 가리킬 수 있다. **표면형이 지칭하는 구체 대상**으로 결정한다.
`東京都`가 행정구역을 지칭하면 LOC, 지방정부 조직을 지칭하면 ORG.

### 핵심 경계 케이스

| 규칙 | 판정 | 예시 |
|---|---|---|
| 철도회사·방송사·공사·공단 | `ORG` | `JR東日本`, `NHK`, `日本専売公社` |
| 철도 노선·역·공항 | `ORG` | `JR京都線`, `東京駅`, `成田空港` |
| 병원 (운영체와 무관하게 개별 시설) | `ORG` | `セントメアリー病院`, `東京大学病院` |
| 초·중·고등학교 | `ORG` | `藤岡高校`, `千代田小学校` |
| 대학 법인 본체 | `ORG` | `早稲田大学`, `東京大学` |
| 대학 부속 시설(캠퍼스·植物園·校舎) | `ORG` | `旭川キャンパス`, `東京大学植物園`, `大学本部校舎` |
| 정부·군대·국제기관 | `ORG` | `日本政府`, `米軍`, `北大西洋条約機構` |
| 스포츠 리그·팀·교향악단 | `ORG` | `ラ・リーガ`, `FCバルセロナ`, `NHK交響楽団` |
| 점포·박물관·미술관·도서관·종교시설 | `ORG` | `東京国立博物館`, `国会図書館`, `明治神宮`, `清水寺` |
| 국가·행정구역(자체 표기) | `LOC` | `日本`, `東京都`, `北海道`, `Việt Nam` |
| 자연 지명 | `LOC` | `富士山`, `琵琶湖`, `Sông Hồng`, `太平洋` |
| 행정 지명 복합체 | `LOC` 단일체 | `東京都千代田区`, `Thành phố Hồ Chí Minh` |
| 주소 (번지·건물·층수 포함) | `LOC` | `東京都千代田区1丁目2-3 ビル7F` |
| 모든 날짜 | `DAT` | `1985年4月3日`, `2016年1月29日`, `어제` |

## 원본 → canonical 매핑

### HF JA (일본어) → canonical

| 원본 JA | canonical | 비고 |
|---|---|---|
| `人名` | `PER` | |
| `地名` | `LOC` | 자연·행정 지명 |
| `施設名` | **`ORG`** | 이슈 #27 변경 (이전 `LOC`) |
| `法人名` | `ORG` | |
| `政治的組織名` | `ORG` | |
| `その他の組織名` | `ORG` | |
| `製品名` | `PROD` | |
| `イベント名` | `EVT` | |

구현: 1회성 재덤프 스크립트(`/tmp/regen_stockmark.py`, 이슈 #27 시점,
작업 후 폐기). `src/labelers/ja/dataset_loader.py`는 canonical 변환된
JSONL 덤프(`data/stockmark/{train,test}.jsonl`)를 그대로 읽기만 한다.

재덤프는 다음 2단으로 진행:

1. **결정적 source → canonical 매핑**: 위 표 그대로 적용. 일본어 한자
   substring 기반 시설 키워드 후처리는 `香港`(港)·`茨城`(城)·
   `吉祥寺`(寺)·`モスクワ`(モスク)·`ノックスビル`(ビル) 등 false
   positive 가 커서 채택하지 않는다.
2. **LLM 검증** (gemma-4-31B-it-AWQ-8bit): 1단계 결과의 LOC/ORG 라벨
   엔티티를 LLM 으로 재분류, 불일치 케이스 통계·샘플 보고. 명백한
   라벨 오류만 사용자 검토 후 수정 적용.

### 8종 canonical → 5종 canonical (이슈 #21 1회성 축소 + #27 재배치)

| 8종 | 5종 (현재) | 변경 경로 |
|---|---|---|
| `PER` | `PER` | identity |
| `LOC` | `LOC` | identity (지명·주소만) |
| `FAC` | **`ORG`** | #21에서 LOC로 1차 병합 → #27에서 ORG로 재배치 |
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

- **데이터 파일** (모두 10종 canonical, #27 LOC/ORG 경계 적용):
  - `data/stockmark/{train,test}.jsonl` (Stockmark HF → canonical 덤프)
  - `data/wikiann_vi/*.jsonl` (`gold_spans_8type*[*].type`만 canonical.
    원본 `gold_spans`의 WikiANN 3종은 원문 유지 — 시설 매핑은 별도 처리)
  - `data/pii/stockmark_pii_{train,test}.jsonl`, `.stats.json`, `.verify.json`

- **코드**:
  - `src/augmenters/wikiann_vi/prompts.py` 5종 few-shot
  - `src/augmenters/wikiann_vi/wikidata_anchor.py`의 `WIKIDATA_TO_CANONICAL`
    (시설 Q-ID는 ORG로 재매핑 — #27)
  - `src/augmenters/pii/` 프롬프트·verifier·generators (10종 평면)
  - `src/labelers/ja/ner_prompts.py` 10종 평면 (영문 라벨 출력, 시설=ORG)
  - `src/labelers/ja/dataset_loader.py` HF 원본 JA → canonical 매핑
  - `src/labelers/vi/ner_prompts.py` 시설=ORG 정의 동기화

- **적용 제외**:
  - `src/llm_eval/**`, `src/classifier/**` — 후속 정리 예정
  - `src/labelers/vi/ner_prompts.py`의 WikiANN 3종(`PER/LOC/ORG`) —
    canonical과 철자 겹치지만 스키마 context가 다르므로 별도 관리
  - `src/labelers/ko/**`, `docs/manual/data/korean-*.md` — 국문 데이터는
    별도 KLUE 스키마(`PS/LC/OG/DT/TI/QT`) 사용

## 평가 시 주의

### DAT/PII FP (이슈 #21)

10종 평면 스키마로 **원본(PII 미주입) 데이터**를 평가할 때:

- 원본 Stockmark gold에는 `DAT` / 4 PII 라벨이 없다. 프롬프트는 모든
  날짜를 `DAT`로 추출하도록 지시하므로 Wikipedia 텍스트의 연도·기간은
  **FP(false positive)** 로 집계된다.
- 이는 설계 사항: `DAT`는 "맥락 무관 모든 날짜" 라벨이므로 gold가 비어도
  예측을 유지한다.
- 해석 시 주의: overall span F1은 **DAT 예측 건수만큼 precision 하락**한다.
  DAT 자체의 label-level precision은 gold가 0이므로 항상 0.
- 대책: PII-주입 평가셋(`data/pii/stockmark_pii_test.jsonl`)에서는 gold에
  DAT/PII가 포함되므로 정상 평가가 가능하다. 원본 Stockmark에서 NER 5종
  성능만 보려면 per-entity 테이블에서 `DAT`·PII 4종 행을 제외하고 읽는다.

### LOC/ORG 경계 재정의 영향 (이슈 #27)

- Stockmark gold(`施設名` → ORG)와 silver는 동일 매핑이지만, **이전 측정
  결과(#27 이전)와는 LOC/ORG 분포가 달라 직접 비교할 수 없다**. 시설
  엔티티가 LOC에서 ORG로 이동한 만큼 두 라벨의 P/R/F1이 모두 변동한다.
- WikiANN-vi 원본 `gold_spans`(WikiANN 3종)와 silver `gold_spans_8type*`
  (canonical)을 직접 비교하면 시설 엔티티 LOC↔ORG 미스매치가 발생한다.
  비교 시 시설 Q-ID 매핑 차이를 명시하고 별도 집계한다.
