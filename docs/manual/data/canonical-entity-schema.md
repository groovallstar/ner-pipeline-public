# Canonical Entity Schema (JA · VI)

`src/ner/labelers/ja/`·`src/ner/augmenters/pii/` (Stockmark) 와 `src/ner/labelers/vi/`·
`src/ner/augmenters/wikiann_vi/` (WikiANN) 가 공유하는 통합 엔티티 라벨 공간.
NER/PII 구분 없이 **10종 평면 목록**을 OntoNotes 관용 영문 축약으로 표기한다.

- 적용 코드: `src/ner/labelers/{ja,vi}/`, `src/ner/augmenters/pii/`,
  `src/ner/augmenters/wikiann_vi/`
- 적용 데이터: `data/stockmark/` (JA NER 5종 + PII 주입 10종),
  `data/wikiann_vi/` (VI 재라벨 NER 5종 + PII 주입 10종)
- KO: canonical **10종 평면 완성**. NER 5종(`PER/LOC/ORG/PROD/EVT`) +
  `DAT`는 KLUE 유래(라벨러 `src/ner/labelers/ko/**`, `PROD/EVT`는 LLM
  재라벨 증분). PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`)은 합성 PII를
  **llm 자연삽입**으로 증분(`data/klue/pii_all.jsonl`). 분류기 미완(후속 이슈).

## 변경 이력

- **2026-06-16 (이슈 #115)**: KO canonical **10종 완성** — PII 4종
  (`EMAIL/PHONE/ID_NUM/CREDIT_CARD`)을 합성 PII **llm 자연삽입**으로 증분
  (`augmenters/pii --lang ko --mode llm`, gemma-4-31B). 접두 패턴 편향을
  피하려 문중 자연삽입(경직 접두 직후 PII 0%), `extract_spans` string-match로
  offset 정확(0 mismatch). verify 미사용(사람 KLUE gold 보존). gold
  `data/klue/pii_all.jsonl` 25989행, PII 각 ~8.3k, 원문 NER 보존 99.3%.
  주민등록번호는 체크섬 무효로 실유효 번호 비생성. ko 생성기 `generators/ko.py`.
- **2026-06-16 (이슈 #111)**: KO NER 5종 완성 — KLUE 문장 LLM 재라벨로
  `PROD/EVT` 증분(`src/ner/labelers/ko/ko_prod_evt_relabel.py`, gemma-4-31B). canonical
  §3.1~§3.3 회색지대 기준을 ko 프롬프트에 이식(운영리그=`ORG` vs 특정
  연도판=`EVT`, 법령·시대구분·추상 쟁점 비-entity). 병합은
  containment-replace(outermost flat, overlap 0) — flat BIO 분류기 제약.
  gold `data/klue/origin.jsonl` 5종 승격(`PROD` 3313·`EVT` 1191 추가,
  KLUE 1192 sub-span outermost 교체).

- **2026-06-15 (이슈 #109)**: KO 라벨러·seed gold를 canonical 명칭으로
  정렬. `PS→PER, LC→LOC, OG→ORG, DT→DAT` 개명, KLUE `TI/QT` 드롭.
  `tag_aligner` ko/ja normalize 맵 분리(ja 불변). KLUE 유래 4종 seed
  gold(`src/ner/labelers/ko/klue_to_canonical_gold.py` → `data/ko_klue/`) 생성.
  `PROD/EVT`·PII 6종은 후속 증분.
- **2026-06-09 (이슈 #85)**: JA EVT scope 재정의 — 회색지대
  규정(§3.3). **EVT 포함 명문화**: 자연재해·대형사고·named 경제/정치
  위기·주기적 복합 행사명사(선거·투표·`国勢調査`)·경기/컵/토너먼트/
  `歌合戦`/`甲子園`(운영리그는 `ORG`). **EVT 제외(비-entity)**: 추상 논쟁
  topic(`〜問題`)·세기 시대(`大航海時代`)·다년 지속 process·state(`冷戦`·
  `宗教改革`·`産業革命`·`ホロコースト`)·서비스/코드네임. support EVT
  987→992·ORG 5310→5303·PROD 943→942 — 동시 변동으로 ΔEVT 격리 불가,
  성능 레버 아닌 정의·일관성 보정. **#84 EVT −2.4pp 규명**: baseline
  0.8593(`loccensus_refonly`, support 968) vs prodclean 0.8353 은
  cross-baseline — #84 격리 효과는 prodschema→prodclean −0.13pp(seed42,
  노이즈)라 multi-seed 폐기.
- **2026-06-08 (이슈 #84)**: JA PROD 정의 positive 재규정 — 어노테이션
  대상만 명시(상품·작품·패키지SW·형식명 제조물). **서비스·온라인 운영물
  (웹·SaaS·MMO·통신·금융·멤버십·검정)·기술 표준/포맷/프로토콜/라이선스는
  PROD 아님.** 오라벨 정리: 회사→`ORG`, 계획·전시·시리즈→`EVT`,
  상·훈장·규격·생물통칭→비-entity. 순 PROD 1042→943(이탈 99 = 재라벨 20
  + 제거 79). prodclean 10-fold: PROD F1 0.8090→0.8168 / overall 0.9222→
  0.9251(baseline 0.9256 회복) — support 변동으로 비교 불가, 성능 레버
  아닌 일관성 보정. EVT −2.4pp 는 #85 에서 cross-baseline 으로 규명(§3.3). §3.2 추가.
- **2026-06-08 (이슈 #84)**: JA PROD 회색지대 4범주 규정 (#73 BORDERLINE
  47 의 구조적 gold 비일관 정정). 제조 명명 탈것·무기·함정·항공기 =
  `PROD` / 법령·법안·규정·칙유 = 비-entity / 전시·투어·프리미어 = `EVT` /
  named 프로젝트·계획·전략 = `EVT`. §3.1 추가. (PROD = gold 천장 재확인
  후 일관성 보정이지 F1 레버 아님 — #73·#45·재감사 일치. 기존 gold 소급
  relabel 은 로컬 한정·미커밋, 코퍼스 재생성 시 규칙 반영.)
- **2026-06-08 (이슈 #82)**: JA attributive nationality 규정 추가 (LOC
  GAP 정정). 국적·언어·계통 복합어(지명+人/語/系/製)와 국적·소속 수식
  명사구(`日本企業`·`日本社会`)는 LOC 아님 — 장소 *자체* 를 지칭하는
  단독·조사 결합(`日本では`)만 LOC. `[지명]+代表→ORG`(#51)와 일관.
  §2.3 추가. (gold 50/50 주석 모순의 근본 원인 규명 → LOC = gold 천장
  확정. 기존 gold 소급 정리는 미적용 — 코퍼스 재생성 시 규칙 반영.)
- **2026-05-07 (이슈 #51)**: JA 모호 케이스 4영역 명시화 — 경기장·
  서킷·도시공원, 城·城跡 史跡, `[지명]+代表` 스포츠 대표팀,
  부동산·산업 단지, 약어 단독은 회사명 우선(ORG vs PROD). §2.3 / §3 /
  §5.1 추가.
- **2026-04-30**: VI 섹션 통합. 베트남어 WikiANN → canonical 매핑·
  LOC/ORG 경계 사례·PROD/EVT 분기 규칙을 본 문서에 추가. 파일명을
  다국어 SoT 로 명시화 (JA 한정 표기 폐기).
- **2026-04-28 (이슈 #27)**: JA LOC/ORG 경계 재정의.
  - `LOC` = **지리적 위치만** — 국가·행정구역·자연지명(산·강·바다·
    섬·호수)·주소(번지·건물·층수 포함). **인공 시설은 모두 제외.**
  - `ORG` = **모든 인공 시설·조직 통합** — 기존 ORG(기업·정당·정부·
    군대·국제기관·스포츠팀·학회 등) + **역·공항·병원·초·중·
    고등학교·대학(본체·캠퍼스·부속 시설 모두)·점포·박물관·
    미술관·도서관·연구소·종교시설**.
  - `大学` 2단 규칙(법인 본체 ORG / 부속 시설 LOC) 폐기 — 본체·
    캠퍼스·시설 모두 ORG.
  - HF JA 매핑 변경: `施設名 LOC → ORG`.
- **2026-04-24 (이슈 #21)**: 축소 — 13종 → **10종 평면 목록**.
  - NER 8종 → 5종: `CORP/POL/ORG → ORG`, `FAC → LOC`
    (이후 #27에서 `FAC`가 차지하던 시설은 ORG로 재배치).
  - NER/PII 구분 섹션 제거 — 모든 라벨을 단일 테이블에 평면 나열.
  - `src/ner/labelers/ja/ner_prompts.py`와 `src/ner/labelers/ja/dataset_loader.py`
    가 canonical 스키마를 직접 사용하도록 확장.
- **2026-04-23 (이슈 #17)**: PII 재설계 — `ADDRESS` → `LOC` 흡수,
  `DOB` → `DAT` 개명·의미 확장.
- **2026-04-23 (이슈 #13)**: 일본어 원문 라벨 → OntoNotes 영문 축약 통일.

## 배경

JA Stockmark HF 원본은 일본어 원문 라벨(`人名`, `法人名`, `地名` 등)을,
VI WikiANN 은 3종 영문 라벨(`PER/LOC/ORG`)을 사용하지만 프로젝트 통합
라벨 공간은 다음 이유로 영문 canonical 10종 평면 목록을 쓴다:

1. **다국어 라벨 공간 정합** — JA/VI 모두 동일한 영문 3축 구조
   (PER/LOC/ORG) 상위에서 비교 가능
2. **리포트 가독성** — 일본어·베트남어 라벨이 혼입된 영문 서사 방지
3. **라벨러·증강기 코드 재사용성** — 동일 타입을 여러 언어에서 재활용할
   때 변환기 불필요
4. **스키마 평면화** — 실제 라벨러·증강기·평가기는 `type` 문자열만
   다루므로 NER vs PII 메타 분류는 문서상 구분에 지나지 않음

## 1. 10종 canonical 정의 (평면 목록)

| Label | 의미 | 정의 | JA 예시 | VI 예시 |
|---|---|---|---|---|
| `PER` | 인물 | 사람 이름(풀네임·성·이름·별명·예명). 직함·호칭은 제외 | `織田信長`, `田中` | `Nguyễn Xuân Phúc`, `Hồ Chí Minh` |
| `LOC` | 지명·주소 | 국가·행정구역·자연지명(산·강·바다·섬·호수)·주소(번지·건물·층수 포함). **인공 시설은 ORG로 분류** | `東京`, `富士山`, `東京都千代田区1丁目2-3` | `Hà Nội`, `Vịnh Hạ Long`, `Sông Hồng` |
| `ORG` | 조직·시설 | 기업·정당·정부기관·군대·부대·국제기관·재판소·의회·스포츠리그/팀·학회·협회·NGO + **철도회사·방송사·공사·공단·역·공항·병원·초·중·고등학교·대학(본체·캠퍼스·부속 시설 모두)·점포·박물관·미술관·도서관·연구소·종교시설·교향악단** | `トヨタ自動車`, `早稲田大学`, `東京駅`, `セントメアリー病院`, `FCバルセロナ`, `明治神宮` | `Vietnam Airlines`, `Đại học Quốc gia Hà Nội`, `Sân bay Nội Bài`, `Chùa Một Cột`, `Hà Nội FC` |
| `PROD` | 제품·작품 | 시판 물품(기기·하드웨어·식품·약품)·창작 작품(영화·음악·서적·만화·게임·방송 프로그램)·패키지 소프트웨어/OS·형식·모델명 제조물(탈것·무기·함정·항공기·위성·기관차) | `iPhone`, `プリウス`, `NHKスペシャル`, `IV号戦車` | `Galaxy S24`, `Doraemon` |
| `EVT` | 행사·사건 | 1회성 공식 행사·전쟁·조약·대회·자연재해·named 위기·주기적 선거/투표 (지속 운영리그·팀은 `ORG`, 경기·컵은 EVT — 회색지대 §3.3) | `関ヶ原の戦い`, `第45回NHK紅白歌合戦`, `東日本大震災`, `総選挙` | `Chiến tranh Việt Nam`, `UEFA Champions League 2007-08` |
| `DAT` | 날짜 | 연도·월·일·기간·상대날짜·시대·요일. 출생일·사건일·이적일·일반 표기 구분 없이 **모든 날짜** | `1985年4月3日`, `昨日`, `平安時代` | `năm 1985`, `hôm qua` |
| `EMAIL` | 이메일 주소 | `local@domain.TLD` 완전 형식 (TLD 포함) | `taro@example.com`, `hanako@yahoo.co.jp` | (공통) |
| `PHONE` | 전화번호 | 국내·국제 전화번호 (하이픈·공백·국가번호 허용) | `090-1234-5678`, `+81 80 1234 5678` | `+84 90 1234567` |
| `ID_NUM` | 개인 식별 번호 | 마이넘버·CCCD·세무번호·사원번호·주민등록번호 등 개인 식별 숫자열 | `284257239645` | `079123456789` |
| `CREDIT_CARD` | 신용카드 번호 | 13~19자리 카드 번호 (공백·하이픈 구분자 허용) | `4065 0551 3022 4539` | (공통) |

> `augmenters/pii` 의 `generate_address()` 함수가 주소 문자열을 생성하지만,
> 출력 라벨은 `DEFAULT_MERGE_RULES` 의 `'ADDRESS': 'LOC'` 무조건 병합
> 규칙에 의해 전부 `LOC` 로 변환되어 학습 데이터에 반영된다.

## 2. LOC vs ORG 경계 규칙

### 2.1 2단 판정 원칙 (공통)

NER 5종 경계는 다음 두 축으로 단순화한다:

1. **LOC** — **지리적 위치 자체**: 국가·행정구역·자연지명·주소(번지·
   건물·층수 포함). 사람이 만든 시설(역·공항·건물·교육기관 등)은
   **모두 제외**.
2. **ORG** — **조직·시설 일체**: 기업·법인·정당·정부·군대·국제
   기관·스포츠팀·학회·협회 + 역·공항·병원·학교·대학·점포·
   박물관·도서관·연구소·종교시설 등 모든 인공 시설.

행정 지명 복합체(예: `東京都`, `Thành phố Hồ Chí Minh`)는 행정구역(LOC)과
지방정부 조직(ORG)을 모두 가리킬 수 있다. **표면형이 지칭하는 구체
대상**으로 결정한다. `東京都` 가 행정구역을 지칭하면 LOC, 지방정부 조직을
지칭하면 ORG.

### 2.2 라벨 우선순위 (모호할 때 위에서 아래로)

| 순위 | 판정 | 적용 대상 |
|---|---|---|
| 1 | **`ORG`** | 조직·법인·공권력·인공 시설 모두 |
| 2 | **`LOC`** | 지리적 위치만 — 국가·행정구역·자연지명·주소 |
| 3 | **`PROD`** | 상품·서비스·작품·방송 프로그램 |
| 4 | **`EVT`** | 1회성 행사·전쟁·조약·대회 |

추가 원칙:
- **대학**: 본체·캠퍼스·부속시설 모두 ORG. `○○大学病院` 도 ORG.
- **행정구역 복합체**: `東京都千代田区` 같은 복합체는 단일 LOC 한 덩어리.
- **복합 조직명**: 리그명+팀명 등은 분리하지 않고 한 덩어리.
- **번지 포함 주소**: `東京都千代田区1丁目2-3 ビル7F` 처럼 번지·층 포함도
  LOC (별도 ADDRESS 타입 없음).

### 2.3 JA 핵심 경계 케이스

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
| 스포츠 대표팀 (`[지명]+代表` 패턴) | `ORG` | `日本代表`, `チェコ代表`, `アメリカ代表` |
| 경기장·서킷·놀이공원·도시공원 | `ORG` | `東京ドーム`, `カタロニア・サーキット`, `バッテリー・パーク`, `ディズニーランド` |
| 城·城跡 (史跡 시설) | `ORG` | `江戸城`, `大阪城`, `寺池城`, `佐沼城` |
| 부동산·산업 단지·아파트 단지 | `ORG` | `ワンハンドレッドヒルズ`, `六本木ヒルズ`, `幕張新都心` |
| 점포·박물관·미술관·도서관·종교시설 | `ORG` | `東京国立博物館`, `国会図書館`, `明治神宮`, `清水寺` |
| 국가·행정구역(자체 표기, 장소 지칭) | `LOC` | `日本では`, `東京都`, `北海道` |
| 국적·언어·계통 복합어 (지명+人/語/系/製) | **비-LOC** | `日本人`, `ドイツ語`, `日系`, `アメリカ製` |
| 국적·소속 수식 명사구 (장소 자체 비지칭) | **비-LOC** | `日本企業`, `日本社会`, `アメリカ大統領` |
| 자연 지명 | `LOC` | `富士山`, `琵琶湖`, `太平洋` |
| 행정 지명 복합체 | `LOC` 단일체 | `東京都千代田区`, `茨城県神栖市` |
| 주소 (번지·건물·층수 포함) | `LOC` | `東京都千代田区1丁目2-3 ビル7F` |
| 모든 날짜 | `DAT` | `1985年4月3日`, `2016年1月29日`, `昨日` |

### 2.4 VI 핵심 경계 케이스

**LOC (자연·행정만)**:
- 국가·도시·tỉnh·huyện·xã
- 자연지명: `Sông X` / `Núi X` / `Biển X` / `Đảo X` / `Vịnh X` / `Hồ X`
- 주소 (số nhà, tầng, tòa nhà 포함)

**ORG (모든 조직·인공시설)**:
- 영리법인: `Công ty X` / `Tập đoàn X` / `Ngân hàng X` / `Hãng X` /
  `Đài truyền hình X`
- 정치·정부·군: `Đảng X` / `Bộ X` / `Cục X` / `Quốc hội` / `Quân đội X` /
  `Tòa án`
- 국제기관: `Liên Hợp Quốc`, `ASEAN`, `WTO`
- 학교 (대학·초중고 모두): `Đại học X` / `Trường THCS·THPT X` /
  `Học viện X`
- 시설: `Bệnh viện X` / `Sân bay X` / `Ga X` / `Cảng X` / `Bảo tàng X` /
  `Thư viện X` / `Chùa X` / `Nhà thờ X` / `Đền X` / `Sân vận động X`
- 스포츠: `Hà Nội FC` / `V.League` / `Câu lạc bộ X` (정기 리그 = ORG,
  특정 연도판 = EVT)

## 3. PROD vs EVT vs ORG 경계 (공통)

- 정기 리그·정기 대회 → `ORG`. **특정 연도판** ("World Cup 2022",
  "UEFA Champions League 2007-08") → `EVT`
- 음악·영화·책·만화·게임·TV 프로그램 → `PROD`
- "Danh sách..." (Wikipedia "List of") 는 entity 아님 → 무시
- 학명 Latin binomial ("Bulbophyllum X") 는 entity 아님 → 무시
- 모델 번호 단독 ("RV522") → `PROD` 아님. 브랜드+모델 결합
  ("Galaxy S24") 만 `PROD`
- 인프라 (철도·지하철 노선): 운영주체 = `ORG`, 경로·노선 자체 = `LOC`
- 약어 단독 (`LKAB`, `IBM`, `JR`, `NHK` 등) → 회사·조직 가능성 우선
  `ORG`. `PROD` 로 분류하려면 명시적 제품 컨텍스트 (브랜드+모델 결합,
  또는 동사 `開発した`/`販売した`/`発売した` 가 직접 수식) 필요
- 동음이의 가타카나 단독 (`タランテラ` = 회사 vs 제품) → 텍스트 내
  동사·수식 문맥으로 판단. `本社`/`会長`/`設立` 등 조직 시그널 →
  `ORG`. `発売`/`新作`/`型番` 등 제품 시그널 → `PROD`. 시그널 부재
  시 `ORG` 우선

### 3.1 PROD 회색지대 4범주 (제조 artifact·법령·전시·프로젝트)

§3 기본 원칙으로 가려지지 않던 PROD 경계 4범주를 명시한다. canonical
미규정이 同 표면 라벨/무라벨 혼재(gold 비일관)를 낳아 PROD 측정 천장의
구조적 원인이었다 (#73 BORDERLINE 47).

| 규칙 | 판정 | 예시 |
|---|---|---|
| 제조 명명 탈것·무기·함정·항공기·우주선·기관차 (형식·급·모델명 결합) | `PROD` | `209型潜水艦`, `IV号戦車`, `四式戦闘機`, `テゲトフ級戦艦`, `イージスシステム`, `スペースシャトルエンデバー`, `ラプターメタンエンジン`, `報国丸級貨客船` |
| 법령·법률·법안·규정·칙유 (법적 문서·제도) | **비-entity** | `証券取引法`, `民法`, `マスキー法`, `平和安全法制`, `軍人勅諭`, `○○法案`, `律令法` |
| 전시·투어·프리미어·1회성 방송 이벤트 | `EVT` | `レモン展`, `UHHA!…TOUR`, `国際プレミア`, `ワールド・エコノ・ムーブ` |
| named 프로젝트·계획·전략·프로그램 | `EVT` | `Eggsプロジェクト`, `Google Summer of Code Project`, `子育て生活もっとハッピー化計画!`, `石油食料交換プログラム`, `高齢者保健福祉推進10ヵ年戦略` |

보조 원칙:
- **generic vs named**: 형식·고유명 없는 일반명사 단독(`新幹線`·
  `スペースシャトル`·`プロジェクト`)은 비-entity. 형식·급·고유명
  결합(`955形新幹線高速試験電車`)만 해당 타입 — §3 "모델번호 단독 vs
  브랜드+모델" 과 동형.
- **番組(작품) vs 방송 이벤트**: 방송 *프로그램* 은 `PROD`(`NHKスペシャル`),
  1회성 방송 행사·방식은 `EVT`. `プレミアム・コレクションDVD-BOX`(제품)은
  `プレミア` 표면이라도 `PROD`.
- **法 표면 ≠ 법령**: `魔法にかけられて`(영화)·`法政大学校歌`(교가)·
  `ホール・エルー法`(공정)은 法 포함이라도 비-법령 → 기존 판정 유지.
- **조약·랭킹은 기존 규정 유지**: 조약 → `EVT`(§1), "List of"·
  베스트리스트 → 비-entity(§3).

### 3.2 PROD 범위 — 서비스·온라인 운영물·기술 표준 제외 (이슈 #84)

PROD 는 **유형 제품·창작 작품·패키지 소프트웨어·형식명 제조물**에
한정한다. 무형 서비스·온라인 운영물·기술 사양은 PROD 가 아니며 아래로
귀속한다.

| 대상 | 귀속 | 예시 |
|---|---|---|
| 통신·SaaS·웹/온라인 매체·금융상품·멤버십·검정·강좌 | 비-entity | `docomo Wi-Fi`, `Skype`, `PRESIDENT Online`, `生活キャッシング`, `ヤマハ音楽能力検定` |
| 온라인 운영형 게임(MMO) | 비-entity | `TERA`, `ウルティマオンライン`, `R2` |
| 기술 표준·규격·포맷·프로토콜·라이선스 | 비-entity | `PDF`, `XML Schema`, `NVMe`, `GNU Free Documentation License` |
| 회사·사업체·레코드 레이블 | `ORG` | `スズキ`, `マッド・ハッター・ラベル` |
| named 계획·프로젝트·전시·레이싱 시리즈 | `EVT` | `イージス計画`, `社寺参詣曼荼羅`, `フェラーリ・チャレンジ` |
| 상·훈장·작위·규격기·생물 통칭 | 비-entity | `銀獅子賞`, `レジオンドヌール勲章`, `国際キログラム原器`, `blue lace flower` |

보조: 番組(정규 방송 프로그램)·패키지 게임은 PROD 유지. 방송 플랫폼·
위성방송 서비스(`モバHO!`)·게임 서비스 플랫폼(`Xbox Live`)은 비-entity.
형식·급·모델명 제조물(`IV号戦車`·`MBSat`)은 PROD(§3.1 A).

### 3.3 EVT 회색지대 규정 — scope 재정의 (이슈 #85)

§1 EVT(1회성 행사·전쟁·조약·대회)로 가려지지 않던 EVT 경계를 규정한다.
canonical 미규정이 同 표면형 라벨/무라벨·타입 혼재(gold 비일관)를 낳던
구조를 닫는다.

**EVT 포함 (positive 명문화):**

| 군집 | 판정 | 예시 |
|---|---|---|
| 자연재해·대형사고 (지진·태풍·분화·침몰·추락·폭발 사고) | `EVT` | `東日本大震災`, `関東大震災`, `洞爺丸台風`, `チャレンジャー号爆発事故` |
| named 경제·금융·정치 위기 | `EVT` | `世界恐慌`, `リーマンショック`, `アジア通貨危機`, `湾岸危機` |
| **주기적 복합 행사명사** (선거·투표·조사 — 고유명 수식 없어도) | `EVT` | `総選挙`, `市長選挙`, `大統領選挙`, `知事選`, `補欠選挙`, `国民投票`, `国勢調査` |
| 경기·컵·토너먼트·챔피언십·`歌合戦` (1회성/연차 경기 이벤트) | `EVT` | `甲子園`, `UEFAチャンピオンズリーグ`, `日本シリーズ`, `NHK紅白歌合戦`, `天皇賞`(경마) |
| dated bounded 운동·사건·학살 | `EVT` | `五四運動`, `三・一独立運動`, `安保闘争`, `南京大虐殺` |

**EVT 아님 (비-entity):**

| 대상 | 귀속 | 예시 |
|---|---|---|
| 추상 논쟁·쟁점 topic (`〜問題`·`〜論争`) | 비-entity | `靖国神社問題`, `慰安婦問題`, `統帥権干犯問題`, `キプロス問題` |
| 세기 단위 시대구분 (`〜時代`) | 비-entity | `大航海時代`, `ジャーヒリーヤ` |
| 다년 지속 process·state·era (수십 년 지속) | 비-entity | `冷戦`, `宗教改革`, `産業革命`, `ホロコースト` |
| 무형 서비스·제품개발 코드네임 (§3.2 연장) | 비-entity | `ポストカプセル郵便2001`, `プロジェクト・ミッドウェー` |

**보조 원칙:**

- **운영리그 vs 경기**: 정기 운영 리그·팀(지속 조직)은 `ORG`(`Jリーグ`·
  `メジャーリーグ`·`プレミアリーグ`). 1회성/연차 경기 이벤트(컵·토너먼트·
  챔피언십·`チャンピオンズリーグ`·`甲子園`)는 `EVT`. §3 "정기 리그=ORG /
  특정 연도판=EVT" 를 *경기성 명칭*까지 확장.
- **bounded vs 지속**: 날짜로 묶인 1회성 사건(전투·학살·운동·재해)은 EVT,
  수십 년 지속 상태·시대·process(`冷戦`·`時代`·`革命`)는 비-entity. 경계
  예: `南京大虐殺`(6주, EVT) vs `ホロコースト`(다년, 비-entity).
- **주기적 복합 행사명사 예외**: §3.1 "generic 단독=비-entity" 의 예외 —
  `総選挙`·`国勢調査` 등 제도화된 주기적 공식 행사를 지시하는 복합 사건
  명사는 고유명 수식이 없어도 EVT(1회성 행사 지시). 단, 단일 generic
  사건명사(`戦争`·`地震`·`祭`·`大会` 단독)는 named/수식 시에만 EVT.
- **賞 명칭**: 경마·모터스포츠 races(`天皇賞`·`グランプリ`)·시상**식**
  (`授賞式`·`Awards`)은 EVT. §3.2 "상·훈장·규격 → 비-entity"는
  훈장·표창·규격기 한정(시상 행사 아님).

> 측정(evtgray 10-fold, seed42): EVT/ORG/PROD 동시 변동(987→992·5310→
> 5303·943→942)으로 ΔEVT 단독 격리 불가 — 성능 레버 아닌 **gold 정의·
> 일관성 보정**. 상세 이슈 #85.

## 4. 원본 → canonical 매핑

### 4.1 JA: HF Stockmark → canonical

| 원본 JA | canonical | 비고 |
|---|---|---|
| `人名` | `PER` | |
| `地名` | `LOC` | 자연·행정 지명 |
| `施設名` | **`ORG`** | 이슈 #27 변경 (이전 `LOC`) |
| `法人名` | `ORG` | |
| `政治的組織名` | `ORG` | |
| `その他の組織名` | `ORG` | |
| `製品名` | `PROD` | 서비스·온라인 운영물·기술 표준은 §3.2 로 제외 |
| `イベント名` | `EVT` | |

구현: 1회성 재덤프 스크립트(`/tmp/regen_stockmark.py`, 이슈 #27 시점,
작업 후 폐기). `src/ner/labelers/ja/dataset_loader.py` 는 canonical 변환된
JSONL 덤프(`data/stockmark/{train,test}.jsonl`)를 그대로 읽기만 한다.

재덤프는 다음 2단으로 진행:

1. **결정적 source → canonical 매핑**: 위 표 그대로 적용. 일본어 한자
   substring 기반 시설 키워드 후처리는 `香港`(港)·`茨城`(城)·
   `吉祥寺`(寺)·`モスクワ`(モスク)·`ノックスビル`(ビル) 등 false
   positive 가 커서 채택하지 않는다.
2. **LLM 검증** (gemma-4-31B-it-AWQ-8bit): 1단계 결과의 LOC/ORG 라벨
   엔티티를 LLM 으로 재분류, 불일치 케이스 통계·샘플 보고. 명백한
   라벨 오류만 사용자 검토 후 수정 적용.

### 4.2 VI: HF WikiANN → canonical

| 원본 WikiANN | canonical | 비고 |
|---|---|---|
| `PER` | `PER` | identity |
| `LOC` (자연·행정 지명) | `LOC` | 국가·행정구역·자연지명·주소 |
| `LOC` (시설) | **`ORG`** | 역·공항·병원·박물관·종교시설 등 인공 시설 |
| `ORG` | `ORG` | 기업·정당·정부·군·협회·대학·CLB 일체 |
| (WikiANN 미커버) | `PROD`, `EVT` | LLM 재라벨이 신규 추출 |
| (WikiANN 미커버) | `DAT`, `EMAIL`, `PHONE`, `ID_NUM`, `CREDIT_CARD` | PII 주입 또는 LLM 신규 추출 |

매핑 적용은 `src/ner/augmenters/wikiann_vi/` 의 silver 재라벨 파이프라인에서
LLM 이 본 스키마(§1·§2·§3) 기준으로 자동 수행한다. 재라벨 산출은 위 표대로
**NER 5종**(`PER/LOC/ORG/PROD/EVT` — `relabel_8type.py`·`prompts.py` 의
`DEFAULT_ENTITY_TYPES`)이고, PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`)은
별도 `augmenters/pii --lang vi` 주입으로 추가된다(JA `data/stockmark/pii_*` 와 동형).

### 4.3 8종 canonical → 5종 canonical (이슈 #21·#27 1회성 축소·재배치)

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

구현: 1회성 축소·재배치 스크립트(적용 후 삭제 — 현재 코드 부재).

### 4.4 PII 라벨 (#17 정리)

| 과거 PII | canonical |
|---|---|
| `EMAIL` | `EMAIL` (identity) |
| `PHONE` | `PHONE` (identity) |
| `ADDRESS` | **`LOC`** (병합 규칙 흡수) |
| `DOB` | **`DAT`** (일반 날짜) |
| `ID_NUMBER` | **`ID_NUM`** |
| `CREDIT_CARD` | `CREDIT_CARD` (identity) |

## 5. 모호 사례 결정표

### 5.1 JA

| 표면형 | 판정 | 근거 |
|---|---|---|
| `早稲田大学` | `ORG` | 대학 법인 본체 |
| `旭川キャンパス` | `ORG` | 대학 부속 시설도 ORG |
| `東京大学病院` | `ORG` | 대학 부속 병원도 ORG |
| `東京駅` | `ORG` | 역 = 인공 시설 |
| `成田空港` | `ORG` | 공항 = 인공 시설 |
| `セントメアリー病院` | `ORG` | 병원 = 인공 시설 |
| `明治神宮` | `ORG` | 종교 시설 |
| `清水寺` | `ORG` | 종교 시설 |
| `FCバルセロナ` | `ORG` | 스포츠 팀 (정기) |
| `ラ・リーガ` | `ORG` | 정기 리그 |
| `NHK交響楽団` | `ORG` | 상설 단체 |
| `関ヶ原の戦い` | `EVT` | 1회성 사건 |
| `第45回NHK紅白歌合戦` | `EVT` | 특정 연도판 행사 |
| `富士山` | `LOC` | 자연 지명 |
| `琵琶湖` | `LOC` | 자연 지명 |
| `東京都` | `LOC` 또는 `ORG` | 행정구역=LOC, 지방정부 조직=ORG (표면형 의미로 결정) |
| `東京都千代田区` | `LOC` 단일체 | 행정 지명 복합체 |
| `東京都千代田区1丁目2-3 ビル7F` | `LOC` | 번지·층 포함 주소 |
| `iPhone` | `PROD` | 제품 |
| `NHKスペシャル` | `PROD` | 방송 프로그램 |
| `カタロニア・サーキット` | `ORG` | 인공 시설 (서킷) |
| `バッテリー・パーク` | `ORG` | 도시 공원 = 인공 시설 |
| `東京ドーム` | `ORG` | 경기장 (인공 시설) |
| `寺池城` | `ORG` | 城跡 (史跡 시설) |
| `江戸城` | `ORG` | 城 (史跡 시설) |
| `チェコ代表` | `ORG` | `[지명]+代表` 스포츠 대표팀 |
| `日本代表` | `ORG` | 스포츠 대표팀 |
| `ワンハンドレッドヒルズ` | `ORG` | 부동산 단지 |
| `LKAB` | `ORG` | 약어 단독 — 시그널 부재 시 ORG 우선 |
| `タランテラ` (회사) | `ORG` | `本社`/`会長`/`設立` 시그널 |
| `タランテラ` (제품) | `PROD` | `発売`/`新作`/`型番` 시그널 |

### 5.2 VI

| 표면형 | WikiANN 원본 | canonical 5종 | 근거 |
|---|---|---|---|
| `Chùa Một Cột` | LOC | **ORG** | 종교 시설 |
| `Vịnh Hạ Long` | LOC | **LOC** | 자연 지명 |
| `Sân bay Nội Bài` | LOC | **ORG** | 공항 시설 |
| `Bệnh viện Bạch Mai` | ORG | **ORG** | 병원 시설 |
| `Đại học Quốc gia Hà Nội` | ORG | **ORG** | 대학 법인·캠퍼스 모두 ORG |
| `V.League` | ORG | **ORG** | 정기 리그 |
| `UEFA Champions League 2007-08` | (없음) | **EVT** | 특정 연도판 → 1회성 |
| `Hà Nội FC` | ORG | **ORG** | 스포츠 팀 |
| `Đảng Cộng sản Việt Nam` | ORG | **ORG** | 정당 |
| `Bộ Giáo dục và Đào tạo` | ORG | **ORG** | 정부 부처 |
| `Vietnam Airlines` | ORG | **ORG** | 공기업·영리법인 |
| `Doraemon` | (없음) | **PROD** | 만화·작품 |
| `Galaxy S24` | (없음) | **PROD** | 브랜드+모델 결합 |
| `Đường sắt xuyên Sibir` | (없음) | **LOC** | 철도 노선 자체 = LOC |
| `Thành phố Hồ Chí Minh` | LOC | **LOC** | 행정 지명 전체 단일 LOC. `Hồ Chí Minh` 단독은 PER |

## 6. 적용 범위

- **데이터 파일** (#27 LOC/ORG 경계 적용):
  - `data/stockmark/{train,test}.jsonl` — JA Stockmark HF → canonical 덤프, NER 5종
  - `data/stockmark/pii_{train,test}.jsonl`, `.stats.json`, `.verify.json` — JA NER 5종 + PII 5종 = 10종 평면
  - `data/wikiann_vi/{train,valid,test}.jsonl` — VI silver: 재라벨은 NER 5종, PII 5종은 `augmenters/pii --lang vi` 주입으로 추가 (Stockmark 포맷)

- **코드**:
  - `src/ner/labelers/ja/ner_prompts.py`, `src/ner/labelers/ja/dataset_loader.py`
  - `src/ner/labelers/vi/ner_prompts.py`, `src/ner/labelers/vi/dataset_loader.py`
  - `src/ner/augmenters/pii/` 프롬프트·verifier·generators (10종 평면)
  - `src/ner/augmenters/pii/config.py` 의 `DEFAULT_MERGE_RULES`
    (`NAME→PER`, `ADDRESS→LOC`)
  - `src/ner/augmenters/wikiann_vi/` (silver 재라벨 파이프라인)

- **본 스키마 적용 제외**:
  - `src/ner/llm_eval/**` — 후속 정리 예정
  - `src/ner/labelers/ko/**` — 라벨러는 canonical **6종**(`PER/LOC/ORG/DAT/PROD/EVT`)
    출력(KLUE 유래, `TI/QT` 드롭; `PROD/EVT`는 §3.1~§3.3 회색지대 기준, KLUE
    문장 LLM 재라벨). 라벨러는 PII 미생성 — PII 4종은 별도 `augmenters/pii
    --lang ko` 주입으로 gold(`data/klue/pii_all.jsonl`)에 반영(§1). 분류기 미완.
    이슈 #109(4종)·#111(`PROD/EVT`)·#115(PII 주입)
  - 참고: `src/ner/classifier/**` 는 issue #40 에서 본 스키마(canonical 10종 평면)로 정합 완료

## 7. 평가 시 주의

### 7.1 DAT/PII FP — JA 원본 Stockmark (이슈 #21)

10종 평면 스키마로 **원본(PII 미주입) Stockmark 데이터** 를 평가할 때:

- 원본 Stockmark gold 에는 `DAT` / 4 PII 라벨이 없다. 프롬프트는 모든
  날짜를 `DAT` 로 추출하도록 지시하므로 Wikipedia 텍스트의 연도·기간은
  **FP(false positive)** 로 집계된다.
- 이는 설계 사항: `DAT` 는 "맥락 무관 모든 날짜" 라벨이므로 gold 가
  비어도 예측을 유지한다.
- 해석 시 주의: overall span F1 은 **DAT 예측 건수만큼 precision 하락**
  한다. DAT 자체의 label-level precision 은 gold 가 0 이므로 항상 0.
- 대책: PII-주입 평가셋(`data/stockmark/pii_test.jsonl`) 에서는 gold 에
  DAT/PII 가 포함되므로 정상 평가가 가능하다. 원본 Stockmark 에서 NER
  5종 성능만 보려면 per-entity 테이블에서 `DAT`·PII 4종 행을 제외하고
  읽는다.

### 7.2 LOC/ORG 경계 재정의 영향 — JA (이슈 #27)

- Stockmark gold(`施設名` → ORG) 와 silver 는 동일 매핑이지만, **이전
  측정 결과(#27 이전) 와는 LOC/ORG 분포가 달라 직접 비교할 수 없다**.
  시설 엔티티가 LOC 에서 ORG 로 이동한 만큼 두 라벨의 P/R/F1 이 모두
  변동한다.

### 7.3 WikiANN 3종 평가 시 FP — VI

- VI 라벨러는 WikiANN 평가 외에 PII 주입본 평가와 코드를 공유하므로 10종
  출력 공간을 유지한다. WikiANN 3종 gold 로 평가 시 PROD/EVT/PII 5종
  출력은 FP 로 잡혀 precision 이 인위적으로 하락하는데, 이는 의도된
  trade-off.
- WikiANN 시설=LOC 와 canonical 시설=ORG 의 불일치도 FP/FN 으로 잡힌다.
  silver 재라벨 데이터셋(`data/wikiann_vi/`)에서는 이 불일치가 자동
  해소된 상태로 저장된다.
