# issue-16: 스키마 경계 규칙 정비 — CORP/POL/FAC/ORG 룰 명문화 · Stockmark cross-label 정정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/16
- PR: (머지 직전 채움, `closes #16`)
- 브랜치: `feat/issue-16-canonical-boundary-refine`
- 선행 의존성: #17 (완료 · 2026-04-23)
- 재개일: 2026-04-24

## 목적

#13으로 도입한 canonical 스키마(14종 → #17 이후 13종)의 CORP/POL/FAC/ORG
4종 경계가 모호하거나 Stockmark 실데이터 분포와 역전되어 있다. 본 이슈는
실데이터 분포를 근거로 경계 룰을 결정론적 접미사 우선순위로 명문화하고,
문서·프롬프트·데이터 3자를 일관시킨다.

## 실데이터 근거 (재확인, 2026-04-24)

Stockmark `train.jsonl`(약 5K) + `test.jsonl`(약 0.3K) 전수 오디트.

| 표면 접미사 | CORP | POL | FAC | ORG | 비고 |
|---|---:|---:|---:|---:|---|
| `大学` / `大學` | **115** | 0 | 0 | 0 | **대학 본체 100% CORP** — 문서 표기(ORG)와 역전 |
| `病院` | 1 | 0 | 10 | 0 | `セントメアリー病院` 1건만 CORP 오라벨 |
| `駅` | 0 | 0 | 75 | 0 | FAC 100% |
| `鉄道` | 64 | 0 | 2 | 0 | CORP 기본 (예외: `JR京都線` 계열 FAC) |
| `テレビ` | 23 | 0 | 0 | 0 | CORP 100% |
| `放送局` | 1 | 0 | 2 | 0 | 혼재 — 본 이슈 범위 제외 |
| `公社` | 8 | 1 | 0 | 0 | CORP 기본 |
| `公団` | 4 | 0 | 0 | 0 | CORP 100% |
| `政府` | 5 | 51 | 0 | 0 | POL 기본, 5건 CORP 오라벨 |
| `省` | 0 | 92 | 0 | 0 | POL 100% (행정구역 `台湾省` 6건은 LOC로 별개) |
| `庁` | 0 | 29 | 1 | 0 | POL 100% 가까이 |
| `軍` | 0 | 216 | 0 | 0 | POL 100% |
| `部隊` / `師団` | 0 | 28 | 0 | 0 | POL 100% |
| `協力機構` / `条約機構` | 0 | 4 | 0 | 0 | POL 100% |
| `高校` / `高等学校` / `中学校` / `小学校` | 0 | 0 | 45 | 0 | FAC 100% |

→ **접미사 우선순위 룰**이 실데이터와 완전 정합. 대학은 문서가 ORG라 적고
있으나 실데이터는 100% CORP.

### Cross-label 오라벨 실측 (7건)

| 위치 | 엔티티 | 현재 라벨 | 정정 라벨 | 근거 |
|---|---|---|---|---|
| train.jsonl:819 (id=2746825) | `NHK` | ORG | CORP | NHK 단독 13건 중 12건이 CORP |
| train.jsonl:522 (id=2919391) | `香港政府` | CORP | POL | `〜政府` 51건 POL 기준 |
| train.jsonl:1557 (id=1836213) | `中華民国政府` | CORP | POL | 동상 |
| train.jsonl:4166 (id=2115134) | `日本政府` | CORP | POL | 동상 |
| test.jsonl:769 (id=3908159) | `ロシア政府` | CORP | POL | 동상 |
| test.jsonl:979 (id=1587749) | `ロシア政府` | CORP | POL | 동상 |
| test.jsonl:183 (id=2942700) | `セントメアリー病院` | CORP | FAC | `病院` 11건 중 10건이 FAC |

> 이슈 본문은 "NHK 2건"으로 기재했지만 실측 결과 `NHK` 단독 ORG 오라벨은
> **1건**이다. `NHK交響楽団`(train:2784, ORG)은 교향악단(조직체)으로 정규
> ORG 분류에 해당하므로 정정 대상 아님.

## 결정 — 결정론적 접미사 우선순위 룰

모호한 경우 위에서 아래 순으로 적용한다.

1. **POL**: `〜省`, `〜庁`, `〜軍`, `〜部隊`, `〜師団`, `〜条約機構`,
   `〜協力機構`, `〜政府`, `〜党`, `〜議会`, `〜裁判所`
2. **ORG**: `〜大学院 부속 조직`(동아리·研究会·学院), 스포츠리그·팀,
   학회·협회·연맹·NGO
3. **CORP**: `〜大学` / `〜大學` (대학 법인 본체), `〜公社`, `〜公団`,
   `〜鉄道`, `〜放送` / `〜テレビ`, `〜会社` / `〜社`, `〜銀行`, `〜航空`
4. **FAC**: `〜駅`, `〜空港`, `〜港`, `〜病院`, `〜高校` / `〜高等学校` /
   `〜中学校` / `〜小学校`, `〜郵便局`, `〜店`, `〜支店`, `〜研究所`,
   `〜図書館`, `〜美術館`, `〜館`, `〜キャンパス`, `〜校舎`, `〜植物園`,
   `〜博物館`, `〜寺`, `〜神社`
5. **PROD**: 상품·작품·소프트웨어
6. **EVT**: 전쟁·조약·대회·일회성 행사

**대학의 3단 규칙** (단일 표제 "대학"이 CORP·FAC·ORG 중 어느 것인가):

| 대상 | 라벨 | 예시 |
|---|---|---|
| 대학 법인 본체 | **CORP** | `早稲田大学`, `東京大学`, `コロラド・メサ大学` |
| 대학 부속 시설 (건물·캠퍼스·부속 植物園·校舎) | **FAC** | `旭川キャンパス`, `東京大学植物園` |
| 대학 부속 조직 (동아리·研究会·学院) | **ORG** | `〜研究会`, `〜学院`(학술 조직 단위) |

## 스코프

### 포함
- `docs/manual/data/japanese-canonical-entity-schema.md` — CORP/POL/FAC/ORG 경계 룰
  재작성, 대학=CORP 3단 규칙 추가
- `docs/manual/data/japanese-ner.md` — 대학=CORP·접미사 우선순위 정합,
  few-shot 예시(`北海道東海大学 → 그 외 조직` → 法人名) 수정
- `docs/manual/data/vietnamese-ner-8types.md` — `Đại học` CORP 전환, §3.4
  ORG 경계 규칙 재정렬, 모호 사례 10건 표 갱신
- `src/augmenters/wikiann_vi/prompts.py` — few-shot `Đại học Quốc gia Hà Nội`
  ORG → CORP, 우선순위 문구 정합
- `src/augmenters/wikiann_vi/wikidata_anchor.py` — `Q3918/Q38723/Q875538`
  ORG → CORP 블록 이동
- `src/labelers/ja/ner_prompts.py` — 大学=法人名, few-shot `北海道東海大学`
  변경, 접미사 우선순위 정합
- `src/labelers/ja/dataset_loader.py` — `LABEL_CORRECTIONS` 테이블로 HF
  원본의 오라벨 7건을 로딩 시점에 결정론적 정정
- `tests/labelers/ja/test_ja_dataset_loader.py` — `LABEL_CORRECTIONS` 유닛
  테스트 6건 신설

### 제외
- 연맹·협회·위원회 CORP↔POL 재정렬 (114건, 별도 이슈)
- `src/labelers/ko/**`, `docs/manual/data/korean-*.md` (국문 데이터 미사용)
- `src/llm_eval/**`, `src/classifier/**` 로직 변경

## 성공 기준

- [x] CORP/POL/FAC/ORG 경계 룰이 Stockmark 상위 라벨 분포와 100% 정합
- [x] NHK·`〜政府`·`セントメアリー病院` cross-label 7건 정정
- [x] `ruff check` 통과
- [x] `pytest` 통과
- [x] 계획↔구현 섹션 정합

## 작업 계획 (체크박스)

- [x] japanese-canonical-entity-schema.md — 대학=CORP 3단 규칙, 접미사 우선순위 명문화
- [x] japanese-ner.md 정합 — 본문 표·few-shot 예시
- [x] vietnamese-ner-8types.md 정합 — §3.4 재정렬, 모호 사례 표 갱신
- [x] augmenters/wikiann_vi/prompts.py — Đại học=CORP, 우선순위 정합
- [x] augmenters/wikiann_vi/wikidata_anchor.py — `Q3918/Q38723/Q875538` ORG→CORP
- [x] labelers/ja/ner_prompts.py — 大学=法人名, few-shot 정합
- [x] labelers/ja/dataset_loader.py — `LABEL_CORRECTIONS` 도입 + 유닛 테스트
- [x] pytest + ruff check
- [x] 구현 결과·검증 섹션 작성 → 최초 커밋 → PR (`closes #16`)

## 구현 결과

### 1. 스키마 문서 정비

- **`japanese-canonical-entity-schema.md`**: 13종 정의 표에서 CORP(대학 법인 본체),
  FAC(병원·초중고·대학 부속 시설), POL(`〜政府`/`〜条約機構` 등) 경계를
  재작성. 새 섹션 "접미사 우선순위(결정론적 룰)" 추가(POL>ORG>CORP>FAC>
  PROD>EVT 6단계). "대학의 3단 규칙" 하위 섹션 추가(본체=CORP, 부속
  시설=FAC, 부속 조직=ORG). 핵심 경계 케이스 표에 NHK·セントメアリー
  病院·초중고 등 실례 반영.
- **`japanese-ner.md`**: 엔티티 표를 canonical 정의와 정합. "일본어 특화
  분류 규칙" 섹션을 "접미사 우선순위" 6단계 표로 교체. 대학 3단 규칙과
  병원 단일 규칙 서술. few-shot 예시 주석
  `北海道東海大学(CORP)+旭川キャンパス(FAC)`로 수정.
- **`vietnamese-ner-8types.md`**: §2 엔티티 정의에서 CORP에 `trường đại học`
  명시, ORG에서 대학 제거하고 교향악단·부속 조직 명시. §3.4 ORG 경계
  규칙을 CORP(대학 법인)·POL·ORG·FAC 4-way 블록으로 재정렬하고 결정론적
  접미사 우선순위 4단계 추가. §4 모호 사례 표의 `Đại học Quốc gia Hà Nội`
  판정을 ORG → **CORP**로 변경. §8에 2026-04-24 이슈 #16 변경 이력 추가.

### 2. 프롬프트·라벨러 정합

- **`src/augmenters/wikiann_vi/prompts.py`** (SINGLE + BATCH): 타입 정의에서
  CORP에 `trường đại học (pháp nhân)` 명시, ORG에서 `Đại học` 제거. 우선
  순위 1→POL, 2→ORG(스포츠·부속 조직), 3→CORP(대학·영리), 4→FAC로 재정렬.
  few-shot `Đại học Quốc gia Hà Nội` ORG → **CORP**.
- **`src/augmenters/wikiann_vi/wikidata_anchor.py`**: 대학 관련 Q-ID 3건
  (`Q3918 university`, `Q38723 higher education institution`,
  `Q875538 public university`)을 ORG 블록에서 CORP 블록으로 이동. 주석에
  Stockmark 실측 100% CORP 근거 명기. `Q2385804 educational institution`
  은 초중고를 포함하는 상위 클래스이므로 ORG fallback 유지.
- **`src/labelers/ja/ner_prompts.py`** (SINGLE + BATCH + SYSTEM 3개 템플릿):
  法人名 정의에 `〜公社/公団/大学/大學/テレビ` 추가. 施設名 정의에
  `〜病院/高校/中学校/小学校/キャンパス/校舎/植物園` 추가. 政治的組織名
  정의에 `〜政府/条約機構/協力機構` 추가. 판정 우선순위 6단계 표를 대학
  3단 규칙·병원 단일 규칙 각주와 함께 재작성. few-shot
  `北海道東海大学` を その他の組織名 → **法人名**로 변경.

### 3. Stockmark cross-label 런타임 정정

HF 원본(`stockmark/ner-wikipedia-dataset`)은 이 저장소에서 직접 수정할 수
없고 local `data/stockmark/*.jsonl`은 현재 소스 코드에서 참조되지 않으므로,
**`JapaneseDatasetLoader.LABEL_CORRECTIONS`** 딕셔너리를 도입해 로딩 시점에
결정론적 정정을 적용한다. 키는 `(curid, entity_text, original_type)`,
값은 정정된 type. `_to_records()` 에서 모든 엔티티에 대해 조회 후 매칭되면
타입만 교체(텍스트·오프셋은 원본 유지).

정정된 7건 (모두 일본어 원본 라벨 기준):

| id | 엔티티 | 원 라벨 | 정정 |
|---|---|---|---|
| 2746825 | `NHK` | `その他の組織名` | `法人名` |
| 2919391 | `香港政府` | `法人名` | `政治的組織名` |
| 1836213 | `中華民国政府` | `法人名` | `政治的組織名` |
| 2115134 | `日本政府` | `法人名` | `政治的組織名` |
| 3908159 | `ロシア政府` | `法人名` | `政治的組織名` |
| 1587749 | `ロシア政府` | `法人名` | `政治的組織名` |
| 2942700 | `セントメアリー病院` | `法人名` | `施設名` |

### 4. 유닛 테스트

`tests/labelers/ja/test_ja_dataset_loader.py` 신설 (6 케이스):
- NHK ORG → CORP 매핑 확인
- 〜政府 CORP → POL 매핑 확인
- 병원 CORP → FAC 매핑 확인
- 대학(원래 법인명) 원본 라벨 유지 확인
- 비대상 엔티티 원본 유지 확인
- LABEL_CORRECTIONS 테이블 크기(=7) 회귀 방지

## 검증

### 실측 오디트 (Stockmark train+test 5,343행)

| 표면 접미사 | CORP | POL | FAC | ORG | 정정 후 위반 |
|---|---:|---:|---:|---:|---|
| `〜大学` / `〜大學` | **115** | 0 | 0 | 0 | 0 |
| `〜病院` | 0 | 0 | **11** | 0 | 0 (1건 CORP→FAC) |
| `〜駅` | 0 | 0 | 75 | 0 | 0 |
| `〜鉄道` | 64 | 0 | 2 | 0 | 0 (JR京都線 계열 FAC는 정규) |
| `〜テレビ` | 23 | 0 | 0 | 0 | 0 |
| `〜公社` | 8 | 1 | 0 | 0 | 경계 외 1건 (별도 이슈 후보) |
| `〜公団` | 4 | 0 | 0 | 0 | 0 |
| `〜政府` | 0 | **60** | 0 | 0 | 0 (5건 CORP→POL) |
| `〜省` | 0 | 92 | 0 | 0 | 0 (LOC 6건은 행정구역 `台湾省` 계열, 정규) |
| `〜庁` | 0 | 29 | 1 | 0 | 경계 외 1건 |
| `〜軍` | 0 | 216 | 0 | 0 | 0 |
| `〜部隊` / `〜師団` | 0 | 28 | 0 | 0 | 0 |
| `〜協力機構` / `〜条約機構` | 0 | 4 | 0 | 0 | 0 |
| `〜高校/高等学校/中学校/小学校` | 0 | 0 | 45 | 0 | 0 |
| `NHK` 단독 | 13 | 0 | 0 | 0 | 0 (1건 ORG→CORP) |

정정 완료 후 접미사 우선순위 룰 대비 오라벨이 0 또는 소수 경계 외 케이스
(별도 이슈 후보)만 남는다.

### 테스트·린트

- `pytest -q`: **209 passed** (기존 203 + 신규 6). 기존 테스트 회귀 없음.
- `ruff check src/ tests/labelers/ja/`: 수정 파일 **All checks passed**.
  (타 파일의 사전 존재 15건 F401·F541은 본 이슈 범위 외.)

### 런타임 정합 확인

```python
loader = JapaneseDatasetLoader()
recs = loader.load(split='train') + loader.load(split='test')
# 7건 정정 대상이 모두 새 라벨로 로드됨 — 수동 확인 완료.
```

