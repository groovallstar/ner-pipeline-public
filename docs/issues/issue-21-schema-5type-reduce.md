# issue-21: canonical 스키마 8종 → 5종 축소 · 10종 평면 통합 (JA + VI)

- Issue: https://github.com/groovallstar/ner_pipeline/issues/21
- PR: (머지 직전 채움, `closes #21`)
- 브랜치: `feat/issue-21-ja-schema-5type-reduce` (브랜치명은 초기 JA 전용
  범위 시점 기준, 중간에 VI 라벨러 포함으로 확장됨)
- 승인일: 2026-04-24 (초기 JA 범위) · 확장 승인: 2026-04-24 (VI 라벨러)
- 완료일: 2026-04-24
- 선행 의존성: #13 (완료) · #16 (완료) · #17 (완료)
- 무시: #10 (self-close — 본 이슈와 방향 역전)

## 목적

#13(영문 축약) → #16(경계 룰 명문화) → #17(PII 재설계)로 정비된 canonical
NER 8종을 **5종**(`PER LOC ORG PROD EVT`)으로 축소한다. `CORP/POL/ORG`
사이, `LOC/FAC` 사이의 경계 모호성을 접미사 우선순위 룰로 봉합해 온 누적
복잡도를 **구조적으로** 제거한다.

## 설계 근거

| 포인트 | 효과 |
|---|---|
| 경계 모호성 구조적 해소 | #16 접미사 룰(`〜大学` CORP, `〜病院` FAC, `〜政府` POL 등)이 불필요 |
| 대학 3단 → 2단 | 본체·부속 조직 모두 ORG, 부속 시설만 LOC |
| 다국어 정렬 | WikiANN-vi 3종(PER·LOC·ORG) 상위 3축 동형, KLUE(PS·LC·OG) 정합 |
| 프롬프트 단순화 | `src/labelers/ja/ner_prompts.py` (hot path) few-shot·룰 분량 축소 |

## 축소 매핑 (결정론적 1:1)

| 현행 8종 | 축소 5종 |
|---|---|
| PER | PER |
| LOC | LOC |
| **FAC** | **LOC** |
| **CORP** | **ORG** |
| **POL** | **ORG** |
| ORG | ORG |
| PROD | PROD |
| EVT | EVT |

**분포 (병합 후, stockmark train+test)**: PER 2,980 · ORG 4,715 ·
LOC 3,266 · PROD 1,215 · EVT 1,009

## 범위

### 포함

- **1회성 모듈** (완료 후 삭제) — `src/augmenters/migration/reduce_5type.py`
  + CLI + 단위 테스트
- **영구 전환** — `src/augmenters/`
  - `wikiann_vi/prompts.py` — SINGLE/BATCH/SYSTEM·few-shot 5종
  - `wikiann_vi/relabel_8type.py` — 검증 타입 셋 5종 (파일명 유지)
  - `wikiann_vi/wikidata_anchor.py` — `WIKIDATA_TO_CANONICAL` 134 엔트리
    value 재매핑 (`CORP/POL/ORG → ORG`, `FAC → LOC`)
  - `wikiann_vi/kappa.py`, `wikiann_vi/merge_confidence.py` — 테스트
    fixture 5종 갱신
  - `pii/` — 프롬프트·verifier·generators 라벨 공간을 **10종 평면
    목록**으로 전환 (NER 축소분 반영. PII 4개 라벨 자체는 #17 결과 유지,
    문서·프롬프트에서는 NER/PII 구분 없이 나열)
- **영구 전환** — `src/labelers/ja/` — 룰·few-shot 5종 기준 재작성
  (`ner_prompts.py` + `dataset_loader.py`의 HF JA → canonical 매핑)
- **영구 전환** — `src/labelers/vi/ner_prompts.py` (이슈 진행 중 확장):
  WikiANN 3종에서 canonical **10종 평면 목록**으로 라벨 공간 확장.
  `{ollama,vllm,openai}_ner_labeler.py`는 `DEFAULT_ENTITY_TYPES`를 임포트
  하므로 자동 반영(수정 불필요)
- **데이터 마이그레이션** (1회):
  - `data/stockmark/{train,test}.jsonl`
  - `data/pii/stockmark_pii_1000.jsonl` (+ `.stats.json`, `.verify.json`)
  - `data/wikiann_vi/*.jsonl` (FAC/CORP/POL/ORG 통합)
- **테스트**:
  - 기존 `tests/augmenters/wikiann_vi/`, `tests/augmenters/pii/`,
    `tests/labelers/ja/` fixture·assertion 5종 전환
  - `tests/labelers/vi/` 는 WikiANN 3종 fixture가 canonical 10종의
    부분집합이므로 변경 불필요
  - 신규 `tests/augmenters/migration/test_reduce_5type.py` (1회성)
- **문서 갱신** (과거 포함, 조용한 치환):
  - `docs/manual/data/japanese-canonical-entity-schema.md` — 13종 → **10종
    단일 목록** (`PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM
    CREDIT_CARD`). NER/PII 구분 섹션을 없애고 10종 평면 테이블로 재작성.
    경계 규칙·대학 2단 규칙 축소판 포함
  - `docs/manual/data/japanese-ner.md`
  - `docs/manual/data/vietnamese-ner.md` — §변경 이력 추가(10종 확장)
  - `docs/manual/data/vietnamese-ner-8types.md` — 파일명 유지, §변경 이력
    추가
  - `docs/reports/vietnamese-ner-schema-expansion-2026-04.md` — 표·본문
    라벨 일괄 치환
  - `docs/issues/issue-21-schema-5type-reduce.md` — 본 파일

### 제외

- `src/labelers/ko/` (KLUE 6종 별도 스키마)
- `src/llm_eval/**`, `src/classifier/**`
- HF 원본 데이터 직접 수정
- 이슈 #10, #8은 별도 트랙

## 성공 기준

- [ ] 테스트: 기존 pytest 전부 pass + 신규 `test_reduce_5type.py` pass
- [ ] 데이터: `data/stockmark/{train,test}.jsonl` 모든 `type`이 5종
      (`PER LOC ORG PROD EVT`)
- [ ] 데이터: `data/pii/stockmark_pii_1000.jsonl` `entities[*].label` 전량
      10종 (`PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD`)
      중 하나
- [ ] 데이터: `data/wikiann_vi/*.jsonl` 의 `gold_spans_8type_merged[*].type`
      5종만 (span 총 수 불변)
- [ ] 기능: `augmenters/wikiann_vi` 스모크 실행 시 5종만 출력
- [ ] 기능: `augmenters/pii` end-to-end 실행 시 10종 평면 라벨만 출력
- [ ] 기능: `python -m llm_eval --lang ja` 스모크 실행 시 5종만 출력
- [ ] 정리: `src/augmenters/migration/reduce_5type.py` 및 해당 테스트 삭제
- [ ] 문서: 5개 문서 5종 반영 + 이슈 md 최초 커밋

## 구현 단계 (6단계)

- [ ] 1. `augmenters/migration/reduce_5type.py` (1회성) — 결정론적 치환
      함수 + CLI. 멱등성(이미 5종이면 skip+경고). 검증(변경 전후 span 수
      동일). 단위 테스트
- [ ] 2. `augmenters/wikiann_vi/` 5종 전환 — prompts/wikidata_anchor/
      relabel_8type/kappa/merge_confidence 및 테스트
- [ ] 3. `augmenters/pii/` 5종 전환 — 프롬프트·verifier·generators 및
      테스트
- [ ] 4. `src/labelers/ja/ner_prompts.py` 5종 전환 — 룰 축소·few-shot
      재작성 및 테스트
- [ ] 5. 데이터 마이그레이션 실행 + 검증 (stockmark → pii → wikiann_vi
      순서)
- [ ] 6. 문서 갱신(canonical schema·japanese-ner·vietnamese-ner-8types·
      보고서) + `migration/` 패키지 삭제 + 이슈 md 최초 커밋

## 위험·의존성

- **#16 cross-label 정정 의미 약화**: 7건 중 `香港政府` 등 대부분이 병합
  후 동일 라벨이 됨 — 벤치 효과는 사라지나 의미적 정확성 기록 가치는 유지.
- **정보 손실**: 공권력(POL) vs 법인(CORP) vs 비영리(ORG) 구분 소멸.
  이 프로젝트의 NER 벤치·PII 증강 용도에서는 영향 작음.
- **마이그레이션 순서 의존**: stockmark(원본 JSONL) → pii(stockmark 기반
  증강본) → wikiann_vi(별도) 순으로 수행. 각 단계에서 스키마 검증 통과
  확인 후 다음으로.
- **hot path 라벨러 프롬프트 변경 (JA)**: 프롬프트 변경은 LLM 출력 분포에
  영향을 주므로 단위 테스트만으로 부족할 수 있음 — 5단계 직후 JA ollama
  스모크 실행으로 라벨 공간 확인.

---

## 변경 요약

NER 8종(`PER CORP LOC FAC PROD EVT POL ORG`)을 canonical **5종**
(`PER LOC ORG PROD EVT`)으로 축소. NER/PII 구분 없는 **10종 평면 목록**
(`PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD`)을
canonical로 확정. JA·VI 라벨러 모두 canonical 영문 라벨을 직접 출력하고,
dataset 로더는 `data/stockmark/`·`data/wikiann_vi/`의 canonical 덤프
JSONL만 읽는다(HF 원본 로딩 및 런타임 라벨 매핑은 제거). 1회성
`src/augmenters/migration/reduce_5type.py`로 기존 JSONL·stats·kappa
파일을 결정론적으로 치환 후 해당 모듈을 삭제했고, 이후 로더 리팩터
단계에서는 매핑 코드도 제거해 덤프 → 평가 경로를 단일 책임으로 정리.

## 구현 결과 (계획 대비)

- [x] 1. `augmenters/migration/reduce_5type.py` (1회성) + 25개 단위 테스트
- [x] 2. `augmenters/wikiann_vi/` 5종 전환 — prompts·wikidata_anchor
      (134 엔트리 재매핑)·relabel_8type·kappa·merge_confidence 및 테스트
      fixture
- [x] 3. `augmenters/pii/` 라벨 공간 10종 평면 전환 (NER 축소 반영·PII
      4라벨은 #17 그대로). PII 모듈 자체는 라벨 중립이라 코드 수정은
      fixture만, 테스트에서 CORP→ORG·FAC→LOC
- [x] 4. `src/labelers/ja/ner_prompts.py` 5종 NER + 5 PII/날짜 영문 출력
      으로 재작성. 추가 범위: `dataset_loader.py`에 `JA_TO_CANONICAL`
      매핑 추가 + `LABEL_CORRECTIONS` 7건 → 1건(병원 단일 규칙만 유효)
      축소. `AGENTS.md` `DEFAULT_ENTITY_TYPES` 갱신
- [x] 5. 데이터 마이그레이션 실행 (계획된 순서 stockmark → pii →
      wikiann_vi 순으로 완료). JA 라벨러 스모크(vLLM
      `cyankiwi/gemma-4-31B-it-AWQ-8bit`, 10 샘플)에서 10종 평면 라벨만
      출력, PII 클래스 오탐은 `DAT`만 관찰 (설계 의도; 아래 검증 참조)
- [x] 6. 문서 갱신(canonical-entity-schema·japanese-ner·vietnamese-ner-
      8types·보고서·AGENTS.md) + `migration/` 패키지 삭제 + 이슈 md 최초
      커밋 준비

추가 발견·계획 변경:

- **`src/labelers/ja/dataset_loader.py` 편입**: 원 계획은 `labelers/ja/
  ner_prompts.py`만 포함이었으나, 라벨러 출력을 영문 canonical로 바꾸면서
  HF 원본 JA 라벨과의 정합을 맞추기 위해 동일 디렉터리의 `dataset_loader.
  py`와 테스트를 함께 갱신. 파일 수 휴리스틱이 아니라 "ORG/LOC 출력 ↔
  gold 매칭" 단일 관심사의 불가분 변경.
- **LABEL_CORRECTIONS 7건 → 1건**: 축소 후 대부분 정정이 동일 canonical
  로 수렴(e.g. 法人名↔政治的組織名 모두 ORG). 병원 오라벨(法人名 → 施設名,
  canonical로는 ORG → LOC)만 유효 정정으로 남음.

## 검증

**단위 테스트 (메인 세션, `uv run pytest -q`)**

- 마이그레이션 단계: 25 passed (5종 매핑·JSONL 멱등성·PII stats 병합·
  Kappa confusion 병합)
- wikiann_vi: 61 passed (5종 값 전체 커버, CORP/POL/FAC 0건 assertion)
- pii: 37 passed (NER 축소 라벨 사용, PII 4라벨 identity)
- labelers/ja: 12 passed (JA_TO_CANONICAL 맵·LABEL_CORRECTIONS 1건·대학
  ORG·병원 LOC)
- 전체 (migration 삭제 후): **215 passed**
- `ruff check src/augmenters src/labelers/ja tests/augmenters tests/labelers/ja`: All checks passed

**데이터 마이그레이션 (실행 결과)**

| 파일 | 레코드 | 변경 span |
|---|---:|---:|
| `data/stockmark/train.jsonl` | 4,274 | 3,779 |
| `data/stockmark/test.jsonl` | 1,069 | 995 |
| `data/pii/stockmark_pii_1000.jsonl` | 993 | 682 |
| `data/wikiann_vi/gemma_8type_*.jsonl` (3 split) | 40,000 | 5,912 |
| `data/wikiann_vi/qwen_8type_*.jsonl` (3 split) | 40,000 | 4,268 |
| `data/wikiann_vi/vi_wikiann_8type_recall_*.jsonl` (3 split) | 40,000 | 5,490 |
| `data/pii/stockmark_pii_1000.{stats,verify}.json` | 2 | per_label 병합 |
| `data/wikiann_vi/kappa_*.json` (3 split) | 3 | confusion·per_type 병합 |

**라벨 공간 검증** (마이그레이션 후)

- `data/stockmark/*`: 5종만 (`PER 2,980 · ORG 4,715 · LOC 3,266 ·
  PROD 1,215 · EVT 1,009` — 축소 전 합계와 일치)
- `data/pii/stockmark_pii_1000.jsonl`: 10종만 (`PER · LOC · ORG · PROD ·
  EVT · DAT · EMAIL · PHONE · ID_NUM · CREDIT_CARD`) + `stats/verify` JSON의
  `per_label` 키도 동일
- `data/wikiann_vi/gold_spans_8type*`: 5종만. 원본 `gold_spans`
  (WikiANN 3종)는 무변경

**JA 라벨러 스모크 (vLLM, 10+5 샘플)**

원본 Stockmark test 10 샘플:
- 예측 라벨 전량 10종 평면 내: `PER 8 · ORG 8 · LOC 2 · PROD 3 · DAT 3 · EVT 1`
- PII 클래스 오탐(EMAIL/PHONE/ID_NUM/CREDIT_CARD): **0건**
- DAT 예측 3건: 설계 의도(모든 날짜 추출). Stockmark gold는 DAT 비포함
  이므로 해당 예측은 span F1 산출 시 FP로 잡혀 precision이 DAT 예측
  건수만큼 하락함을 `japanese-canonical-entity-schema.md` §평가 시 주의에 명시

PII-주입 5 샘플:
- gold 라벨 `ORG 5 · PER 1 · EMAIL 3 · PHONE 1 · DAT 2 · LOC 1`
- 예측 라벨 `ORG 6 · PER 1 · EMAIL 3 · DAT 7 · PHONE 1 · LOC 3` — 10종
  외 라벨 0건, PII(EMAIL/PHONE) cardinality 일치. DAT 과다 예측은 원본
  데이터 스모크와 동일 경향(설계 의도)

## 후속 확장 (VI 라벨러 + 로더 리팩터)

주 커밋 후 동일 이슈·동일 브랜치에서 VI 경로를 JA와 동일 스키마로 정렬
하고, 데이터 로더를 "canonical 덤프 전용"으로 재정렬했다.

### 변경

- **`src/labelers/vi/ner_prompts.py`**: WikiANN 3종에서 canonical **10종
  평면 목록**으로 확장. SINGLE/BATCH/SYSTEM+USER 프롬프트와 few-shot을
  PER/LOC/ORG/PROD/EVT + DAT·EMAIL·PHONE·ID_NUM·CREDIT_CARD 기준으로
  재작성. ollama/vllm/openai 라벨러는 `DEFAULT_ENTITY_TYPES` 임포트로
  자동 반영.
- **`src/labelers/ja/dataset_loader.py`**: HF Stockmark 로딩과
  `JA_TO_CANONICAL`·`LABEL_CORRECTIONS` 런타임 매핑 제거. `load()`는
  `data/stockmark/{train,test}.jsonl` canonical 덤프만 읽고, 파일이
  없으면 `FileNotFoundError`로 즉시 실패(폴백 없음).
- **`src/labelers/vi/dataset_loader.py`**: HF WikiANN 로딩 제거. `load()`
  는 `data/wikiann_vi/vi_wikiann_8type_recall_{split}.jsonl`의
  `gold_spans_8type_merged` 필드(recall 정책 병합, canonical 5종)를
  기본으로 읽는다. `span_key` 파라미터로 다른 필드(`gold_spans` 3종 등)
  선택 가능. `bio_to_offset_spans` / `offset_spans_to_bio` 유틸은
  augmenters 재라벨 파이프라인용으로 유지.
- **호출자 업데이트**:
  - `src/augmenters/wikiann_vi/__main__.py`: 재라벨 파이프라인은 HF 원본
    3종 BIO를 필요로 하므로 본 CLI에 `_load_wikiann_hf` 헬퍼를 내장
    (`labelers.vi` 의 `bio_to_offset_spans` 재사용). labelers 쪽 HF
    의존을 의도적으로 제거한 뒤의 단일 책임 분리.
  - `src/augmenters/pii/loaders.py`: `load_stockmark` 의 `seed`·
    `test_size` 파라미터 삭제 — canonical 덤프에서 분할이 이미 고정.
  - `src/augmenters/pii/__main__.py`: `load_stockmark(seed=...)` 호출
    지점에서 인자 제거.
  - `src/llm_eval/__main__.py`: JA/VI 경로 모두 `load(name=...)` 등
    HF 인자 호출을 제거하고 canonical 덤프 경로를 출력 로그로 공개.

### 방침

- **폴백 금지**: 덤프가 없으면 명시적 에러로 실패. 히스토리 상 매핑
  혼선(예: `JA_TO_CANONICAL` 정의 변경 시 과거 덤프와의 해석 불일치)을
  구조적으로 제거.
- **매핑 단일 책임**: HF→canonical 변환은 1회성 덤프 시점에 이뤄지고,
  평가 로더는 덤프 결과만 소비. 재덤프가 필요하면 별도 1회성 도구로
  처리.

### 검증 (후속 단계)

- 전체 pytest: **216 passed**
  - 신규 `TestLoadCanonicalDump` (VI) 5 케이스
  - 재작성 `tests/labelers/ja/test_ja_dataset_loader.py` 10 케이스
    (JSONL 경로 로드·split 가드·파일 부재 예외·키 별칭)
- `ruff check src/labelers src/augmenters tests/labelers tests/augmenters
  src/llm_eval/__main__.py`: clean
- 문서 갱신: `src/labelers/AGENTS.md`, `src/labelers/ja/AGENTS.md`,
  `src/augmenters/AGENTS.md`, `docs/manual/data/vietnamese-ner.md`
  (§변경 이력), 본 이슈 md 최신화

## 관련 커밋

<!-- PR 직전 채움 -->
- `f2fe819` feat(schema): NER 8종 → 5종 축소 · canonical 10종 평면 통합
- (후속) VI 라벨러 10종 확장 + 로더 canonical 덤프 전용 리팩터
