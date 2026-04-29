# issue-13: augmenters 8종 + PII 6종 스키마 OntoNotes 영문 축약 통일

- Issue: https://github.com/groovallstar/ner_pipeline/issues/13
- PR: https://github.com/groovallstar/ner_pipeline/pull/14
- 브랜치: `feat/issue-13-schema-english-abbrev`
- 승인일: 2026-04-22
- 완료일: 2026-04-22

## 목적

`augmenters/` 범위에서 사용하는 JA Stockmark 8종 + PII 6종 스키마를
OntoNotes 관용 영문 축약으로 통일한다. 현재 JA 8종은 일본어 원문
(`人名`, `法人名`, `地名` 등)으로 고정되어 있어 다국어 통합 라벨 공간이
불균일하고, 리포트 가독성·라벨러 재사용성이 떨어진다.

본 이슈는 **범위를 `src/augmenters/` 하위로 한정**하여 canonical label
space만 영문 축약으로 전환한다. 이미 생성된 JSONL은 **결정론적 문자열
치환으로 마이그레이션**(LLM 재호출 없음)하고, 원본 HF 데이터셋은 덤프
시점 이후로 직접 사용하지 않는다.

## Canonical 14종

정의·경계 규칙·예시·원본 매핑은 독립 문서로 분리:
**[`docs/manual/data/japanese-canonical-entity-schema.md`](../manual/data/japanese-canonical-entity-schema.md)**

요약:
- 8종: `PER CORP LOC FAC PROD EVT POL ORG`
- PII 6종: `EMAIL PHONE ADDRESS DOB ID_NUM CREDIT_CARD`
- WikiANN 3종 축소 매핑: `CORP ∪ POL ∪ ORG → ORG`
- 유일한 PII 변경: `ID_NUMBER` → `ID_NUM`

## 범위

- 포함 (`src/augmenters/` 및 `data/`):
  - **1회성 모듈 (사용 후 제거)** — `src/augmenters/migration/` 신설 패키지:
    - `stockmark_dump.py` — Stockmark HF 원본을 canonical 8종 JSONL로
      덤프 → `data/stockmark/{train,test}.jsonl`(gitignore)
    - `migrate_jsonl.py` — 기존 JSONL의 `type`/`label` 문자열 결정론적
      치환. 대상: `data/wikiann_vi/*.jsonl`,
      `data/pii/*.jsonl`(`.stats.json`·`.verify.json` 포함)
    - 본 이슈 하위 작업 마지막 단계에서 `migration/` 패키지 및 관련
      테스트 일괄 삭제
  - **영구 전환** — 기존 `src/augmenters/` 모듈 라벨 영문화:
    - `wikiann_vi/prompts.py` — SINGLE/BATCH/SYSTEM 프롬프트 본문과
      few-shot 예시의 모든 타입 문자열
    - `wikiann_vi/relabel_8type.py` — 출력 파서 기본값 · 검증 타입 셋
    - `wikiann_vi/wikidata_anchor.py` — **상수명 전환**:
      `WIKIDATA_TO_STOCKMARK` → `WIKIDATA_TO_CANONICAL`, 134 엔트리
      value 전량 영문 canonical로 치환. JA 라벨은 주석으로도 남기지
      않음
    - `wikiann_vi/kappa.py` · `wikiann_vi/merge_confidence.py` —
      JSONL의 `type` 필드가 영문 canonical인 전제 유지(로직은 문자열
      중립이라 별도 수정 거의 없음. 테스트 fixture 갱신)
    - `pii/` — 프롬프트·verifier·generators의 `ID_NUMBER` → `ID_NUM`
      및 8종 라벨 문자열 영문화 (JA 라벨 예시 포함된 곳 전수 치환)
  - **테스트**:
    - 기존 `tests/augmenters/wikiann_vi/`, `tests/augmenters/pii/`
      fixture·assertion 전면 갱신
    - 신규 `tests/augmenters/migration/` (1회성 — `migration/` 삭제와
      함께 제거)
- 제외:
  - `src/labelers/**`, `src/llm_eval/**`, `src/classifier/**`
  - WikiANN 3종 스키마(`src/labelers/vi/ner_prompts.py`의 PER/LOC/ORG)
  - HF 원본 데이터 직접 수정 (덤프만)
- 문서 갱신 (과거 포함, **후속 이슈 언급 없이 조용한 치환**):
  - `docs/manual/data/japanese-ner.md` — 본문 태그·예시·매핑표를
    영문 canonical로 전량 치환. `§변경 이력` 한 줄 추가
  - `docs/manual/data/vietnamese-ner-8types.md` — 본문 태그·예시·
    매핑표 영문 치환. §8 변경 이력 한 줄 추가
  - `docs/reports/vietnamese-ner-schema-expansion-2026-04.md` —
    표·본문 라벨 일괄 영문 치환 (수치는 동일. 서사 변경 없음)
  - `docs/issues/issue-13-schema-english-abbrev.md` — 본 파일
  - 이전 이슈 문서(`issue-10`, `issue-8`)는 당시 시점 보존(수정 없음)

## 성공 기준

- [ ] 테스트: 기존 pytest 전부 pass + 신규 migration 단위 테스트
- [ ] 데이터: `data/stockmark/test.jsonl` 생성, 모든 `type`이
      8종 canonical 중 하나 (검증 스크립트 통과)
- [ ] 데이터: `data/wikiann_vi/vi_wikiann_8type_recall_test.jsonl`
      마이그레이션 후 `gold_spans_8type_merged[*].type` 전량 영문
      canonical + span 수 동일 (11,437 span)
- [ ] 데이터: `data/pii/stockmark_pii_1000.jsonl` 마이그레이션 후
      `entities[*].label` 전량 영문 canonical (1000 records 전수)
- [ ] 데이터: `stockmark_pii_1000.stats.json`, `.verify.json` 내
      라벨 키·값 영문화
- [ ] 기능: `augmenters/wikiann_vi` 스모크 실행 시 영문 라벨 출력
- [ ] 기능: `augmenters/pii` end-to-end 실행 시 영문 라벨만 출력
- [ ] 정리: `src/augmenters/migration/` 및 해당 테스트 삭제 완료
- [ ] 문서: 3개 과거 문서 영문 치환 + 이슈 md 최초 커밋

## 구현 단계 (3~6)

- [ ] 1. `augmenters/migration/stockmark_dump.py` (1회성) — HF 원본
      로드 → JA→EN 매핑 적용 → `data/stockmark/{train,test}.jsonl`
      덤프. 단위 테스트(매핑 완전성·덤프 무결성)
- [ ] 2. `augmenters/migration/migrate_jsonl.py` (1회성) — 치환 함수 +
      CLI. 멱등성 보장(이미 영문이면 skip+경고). 검증 로그(변경 전후
      span 수 동일 확인)
- [ ] 3. `augmenters/wikiann_vi/` 영문화 — prompts(few-shot 포함)/
      relabel_8type/`WIKIDATA_TO_STOCKMARK`→`WIKIDATA_TO_CANONICAL`
      리네이밍·값 치환/kappa/merge_confidence. 테스트 fixture 갱신
- [ ] 4. `augmenters/pii/` 영문화 — prompts/verifier/generators의
      JA 예시·`ID_NUMBER` 문자열 일괄 치환. 테스트 갱신
- [ ] 5. JSONL 마이그레이션 실행:
      - `data/stockmark/{train,test}.jsonl` 생성
      - `data/wikiann_vi/vi_wikiann_8type_recall_test.jsonl` 치환
      - `data/pii/stockmark_pii_1000.jsonl` + `.stats.json` +
        `.verify.json` 치환
      - `migration/` 패키지·테스트 삭제 커밋
- [ ] 6. 과거 문서 영문 치환 (japanese-ner.md, vietnamese-ner-8types.md,
      vietnamese-ner-schema-expansion-2026-04.md). 이슈 md에 결과·검증
      섹션 추가 → 최초 커밋 → PR 생성(`closes #13`)

## 리네이밍 확인표

| 위치 | before | after | 비고 |
|---|---|---|---|
| `wikidata_anchor.py` | `WIKIDATA_TO_STOCKMARK` | `WIKIDATA_TO_CANONICAL` | 상수. 134 엔트리 값도 함께 치환 |
| `wikidata_anchor.py` | `_stockmark_type_from_claim` 등 내부 함수명에 `stockmark` 포함 시 | `_canonical_type_from_claim` | grep 후 전수 확인 |
| `prompts.py` | docstring 내 "Stockmark 8종" 서술 | "8종 canonical (OntoNotes 스타일)" | 서술 일관성 |
| JSONL 필드 | `gold_spans_8type_merged` | (그대로 유지) | 필드명은 변경 범위 밖 |

## 위험·의존성

- **프롬프트 성능**: 일본어 프롬프트 본문 맥락에 영문 canonical 라벨을
  few-shot 출력 예시로 넣었을 때 LLM(Gemma/Qwen)이 일본어 라벨로 회귀
  출력할 위험. 200 샘플 스모크로 출력 형식 붕괴 없음을 확인. 회귀 시
  후처리 JA→EN 폴백 매핑(runtime)을 `relabel_8type.py`에 추가
- **JSONL 멱등성**: 치환된 파일 재실행 시 매핑 실패 → `log.warning`
  내고 skip. 변경 전후 span 수 동일성 assert
- **리포트 표 정합성**: 표 캡션·본문 라벨만 치환. 수치·서사는 변경
  하지 않음. diff 검토로 의도치 않은 수치 변경 0건 확인
- **라벨 문자열 누락**: 전역 검색
  `rg '人名|法人名|地名|施設名|製品名|イベント名|政治的組織名|その他の組織名|ID_NUMBER'`
  를 마지막 단계에서 돌려 누락 없음을 확인
- **`data/pii/*.stats.json` 구조**: JSON 내 라벨 키·값 모두 치환 필요.
  스크립트는 key·value 양쪽 재귀 치환하도록 작성

## 참고

- 선행 이슈: #10 (머지됨, 커밋 `dba2ce0`)
- OntoNotes 5 태그 레퍼런스: https://catalog.ldc.upenn.edu/LDC2013T19

---

## 변경 요약

`augmenters/` 범위에 한정해 NER 8종 + PII 6종 라벨을 OntoNotes 관용 영문
축약(`PER/CORP/LOC/FAC/PROD/EVT/POL/ORG` + `EMAIL/PHONE/ADDRESS/DOB/
ID_NUM/CREDIT_CARD`)으로 통일. 1회성 마이그레이션 패키지로 기존 데이터
파일을 결정론적 치환(LLM 재호출 없음). 패키지는 사용 후 제거.

## 구현 결과 (계획 대비)

- [x] 1. `augmenters/migration/stockmark_dump.py` (1회성) — HF Stockmark
      → `data/stockmark/{train,test}.jsonl` canonical 덤프 + 13 단위
      테스트
- [x] 2. `augmenters/migration/migrate_jsonl.py` (1회성) — JSONL/JSON
      결정론적 치환 + 멱등성 + span 수 보존 assert + 15 단위 테스트
- [x] 3. `augmenters/wikiann_vi/` 영문화 — prompts/relabel/anchor/kappa/
      merge + 4개 테스트 fixture sed 치환. 상수 `WIKIDATA_TO_STOCKMARK`
      → `WIKIDATA_TO_CANONICAL`, 134 Q-ID 값 영문화
- [x] 4. `augmenters/pii/` 영문화 — config·injector·llm_injector·
      generators + 5개 테스트 fixture sed 치환. `ID_NUMBER` → `ID_NUM`,
      `'NAME': '人名'` → `'NAME': 'PER'`, `ADDRESS → 地名` 머지 비교 →
      `ADDRESS → LOC`
- [x] 5. JSONL 마이그레이션 실행:
      - `data/stockmark/{train,test}.jsonl` 신규 (4275·1069 records)
      - `data/wikiann_vi/vi_wikiann_8type_recall_test.jsonl`
        11,437 spans canonical (`gold_spans_8type_merged[*].type`)
      - `data/pii/stockmark_pii_1000.jsonl` 3,309 entities
        (2,428 migrated + 881 already canonical = 3,309 ✓)
      - `data/pii/stockmark_pii_1000.{stats,verify}.json` per_label key
        영문화
      - `migration/` 패키지·테스트 삭제 (5 files, -838 lines)
- [x] 6. 과거 문서 영문화 + 이슈 md 최종 커밋 + PR
      - `docs/manual/data/japanese-ner.md`: 표·본문 라벨 일괄 치환,
        설계 의사결정 §7 #3 항목을 canonical 사용으로 갱신
      - `docs/manual/data/vietnamese-ner-8types.md`: §2 도입부 ·
        §6 인코딩 규칙 · §8 변경 이력 갱신
      - `docs/reports/vietnamese-ner-schema-expansion-2026-04.md`:
        제목 · 헤더 메타에 라벨 표기 한 줄 추가, 본문 라벨 일괄 치환

## 검증

- 테스트: `uv run pytest tests/ -q` — **203 passed** (회귀 0건)
- 린트: `uv run ruff check src/augmenters/ tests/augmenters/` — All
  checks passed
- 누락 검증: `rg '人名|法人名|地名|施設名|製品名|イベント名|政治的組織名|その他の組織名|ID_NUMBER|WIKIDATA_TO_STOCKMARK' src/augmenters tests/augmenters docs/{specs,reports,manual,issues}/`
  — 매치 1건 (`src/augmenters/pii/llm_injector.py:35`의 LLM 프롬프트
  자연 일본어 서술 `人名・地名・組織名` — 라벨 값 아님, 의도적 보존)
- 마이그레이션 결정론성: span 수 변경 0건 (사전 11,437 spans = 사후
  11,437 spans, PII 3,309 = 3,309)
- kappa·anchor 수치: 결정론적 치환이라 재측정 불요. 기존 리포트 수치
  (κ=0.6571, anchor=0.9512) 그대로 유효

## 관련 커밋

- `4db62f6` feat(augmenters/migration): Stockmark 덤프 CLI 및 canonical
  스키마 정의
- `e76e164` feat(augmenters/migration): JSONL·JSON canonical 마이그레이션
  유틸 추가
- `4c9565f` feat(augmenters/wikiann_vi): 라벨 스키마를 canonical 영문
  축약으로 전환
- `968e1d9` feat(augmenters/pii): PII 라벨 스키마를 canonical 영문 축약
  으로 전환
- `8ced723` chore(augmenters/migration): 1회성 마이그레이션 패키지 제거
- `<TBD>` docs: 과거 스펙·리포트 영문 치환 + 이슈 md 최초 커밋

## 후속 작업·알려진 한계

- `data/`는 gitignore 대상이므로 본 PR에는 마이그레이션 결과 데이터
  파일이 포함되지 않는다. 신규 환경에서는 `git show <commit>`으로
  migration 패키지를 복원해 재실행하거나, 새 영문 프롬프트 기반
  파이프라인을 다시 돌리면 된다.
- `src/augmenters/pii/llm_injector.py:35`의 LLM 프롬프트 본문은 일본어
  자연어 서술이며 라벨 토큰이 아니므로 그대로 유지.

---

## train·validation·test 전 스플릿 재라벨 확장 (2026-04-22)

초기 계획은 test 스플릿만 canonical 8종으로 재라벨하는 것이었으나,
`data/` 경로 정리(stockmark_en→stockmark, wikiann_vi_relabel→wikiann_vi)
후속 작업으로 **train(20K) + validation(10K) + test(10K) 세 스플릿
전체**를 canonical 8종으로 재라벨했다.

### Qwen 모델 선정 — Qwen3.6-35B-A3B-AWQ-4bit

MoE 아키텍처(3B active) Qwen3.6-35B-A3B-AWQ-4bit를 validator로 채택.
단일 요청 레이턴시 0.47s, 처리량 88 tok/s로 Gemma(19 tok/s) 대비 약
4.6배 빠른 디코딩을 확보해 3 스플릿(40K) 재라벨을 단시간에 완료.
thinking 모드는 vLLM 컨테이너의 기본 `--default-chat-template-kwargs
'{"enable_thinking": false}'`로 off.

시작 스크립트: `docker/vllm/start-qwen3.6-35b-a3b-awq.sh` 신규.

### 재라벨 결과 (per-model, canonical 8종)

**Gemma (`cyankiwi/gemma-4-31B-it-AWQ-8bit`, concurrency=16)**

| 스플릿 | 레코드 | 라벨링 성공 | 총 스팬 | PER | LOC | PROD | ORG | POL | FAC | CORP | EVT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train (20K) | 20,000 | 17,282 (86.4%) | 23,203 | 8,195 | 7,935 | 2,204 | 1,585 | 1,453 | 912 | 601 | 318 |
| validation (10K) | 10,000 | 8,602 (86.0%) | 11,454 | 4,078 | 3,921 | 1,091 | 746 | 777 | 418 | 280 | 143 |
| test (10K) | 10,000 | 8,620 (86.2%) | 11,627 | 4,206 | 3,885 | 1,131 | 768 | 724 | 465 | 282 | 166 |

**Qwen3.6 (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, concurrency=16)**

| 스플릿 | 레코드 | 라벨링 성공 | 총 스팬 | PER | LOC | PROD | ORG | POL | FAC | CORP | EVT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train (20K) | 20,000 | 12,871 (64.4%) | 18,152 | 7,137 | 6,251 | 1,182 | 1,171 | 936 | 711 | 491 | 273 |
| validation (10K) | 10,000 | 6,357 (63.6%) | 8,946 | 3,589 | 3,027 | 597 | 543 | 502 | 326 | 240 | 122 |
| test (10K) | 10,000 | 6,436 (64.4%) | 9,116 | 3,750 | 3,011 | 609 | 543 | 476 | 351 | 235 | 141 |

라벨링 성공률은 Gemma 86% vs Qwen3.6 64%로 차이가 있다. Qwen이
보수적으로 엔티티 없음(빈 배열)을 더 자주 반환하는 경향이 관측됨.

### recall 병합 결과 (Gemma ∪ Qwen3.6)

| 스플릿 | 총 스팬 | PER | LOC | PROD | ORG | POL | FAC | CORP | EVT |
|---|---|---|---|---|---|---|---|---|---|
| train | 22,649 | 8,111 | 7,862 | 2,160 | 1,450 | 1,290 | 894 | 564 | 318 |
| validation | 11,179 | 4,048 | 3,874 | 1,068 | 678 | 689 | 410 | 269 | 143 |
| test | 11,382 | 4,171 | 3,850 | 1,113 | 709 | 651 | 456 | 267 | 165 |

### Cohen kappa (Gemma vs Qwen3.6)

| 스플릿 | κ | p_o | p_e | A∪B 스팬 |
|---|---|---|---|---|
| train | 0.5879 | 0.6727 | 0.2057 | 24,393 |
| validation | 0.5733 | 0.6613 | 0.2062 | 12,114 |
| test | 0.5766 | 0.6642 | 0.2069 | 12,317 |

Landis-Koch "moderate agreement" (0.41~0.60) 구간. 타입별로는
**PER 0.77~0.78 최고**, **PROD 0.45~0.47 최저**. PROD 경계가 두
모델 간 가장 큰 차이 지점.

### 산출물 (gitignore 대상, 로컬 생성)

```
data/wikiann_vi/
├── gemma_8type_{train,validation,test}.jsonl
├── qwen_8type_{train,validation,test}.jsonl
├── vi_wikiann_8type_recall_{train,validation,test}.jsonl
└── kappa_{train,validation,test}.json
```

### 추가 커밋

- `<TBD>` feat(docker/vllm): Qwen3.6-35B-A3B AWQ 시작 스크립트 추가
- `<TBD>` chore(claude): `.claudeignore`에 `data/`/`results/`/`logs/` 추가
- `<TBD>` docs: 이슈 #13 train·validation·test 전 스플릿 재라벨 결과 반영
