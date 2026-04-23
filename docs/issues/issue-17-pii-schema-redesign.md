# issue-17: PII 스키마 재설계 — ADDRESS → LOC 흡수, DOB → DAT 개명·의미 확장

- Issue: https://github.com/groovallstar/ner_pipeline/issues/17
- PR: (머지 직전 채움, `closes #17`)
- 브랜치: `feat/issue-17-pii-schema-redesign` (Phase 1 + Phase 2 통합)
- 승인일: 2026-04-23
- 완료일: 2026-04-23

> **운영 변경** (2026-04-23): 사용자 지시로 Phase 2(VI 측 PII 정합, 좁은
> 범위)를 Phase 1과 동일 브랜치에 이어서 작업하고 **단일 PR로 통합**한다.
> VI 라벨러 전면 Stockmark 정합(`labelers/vi/ner_prompts.py` 3종 → 13종
> 확장)은 VI 라벨러 확장 작업(별도 이슈)에서 처리.

## 목적

#13에서 도입한 canonical 14종 중 PII 6종의 두 항목을 정비해 스키마를 13종으로 축소하고, 날짜 라벨 공간을 `DAT`로 일원화한다.

- `ADDRESS`는 `LOC`와 경계가 모호하고 PII 파이프라인 밖에서는 의미 분리 근거가 약하다. 이미 `augmenters/pii/injector.py`에 단일 토큰 `ADDRESS → LOC` 병합 규칙이 존재 — 이를 전면화한다.
- `DOB`는 출생일 맥락으로 한정되어 일반 날짜와 이분화된 상태다. `DAT`로 개명·의미 확장해 "모든 날짜(연·월·일·기간·상대날짜)"를 포괄하는 단일 카테고리로 만든다.

## 운영 방식 — 이슈 1개 × 2 PR

이슈 #17은 하나로 유지하되, 작업·PR은 JA → VI 순으로 쪼갠다. 중간 상태에서 한꺼번에 전체를 바꾸지 않고, JA 측 정비를 먼저 확정한 뒤 VI를 후속으로 정합한다.

| Phase | 대상 | 브랜치 | PR 참조 | 본 md 섹션 |
|---|---|---|---|---|
| **1** | JA PII 스키마 재설계 | `feat/issue-17-pii-schema-redesign` | `refs #17` | 본문 "Phase 1" 섹션 |
| **2** | VI PII 스키마 정합 + VI 라벨러 확장 | `feat/issue-17-pii-schema-vi` | `closes #17` | 본문 "Phase 2" 섹션(개요만) |

본 md는 Phase 1 PR 작성 직전에 **Phase 1 구현 결과·검증 섹션을 채워 최초 커밋**한다. Phase 2 착수 시 본 md에 Phase 2 구현 결과 섹션이 추가된다.

## 핵심 결정

- **ADDRESS 완전 제거** (canonical 스키마·JA·VI 모두): 기존 ADDRESS 라벨은 전부 LOC로 이월
- **DOB → DAT 개명 + 의미 확장**: "출생일만" → "모든 날짜"
- **`generate_address()` 함수는 유지**: 출력 라벨만 LOC로. 합성 PII 학습 데이터에서 주소 패턴이 사라지지 않도록 함
- **JA 프롬프트의 "`DATE` というタイプは使わない" 금지문 제거** (Phase 1)
- **VI 라벨러·프롬프트는 Phase 2에서** JA와 동일 canonical로 맞춤. 현재 VI 라벨러는 3종(PER/LOC/ORG)이고 Stockmark 8종 정합 작업이 진행 중이므로, PII 관련 변경을 그 시점에 일괄 처리해 어정쩡한 중간 상태를 피한다

---

## Phase 1 — JA (현재)

### 범위

#### 포함
- `docs/manual/data/canonical-entity-schema.md` — 14종 → 13종 (ADDRESS 삭제, DOB → DAT)
- `docs/manual/data/japanese-ner.md` — PII 라벨 언급 부분 정합
- `src/augmenters/pii/config.py` — `DEFAULT_LABELS`에서 ADDRESS 제거, DOB → DAT 개명
- `src/augmenters/pii/injector.py` — `_SUFFIX_JA`에서 ADDRESS 템플릿 삭제, DOB → DAT 개명. `apply_label_merge`의 ADDRESS → LOC 병합 로직 삭제 (완전 흡수되므로 병합 개념 불필요)
- `src/augmenters/pii/generators/base.py` — label 라우팅에서 ADDRESS 분기 제거, DOB → DAT
- `src/augmenters/pii/generators/ja.py` — `generate_dob` → `generate_dat` 개명. `generate_address`는 유지(출력 라벨은 LOC로 라우팅)
- `src/augmenters/pii/AGENTS.md` — 라벨 목록·병합 규칙 서술 갱신
- `src/labelers/ja/ner_prompts.py` — ADDRESS 라벨 정의·예시·경계 규칙 전부 삭제 (L10, 28, 69, 71-79, 156). DOB 정의를 DAT로 개명·확장, "DATE 금지" 문구 제거, few-shot에 일반 날짜(`"2016年1月29日"` 등) DAT 예시 포함
- `data/pii/stockmark_pii_1000.jsonl`·`.stats.json`·`.verify.json` — idempotent 마이그레이션 스크립트 작성·실행 (ADDRESS → LOC, DOB → DAT)

#### 제외 (Phase 2로 이월)
- `src/labelers/vi/**`
- `src/augmenters/wikiann_vi/prompts.py`
- `src/augmenters/pii/generators/vi.py`
- `docs/manual/data/vietnamese-ner-8types.md`, `vietnamese-ner.md`
- `data/pii/` 내 VI 파일 (현재 존재하지 않음)

#### 공통 제외
- `src/labelers/ko/**`, `docs/manual/data/korean-*.md` — 국문 데이터는 미사용
- `src/llm_eval/**`, `src/classifier/**` 로직 변경 없음

### 성공 기준

- [ ] `docs/manual/data/canonical-entity-schema.md` — ADDRESS 부재, DOB 부재, DAT 존재 (13종)
- [ ] `data/pii/*.jsonl` (JA)에서 `ADDRESS`·`DOB` 라벨 0건 (전부 LOC·DAT로 이월)
- [ ] `pytest` 통과 (augmenters/pii 테스트 포함)
- [ ] `ruff check` 통과
- [ ] 본 md ↔ 구현 결과 섹션 정합

### 구현 단계

전체 변경은 **단일 원자 커밋**으로 처리 (사용자 지시: 2026-04-23).

1. **스키마 문서 13종화**
   - `canonical-entity-schema.md` PII 표에서 ADDRESS 행 제거, DOB 행을 DAT로 개명·의미 확장
   - 경계 규칙 섹션의 `ADDRESS vs LOC`, `DOB vs 일반 연도` 행 제거 → `DAT` 규칙 1행(모든 날짜) 추가
   - 원본 → canonical 매핑 표에서 `ADDRESS → ADDRESS` 제거, `DOB → DAT` 반영
2. **augmenters/pii 코드 갱신**
   - `config.py DEFAULT_LABELS` 에서 `'ADDRESS'` 제거, `'DOB'` → `'DAT'`
   - `injector.py _SUFFIX_JA`에서 `'ADDRESS'` 키 삭제, `'DOB'` → `'DAT'` 개명 (템플릿 문구도 "생년월일"에서 "일자" 같은 일반 날짜 문맥으로)
   - `injector.py apply_label_merge`의 ADDRESS → LOC 로직 삭제
   - `generators/base.py route_label` — ADDRESS 분기 제거, DOB → DAT
   - `generators/ja.py` — `generate_dob` → `generate_dat` 개명. `generate_address` 유지(라벨 라우팅만 LOC)
   - `AGENTS.md` 내부 PII 토큰 목록·병합 규칙 서술 갱신
3. **labelers/ja/ner_prompts.py 재작성**
   - ADDRESS 라벨 정의·예시·경계 규칙 전부 삭제
   - DOB 정의를 DAT로 개명하며 "출생일 맥락만" 조건 삭제
   - `"DATE というタイプは使わない"` 금지문 제거, DAT few-shot 예시 추가
4. **data/pii JSONL 마이그레이션 스크립트 작성·실행**
   - `src/augmenters/pii/tools/migrate_labels.py` (idempotent): `type`/`label` 필드 `ADDRESS → LOC`, `DOB → DAT` 치환. `.stats.json`·`.verify.json`의 라벨 키도 갱신
   - 실행: `data/pii/stockmark_pii_1000.jsonl` + 부수 파일
   - span 수·오프셋 완전 동일 확인 (단사 치환)
5. **japanese-ner.md 정합**
   - PII 라벨 언급 섹션에서 ADDRESS 제거, DOB → DAT
6. **본 md Phase 1 구현 결과 섹션 작성 → PR (`refs #17`)**

### 위험·의존성

- 기존 생성 데이터(`data/pii/stockmark_pii_1000.jsonl`) 마이그레이션 스크립트 idempotent 설계 (재실행 시 변경 없음 보장)
- JA 프롬프트 "DATE 금지" 문구 반대로 뒤집는 규모 변경 — 기존 few-shot 예시 의미 정합 재확인 (`"2016年1月29日移籍"` 같은 건 현재 미라벨 → DAT로 전환)
- `tests/augmenters/pii/`에 ADDRESS·DOB 라벨 검증 케이스가 있다면 갱신 필요

---

## Phase 2 — VI (좁은 범위)

Phase 1과 동일 브랜치에서 이어서 처리. 어정쩡한 중간 상태를 피하기 위해
`labelers/vi/ner_prompts.py` 3종 → 13종 전면 확장은 VI 라벨러 확장 작업
(별도 이슈)에 위임하고, 본 Phase 2는 PII 주입 내부 코드 정합에만 집중.

### 범위

#### 포함
- `src/augmenters/pii/generators/vi.py` — `generate_dob` → `generate_dat`
  개명 + docstring
- `src/augmenters/pii/generators/base.py` — Phase 1에서 임시로 두었던
  `getattr(mod, 'generate_dat', …or generate_dob)` 폴백 제거, `mod.generate_dat`
  단순 호출로 복원
- `docs/manual/data/canonical-entity-schema.md` — 변경 이력에 Phase 2 완료 기록
- `src/augmenters/AGENTS.md` — VI Phase 2 호환 주석 제거

#### 제외 (별도 이슈)
- `src/labelers/vi/ner_prompts.py` — 3종 유지 (VI 라벨러 확장 작업에서 일괄
  13종화)
- `src/augmenters/wikiann_vi/prompts.py` — 8종 재라벨 프롬프트, PII 무관
- `docs/manual/data/vietnamese-ner-8types.md` — PII 관련 직접 참조 없음 확인
- `data/pii/` VI 파일 — 아직 존재하지 않음

### 성공 기준

- [x] `generate_dat`가 JA/VI 모두 존재해 `generate_pii(label='DAT')`가
  언어별 분기 없이 동작
- [x] `pytest` 통과 (Phase 1과 동일 203건)
- [x] `ruff check src/augmenters/` 통과

### Phase 2 구현 결과 (계획 대비)

- [x] `generators/vi.py` `generate_dob` → `generate_dat` 개명 + docstring
- [x] `generators/base.py` DAT dispatch 단순화 (폴백 제거)
- [x] `augmenters/AGENTS.md` VI 호환 주석 제거
- [x] `canonical-entity-schema.md` 변경 이력에 Phase 2 완료 반영

### Phase 2 검증

- **테스트**: `python -m pytest tests/ -q` → **203 passed** (Phase 1 이후
  변동 없음 — getattr 폴백 제거가 동작에 영향 없음 확인)
- **Lint**: `ruff check src/augmenters/` → **All checks passed**
- **호환성**: VI PII 주입 데이터는 아직 생성 전이므로 마이그레이션 불필요.
  JA 쪽은 Phase 1에서 이미 치환 완료

---

## 후속 복귀

- **이슈 #17 PR 머지 후 #16 (CORP/POL/FAC/ORG 경계 정비) 재개**

---

## Phase 1 변경 요약

canonical을 14종 → 13종으로 축소 (ADDRESS 제거, DOB → DAT 개명·의미 확장). `generate_address()` 함수는 유지하되 출력 라벨을 `DEFAULT_MERGE_RULES`의 `'ADDRESS': 'LOC'` 무조건 병합으로 LOC에 흡수. DAT는 "모든 날짜"로 의미 확장해 JA 프롬프트의 "DATE 금지" 문구를 제거. data/pii JSONL·stats·verify 3종을 idempotent CLI(`augmenters.pii.tools.migrate_labels`)로 일괄 치환(ADDRESS×149 → LOC, DOB×188 → DAT).

## Phase 1 구현 결과 (계획 대비)

- [x] 1. **스키마 문서 13종화** — canonical-entity-schema.md 재구조화 (NER 8종 + 일반 날짜 1종 + PII 4종). ADDRESS→LOC 병합 주석, 매핑 표 갱신, 변경 이력 추가
- [x] 2. **augmenters/pii 코드 갱신**
  - `config.py`: `DEFAULT_PII_LABELS`에서 DOB → DAT, `DEFAULT_MERGE_RULES`에 `'ADDRESS': 'LOC'` 추가
  - `injector.py`: `_JA_CONNECTORS`·`_VI_CONNECTORS` DOB 키 → DAT, 템플릿 "生年月日："/"Ngày sinh:" → "日付："/"Ngày:". `apply_label_merge` `is_simple_place` 조건부 로직 제거(무조건 ADDRESS→LOC). `is_simple_place` 함수·`_SIMPLE_PLACE_SPLIT` 정규식 삭제, 미사용 `re` import 제거
  - `generators/base.py`: 'DOB' 분기 → 'DAT'로 변경, `generate_dat` 우선 + `generate_dob` getattr 폴백 (VI Phase 2 호환)
  - `generators/ja.py`: `generate_dob` → `generate_dat` 개명, docstring 갱신
  - `AGENTS.md`: 라벨 스키마·병합 규칙 설명 13종 기준으로 재작성
- [x] 3. **labelers/ja/ner_prompts.py 재작성** — `DEFAULT_ENTITY_TYPES`에서 ADDRESS 제거·DOB → DAT, SINGLE_PROMPT/SYSTEM_PROMPT의 ADDRESS 정의 삭제·DOB를 DAT로 개명·의미 확장, "DATE 금지" 문구 제거, few-shot 주소 예시를 `地名`으로, DAT 예시로 2016年1月29日 등 일반 날짜 추가
- [x] 4. **data/pii JSONL 마이그레이션 스크립트 + 실행** — `src/augmenters/pii/tools/migrate_labels.py` 신규 작성 (idempotent). `python -m augmenters.pii.tools.migrate_labels data/pii/`로 stockmark_pii_1000.{jsonl,stats.json,verify.json} 3종 치환 완료. 재실행 시 "no-op" 확인
- [x] 5. **japanese-ner.md 정합** — ADDRESS/DOB 참조 없음 확인, 변경 불필요
- [x] 6. **본 md Phase 1 구현 결과 섹션 작성 → PR (`refs #17`)**

## Phase 1 검증

- **테스트**: `python -m pytest tests/ -q` → **203 passed** (augmenters/pii·labelers·bio_dataset·span_f1 전부 통과)
- **Lint**: `ruff check src/augmenters/pii/ src/labelers/ja/ner_prompts.py tests/augmenters/pii/` → **All checks passed**
- **data/pii 마이그레이션 결과** (span 오프셋·text 필드 불변, 라벨만 변경):
  | 파일 | ADDRESS → LOC | DOB → DAT |
  |---|---:|---:|
  | `stockmark_pii_1000.jsonl` | 149건 | 188건 |
  | `stockmark_pii_1000.stats.json` | key 1(병합) | key 1(개명) |
  | `stockmark_pii_1000.verify.json` | key 1(병합) | key 1(개명) |
  - LOC 총합: 340 + 149 = **489건** 확인 완료
  - DAT: 188건 (DOB 전수 이월)
  - 재실행 시 "no-op (already migrated)" 출력으로 idempotency 확인
- **테스트 갱신**: `tests/augmenters/pii/test_label_merger.py` — `is_simple_place` 테스트 제거, `test_address_complex_kept` (ADDRESS 잔존 기대) 를 `test_address_complex_also_mapped_to_loc` (무조건 LOC 병합)로 반전, `test_dat_label_passthrough` 신규 추가
- **리뷰어 확인 사항**:
  - 프롬프트 변경이 JA 라벨러 벤치마크 정확도에 미치는 영향은 실측 이전이므로 Phase 1 PR에서는 스키마·코드 정합성만 확인하고, 후속 벤치마크 리포트에서 메트릭 변동을 기록
  - `src/classifier/pii_benchmark.py:32`의 독립 라벨 리스트(`"NAME", "PHONE", "ADDRESS", "DOB", "ID_NUMBER", ...`)는 자체 `pii_gen` 데이터 생성기 경로라 본 이슈 스코프(`classifier/` 로직 변경 없음) 제외. 필요 시 후속 이슈
  - 벤치마크 리포트(`docs/reports/japanese-ner-pii-benchmark-2026-04.md`)는 pre-#17 실측 스냅샷이라 historical 기록으로 보존

## Phase 1 관련 커밋

- 커밋 직후 채움: `<hash>`

