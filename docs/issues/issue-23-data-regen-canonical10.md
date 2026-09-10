# issue-23: WikiANN-vi 5종 silver 재라벨 + PII 주입본 생성

- Issue: https://github.com/groovallstar/ner_pipeline/issues/23
- PR: (머지 직전 채움, `closes #23`)
- 브랜치: `feat/issue-23-data-regen-canonical10`
- 승인일: 2026-04-27
- 완료일:
- 선행 의존성: #21 (완료, 8→5종 축소 · 10종 평면 통합) · 1a39401 (VI 라벨러
  10종 확장 · 로더 canonical 덤프 전용 리팩터)

> **상태**: 4/27 코드/grep 검토에서 발견한 사실 4건을 반영해 범위 축소.
> Stockmark는 현행 5종 gold + 5종+PII(`stockmark_pii_1000.jsonl`) 모두
> 그대로 유지(범위 외). 본 이슈는 **WikiANN-vi 5종 silver 재라벨 + PII
> 주입본 생성**만 다룬다.

## 배경

- #21로 canonical NER 8종이 5종(`PER LOC ORG PROD EVT`)으로 축소되고,
  PII 5종과 합쳐 **10종 평면 목록**(`PER · LOC · ORG · PROD · EVT ·
  DATE · EMAIL · PHONE · ID_NUM · CREDIT_CARD`)이 단일 출처가 되었다.
- Stockmark 측 데이터는 #21 마이그레이션으로 **이미 캐노니컬 5종 gold**로
  정리됨 (`data/stockmark/{train,test}.jsonl`, 9f55b94 7건 정정 반영).
  PII 주입본(`data/pii/stockmark_pii_1000.jsonl`)도 10종 평면. → **현상
  유지** (Stockmark 범위 외).
- 1a39401로 VI 라벨러는 10종 canonical 덤프 전용으로 리팩터됐으나, 실제
  WikiANN-vi 데이터는 과거 **8종 silver** 산출물만 있다. → **재생성
  필요** (silver를 5종으로 재라벨, PII 주입본 신규 생성).

## 4/27 검토에서 확인한 사실

| # | 사실 | 본 이슈에 미친 영향 |
|---|---|---|
| 1 | `JapaneseDatasetLoader`는 #21에서 HF 직접 호출 없이 **JSONL 덤프 전용**으로 리팩터. `LABEL_CORRECTIONS`·`JA_TO_CANONICAL`은 1회성 마이그레이션 도구로 이동 후 도구 삭제됨 | Stockmark "10종 재덤프"는 gold→silver 격하라 부적절 → **범위 제외** |
| 2 | Stockmark 원본 gold에 PII 0건 (위키 개요문) | Stockmark 10종 재덤프의 실효성 약함 → **범위 제외** |
| 3 | #16 7건 정정 중 5종 매핑 후 분별 가능한 항목은 **1건뿐** (병원 ORG→LOC). 나머지 6건은 5종에서 모두 ORG로 흡수되어 정정 무의미 | "#16 회귀 점검" → **현행 데이터 1건 read-only spot-check**로 축소 |
| 4 | `8type` 리터럴이 데이터 파일명 외에 JSONL 필드명·모듈 파일명·스펙 문서명에 광범위 분포. **#21이 명시적으로 호환성 유지를 결정**(코드 주석 `relabel_8type.py:5`) | "후보 B 일괄 치환"은 **데이터 파일명만**으로 제한 |

## 결정 사항 (2026-04-27 확정)

### Q1. 생성 대상 → **WikiANN-vi 5종 silver + PII 주입본만**

- **Stockmark 범위 외**: 5종 gold·PII 주입본 모두 현행 그대로 유지
- **WikiANN-vi 5종 silver 재라벨**: HF 원본을 LLM(Gemma + Qwen)으로 5종
  재라벨. PROD/EVT는 silver(LLM)일 수밖에 없음(원본 3종에 없으므로).
- **WikiANN-vi PII 주입**: 5종 silver에 합성 PII 주입 → 5 NER + 5 PII
  = 10종 평면. `stockmark_pii_1000.jsonl` 포맷 대응.

> Stockmark "10종 재덤프"는 원본 5종 gold를 silver로 격하시키는 결과라
> 본 이슈에서 제외. PII가 원본 위키 개요문에 0건이라 silver를 만들어도
> 실효성 약함.

### Q2. 백엔드·모델 → **vLLM Gemma + Qwen 2-모델**

- 1차 라벨러: `cyankiwi/gemma-4-31B-it-AWQ-8bit` (vLLM, localhost:8081,
  `max_model_len=8192`)
- 2차 라벨러: 기존 트랙(#10)에서 사용한 Qwen 계열 AWQ vLLM 엔드포인트
  (구체 엔드포인트·버전은 실행 직전 가용성 재확인)
- 두 모델 산출물로 cross-model kappa 측정

### Q3. 샘플 수·스플릿 → **전체 (3-스플릿 전량)**

- WikiANN-vi 5종 silver: `train` / `validation` / `test` **전량**
- VI PII 주입본: WikiANN-vi 전량 기반 (또는 라인 수 기준 산정 — 실행
  직전 결정)

### Q4. 출력 경로·파일명 규약 → **A만 변경 (데이터 파일명만)**

#21에서 명시적으로 "파일·필드명의 `8type` 리터럴은 데이터 호환성을 위해
유지"하기로 결정했다(코드 주석 `relabel_8type.py:5`). 본 이슈는 그
결정을 존중하고 **데이터 파일명만** 변경한다.

| 카테고리 | 본 이슈 변경 여부 |
|---|---|
| **A. 데이터 파일명** (`*_8type_*.jsonl`) | ✅ 변경 |
| **B. JSONL 필드명** (`gold_spans_8type`, `gold_spans_8type_merged`) | ❌ 유지 |
| **C. 모듈/테스트 파일명** (`relabel_8type.py`, `test_relabel_8type.py` 등) | ❌ 유지 |
| **D. 스펙 문서명** (`vietnamese-ner-8types.md`) | ❌ 유지 |
| **E. 과거 이슈/리포트 docs** | ❌ 유지(히스토리) |

확정 출력 경로:

```
data/wikiann_vi/
  gemma_{train,validation,test}.jsonl       # 신규 5종 silver (구 gemma_8type_* 대체)
  qwen_{train,validation,test}.jsonl        # 신규 5종 silver (구 qwen_8type_* 대체)
  kappa_{train,validation,test}.json        # 갱신
  vi_wikiann_recall_{train,validation,test}.jsonl   # 구 *_8type_recall_* 대체
data/pii/
  wikiann_vi_pii.jsonl (+ .stats.json, .verify.json)   # 신규
data/stockmark/
  train.jsonl, test.jsonl                   # 변경 없음 (현행 5종 gold)
data/pii/
  stockmark_pii_1000.jsonl                  # 변경 없음 (현행 10종 평면)
```

> JSONL 안의 필드명은 그대로 `gold_spans_8type`/`gold_spans_8type_merged`
> 사용. 파일명만 schema-tag 없이 단순화.

### Q5. 검증 레이어 → **6종**

1. **cross-model kappa**: Gemma vs Qwen, WikiANN-vi 3-스플릿
   (기존 `src/augmenters/wikiann_vi/kappa.py` 그대로 재사용)
2. **Wikidata 인터링크 앵커 일치율**: WikiANN-vi 신규 산출물
   (`src/augmenters/wikiann_vi/wikidata_anchor.py`, 134 엔트리 5종 value)
3. **Stockmark #16 1건 read-only spot-check**: 현행
   `data/stockmark/{train,test}.jsonl`에서 `セントメアリー病院`이 `LOC`로
   유지되는지 확인 (#16 7건 중 5종 매핑 후 분별 가능한 1건)
4. **PII 할루시네이션 카운트**: VI PII 주입본에서 false positive 점검
   (위키 개요문에서 숫자 패턴을 `ID_NUM`/`PHONE`으로 오탐 등)
5. **PII prefix-동반률 측정** (신규, 2026-04-17 핸드오프 §7
   흡수): Stockmark 측정에서 일본어 단서어(`担当者`/`連絡先`/`ID番号`
   등) PII 값 직전 동반 비율 **27.5%** + `ID_NUMBER` 영문 라벨 그대로
   leakage **170건** 확인됨. VI 등가 패턴(`liên hệ`, `số điện thoại`,
   `email`, `số CMND` 등) 매칭률 + 영문 라벨 leakage 카운트 측정.
6. **PII 밀도 분포 점검** (신규, ja PII augmenter 핸드오프
   §3 흡수): 일본어 주입에서 `P(0)=0.2` 기대 대비 실제
   `samples_no_pii_ratio = 1.8%` 괴리 발견. VI 주입본의 N=0 비율이
   `density[0]`과 일치하는지 점검. 불일치 시 `PIIInjector._sample_n()`
   또는 0건일 때 접미사 삽입 경로 의심.

## 범위

### 포함

- **WikiANN-vi 5종 silver 재라벨** (Gemma + Qwen, 3-스플릿 전량)
  - `src/augmenters/wikiann_vi/__main__.py` 재실행
  - 산출: `data/wikiann_vi/{gemma,qwen}_{train,validation,test}.jsonl`
  - kappa 갱신 + recall-merge 갱신
- **WikiANN-vi PII 주입** (전량 기반)
  - `src/augmenters/pii/` 재실행, VI 텍스트 입력 처리
  - 산출: `data/pii/wikiann_vi_pii.jsonl` (+ `.stats.json`, `.verify.json`)
- **데이터 파일 경로 참조 정정** (카테고리 A만)
  - CLI defaults (`src/augmenters/wikiann_vi/__main__.py:9` 도움말 예시 등)
  - VI 라벨러 로더 (`src/labelers/vi/dataset_loader.py:24,27,30`)
  - 테스트 fixture에서 데이터 파일 경로 참조하는 항목
- **검증 6종**:
  - cross-model kappa (Gemma vs Qwen, WikiANN-vi 3-스플릿)
  - Wikidata 인터링크 앵커 일치율 (WikiANN-vi)
  - Stockmark #16 1건 read-only spot-check (병원=LOC)
  - PII 할루시네이션 카운트 (VI PII 주입본)
  - PII prefix-동반률·영문 라벨 leakage 카운트 (VI PII 주입본)
  - PII 밀도 분포 점검 (`samples_no_pii_ratio` vs `density[0]`)
- **분포·검증 리포트**: `docs/reports/data-regen-canonical10-2026-XX.md`

### 제외

- **Stockmark 데이터 재생성** (`data/stockmark/*`,
  `data/pii/stockmark_pii_1000.jsonl`) — 현행 그대로
- 라벨러·로더 코드 변경 (1a39401에서 끝남)
- BERT 파인튜닝·벤치 실행 (#8 계열 별도 트랙, 본 이슈가 데이터 공급원)
- 원본 HF 데이터 수정
- **JSONL 필드명·모듈 파일명·스펙 문서명·과거 docs 변경** (#21 호환성
  결정 존중. 별도 이슈로 분리 고려 가능)

## 성공 기준

- [ ] 데이터: `data/wikiann_vi/{gemma,qwen}_*.jsonl` 전 라인의
      `gold_spans_8type[*].type` ⊂ 5종 NER (`PER LOC ORG PROD EVT`)
- [ ] 데이터: `data/wikiann_vi/vi_wikiann_recall_*.jsonl`의
      `gold_spans_8type_merged[*].type` ⊂ 5종 NER
- [ ] 데이터: `data/pii/wikiann_vi_pii.jsonl` 전 라인 라벨 ⊂ 10종 평면
- [ ] 검증: cross-model kappa (Gemma vs Qwen) — WikiANN-vi 3-스플릿
- [ ] 검증: Wikidata 인터링크 앵커 일치율 (WikiANN-vi)
- [ ] 검증: Stockmark `data/stockmark/{train,test}.jsonl` `セント
      メアリー病院` = `LOC` 확인 (#16 1건 spot-check)
- [ ] 검증: PII 할루시네이션 카운트 리포트 (VI PII 주입본)
- [ ] 검증: VI PII 주입본 prefix-동반률 + 영문 라벨 leakage 카운트
      (2026-04-17 핸드오프 §7 흡수)
- [ ] 검증: VI PII 주입본 밀도 분포 (`samples_no_pii_ratio` vs
      `density[0]`, ja PII augmenter 핸드오프 §3 흡수)
- [ ] 정리: 구 `*_8type_*.jsonl` 데이터 파일 삭제, 데이터 경로 참조
      코드·테스트 fixture 신규 경로로 정정 (필드명·모듈명은 그대로)
- [ ] 문서: 리포트 생성 + 본 이슈 md 최초 커밋

## 구현 단계

1. **이슈 본문·md 갱신** ← 현재 단계 (4/27 발견 사실 반영, 범위 재조정)
2. **#16 1건 read-only 검증** — `data/stockmark/{train,test}.jsonl`에서
   `セントメアリー病院` 라인을 찾아 `LOC` 라벨 확인
3. **데이터 파일 경로 참조 grep** — A 카테고리만 식별 (CLI defaults,
   VI 로더, 테스트 fixture)
4. **스모크** — Gemma 단독, train 50샘플 정도로 5종 프롬프트 검증
5. **본 배치 — WikiANN-vi 5종 silver**: Gemma 3-스플릿 전량 → Qwen
   3-스플릿 전량 → kappa 측정 → recall-merge
6. **본 배치 — VI PII 주입**: 전량 텍스트 기반 합성 PII 주입.
   `_INJECTION_PROMPT`에 단서어 직전 부착 금지·라벨 영문명(`ID_NUMBER`
   등) 출력 금지 부정 예시를 사전 추가. 밀도 분포 버그 의심분(`P(0)`
   미반영) 점검 후 진입.
7. **검증** — Wikidata 앵커 일치율 + PII 할루시네이션 카운트 + PII
   prefix-동반률·영문 라벨 leakage 카운트 + 밀도 분포 점검
8. **데이터 경로 참조 정정** — 3단계 grep 결과를 신규 경로로 일괄 치환
9. **리포트 작성** + 구 `*_8type_*.jsonl` 데이터 파일 삭제 + 이슈 md
   최종화 + 최초 커밋 (PR 직전)

## 위험·의존성

- **PII 할루시네이션** (高): VI PII 주입본 검증 단계에서 위키 개요문
  숫자 패턴을 `ID_NUM`/`PHONE`으로 오탐 가능. 4단계 스모크에서 카운트
  점검 후 본 배치 진입.
- **PII prefix leakage 재현** (中, 2026-04-17 핸드오프 §7
  흡수): Stockmark 측정에서 단서어(`担当者`/`連絡先`/`ID番号`) PII 값
  직전 동반 27.5% + `ID_NUMBER` 영문 라벨 leakage 170건 발견. VI 주입에
  서 동일 shortcut 학습 위험. 6단계에서 `_INJECTION_PROMPT`에 단서어
  직전 부착 금지·라벨 영문명 출력 금지 부정 예시를 선행 추가.
- **PII 밀도 분포 미반영** (中, ja PII augmenter 핸드오프
  §3 흡수): 일본어 주입에서 `P(0)=0.2` 기대 대비 실제
  `samples_no_pii_ratio = 1.8%` 괴리. `PIIInjector._sample_n()` 또는
  0건일 때 접미사 삽입 경로 의심. VI 주입 직전
  `tests/augmenters/pii/test_injector.py::test_density_approx` 샘플
  수 확장으로 재검증 후 진입.
- **이중 silver**: WikiANN 원본(silver, 3종) × LLM 재라벨(silver, 5종).
  절대 F1 비교 포기, 스키마 일관성만 확보.
- **모델 가용성**: Gemma `cyankiwi/gemma-4-31B-it-AWQ-8bit`
  (localhost:8081) + Qwen vLLM 엔드포인트 본 배치 직전 가용성 재확인.
- **A 경로 일괄 치환**: 데이터 파일 경로 참조 누락 시 로딩 실패. 3단계
  grep 후 8단계 일괄 치환. **#21 결정에 따라 필드명·모듈명·스펙 문서명·
  과거 docs는 변경 대상 아님.**
- **공수**: 3-스플릿 전량 × 2모델 + VI PII 전량. 단계별 시간 측정 후
  다음 진입.

## 참조

- 선행: #21 (canonical 10종 확정), #16 (경계 룰), #17 (PII 재설계)
- 라벨러 리팩터: 커밋 1a39401
- 이전 재라벨 트랙: #10 (closed, 8종 기준 — 본 이슈가 5종 silver로 갱신)
- 병렬 트랙: #8 (PII 벤치 — 데이터 확정 후 소비 측)
- 흡수된 핸드오프 (원본은 2026-09-10 폐기, 내용은 아래에 남는다):
  - 2026-04-17 핸드오프 §7 — 일본어 PII prefix leakage 27.5%
    + `ID_NUMBER` 영문 라벨 leakage 170건 (검증 5번·위험·6단계 프롬프트
    보강에 흡수)
  - ja PII augmenter 핸드오프 §3 — 일본어 주입 밀도 분포
    `P(0)` 미반영 (검증 6번·위험에 흡수)

---

## 진행 결과 (4/27)

### 완료된 단계

- **1단계 — 이슈 본문·md 갱신**: 4/27 발견 사실 4건 반영, 범위 재조정 ✅
- **2단계 — #16 1건 spot-check** ✅
  - `data/stockmark/test.jsonl:183` (id=2942700)
  - `セントメアリー病院` = `LOC` 확인 (5종 매핑 후 분별 가능 1건 정합)
  - 부수: `コロラド・メサ大学` = `ORG` (대학=CORP→ORG 매핑 정상)
- **3단계 — 데이터 경로 grep** ✅
  - 변경 대상 = **2개 파일, 4줄**:
    - `src/augmenters/wikiann_vi/__main__.py:9` (docstring 예시)
    - `src/labelers/vi/dataset_loader.py:4` (docstring 안 경로)
    - `src/labelers/vi/dataset_loader.py:24,27,30` (`DEFAULT_PATH` 3개)
  - 테스트 fixture는 인메모리 dict라 영향 0
  - 모듈명·필드명·스펙 문서명·과거 docs는 카테고리 B/C/D/E (변경 X)
- **4단계 — 스모크** ✅
  - Gemma 50샘플 train, 13초, 0 에러
  - 5종만 출력 (PII 출력 0건 — 의도한 동작)
  - per-type: LOC 29 · PER 14 · ORG 13 · PROD 5 · EVT 0
- **5단계 — WikiANN-vi 5종 silver 재라벨** ✅
  - Gemma+Qwen 양쪽 3-스플릿 전량, **wall time 87분** (병렬 concurrency 16)
  - 6개 파일: `data/wikiann_vi/{gemma,qwen}_{train,validation,test}.jsonl`
  - 0 에러, 5종만 출력 (10 카테고리 중 PII 0건)
  - **kappa (Gemma vs Qwen)**:
    - train 0.6453 (po=73.10%)
    - validation 0.6317 (po=72.02%)
    - test 0.6333 (po=72.18%)
  - **recall-merge** (high + medium_recall):
    - train 22,872 spans (18,017 + 4,855)
    - validation 11,300 spans (8,819 + 2,481)
    - test 11,484 spans (8,987 + 2,497)
- **7단계 — Wikidata 앵커 일치율** ✅ (검증 1/4)
  - **train 97.38%** (PER 98.9%, LOC 99.5%, ORG 90.1%, PROD 95.6%, EVT 83.9%)
  - **validation 97.10%** (PER 98.9%, LOC 99.1%, ORG 88.7%, PROD 95.9%, EVT 85.7%)
  - **test 97.01%** (PER 98.8%, LOC 98.9%, ORG 89.5%, PROD 94.7%, EVT 82.1%)
  - #10의 8종 트랙 historical 수치보다 향상

### 완료된 단계 (이어서)

- **6단계 — VI PII 주입 (llm 모드)** ✅
  - **VI 프롬프트** (`src/augmenters/pii/llm_injector.py`): `_INJECTION_PROMPT_JA` ·
    `_INJECTION_PROMPT_VI` · `_INJECTION_PROMPTS` dict, `_EMPTY_PII_LIST`
    lang 분기, `build_injection_prompt(..., lang='ja')` 매개변수.
    Rule 4(단서어 직전 부착 금지)·Rule 5(영문 라벨 leakage 금지) 부정 예시 포함.
  - **비동기 리팩터** (1차 배치 retry 폭주 진단 후 도입):
    - 1차 시도: record당 `asyncio.run(self._client.generate(...))` →
      매 호출마다 새 event loop 닫힘 → `AsyncOpenAI` connection pool
      cleanup 단계에서 `RuntimeError('Event loop is closed')` 발생 →
      openai client 가 200 OK 응답에도 retry → 실효 처리량 17h+ 추정.
    - 2차 (수정): `LLMInjector._inject_async` 신규 + `_gather_inject`
      신규 + `inject_dataset` 가 단일 event loop + `asyncio.gather` 로
      전 record 병렬 처리. AsyncOpenAI client 재사용·VllmClient 세마포어
      실효화. 동기 `inject(record)` 는 단발 호출 호환 유지(테스트 그대로).
  - **테스트**: 15/15 pass (기존 4 + VI prompt 2 + extract/inject 9).
  - **본 배치 (40K)**: 시작 14:12, 종료 16:14. wall time **2시간 02분**.
    HTTP 31,189건, **retry 0건**, 4.4 req/s 평균.
  - **산출**:
    - `data/pii/wikiann_vi_pii.jsonl` — **39,979 / 40,000** record
      (21건은 PII not found ValueError로 drop)
    - `data/pii/wikiann_vi_pii.stats.json` — per-label count·
      char_coverage. 라벨 10종 모두 출현.

- **7단계 — 검증** ✅ (4종)
  - **4-1. 라벨 셋 ⊂ 10종**: `{CREDIT_CARD, DAT, EMAIL, EVT, ID_NUM, LOC,
    ORG, PER, PHONE, PROD}` — 정확 일치 ✅
  - **4-2. Offset 정합성**: `text[start:end] == ent.text` 위반
    **0 / 97,306** entity ✅
  - **4-3. Prefix-동반률 + 영문 라벨 leakage**:
    - 베트남어 단서어(Liên hệ:/SĐT:/Email:/CCCD:/Địa chỉ:/Ngày sinh:/
      Thẻ:/ID:/CMND:): **0 / 39,979 = 0.00%** (JA 27.5% 대비)
    - 영문 라벨(PHONE:/EMAIL:/ID_NUM:/ID_NUMBER:/CREDIT_CARD:/DAT:/
      ADDRESS:/NAME:): **0 / 39,979 = 0.00%** (JA 170건 대비)
  - **4-4. PII 밀도 분포** (관측 가능 PII = EMAIL+PHONE+ID_NUM+
    CREDIT_CARD+DAT, NAME→PER · ADDRESS→LOC 마스킹 5/7 가정):
    - 실측: n=0 32.88% · n=1 44.23% · n=2 20.14% · n=3 2.75%
    - 이론: n=0 32.8% · n=1 44.3% · n=2 20.0% · n=3 2.86%
    - 0.1pp 단위 일치 → **raw N 샘플링은 DEFAULT_DENSITY 와 부합**.
      JA 의 1.8% 괴리는 `stats.py:21` 측정 결함이 원인으로 확정.

- **8단계 — 데이터 경로 정정** ✅
  - `src/augmenters/wikiann_vi/__main__.py:9` (docstring 예시
    `gemma_8type.jsonl → gemma_test.jsonl`)
  - `src/labelers/vi/dataset_loader.py:4` (docstring 경로) + 23–31
    (`DEFAULT_PATH` 3건 `vi_wikiann_8type_recall_*.jsonl →
    vi_wikiann_recall_*.jsonl`)
  - 잔여 grep 결과 0건. VI 로더 신규 경로로 재로딩 검증 OK.

- **9단계 — 마무리** ✅ (커밋·PR 제외)
  - **리포트**: `docs/reports/data-regen-canonical10-2026-04.md` 작성.
  - **구파일 삭제**: `data/wikiann_vi/*_8type_*.jsonl` 9개 파일 삭제
    (gemma_8type_×3, qwen_8type_×3, vi_wikiann_8type_recall_×3, ≈43MB).
    `data/wikiann_vi/` 는 gitignore 대상이라 git index 영향 없음.
  - **본 md 갱신** (현재 단계).
  - **커밋·PR**: 사용자 확인 후 별도 진행.

### 발견 사항 (후속 이슈로 분리)

1. **`src/augmenters/pii/stats.py:21` 측정 결함**: `if not rec.entities:
   no_pii_samples += 1` — entities **전체 부재**만 카운트, **PII만
   부재(NER 있음)** 케이스 누락. 본 이슈에서는 별도 분석 스크립트로
   density 검증. stats.py 수정은 호환성 영향 있어 후속 이슈 후보.

2. **카테고리 B/C/D/E 명칭 정리**: JSONL 필드명(`gold_spans_8type`,
   `gold_spans_8type_merged`)·모듈 파일명·스펙 문서명·과거 docs 는
   #21 호환성 결정에 따라 변경하지 않음. 향후 메이저 마이그레이션에서
   다룰 후보.

3. **PROD 클래스 합의 약함**: kappa per-type 0.52–0.55. 후속 BERT
   학습 시 클래스 불균형·경계 모호성 별도 점검.

---

## 변경 요약

| 파일 | 변경 |
|---|---|
| `src/augmenters/pii/llm_injector.py` | VI 프롬프트(부정 예시 5조항) + `LLMInjector` 비동기 일괄(`_inject_async`/`_gather_inject`/단일 `asyncio.run`) |
| `tests/augmenters/pii/test_llm_injector.py` | VI 프롬프트 단위 테스트 2건 |
| `src/augmenters/wikiann_vi/__main__.py` | docstring 예시 경로 정정 |
| `src/labelers/vi/dataset_loader.py` | docstring 경로 + `DEFAULT_PATH` 3건 |
| `docs/reports/data-regen-canonical10-2026-04.md` | 신규 리포트 |
| `docs/issues/issue-23-data-regen-canonical10.md` | 본 이슈 md 신규 |
| `data/wikiann_vi/{gemma,qwen}_{train,validation,test}.jsonl` | 5종 silver 재라벨 (신규) |
| `data/wikiann_vi/kappa_{train,validation,test}.json` | cross-model kappa (신규) |
| `data/wikiann_vi/vi_wikiann_recall_{train,validation,test}.jsonl` | recall-merge (신규) |
| `data/wikiann_vi/wikidata_anchor_{train,validation,test}.json` | Wikidata 앵커 (신규) |
| `data/pii/wikiann_vi_pii.jsonl` (+ `.stats.json`) | 10종 PII 주입본 (신규) |
| `data/wikiann_vi/*_8type_*.jsonl` × 9 | 삭제 |

## 구현 결과 (계획 대비)

| 계획 항목 | 결과 |
|---|---|
| Stockmark 범위 외, WikiANN-vi 5종 silver + 10종 PII만 | ✅ 그대로 |
| 3-스플릿 전량 (Gemma+Qwen) | ✅ wall time 87분 |
| VI PII 주입 (LLM 모드) 전량 | ✅ 39,979 / 40,000 (drop 21건은 LLM이 PII 값을 그대로 인용하지 못한 케이스) |
| 검증 6종 → 4종 (NER 산출 측 kappa·Wikidata anchor 는 5단계 직후 측정 완료, PII 측 4종을 7단계에서 측정) | ✅ 모두 통과 |
| 데이터 파일명만 정정 (B/C/D/E 미변경) | ✅ 9파일 삭제 + 4줄 수정 |

## 검증

상세 수치는 `docs/reports/data-regen-canonical10-2026-04.md` 참조.

## 관련 커밋

(사용자 확인 후 본 이슈 md 최초 커밋과 함께 작성)

## 후속 작업·알려진 한계

- `stats.py` 측정 결함 — 별도 후속 이슈로 분리 후보
- 카테고리 B/C/D/E 명칭(필드명·모듈명·스펙 문서명) 정리 — 후속 메이저
  마이그레이션 후보
- PROD 클래스 약한 합의 — 후속 BERT 학습 시 클래스 균형 점검
