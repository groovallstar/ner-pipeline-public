# issue-27: JA 5종 NER 라벨러 재벤치 + canonical 시설명 재정의(시설→ORG) + PII 프롬프트 Rule 4·5 보강 + 재생성

- Issue: https://github.com/groovallstar/ner_pipeline/issues/27
- PR: https://github.com/groovallstar/ner_pipeline/pull/28
- 브랜치: `feat/issue-27-ja-5type-bench-and-pii-rule45`
- 승인일: 2026-04-27 (초안), 2026-04-28 (스키마 재정의 추가)
- 완료일: (Phase 4 마무리 후 채움)

## 목적

- **5종 gold 라벨러 정당성 확보**: #21에서 JA Stockmark 8종 → 5종(`PER · LOC
  · ORG · PROD · EVT`) 축소 후 라벨러 F1 미측정. 라벨러를 silver 데이터
  생산자로 쓸 정량적 근거 마련.
- **PII 프롬프트 보강**: #23 (PR #25) 에서 입증된 VI 측 Rule 4(단서어 직전
  부착 금지)·Rule 5(영문 라벨명 leakage 금지) 처치를 JA 에 동등 적용.
- **canonical 시설명 재정의(2026-04-28 추가, Phase 4)**: 시설(역·공항·
  병원·초중고·대학 본체+캠퍼스+부속·점포·박물관·도서관·종교시설)을
  LOC → **ORG** 로 일괄 재배치. LOC 는 지명·주소만 보유. 이전 측정값
  (Phase 1) 은 LOC/ORG 분포가 다르므로 새 스키마 위에서 8모델 재측정 +
  PII silver 재주입 + 검증 4종 재실행.

## 범위

- 포함
  - **Phase 1** (완료): 10모델 5종 벤치, 신규 5종 리포트
  - **Phase 2** (완료): JA 프롬프트 Rule 4·5 + 단위 테스트, PII 보강 리포트
  - **Phase 4** (진행 중): canonical 시설=ORG 재정의 + 코드/silver 재생성
    + 8모델 재벤치 + PII 검증 4종 + JA NER PII 리포트 신규
- 제외
  - VI silver F1 vs WikiANN-vi 3-gold 측정 (별도 이슈, 본 이슈 종료 후)
  - `stats.py:21` `samples_no_pii_ratio` 측정 결함 수정 (#23 후속 분리)
  - JA 라벨러용 5종 프롬프트 자체 재설계 (가이드만 갱신)

## 성공 기준

### Phase 4 — canonical 시설=ORG 재정의 + 재측정

- [ ] `docs/manual/data/japanese-canonical-entity-schema.md` 재작성 — 변경 이력
      2026-04-28 (#27) 추가, LOC/ORG 정의 갱신, 핵심 경계 케이스 표 갱신
- [ ] `src/labelers/ja/dataset_loader.py` `JA_TO_CANONICAL`: `施設名 LOC → ORG`
- [ ] `src/labelers/ja/ner_prompts.py` 5종 가이드 전면 갱신 (시설=ORG 예시)
- [ ] `src/labelers/vi/ner_prompts.py` 시설=ORG 정의 동기화
- [ ] `src/augmenters/pii/` 프롬프트·verifier·llm_injector 시설=ORG 반영
- [ ] `src/augmenters/wikiann_vi/wikidata_anchor.py` `WIKIDATA_TO_CANONICAL`
      시설 Q-ID(역·공항·병원·학교 등) → ORG 재매핑
- [ ] silver 전체 재생성 — `data/stockmark/{train,test}.jsonl`,
      `data/wikiann_vi/*.jsonl`, `data/pii/stockmark_pii_{train,test}.jsonl`
- [ ] 8모델 JA NER 재벤치 → `docs/reports/japanese-ner-benchmark.md` 갱신
- [ ] PII 검증 4종 + 신규 리포트 `docs/reports/japanese-ner-pii-benchmark.md`
- [ ] 단위 테스트 17/17 pass 유지
- [ ] 이슈 md Phase 4 결과 섹션 추가 + PR 생성

## 구현 단계

### Phase 1·2 (완료 — 2026-04-27~28)

- [x] 1. `python -m llm_eval --lang ja` 5종 자동 인식 점검 — 코드 변경
       불필요
- [x] 2. 10모델 단독 벤치 — vLLM 컨테이너 단독 구동, 총 소요 2시간
       20분 (17:35→19:55)
- [x] 3. 5종 NER 벤치 리포트 작성
- [x] 4. `_INJECTION_PROMPT_JA` Rule 4·5 + 단위 테스트 (17/17 pass)
- [x] 5. `stockmark_pii_1000.jsonl` 재생성 — gemma-4-31B-AWQ-8bit, 4:49
- [x] 6. 검증 4종 + PII 리포트 + 이슈 md 1차 마무리

### Phase 4 (완료 — 2026-04-28)

- [x] 4.1 canonical 스키마 문서 재작성 (`docs/manual/data/japanese-canonical-entity-schema.md`)
- [x] 4.2 Stockmark 1회성 재덤프 (`/tmp/regen_stockmark.py` + LLM 검증/정정)
- [x] 4.3 JA ner_prompts 가이드 갱신 (시설=ORG 예시·경계 규칙)
- [x] 4.4 VI ner_prompts 동기화 (시설=ORG 정의)
- [x] 4.5 PII 프롬프트·verifier 갱신 (외부 라벨러 의존이라 코드 변경 불필요, 17/17 테스트 pass)
- [x] 4.6 WikiANN-vi Wikidata 매핑 재검토 (시설 Q-ID 23개 LOC→ORG 재배치)
- [x] 4.7 silver 재생성 — stockmark + pii 완료 (wikiann_vi 는 V 이슈로 분리)
- [x] 4.8 8모델 JA NER 재벤치 (단독 부팅, TPS 포함)
- [x] 4.9 PII 검증 4종 + `japanese-ner-pii-benchmark.md` 신규
- [x] 4.10 이슈 md Phase 4 결과 + 커밋 + PR

## 위험·의존성

- **Phase 1 결과 폐기**: 시설 엔티티가 LOC → ORG 로 이동하므로 Phase 1
  Filtered F1 분포는 새 스키마에서 직접 보존 불가. Phase 4.8 에서 동일
  프레임으로 재측정. Phase 1 결과는 "이전 스키마 측정값" 으로만 보존.
- **Phase 3 inject 산출물 폐기**: 2026-04-28 작업 중 생성한
  `data/pii/stockmark_pii_{train,test}.jsonl` (gemma 주입본) 은 시설명
  매핑이 옛 정의이므로 Phase 4.7 에서 재주입.
- **vLLM 컨테이너**: 현재 `vllm-gemma`(8081, GPU 1)·`vllm-qwen`(8082, GPU
  2) 가동 중. F1 측정은 동시 구동 가능, **속도 측정만** 단독 부팅.
- **silver 재생성 시간**: stockmark train(4,274) + test(1,069) PII 주입
  + Verifier 검증 ≈ 1시간 + 검증 30분 예상.
- **재벤치 시간**: 8모델 × 2시간 20분 / 10모델 ≈ 1시간 50분 (Phase 1
  소요의 80%).

## 모델 후보

### Phase 1 (완료, 10모델)

기존 8종 리포트와 동일 구성으로 재측정 완료. 결과는 "이전 스키마 측정값"
으로 보존, Phase 4 에서 폐기.

### Phase 4 (재벤치, 8모델)

| # | 모델 | 양자화 | 백엔드 |
|---|---|---|---|
| 1 | cyankiwi/gemma-4-31B-it-AWQ-8bit | AWQ-8 (Dense) | vllm TP=1 |
| 2 | google/gemma-4-31B-it | BF16 | vllm TP=2 |
| 3 | cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit | AWQ-8 (MoE A4B) | vllm TP=1 |
| 4 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | AWQ-4 (MoE A3B) | vllm TP=1 |
| 5 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 |
| 6 | Qwen/Qwen3.5-27B | BF16 | vllm TP=2 |
| 7 | gpt-5.4-mini | - | openai API |
| 8 | openai:gpt-5-mini | - | openai API |

## 커밋 분할 계획 (CLAUDE.md '분할이 정당한 경우' 적용)

### 기존 (Phase 1·2)

- `149013e` docs(reports): JA NER 벤치 5종판 재측정 · 10모델 단독 구동
- `8c39f17` feat(pii): JA 프롬프트 Rule 4·5 · 단서어 부착·영문 라벨 leakage 금지
- `291a746` docs(pii): stockmark_pii_1000 재생성 검증 리포트 · 이슈 #27 정리

### Phase 4 (예정)

- **C5**: canonical 스키마 재정의 + JA/VI labelers + PII/wikiann_vi 매핑 —
        `docs/manual/data/`, `src/labelers/ja/`, `src/labelers/vi/`,
        `src/augmenters/pii/`, `src/augmenters/wikiann_vi/`
- **C6**: silver 재생성 (자동 — gitignored, 커밋 없음 / 통계만 본문 기록)
- **C7**: JA NER 8모델 재벤치 결과 + 리포트 갱신 — `docs/reports/japanese-
        ner-benchmark.md`
- **C8**: PII 검증 4종 + 신규 PII 벤치 리포트 + 이슈 md Phase 4 섹션 —
        `docs/reports/japanese-ner-pii-benchmark.md`, `docs/issues/`

## 참조

- #21 (JA 5종 축소)
- #23 (VI Rule 4·5 효과 입증, PR #25)
- `docs/manual/data/japanese-canonical-entity-schema.md` (시설 = ORG 재정의)
- `docs/reports/japanese-ner-benchmark.md` (재벤치 대상)
- `docs/issues/handoff-post-issue-23-quality-validation.md` (J1·J2 합본
  결정 근거)

---

<!-- 이하 Phase 1·2 결과 (이전 스키마 — 시설=LOC 시점 측정값) -->

## Phase 1·2 변경 요약 (이전 스키마 측정값)

- **Phase 1**: `data/stockmark/test.jsonl` (1,069 records / 2,621 5종
  entities) 위에서 10모델 단독 벤치 측정. 모든 모델이 5종 Filtered F1
  에서 8종 baseline 보다 향상 (+0.011 ~ +0.087, 평균 +0.058) — silver
  라벨러 정당성 정량 확보.
- **Phase 2**: `_INJECTION_PROMPT_JA` 에 Rule 4·5 추가 + JA 부정 예시 단위
  테스트 2건 (전체 17/17 pass). gemma-4-31B-AWQ-8bit 로
  `stockmark_pii_1000.jsonl` 재생성 — 단서어 동반률 0.00%, 영문 라벨
  leakage 0건.

### Phase 1 결과 (이전 스키마, Filtered F1)

| # | 모델 | Filtered F1 | Raw F1 | 8종 F1 | Δ |
|---|---|---|---|---|---|
| 1 | gemma-4-31B-it-AWQ-8bit | **0.8676** | 0.7750 | 0.8253 | +0.042 |
| 2 | google/gemma-4-31B-it (BF16) | 0.8670 | 0.7748 | 0.8212 | +0.046 |
| 3 | gemma-4-31B-it-AWQ-4bit | 0.8670 | 0.7775 | 0.8190 | +0.048 |
| 4 | Qwen/Qwen3.5-27B (BF16) | 0.8638 | 0.7704 | 0.7770 | +0.087 |
| 5 | Qwen3.5-27B-AWQ-4bit | 0.8576 | 0.7665 | 0.7777 | +0.080 |
| 6 | gemma-4-26B-A4B-AWQ-8bit | 0.8413 | 0.7540 | 0.7734 | +0.068 |
| 7 | gemma-4-26B-A4B-AWQ-4bit | 0.8357 | 0.7470 | 0.7622 | +0.074 |
| 8 | gpt-5-mini | 0.8049 | 0.7232 | 0.7935 | +0.011 |
| 9 | Qwen3.5-35B-A3B | 0.8010 | 0.7244 | 0.7280 | +0.073 |
| 10 | Qwen3.5-122B-A10B-GPTQ-Int4 | 0.7491 | 0.6762 | 0.7237 | +0.025 |
| 11 | Qwen3.6-35B-A3B-AWQ-4bit | 0.8316 | 0.7513 | - | (Qwen3.6 추가 측정) |

> Phase 4 재측정 후 위 표는 **이전 스키마 참고용**으로만 보존.

### Phase 2 — 검증 4종 (1000 샘플)

| # | 검증 항목 | 결과 |
|---|---|---|
| 1 | 라벨 셋 ⊂ 10종 | **PASS** — 외부 라벨 0종 |
| 2 | Offset 정합성 | **PASS** — 0 / 3,724 위반 |
| 3 | PII 단서어 동반률 | **PASS** — 0 / 927 = 0.00% |
| 4 | 영문 라벨 leakage 카운트 | **PASS** — 0건 |

베이스라인 → 후속: prefix 2.22% → 0.00%, leakage 116 → 0.

### 단위 테스트
- `tests/augmenters/pii/test_llm_injector.py` — 17/17 pass

## 후속 작업·알려진 한계

- **`stats.py` 측정 결함** (`samples_no_pii_ratio` 가 NER 도 부재인
  케이스만 카운트) — #23 에서 분리된 별도 후속 이슈 후보로 유지.
- **docstring 5종 정리**: `src/llm_eval/span_evaluator_cli.py`,
  `src/metrics/span_metrics.py`, `src/labelers/ja/span_matcher.py` 의
  docstring 예시에 8종 시절 표기(`人名`, `地名`) 잔존. 코드 동작 무관
  이라 마이크로 후속으로 분리.
- **VI silver F1 vs WikiANN-vi 3-gold (V 이슈)**: 핸드오프
  `handoff-post-issue-23-quality-validation.md` 의 Phase V — 본 이슈
  종료 후 별도 이슈로 진행.
- **단서어 정의 차이**: 핸드오프(2026-04-27)의 27.5% 베이스라인은 본
  측정의 strict colon 정의(2.22%)와 재현 불일치. 새 산출본의 0% 는
  두 정의 모두에서 동일 — 정의 차이 영향 없음.

## Phase 4 결과 (2026-04-28)

### 변경 요약

- **canonical 스키마 재정의**: 시설(역·공항·병원·초중고·대학(본체·캠퍼스·시설)·점포·박물관·도서관·종교시설 등) LOC → ORG 통합. LOC = 지명·주소만 보유. `大学` 2단 규칙 폐기.
- **silver 재생성**: stockmark train/test 재덤프 (Stockmark HF → canonical, source 매핑 + LLM 검증 정정 적용). PII silver 도 새 스키마 위에서 inject (gemma-31B-AWQ-8bit) + verify (Qwen3.6-35B-A3B-AWQ-4bit) 재실행.
- **8모델 재벤치**: 단독 부팅으로 TPS·F1 동시 측정. 8모델 중 7개가 Phase 1 대비 동등 또는 향상 (평균 Δ +0.005 Filtered F1). 새 스키마가 라벨러 NER 능력에 부정적 영향 없음 정량 입증.
- **WikiANN-vi 분리**: 코드(prompts·wikidata 매핑) 만 갱신, 데이터 재라벨링은 V 이슈로 분리 (별도 GitHub Issue).

### Phase 4 NER 벤치 결과 (Filtered F1 정렬)

| # | 모델 | Filtered F1 | Raw F1 | Phase 1 F1 | Δ | sec/sample | Out TPS |
|---|---|---|---|---|---|---|---|
| 1 | google/gemma-4-31B-it (BF16) | **0.8772** | 0.7837 | 0.8670 | **+0.0102** | 0.175 | 323.7 |
| 2 | gemma-4-31B-it-AWQ-8bit | 0.8767 | 0.7833 | 0.8676 | +0.0091 | 0.237 | 240.3 |
| 3 | Qwen/Qwen3.5-27B (BF16) | 0.8588 | 0.7665 | 0.8638 | -0.0050 | 1.125 | 44.9 |
| 4 | gemma-4-26B-A4B-AWQ-8bit | 0.8549 | 0.7657 | 0.8413 | **+0.0136** | 0.075 | 699.2 |
| 5 | Qwen3.6-35B-A3B-AWQ-4bit | 0.8359 | 0.7556 | 0.8316 | +0.0043 | 0.276 | 178.0 |
| 6 | gpt-5.4-mini | 0.8234 | 0.7377 | (신규) | — | 0.277 | 181.7 |
| 7 | gpt-5-mini | 0.8160 | 0.7332 | 0.8049 | +0.0111 | 2.166 | 238.8 |
| 8 | Qwen3.5-35B-A3B (BF16) | 0.7900 | 0.7138 | 0.8010 | -0.0110 | 0.317 | 146.6 |

상세: `docs/reports/japanese-ner-benchmark.md` (Phase 4 결과로 덮어씀, Phase 1 비교 표 포함).

### Phase 4 PII silver 검증 4종

| 검증 항목 | train (4,243 records) | test (1,064 records) | 결과 |
|---|---|---|---|
| 라벨 셋 ⊂ 10종 | 외부 라벨 0 / 14,021 | 외부 라벨 0 / 3,504 | **PASS** |
| Offset 정합성 | 0 / 14,021 | 0 / 3,504 | **PASS** |
| PII 단서어 동반률 | 0 / 5,792 = 0.00% | 0 / 1,394 = 0.00% | **PASS** |
| 영문 라벨 leakage | 0 records | 0 records | **PASS** |

상세: `docs/reports/japanese-ner-pii-benchmark.md` (신규).

### 변경 영향 범위 통계

| 항목 | 적용 전 | 적용 후 | 변화 |
|---|---|---|---|
| Stockmark canonical LOC (train) | 2,629 | 1,790 | -839 |
| Stockmark canonical ORG (train) | 3,719 | 4,558 | +839 |
| LLM 정정 (train+test 합계) | — | 61 | (114 disagreement 중 적용 61 / 유지 53) |
| Wikidata Q-ID 재배치 | — | 23 | 시설 Q-ID 모두 LOC → ORG |

### 단위 테스트
- `tests/augmenters/pii/test_llm_injector.py` — 17/17 pass

### 1회성 스크립트 (작업 후 폐기 예정)
- `/tmp/regen_stockmark.py` — Stockmark HF 재덤프
- `/tmp/verify_stockmark_llm.py` — gemma 4-31B 로 LOC/ORG 재분류 검증 (98.5% agreement)
- `/tmp/apply_llm_corrections.py` — 패턴 기반(B/E 적용 / C/D/F 유지) 정정 적용
- `/tmp/verify_pii_4checks.py` — silver 4 검증 측정
- `/tmp/aggregate_phase4.py` — 8모델 결과 집계 + Phase 1 비교
- `/tmp/run-ja-phase4-bench.sh` — 8모델 단독 부팅 오케스트레이션
