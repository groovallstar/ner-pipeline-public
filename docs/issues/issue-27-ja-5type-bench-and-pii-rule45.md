# issue-27: JA 5종 NER 라벨러 재벤치 + PII 프롬프트 Rule 4·5 보강 + 재생성

- Issue: https://github.com/groovallstar/ner_pipeline/issues/27
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-27-ja-5type-bench-and-pii-rule45`
- 승인일: 2026-04-27
- 완료일: 2026-04-28

## 목적

- **5종 gold 라벨러 정당성 확보**: #21에서 JA Stockmark 8종 → 5종(`PER · LOC · ORG
  · PROD · EVT`) 축소 후 라벨러 F1 미측정. 라벨러를 silver 데이터 생산자로
  쓸 정량적 근거 마련.
- **PII 프롬프트 보강**: #23 (PR #25) 에서 입증된 VI 측 Rule 4(단서어 직전
  부착 금지)·Rule 5(영문 라벨명 leakage 금지) 처치를 JA 에 동등 적용.
  현행 JA 측정값(prefix 27.5% / leakage 170건) → 0 근사로 개선 검증.

## 범위

- 포함
  - `python -m llm_eval --lang ja` 5종 호환 점검 (필요 시 보강)
  - 10모델 × stockmark 5종 test 1000샘플 span F1 측정 (vLLM 단독 구동)
  - 신규 5종 벤치 리포트 (기존 8종 리포트 제거 후 대체)
  - `_INJECTION_PROMPT_JA` Rule 4·5 추가 + 단위 테스트
  - `data/pii/stockmark_pii_1000.jsonl` 재생성 + 검증 4종
  - PII 프롬프트 보강 리포트
- 제외
  - VI silver F1 vs WikiANN-vi 3-gold 측정 (핸드오프의 V — 별도 이슈)
  - `stats.py:21` `samples_no_pii_ratio` 측정 결함 수정 (#23 후속 분리)
  - JA 라벨러용 5종 프롬프트 자체 재설계 (현 프롬프트 유지)

## 성공 기준

### Phase 1 — 5종 NER 라벨러 벤치
- [ ] 10모델 전부 × stockmark 5종 test 1000샘플 span F1 측정 완료
- [ ] vLLM 컨테이너 단독 구동 (한 번에 1모델, 속도 측정 오염 방지)
- [ ] per-entity F1 (5종) + 8종↔5종 매핑 비교 컬럼
- [ ] 기존 `docs/reports/japanese-ner-benchmark.md` 제거
- [ ] 신규 5종판 리포트 생성 (파일명 추후 결정)

### Phase 2 — PII 프롬프트 Rule 4·5 + 재생성
- [ ] `_INJECTION_PROMPT_JA` Rule 4·5 추가
- [ ] 단위 테스트 17/17 pass (JA 부정 예시 검증 2건 추가)
- [ ] `data/pii/stockmark_pii_1000.jsonl` 재생성 — 라벨 셋 ⊂ 10종, offset
      0 위반
- [ ] 단서어 동반률 ≤ 5% (가능한 0% 근사), 영문 라벨 leakage = 0
- [ ] PII 보강 리포트 (`docs/reports/japanese-pii-prompt-rule45-2026-04.md`)
- [ ] 이슈 md 5단계 섹션 작성 + PR 생성

## 구현 단계 (6단계)

- [ ] 1. `python -m llm_eval --lang ja` 가 5종 stockmark 자동 인식하는지
       점검. 미지원 시 5종 호환 어댑터 보강 (대부분 #21에서 정리됐을 것)
- [ ] 2. 10모델 단독 벤치 — vLLM 컨테이너 한 번에 1개씩 순차 구동·측정·내림
- [ ] 3. 5종 NER 벤치 리포트 작성 (기존 8종 리포트 제거, 신규 5종판 생성)
- [ ] 4. `_INJECTION_PROMPT_JA` Rule 4·5 추가 + 단위 테스트 2건
- [ ] 5. 사용자가 (3) 리포트 검토 후 재생성 모델 선택 → 1000 샘플 재생성
- [ ] 6. 검증 4종 + PII 리포트 + 이슈 md 마무리

## 위험·의존성

- **vLLM 슬롯**: 10모델 컨테이너 한 번에 1개씩 순차 구동. 현재 gemma4
  컨테이너 가동 중 — 측정 시작 전 모두 내리고 모델별 단독 부팅 필요.
  운영 비용: 1~2일 (이미지 전환·VRAM 로드·서비스 재시작 포함).
- **재생성 모델 선택 게이트**: 5단계는 (3) 리포트 검토 후 사용자가 직접
  결정. Phase 1 종료 시점에 일시 정지하고 결과 공유.
- **8종↔5종 직접 비교 한계**: 다른 분류 문제이므로 F1 비교는
  "축소 효과의 trend" 로만 해석. per-entity 매핑 컬럼은 참고용.
- **GPU 경합 금지**: 사용자 지시(2026-04-16 측정 지침) — vLLM 동시 구동
  시 속도 측정 오염. 본 이슈에서도 동일 원칙.

## 모델 후보 (10종 전부 — 기존 8종 리포트와 동일 구성)

| # | 모델 | 양자화 | 백엔드 |
|---|---|---|---|
| 1 | cyankiwi/gemma-4-31B-it-AWQ-8bit | AWQ-8 (Dense) | vllm TP=1 |
| 2 | google/gemma-4-31B-it | BF16 | vllm TP=2 |
| 3 | cyankiwi/gemma-4-31B-it-AWQ-4bit | AWQ-4 (Dense) | vllm TP=1 |
| 4 | openai:gpt-5-mini | - | openai API |
| 5 | cyankiwi/Qwen3.5-27B-AWQ-4bit | AWQ-4 (Dense) | vllm TP=1 |
| 6 | Qwen/Qwen3.5-27B | BF16 | vllm TP=2 |
| 7 | cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit | AWQ-8 (MoE A4B) | vllm TP=1 |
| 8 | cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit | AWQ-4 (MoE A4B) | vllm TP=1 |
| 9 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 |
| 10 | Qwen/Qwen3.5-122B-A10B-GPTQ | GPTQ-Int4 (MoE A10B) | vllm TP=2 |

## 커밋 분할 계획 (CLAUDE.md '분할이 정당한 경우' 적용)

서로 다른 관심사·독립 revert 가능 단위로 4커밋 예정.

- **C1**: (있다면) 평가 코드 5종 호환 보강 — `src/llm_eval/`
- **C2**: 5종 벤치 결과 + 신규 리포트 (기존 8종 리포트 제거 포함) —
       `docs/reports/`, `results/`
- **C3**: 프롬프트 Rule 4·5 + 단위 테스트 — `src/augmenters/pii/llm_injector.py`,
       `tests/`
- **C4**: 재생성 데이터 + 검증 + PII 리포트 + 이슈 md 5단계 섹션 —
       `data/pii/`, `docs/reports/`, `docs/issues/`

## 참조

- #21 (JA 5종 축소)
- #23 (VI Rule 4·5 효과 입증, PR #25)
- `docs/reports/japanese-ner-benchmark.md` (8종 baseline, 본 이슈에서
  제거 후 5종판으로 대체)
- `docs/issues/handoff-post-issue-23-quality-validation.md` (J1·J2 합본
  결정 근거)

---

<!-- 이하 5단계(PR 직전)에서 추가 -->

## 변경 요약

- **Phase 1**: `data/stockmark/test.jsonl` (1,069 records / 2,621 5종
  entities) 위에서 10모델 단독 벤치 측정. vLLM 컨테이너 한 번에 1개씩
  부팅·측정·정지하여 GPU 경합 0. Raw F1 (10종 프롬프트 그대로) 과
  Filtered F1 (예측에서 비-NER PII 5종 제외 후 재계산) 두 지표를 모두
  보고. 모든 모델이 5종 Filtered F1 에서 8종 baseline 보다 향상
  (+0.011 ~ +0.087, 평균 +0.058) — silver 라벨러 정당성 정량 확보.
- **Phase 2**: `_INJECTION_PROMPT_JA` 에 Rule 4(단서어 직전 부착 금지)·
  Rule 5(영문 라벨명 leakage 금지) 추가 + JA 부정 예시 단위 테스트 2건
  (전체 17/17 pass). Phase 1 1위 모델(`gemma-4-31B-it-AWQ-8bit`)로
  `data/pii/stockmark_pii_1000.jsonl` 재생성 (997 records, 4분 49초).
  검증 4종 모두 PASS — 단서어 동반률 0.00%, 영문 라벨 leakage 0건.

## 구현 결과 (계획 대비)

- [x] 1. `python -m llm_eval --lang ja` 5종 자동 인식 점검 — 코드 변경
       불필요. JapaneseDatasetLoader, compute_offset_span_f1, span_matcher,
       report.print_table 모두 5종 정합. 잔존 사항: docstring 일부에 8종
       시절 예시(`人名`, `地名`) 남아있으나 코드 동작 무관 → 본 이슈
       범위 외, 마이크로 후속.
- [x] 2. 10모델 단독 벤치 — vLLM 컨테이너 단독 구동, 총 소요 2시간
       20분 (17:35→19:55). 모든 결과 `results/ja-5type-bench-2026-04/`
       (gitignored) 에 누적.
- [x] 3. 5종 NER 벤치 리포트 작성 — 기존 `japanese-ner-benchmark.md`
       (8종) 덮어쓰기로 5종판 신규 생성 (245줄).
- [x] 4. `_INJECTION_PROMPT_JA` Rule 4·5 + 단위 테스트 (17/17 pass).
- [x] 5. `stockmark_pii_1000.jsonl` 재생성 — 사용자가 Phase 1 결과
       검토 후 1위 모델(gemma-4-31B-it-AWQ-8bit) 선택, 4:49 소요.
- [x] 6. 검증 4종 + PII 리포트 (`docs/reports/japanese-pii-prompt-rule45-
       2026-04.md`) + 이슈 md 마무리.

## 검증

### Phase 1 — 5종 NER 벤치 (Filtered F1 기준)

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

상세: `docs/reports/japanese-ner-benchmark.md`.

### Phase 2 — PII 재생성 검증 4종

| # | 검증 항목 | 결과 |
|---|---|---|
| 1 | 라벨 셋 ⊂ 10종 | **PASS** — 외부 라벨 0종 |
| 2 | Offset 정합성 | **PASS** — 0 / 3,724 위반 |
| 3 | PII 단서어 동반률 | **PASS** — 0 / 927 = 0.00% |
| 4 | 영문 라벨 leakage 카운트 | **PASS** — 0건 |

베이스라인 → 후속: prefix 2.22% → 0.00%, leakage 116 → 0. 상세:
`docs/reports/japanese-pii-prompt-rule45-2026-04.md`.

### 단위 테스트
- `tests/augmenters/pii/test_llm_injector.py` — 17/17 pass

```
$ python -m pytest tests/augmenters/pii/test_llm_injector.py -v
============================== 17 passed in 0.04s ==============================
```

## 관련 커밋

- `149013e` docs(reports): JA NER 벤치 5종판 재측정 · 10모델 단독 구동
- `8c39f17` feat(pii): JA 프롬프트 Rule 4·5 · 단서어 부착·영문 라벨 leakage 금지
- `291a746` docs(pii): stockmark_pii_1000 재생성 검증 리포트 · 이슈 #27 정리

## 후속 작업·알려진 한계

- **`stats.py` 측정 결함** (`samples_no_pii_ratio` 가 NER 도 부재인
  케이스만 카운트) — #23 에서 분리된 별도 후속 이슈 후보로 유지.
- **docstring 5종 정리**: `src/llm_eval/span_evaluator_cli.py`,
  `src/metrics/span_metrics.py`, `src/labelers/ja/span_matcher.py` 의
  docstring 예시에 8종 시절 표기(`人名`, `地名`) 잔존. 코드 동작 무관
  이라 마이크로 후속으로 분리.
- **VI silver F1 vs WikiANN-vi 3-gold (V 이슈)**: 핸드오프
  `handoff-post-issue-23-quality-validation.md` 의 Phase V — 별도 이슈로
  진행. 이 핸드오프 문서는 V 종료 시 머지·삭제.
- **단서어 정의 차이**: 핸드오프(2026-04-27)의 27.5% 베이스라인은 본
  측정의 strict colon 정의(2.22%)와 재현 불일치. 새 산출본의 0% 는
  두 정의 모두에서 동일 — 정의 차이 영향 없음.
