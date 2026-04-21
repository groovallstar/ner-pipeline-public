# issue-5: Stockmark 1000샘플 NER 벤치 + PII 혼합 측정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/5
- PR: https://github.com/groovallstar/ner_pipeline/pull/7
- 브랜치: `feat/issue-5-stockmark-1000-pii-bench`
- 승인일: 2026-04-17
- 완료일: 2026-04-17

## 목적

- 실제 NER 태깅은 BERT 기반 모델이 수행한다. LLM은 학습 데이터 파이프라인에서
  (1) 문장 내 개인정보가 자연스럽게 삽입되도록 합성 주입 생성기,
  (2) 공개 데이터셋 외 수급된 원시 텍스트에 대한 NER 라벨러 역할을 맡는다.
- LLM 라벨러의 품질을 정량화하기 위해 Stockmark 원본 + PII 주입 혼합 데이터셋
  양쪽으로 다중 모델 벤치마크를 수행한다.
- 수급 데이터 라벨링은 2-way 교차 검증 방식을 사용한다: 한 LLM이 라벨링한
  결과를 다른 LLM이 재검증하고, 합의된 스팬만 채택한다.

## 범위

- 포함:
  - `src/llm_eval/benchmark_runner.py` sample 단위 병렬화 (asyncio.Semaphore)
  - `src/llm_eval/__main__.py` CLI (`--concurrency`, `--local-file` 전달)
  - `src/labelers/base_openai_labeler.py`, `base_vllm_labeler.py` async
    `alabel_spans` API + `OPENAI_REASONING_EFFORT` env
  - `src/augmenters/pii/__main__.py` 주입/검증 LLM 분리 CLI
    (`--inject-url/model`, `--verify-url/model`)
  - Stockmark 원본 1000샘플 × 9개 모델, Stockmark+PII 993샘플 × 8개 모델 측정
  - 리포트 2종 `docs/reports/japanese-ner-*-2026-04.md`
- 제외:
  - 원시 벤치 JSON (`results/`가 gitignore 대상) — 로컬 보관
  - PII 데이터셋 JSONL (`data/`가 gitignore 대상) — 로컬 보관, 생성 명령만 리포트에 기록
  - 한국어·베트남어 동일 벤치 (후속 이슈로 분리)

## 성공 기준

- [x] 테스트: 기존 pytest 회귀 없음
- [x] 문서 갱신: `docs/reports/` 리포트 2종 추가
- [x] 메트릭: gpt-5-mini 1000샘플 측정 1시간 이하 달성
  (2:25:56 → 0:34:20, 4.25배 가속)

## 구현 단계 (3~6)

- [x] 1. `benchmark_runner._run_offset_span_async` sample 병렬화 + async
      labeler API (`alabel_spans`) 추가
- [x] 2. `OPENAI_REASONING_EFFORT` env → chat.completions `reasoning_effort`
      전달 (sync/async 양쪽)
- [x] 3. `augmenters.pii` inject/verify URL·model 분리 CLI
- [x] 4. Stockmark 원본 9개 모델 × 1000샘플 측정
- [x] 5. Stockmark+PII 8개 모델 × 993샘플 측정
- [x] 6. 리포트 2종 작성 후 `docs/reports/` 이동

## 위험·의존성

- vLLM 서버 (TP=1/TP=2 두 구성) 가용성 — 측정 중 메모리 부족으로 AWQ/GPTQ
  양자화 모델만 일부 구성에서 실행 가능
- OpenAI API 키 + gpt-5-mini reasoning 지원 — `OPENAI_REASONING_EFFORT=medium`
  기본, minimal 대비 F1 +10%p 향상 확인
- PII 주입 품질은 주입 LLM(gemma-26B-AWQ) 프롬프트에 의존. 교차 검증
  (Qwen-27B-AWQ, policy=drop_span)으로 허위 양성 스팬 제거

---

## 변경 요약

- `benchmark_runner`의 offset-span 경로를 `asyncio.Semaphore` 기반 sample
  단위 병렬로 전환하고, vLLM/OpenAI labeler에 `alabel_spans()` async API를
  추가해 외부 event loop 내에서도 호출 가능하도록 구조 정리
- `augmenters.pii` CLI에 `--inject-url/model`, `--verify-url/model` 분리
  인자를 추가해 주입 모델과 교차 검증 모델을 독립적으로 지정할 수 있음
- Stockmark 원본 1000샘플 × 9개 모델, Stockmark+PII 993샘플 × 8개 모델을
  측정하고 `docs/reports/`에 리포트 2종 정착

## 구현 결과 (계획 대비)

- [x] 1. `_run_offset_span_async` 병렬화 + `alabel_spans` async API (완료)
- [x] 2. `OPENAI_REASONING_EFFORT` env 지원 (완료)
- [x] 3. `augmenters.pii` inject/verify LLM 분리 CLI (완료)
- [x] 4. Stockmark 원본 9개 측정 (완료)
- [x] 5. Stockmark+PII 8개 측정 (완료)
- [x] 6. 리포트 `docs/reports/` 이동 (완료)

## 검증

- 테스트: 기존 pytest 회귀 없음 (병렬화는 기존 `_chat`, `match_spans`
  재사용만으로 구현)
- 메트릭 (gpt-5-mini, 1000샘플):
  - 직렬: 8.756 sec/sample, 2:25:56
  - 병렬=32: 2.060 sec/sample, 0:34:20 — 4.25배 가속
- 품질 메트릭 (Stockmark 원본 F1):
  - 1위 gemma-4-31B-it 0.8212 (BF16, TP=2)
  - 속도 1위 gemma-4-26B-A4B-AWQ 1000샘플 59초 (TP=1)
- 품질 메트릭 (Stockmark+PII F1):
  - 1위 gemma-4-31B-it 0.8955
  - gpt-5-mini 안전 필터로 PII 7종 전부 F1=0 (부적합 확인)
- 리뷰어 확인 사항:
  - 원시 벤치 JSON은 `results/`(gitignore 대상)에 로컬 보관, 커밋하지 않음
  - PII 데이터셋 JSONL은 `data/`(gitignore 대상)에 로컬 보관, 재현 명령은
    `docs/reports/japanese-ner-pii-benchmark-2026-04.md` 상단 기재

## 관련 커밋

- `d695bc8`: feat(llm_eval): sample 단위 병렬 벤치 러너 + reasoning_effort env
- `7c3452c`: feat(augmenters/pii): 주입·검증 LLM 분리 지정 지원
- `5cc8329`: docs(entities/ja): Stockmark 1000샘플 벤치 리포트 2종 추가

## 후속 작업·알려진 한계

- gpt-5-mini PII 미인식 원인 심층 분석 (프롬프트 개선 또는 structured output
  모드) — 별도 이슈
- vLLM 모델 4개(BF16)를 병렬=32로 재측정해 일관된 속도 비교표 확보
- verifier 경로에도 sample 단위 병렬화 적용 (현재 직렬, 1시간+ 소요)
- `reasoning_effort="low"` 절충 실험 (minimal보다 품질 보존, medium보다 빠름)
- 한국어(KLUE)·베트남어 동일 벤치마크 셋업 — 별도 이슈

## 부록: 이슈 문서 규칙 변경 (2026-04-20)

issue-5 마무리 과정에서 `docs/issues/` 운영 규칙을 아래와 같이 변경하고,
이 파일이 신규 규칙 적용 1호 샘플로 재구성됐다.

- 기존: `docs/issues/issue-{N}-{slug}/plan.md` + `report.md` 2-파일 구조,
  3단계에서 `plan.md`를 먼저 커밋 → 5단계에서 `report.md` 추가 커밋
- 변경: `docs/issues/issue-{N}-{slug}.md` 단일 파일 구조, 계획 단계에서는
  로컬 기록만 하고 **커밋하지 않음** → 구현 완료 후 같은 파일에 결과
  섹션을 덧붙이고 계획~구현 흐름 정합성을 확인한 뒤 **이때 최초 커밋**
- 반영 위치: `docs/issues/README.md` 템플릿·작성 시점 표, 루트
  `CLAUDE.md` "이슈 문서 구조"·"이슈 진행 절차" 섹션
