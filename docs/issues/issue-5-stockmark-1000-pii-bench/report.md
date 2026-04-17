# issue-5 결과 보고서

- Issue: https://github.com/groovallstar/ner_pipeline/issues/5
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-5-stockmark-1000-pii-bench`
- 완료일: 2026-04-17

## 변경 요약

- `benchmark_runner`의 offset-span 경로를 `asyncio.Semaphore` 기반 sample
  단위 병렬로 전환하고, vLLM/OpenAI labeler에 `alabel_spans()` async API를
  추가해 외부 event loop 내에서도 호출 가능하도록 구조 정리
- `augmenters.pii` CLI에 `--inject-url/model`, `--verify-url/model` 분리
  인자를 추가해 주입 모델과 교차 검증 모델을 독립적으로 지정할 수 있음
- Stockmark 원본 1000샘플 × 9개 모델, Stockmark+PII 993샘플 × 8개 모델을
  측정하고 `docs/entities/languages/`에 리포트 2종 정착

## 구현 결과 (계획 대비)

- [x] 1. `_run_offset_span_async` 병렬화 + `alabel_spans` async API (완료)
- [x] 2. `OPENAI_REASONING_EFFORT` env 지원 (완료)
- [x] 3. `augmenters.pii` inject/verify LLM 분리 CLI (완료)
- [x] 4. Stockmark 원본 9개 측정 (완료)
- [x] 5. Stockmark+PII 8개 측정 (완료)
- [x] 6. 리포트 `docs/entities/languages/` 이동 (완료)

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
    `docs/entities/languages/japanese-ner-pii-benchmark-2026-04.md` 상단 기재

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
