# issue-5: Stockmark 1000샘플 NER 벤치 + PII 혼합 측정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/5
- 브랜치: `feat/issue-5-stockmark-1000-pii-bench`
- 승인일: 2026-04-17

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
