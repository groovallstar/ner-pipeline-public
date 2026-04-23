# issue-8: WikiANN-vi 1000샘플 NER 벤치 + 베트남어 PII 혼합 측정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/8
- PR: (미생성)
- 브랜치: `feat/issue-8-vi-wikiann-pii-bench` (예정)
- 승인일: 2026-04-21
- 완료일: -

## 목적

- issue-5(일본어)와 동일한 **주입·검증 LLM 상위 2종 선정 파이프라인**을
  베트남어(WikiANN-vi)로 확장한다.
- 베트남어 BERT PII 파인튜닝(후속 이슈)에 투입할 **합성 학습 데이터 JSONL
  생산 경로**(주입 LLM → 검증 LLM → 스팬 확정)를 정착시킨다.
- 원본 엔티티 3종(PER/LOC/ORG) 기반선과 합성 PII 6종 포함 혼합 F1을
  함께 측정해, 주입 과정이 기존 엔티티를 얼마나 훼손하는지 분리 해석한다.

> **범위 조정 (2026-04-21)**: VI loader 및 VI 라벨러 4종은 #10(8종
> 재라벨 선행 이슈)로 이전했다. 본 이슈는 #10의 loader·라벨러 결과물을
> 전제로 PII 주입·벤치·리포트에 집중한다.
>
> **선행 의존성 해소 (2026-04-22)**: #10 완료(커밋 a773ac6, 039839c 외).
> `labelers/vi/dataset_loader.py` 및 VI 라벨러 4종이 `feat/issue-10-vi-ner-
> 8type-relabel` 브랜치에서 구현·테스트 완료. 본 이슈 착수 전 #10 PR 머지
> 후 최신 develop을 베이스로 `feat/issue-8-vi-wikiann-pii-bench` 재분기 권장.

## 범위

- 포함:
  - `src/llm_eval/__main__.py` — `--lang vi` 경로에 `eval_mode=offset_span`
    옵션과 `--local-file` 지원 추가. 기존 BIO 경로는 유지(회귀 없음).
  - `src/augmenters/pii/verifier.py` — JA 하드코딩 제거, `lang` 파라미터로
    `labelers.{ja,vi}.vllm_ner_labeler` 선택.
  - `src/augmenters/pii/llm_injector.py` — `_INJECTION_PROMPT_VI` 추가,
    `build_injection_prompt(..., lang='vi')` 분기. 예시 문장·PII 트리거
    phrase는 베트남어 자연 문맥("Liên hệ", "SĐT", "Địa chỉ" 등) 사용.
  - `src/augmenters/pii/generators/vi.py` — EMAIL·CREDIT_CARD 생성기 추가
    또는 공용 `generate_pii` fallback 경로 확인.
  - 벤치 실행: WikiANN-vi test 1000샘플 × 주요 모델 5~8종 (원본 + PII).
  - 리포트 3종 `docs/reports/vietnamese-ner-*-2026-04.md`
    (원본 / PII 혼합 / 메타).
- 제외:
  - `src/labelers/vi/dataset_loader.py` 및 `src/labelers/vi/*` 라벨러 —
    **#10로 이전**.
  - 8종 스키마 재라벨 작업 — **#10**.
  - 원시 벤치 JSON(`results/`)·PII JSONL(`data/`) 커밋 — 로컬 보관.
  - 규칙 기반 vs LLM 기반 학습 데이터 BERT 파인튜닝 비교 — 별도 이슈.
  - VLSP2018/PhoNER 등 대체 데이터셋 도입.

## 성공 기준

- [ ] 테스트: 기존 pytest 회귀 없음 (#10의 loader·라벨러 단위 테스트는
      그쪽 이슈에서 담당).
- [ ] 기능: `python -m llm_eval --lang vi --local-file data/pii/vi_wikiann_pii.jsonl`
      경로로 offset-span F1 측정 가능.
- [ ] 기능: `python -m ner.augmenters.pii --lang vi --source hf --hf-name unimelb-nlp/wikiann --verify vllm`
      end-to-end 동작.
- [ ] 문서 갱신: 리포트 3종 + `docs/issues/issue-8-*` 결과 섹션.
- [ ] 메트릭: WikiANN-vi 1000샘플 측정 시간이 일본어와 동일한 수준의
      주입·검증 2회 호출 구조에서 실용적 범위(1~2시간 이하) 유지.

## 구현 단계 (3~6)

> 선행: #10 1단계(`labelers/vi/dataset_loader.py` · `labelers/vi/*`) 완료 후 착수.

- [ ] 1. `llm_eval/__main__.py` · `benchmark_runner.py` VI offset-span 경로
      배선, `--local-file` VI 지원. JA 경로 동일 스위치 재사용.
- [ ] 2. `augmenters/pii/verifier.py` 랭 파라미터화 +
      `augmenters/pii/llm_injector.py` VI 프롬프트 분기 +
      `generators/vi.py` 누락 PII 추가.
- [ ] 3. WikiANN-vi 원본 1000샘플 × 5~8개 모델 측정
      (gemma-4-31B 계열 AWQ / Qwen3.5-27B / gpt-5-mini 최소 포함).
- [ ] 4. 상위 모델로 PII 주입 + 교차 검증 JSONL 생성 →
      혼합 1000샘플 × 동일 모델 세트 측정.
- [ ] 5. `docs/reports/vietnamese-ner-benchmark-2026-04.md` ·
      `vietnamese-ner-pii-benchmark-2026-04.md` ·
      `vietnamese-pii-model-selection-2026-04.md` 작성.

## 위험·의존성

- **선행 의존**: #10의 VI loader·라벨러가 반드시 먼저 완료되어야 본
  이슈의 1~4단계가 동작한다.
- **WikiANN 품질**: 자동 생성 실버 라벨 — 일부 잡음 포함. 측정 절대값보다
  동일 셋 내 모델 간 상대 순위 해석에 집중한다.
- **VI 주입 프롬프트 신뢰도**: JA 프롬프트 번역만으로는 자연스러움이
  떨어질 수 있음 — JA와 동일하게 `policy=drop_span`으로 허위 양성 억제.
- **모델 가용성**: 일본어 벤치와 동일한 vLLM TP=1/TP=2 구성 제약.
- **PII 생성기 공백**: `generators/vi.py`에 EMAIL·CREDIT_CARD 없음.
  기본 로케일 무관 생성기 fallback 동작 확인 또는 VI 전용 추가.

## 참고

- 선행 리포트: `docs/reports/japanese-ner-*-2026-04.md`, 메타
  `japanese-pii-model-selection-2026-04.md`
- 선행 이슈: #5 / `docs/issues/issue-5-stockmark-1000-pii-bench.md`
- 선행 이슈: #10 / `docs/issues/issue-10-vi-ner-8type-relabel.md` —
  VI loader·라벨러·8종 재라벨 작업 이전
