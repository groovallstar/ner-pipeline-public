# issue-8: WikiANN-vi 1000샘플 NER 벤치 + 베트남어 PII 혼합 측정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/8
- PR: (미생성)
- 브랜치: `feat/issue-8-vi-pii-bench`
- 승인일: 2026-04-21
- 재정렬: 2026-04-30 (선행 이슈 #10·#23·#27·#30 완료 후 잔여 작업 재산출)
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
  - 리포트 2종 (JA 패턴 미러링, 상시 갱신형 — 접미사 없음):
    - `docs/reports/vietnamese-ner-benchmark.md` (원본 NER)
    - `docs/reports/vietnamese-ner-pii-benchmark.md` (PII 혼합)
    - 메타: 기존 `docs/reports/augmentation-model-selection.md` 의 VI 섹션
      append (다국어 통합 운용)
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
- [ ] 문서 갱신: 리포트 2종 + `augmentation-model-selection.md` VI 섹션
      + `docs/issues/issue-8-*` 결과 섹션.
- [ ] 메트릭: WikiANN-vi 1000샘플 측정 시간이 일본어와 동일한 수준의
      주입·검증 2회 호출 구조에서 실용적 범위(1~2시간 이하) 유지.

## 구현 단계 (3~6)

> 선행: #10 1단계(`labelers/vi/dataset_loader.py` · `labelers/vi/*`) 완료 후 착수.

### 2026-04-30 감사: 이슈 작성 시점 대비 현재 상태

이슈 본문(2026-04-21) 이후 #10·#23·#27·#30 진행 과정에서 코드 인프라가
크게 진척되어, 원래 1·2 단계 중 다수가 이미 구현됨. 잔여 코드 작업은
verify 라벨러 lang 분기 한 곳뿐.

| 원래 요구 | 현재 상태 |
|---|---|
| `llm_eval` `--lang vi` offset_span 모드 | ✅ 본 브랜치 commit `096c9ea` |
| `llm_eval --local-file` VI 지원 | ✅ `_load_gold` + `VietnameseDatasetLoader.load_local` 기 구현 |
| `verifier.py` JA 하드코딩 제거 | ✅ `SpanLabeler` Protocol 기반 lang-agnostic (#27 후속) |
| `llm_injector.py` VI 프롬프트 분기 | ✅ `_INJECTION_PROMPT_VI` + `build_injection_prompt(lang='vi')` |
| `pii/__main__.py` verify 라벨러 lang 분기 | ✅ 본 브랜치 (`_build_verify_labeler`) |
| `generators/vi.py` EMAIL/CREDIT_CARD | ✅ `generators/base.py::generate_pii` 가 lang-agnostic fallback 처리 |

### 재정렬된 잔여 단계

- [x] 1. `llm_eval/__main__.py` VI offset_span 디스패치 (commit `096c9ea`).
- [x] 2. `pii/__main__.py` verify 라벨러 `_build_verify_labeler` 헬퍼로
      lang 분기. (`augmenters/pii` 모듈 한정 1커밋)
- [x] (추가) VI vllm 라벨러 BaseVllmLabeler 이행 (asyncio 루프 바인딩
      버그 해소, commit `09bde8c`).
- [x] (추가) VI OpenAI 라벨러 BaseOpenAILabeler 이행 (commit `0014bb5`).
- [x] 3. WikiANN-vi test 1000샘플 × **3개 모델 원본 측정 완료** (Phase 1).
      잔여 3개 (BF16/MoE) 는 vLLM 컨테이너 swap 필요 — 후속 작업 (사용자
      결정 대기). 측정 모델:
      - `vllm:cyankiwi/gemma-4-31B-it-AWQ-8bit` ✓ F1 0.876
      - `vllm:cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` ✓ F1 0.812
      - `openai:gpt-5.4-mini` ✓ F1 0.791
- [x] 4. 합성 PII 주입 + 교차 검증 (Qwen3.6-35B-A3B-AWQ-4bit verifier):
      - 모델 벤치 (test 1000): Qwen 0.920, gemma 0.914, gpt 0.782 (F1)
      - 전체 silver (40K): train 20K + valid 10K + test 10K 모두 산출
        `data/wikiann_vi/pii_{train,valid,test}.jsonl` 신뢰율 93.3%
        (confirmed 87,924 / missed 5,447 / conflict 914 / kept 40,000)
- [x] 5. 리포트 2종 + 메타 갱신:
      - `docs/reports/vietnamese-ner-benchmark.md` (원본 NER, 3 모델)
      - `docs/reports/vietnamese-ner-pii-benchmark.md` (PII 혼합)
      - `docs/reports/augmentation-model-selection.md` §2 갱신 (VI PII
        sub-section + 운영 매트릭스 PII inject 컬럼)

## 결과·검증 (2026-04-30)

### 코드 인프라
- VI labelers (vllm + openai) 모두 Base*Labeler 로 이행 — 250+ 줄 중복
  제거. asyncio 루프 바인딩 버그 해소.
- `pii/__main__.py::_build_verify_labeler` 헬퍼로 lang 분기 (ja/vi).
- `llm_eval` `_eval_mode_for_lang` 헬퍼로 vi → offset_span 매핑.

### 측정 (test 1000 샘플)

| 라운드 | 모델 | F1 | P | R |
|---|---|---:|---:|---:|
| 원본 NER | gemma-4-31B-AWQ-8bit | 0.876 | 0.812 | 0.952 |
| 원본 NER | Qwen3.6-35B-A3B-AWQ-4bit | 0.812 | 0.848 | 0.779 |
| 원본 NER | gpt-5.4-mini | 0.791 | 0.713 | 0.887 |
| PII 혼합 | Qwen3.6-35B-A3B-AWQ-4bit | 0.920 | 0.919 | 0.921 |
| PII 혼합 | gemma-4-31B-AWQ-8bit | 0.914 | 0.859 | 0.977 |
| PII 혼합 | gpt-5.4-mini | 0.782 | 0.696 | 0.892 |

### PII silver 산출 (40K 전체)

| Split | kept | confirmed | missed | conflict | 신뢰율 |
|---|---:|---:|---:|---:|---:|
| train | 20,000 | 44,122 | 2,659 | 459 | 93.4% |
| valid | 10,000 | 21,826 | 1,452 | 200 | 93.0% |
| test | 10,000 | 21,976 | 1,336 | 255 | 93.3% |
| **계** | **40,000** | **87,924** | **5,447** | **914** | **93.3%** |

산출물 `data/wikiann_vi/pii_{train,valid,test}.jsonl` (모두 `.gitignore`).
suffix-mode + Qwen3.6-35B-A3B-AWQ-4bit verifier. 산출 wall-clock 약
2시간 40분 (train 80분 + valid·test 병렬 80분).

### 잔여 / 후속

- **Phase 2 모델 라인업 확장** (3 모델, vLLM 컨테이너 swap 필요): 별도
  세션에서 진행 시 본 리포트 §1 표 확장 + Phase 1 vs Phase 2 비교 추가.
- pytest 240 passed (3 신규 테스트 포함, 회귀 0).
- ruff: 본 PR 변경 외 사전 존재 lint 17건은 별도 처리.

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

- 선행 리포트: `docs/reports/japanese-ner-benchmark.md` ·
  `docs/reports/japanese-ner-pii-benchmark.md`, 메타
  `docs/reports/augmentation-model-selection.md`
- 선행 이슈: #5 / `docs/issues/issue-5-stockmark-1000-pii-bench.md`
- 선행 이슈: #10 / `docs/issues/issue-10-vi-ner-8type-relabel.md` —
  VI loader·라벨러·8종 재라벨 작업 이전
