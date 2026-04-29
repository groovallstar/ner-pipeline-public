# 핸드오프: #23 머지 후 NER/PII 품질 재검증 (3개 후속 이슈)

> 작성일: 2026-04-27
> 선행 이슈: #23 (PR #25 머지 대기 중)
> 목적: #21 5종 축소 + #23 VI 데이터 재생성으로 변화한 품질 지표를 정량
>   재측정하고, VI 작업에서 입증된 PII 프롬프트 보강 기법을 JA에 이식.

## 0. 진입 조건

- PR #25 (`feat/issue-23-data-regen-canonical10` → `develop`) 머지 완료
- `data/wikiann_vi/{gemma,qwen}_*.jsonl` 6개 + recall-merge 3개 + Wikidata
  앵커 3개 + `data/pii/wikiann_vi_pii.jsonl` (39,979 record) 존재
- `data/stockmark/{train,test}.jsonl` 5종 gold 그대로 (#21 산출)
- `_INJECTION_PROMPT_VI` (Rule 4·5 부정 예시 포함) + `LLMInjector` 비동기
  일괄 처리 코드 머지됨

## 1. 권장 진행 순서

```
[병렬 진행 — 즉시]
  J2  JA PII 프롬프트 보강 + 재생성             (~0.5일, 코드 변경 + 1000샘플 재생성)
  V   VI silver F1 vs WikiANN-vi 3-gold 측정    (~1일, 새 평가 스크립트)

[그다음 — vLLM GPU 슬롯 확보 후]
  J1  JA 5종 NER 벤치 재측정                    (~1~2일, 10모델 × 1000샘플)
```

근거: J2는 이번 PR의 직접 수확(VI Rule 4·5)을 JA에 옮기는 작업이라 코드
재사용도가 가장 높고, V는 (b) silver 신뢰도 정량화 목적상 가장 직결.
J1은 vLLM 다중 모델 서빙·전환을 요해 운영 비용이 높다.

---

## 2. 이슈 J2 — JA PII 프롬프트 보강 + `stockmark_pii_1000.jsonl` 재생성

### 배경

- 현재 `_INJECTION_PROMPT_JA` 는 부정 예시(Rule 4·5)가 **없다**.
- 그 결과 기존 `data/pii/stockmark_pii_1000.jsonl` 측정값:
  - **단서어 직전 PII 동반: 27.5%** (`担当者:` / `連絡先:` / `ID番号:`
    / `電話:` / `住所:` / `メール:` 등 직후에 PII 값이 위치)
  - **영문 라벨 leakage: 170건** (`ID_NUMBER` / `PHONE` / `EMAIL` 등 영문
    라벨명이 본문에 그대로 출력)
- VI 측에서 같은 부정 예시를 추가했더니 **각각 0/0** 으로 떨어짐 (#23 검증).
- 같은 처치를 JA에 적용해 동일 효과를 검증한다.

### 작업 단계

1. **GitHub Issue 등록**: `feat: JA PII 프롬프트 보강 + 1000샘플 재생성`,
   라벨 `area:augmenters`.
2. **브랜치**: `feat/issue-{N}-ja-pii-prompt-rule45`.
3. **코드 변경** (`src/augmenters/pii/llm_injector.py`):
   - `_INJECTION_PROMPT_JA` 에 Rule 4(단서어 직전 부착 금지) 등가 추가:
     - 부정 예시 키워드: `担当者:` · `連絡先:` · `ID番号:` · `電話番号:`
       · `住所:` · `メール:` · `クレカ番号:` · `カード番号:` · `生年月日:`
     - 자연 임베딩 예시 1~2건 (예: 「連絡先は taro@example.jp までお願い
       します」)
   - Rule 5(영문 라벨명 leakage 금지) 등가 추가:
     - 금지 라벨명: `NAME` · `PHONE` · `EMAIL` · `ID_NUM` · `ID_NUMBER`
       · `CREDIT_CARD` · `DAT` · `ADDRESS`
4. **단위 테스트** (`tests/augmenters/pii/test_llm_injector.py`): VI와 동일
   구조로 JA 프롬프트 부정 예시 검증 2건 추가.
5. **재생성**: `python -m augmenters.pii --source stockmark --lang ja
   --mode llm --inject-url http://localhost:8081/v1 --inject-model
   <gemma 또는 qwen 31B AWQ-8> --output data/pii/stockmark_pii_1000.jsonl
   --pii-max 3 --llm-concurrency 16 --seed 42`. 1000 sample, 약 5~10분.
6. **검증 4종** (#23 7단계와 동일 스크립트 재사용):
   - 라벨 셋 ⊂ 10종 평면
   - offset 정합성
   - JA 단서어 동반률 (현행 27.5% → 신규 X%)
   - 영문 라벨 leakage 카운트 (현행 170 → 신규 Y)
7. **리포트**: `docs/reports/japanese-pii-prompt-rule45-2026-04.md` 또는
   `docs/reports/japanese-pii-regenerate-{YY-MM}.md`. 표 형식 — 이전/이후
   prefix·leakage·라벨별 카운트·offset 위반 비교.
8. **이슈 md**: `docs/issues/issue-{N}-ja-pii-prompt-rule45.md` (계획·결과).

### 성공 기준

- JA 프롬프트 부정 예시 단위 테스트 2건 추가, 전체 17/17 pass
- `data/pii/stockmark_pii_1000.jsonl` 재생성, 라벨 ⊂ 10종 + offset 0 위반
- prefix-동반률 ≤ 5% (가능한 한 0% 근사) — VI 결과와 비교 가능
- 영문 라벨 leakage = 0 (절대값 — 이건 0 이 정상)
- 리포트 + 이슈 md 작성

### 위험·주의

- JA Stockmark 원본은 위키 개요문이라 실제 PII 가 0건. 합성 PII 만 평가
  대상이며, 본 작업은 **합성 분포의 자연스러움**을 보강하는 것이지 기반
  분포 자체를 바꾸지 않는다.
- 모델 선택 시 31B AWQ-8(현재 8081) 또는 Qwen3.5-27B-AWQ-4 선택 가능.
  JA 라벨러 측 F1 최상위 모델(`gemma-4-31B-it-AWQ-8bit`) 동일 사용 권장.

---

## 3. 이슈 V — VI silver F1 vs WikiANN-vi 3-gold 측정

### 배경

- VI 5종 silver(`gemma_*.jsonl` · `qwen_*.jsonl` · `vi_wikiann_recall_*.jsonl`)
  의 **절대 정확도** 는 #23 시점에 측정 안 함. 가용한 메트릭은 cross-model
  kappa(0.63~0.65) + Wikidata 앵커(97% on 60% 매핑 가능 부분).
- WikiANN-vi 원본은 PER/LOC/ORG **3종을 인간 gold(BIO)** 로 보유. 5종
  silver 중 이 3종 부분은 **직접 gold 비교 가능**.
- 목적(b): silver 를 production 학습 데이터로 쓰는 정량적 근거 제시.

### 작업 단계

1. **GitHub Issue 등록**: `feat: VI silver F1 vs WikiANN-vi 3-gold 정량
   검증`, 라벨 `area:llm-eval`.
2. **브랜치**: `feat/issue-{N}-vi-silver-quality-vs-3gold`.
3. **3-gold 추출 스크립트** (신규):
   - HF `unimelb-nlp/wikiann/vi` 의 train/validation/test 를 BIO 토큰
     라벨 → offset span 변환
   - 라벨 매핑: WikiANN-vi `B-PER`/`I-PER` → `PER` 등 (이미 ORG/LOC/PER
     3종)
   - 출력 메모리 객체 또는 임시 JSONL
4. **silver 평가 스크립트** (신규, `src/llm_eval/` 또는 단발 ad-hoc):
   - silver 산출물에서 PER/LOC/ORG 만 필터
   - `metrics/span_metrics.py::compute_offset_span_f1` 호출
   - 측정 대상:
     - Gemma silver (3-스플릿)
     - Qwen silver (3-스플릿)
     - recall-merge high+medium (3-스플릿)
     - recall-merge high-only (3-스플릿)
   - per-type F1·precision·recall + 통합 F1
5. **리포트**: `docs/reports/vietnamese-ner-silver-quality-2026-XX.md`.
   섹션 구성:
   - 1. 3-gold span F1: silver 별 / split 별 / per-type
   - 2. 기존 메트릭 재정리 (kappa·Wikidata anchor)
   - 3. 결정 기준표:
     | 측정 F1 | 의미 | 권장 학습 정책 |
     | --- | --- | --- |
     | ≥ 0.90 | silver 단독 OK | gemma_train.jsonl 단독 |
     | 0.85~0.90 | recall-merge 권장 | high+medium |
     | 0.80~0.85 | high-only 권장 | conservative |
     | < 0.80 | 재라벨 필요 | — |
6. **이슈 md**: `docs/issues/issue-{N}-vi-silver-quality-vs-3gold.md`.

### 성공 기준

- WikiANN-vi 3-gold 변환 코드 (재현 가능)
- silver 4가지 형태(Gemma/Qwen/recall+medium/recall-only) × 3-split ×
  3-type per-type F1 표 — 총 36 셀
- 기존 메트릭(kappa·앵커)과의 정합성 해석
- 리포트 + 이슈 md 작성

### 위험·주의

- WikiANN-vi 의 BIO 정렬은 token 단위라 multi-byte/diacritic 처리 시
  offset 변환 정확성 점검 필요 (`labelers/tag_aligner.py` 참고)
- 3-gold 자체도 silver 라는 비판이 있다 (WikiANN 자동 생성). 다만 인간
  검수가 일부 들어가 있어 본 프로젝트의 LLM silver 보다 상위 신뢰. 절대
  비교 시 이를 명시.

---

## 4. 이슈 J1 — JA 5종 NER 벤치 재측정

### 배경

- `docs/reports/japanese-ner-benchmark-2026-04.md` 는 **8종 시절** 측정.
  최상위 F1 0.8253(gemma-4-31B-it-AWQ-8bit).
- #21에서 8종 → 5종 축소(`PER · LOC · ORG · PROD · EVT`). 일반적으로
  클래스 축소는 경계 모호성을 줄여 F1 ↑ 경향.
- 현재 `data/stockmark/{train,test}.jsonl` 은 5종 변환 완료. **5종 라벨러
  성능은 미측정** — 이를 재측정해 #21 효과를 정량화한다.

### 작업 단계

1. **GitHub Issue 등록**: `feat: JA 5종 축소 후 NER 라벨러 벤치 재측정`,
   라벨 `area:llm-eval`.
2. **브랜치**: `feat/issue-{N}-ja-ner-bench-5type`.
3. **평가**: `python -m llm_eval --lang ja ...` 가 5종 데이터셋을 자동
   인식하는지 확인. 미지원 시 5종 변환 호환 어댑터 보강 (대부분 이미
   #21에서 정리됨).
4. **모델 후보** (JA report 1·2위 + 빠른 후보 위주, 자원 절약):
   - `cyankiwi/gemma-4-31B-it-AWQ-8bit` (이전 1위)
   - `google/gemma-4-31B-it` (BF16, 이전 2위)
   - `cyankiwi/Qwen3.5-27B-AWQ-4bit`
   - `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (속도 우선 후보)
   - `openai:gpt-5-mini` (외부 API 비교)
   - 추가 5+모델은 GPU 시간 가용성 따라 결정
5. **리포트**: `docs/reports/japanese-ner-benchmark-5type-2026-XX.md`.
   기존 JA report 와 같은 형식 + 8종↔5종 비교 컬럼:
   - per-entity F1 (5종)
   - 8종 F1 → 5종 F1 매핑
   - 클래스 매핑 영향(예: `政治的組織名+その他の組織名+法人名 →
     ORG` 통합 효과)
6. **이슈 md**: `docs/issues/issue-{N}-ja-ner-bench-5type.md`.

### 성공 기준

- 5종 기준 1000-sample test span F1 측정값 (모델별)
- 8종 0.8253 → 5종 ?? 변화 정량화
- per-entity F1 5종 × 모델 표
- 리포트 + 이슈 md 작성

### 위험·주의

- vLLM 모델 전환에 시간 소요(이미지·VRAM·서비스 재시작). 가용 슬롯
  확보 후 진입.
- 8종↔5종 직접 F1 비교는 **다른 분류 문제** 라는 점에 주의. 비교는
  "축소 효과의 trend"로만 해석.
- BF16 / GPTQ 122B 등 자원 큰 모델은 후순위. 핵심은 1위 후보 검증.

---

## 5. 진행 추적

각 이슈는 **별도 GitHub Issue + 별도 브랜치**로 진행. 본 핸드오프
문서는 3개 이슈가 모두 종료되면 머지·삭제(또는 `closed` 표기) 한다.

진입 시점에 `git status` 로 PR #25 머지 여부 우선 확인. 머지 전에
J2/V 진입은 금지(코드 충돌 우려).

## 6. 흡수된 직전 발견사항

- VI 측 prefix 27.5% / leakage 170 → 0/0 (#23 7단계, 4-3절)
- 밀도 분포 0.1pp 정합 → raw N 샘플링은 정상, JA 1.8% no-PII 는 stats.py
  측정 결함이 원인 (`stats.py:21` — 후속 별도 이슈 후보)
- 비동기 일괄 리팩터로 LLM 주입 throughput 8.5× 개선 (1차 시도 17h+
  추정 → 2차 122분)
