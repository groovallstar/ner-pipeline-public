# 베트남어 BERT NER 분류기 — 실험 히스토리 상세

- 현 production 요약·영구 인용 표: `docs/reports/vietnamese-bert-classifier-benchmark.md`
- 측정 프로토콜: stratified 5-fold pooled micro-average char-offset span F1
  (`seed=42`, 중복 제거 후 38,371 행)
- 본 문서는 phase별 `직전상태 → 시도 → 결과 → 잔여한계` 연대기. phase별 전체
  P/R/F1·검증·refuter는 각 `docs/issues/issue-N` 에, 현 수치는 benchmark.md 에.
- **gold 계보**: v1 `pii_all.jsonl`(#103) → v2 `pii_all_v2.jsonl`(#108 re-silver)
  → v3 `pii_all_v3.jsonl`(#112). benchmark.md 표 baseline 은 **v1 불변**.

| phase | 이슈 | gold | overall(phobert/xlm-r) | 핵심 |
|---|---|---|---|---|
| 0 | #103·#106 | v1 | 0.9461 / 0.9459 | 누출 제거 + offset-trim + PhoBERT |
| 1 | #108 | v2 | 0.9529 / 0.9543 | PROD 천장 §3 re-silver(+5~7.5pp), EVT 회귀 |
| 2 | #112 | v3 | 0.9513 / 0.9525 | EVT 회귀 해결(recall_strict_evt), PROD give-back |

> **⚠️ 위 overall·NER 수치는 원문 cross-fold 누출 인플레 포함(#124)**: full-text
> dedup 이 원문 중복을 못 막아 NER 5종 절대값이 +1.1~1.4pp(EVT/ORG/PROD +3~5pp)
> 부풀려져 있다. PII 5종·**phase 간 델타는 유효**(같은 누출이 v1/v2/v3 에 동일
> 적재). 정량·정정: benchmark.md §원문 누출 정정 (#124).

## Phase 0 — 캐노니컬 5-fold 베이스라인 확립 (#103, 추론비용 #106)

### 직전 상태
- 구 production 표(0.8985, sweep, v1~v4 ablation)에 두 혼입: fast-tokenizer
  offset 정렬 버그 + WikiANN-vi 내재 train/test 누출.

### 본 단계에서 시도
- **누출 제거(불완전 — #124 정정)**: 40,000→**38,371**(−1,629) "원문 키 중복
  제거". 그러나 원본 WikiANN-vi(40,000행)의 유니크 원문은 **29,343개뿐**(#124
  측정)인데 1,629행만 제거 → 실제로는
  *주입 후* 전체텍스트 중복만 제거됐고 'cross-split 중복 0' 재검증도 주입 텍스트
  (행마다 유니크) 기준이라 **원문 cross-fold 누출이 남았다**(test 행 ~30%). 본
  phase 와 #108/#112 의 **NER 5종 절대값은 +1.1~1.4pp(EVT/ORG/PROD +3~5pp)
  인플레** — 상세·정정은 benchmark.md §원문 누출 정정 (#124).
- **offset-trim 정렬 수정**(`data_utils._encode_vi`): SentencePiece `▁` 선행
  공백·숫자 entity 후행 부호 흡착 교정 → **mmBERT 0.53→0.94 구제**, XLM-R·
  CafeBERT 정렬 0.966→1.0.
- **PhoBERT 통합**(`_encode_phobert`+`pyvi`): 단어분절 char-span 정렬(왕복 0.996).
- 인코더 5종 stratified 5-fold sweep. (#106: 추론 비용 비교 — phobert 메모리
  41%↓·단문 레이턴시 20%↓로 정확도 동률을 비용축에서 결판.)

### 결과 (v1 gold, pooled strict)
- 상위 무승부: **phobert 0.9461 ≈ xlm-r-base 0.9459**. CafeBERT 0.9367(VI
  continued-pretrain 무이득), mmBERT 0.9395, xlm-r-large 0.8364(fold0 all-O 붕괴).
- 천장: **PROD 0.717/0.710, EVT 0.792/0.803**(저support·silver 노이즈). PII 5종
  0.98~1.00 포화, PER/LOC/ORG 0.88~0.93.

### 잔여 한계 → 다음 가설
- PROD/EVT 천장이 (a) silver 노이즈 / (b) 모델 한계 / (c) schema 미규정 중
  무엇인지 미규명 → #108 소규모 감사로 결판.

## Phase 1 — PROD/EVT 천장 §3 감사 + 프롬프트 정렬 re-silver (#108)

### 직전 가설
- 천장 원인 미상. JA #84/#85의 canonical §3 회색지대 기준을 VI gold 생성에
  이식 가능한지 미검증(`wikiann_vi/prompts.py`는 §3 기본만, 3.1/3.2/3.3 누락).

### 본 단계에서 시도
- **§3 기준 gold 감사**(fold0, gemma-4-31B 판정, 272건): 불일치를 silver오류/
  모델오류/schema-갭 3분류 + IAA(temp0 vs temp0.7).
- **VI 프롬프트 §3 정렬**: 서비스(`dịch vụ`) 제거·법령/다년 process=비-entity·
  창작물 positive few-shot — 재라벨·LLM 라벨러 4사이트.
- **격리 re-silver**(`scripts/resilver_vi_isolated.py`): 원문 NER만 2모델
  (Gemma+Qwen) `recall_strict` 합의로 재라벨, 주입 PII·텍스트·split 고정(변수=
  프롬프트 하나) → `pii_all_v2.jsonl`. 5-fold×2 재학습.

### 결과 (v1→v2)
- **천장 분해**(사용자 원질문 "silver냐 모델이냐"): PROD precision(.725→보정
  .897)=silver 창작물 누락, recall(.698→.743)=모델 실측 약점. **schema_gap 0·
  IAA 1.0**(클래스 정의는 병목 아님 — VI gold가 §3.1/3.2 미반영이 원인).
- **PROD pooled 0.7171→0.7920(phobert)·0.7097→0.7614(xlm-r)** (+5~7.5pp, 보정
  추정 0.81 적중). overall 0.9461→0.9529·0.9459→0.9543, LOC·ORG 동반 상승.
- gold PROD 2,005→2,394(+389 창작물 회복), EVT 474→436(−38 과라벨 제거).

### 잔여 한계 → 다음 가설
- **EVT 회귀**: 0.7920→0.7731(phobert)·0.8031→0.7348(xlm-r), 양 모델 일관.
  저support(436) EVT가 새 프롬프트로 2모델 합의가 더 갈려 sparse·noisy → #112.

## Phase 2 — EVT 합의 회귀 해결 (recall_strict_evt) (#112)

### 직전 가설
- EVT 회귀가 (a) 합의 불일치 / (b) 과라벨 제거 / (c) 진짜 손실 중 무엇인지
  분해 필요. 합의 기준 유지하며 few-shot 보강 vs 합의 메커니즘 조정 중 택일.

### 본 단계에서 시도 (기존 re-silver 아티팩트 재사용, 진단은 GPU 없이)
- **진단**: EVT 합의 Jaccard **0.548**. Gemma-only EVT 178건 중 **151(85%)이
  Qwen 무-span**(type 분쟁 아닌 순수 recall 미스). 원인 = VI EVT는 보통명사-핵
  서술구(Cúp·Trận·Công ước·Bão…)가 다수인데 Qwen(MoE active 3B·4bit)의 엄격한
  고유명사 편향이 EVT에만 집중 타격 → strict 교집합이 EVT 를 *EVT에 가장 약한
  모델*의 recall 에 천장. v1→v2 EVT diff: removed 131 중 정당 제거(다년process·
  시대) 19, **legit 손실(연도대회·조약·전쟁·재해) ~98**.
- **메커니즘 조정**(few-shot 아님 — 예시 이미 존재해 헤드룸 불확실): `merge_
  confidence.recall_strict_evt` 정책 — single-model EVT(gemma_only/qwen_only
  대칭) 중 §3 명시 legit 카테고리(연도대회·조약/공의회·전쟁·재해·선거) regex
  매칭분만 구제. **§3 결정론 규칙의 독립 재적용**이라 LLM 판정자 없이 self-
  confirm/유도 회피. 미매칭 long-tail 보수 drop.
- 기존 Gemma/Qwen relabel 재-merge → `pii_all_v3.jsonl`(EVT +183 구제·0 제거·
  구제 오탐 0, 비-EVT는 v2와 바이트 동일). 5-fold×2 재학습.

### 결과 (v2→v3)
- **EVT pooled 0.7731→0.8224(phobert)·0.7348→0.8370(xlm-r)** — 양 모델 **v1
  초과**(+3.0/+3.4pp). EVT precision 회복(xlm v2 0.658→v3 0.806).
- **per-fold std 붕괴**: EVT std phobert 0.089→**0.011**, xlm 0.075→0.043 —
  회귀의 본질이 *평균*이 아니라 *합의 불안정(분산)*이었음을 재학습이 직접 확증.
- **PROD give-back −2.2~2.4pp**(phobert 0.792→0.769, xlm 0.762→0.740 — EVT↔PROD
  경쟁; PROD gold 불변이라 모델 효과). 단 v1 대비 +3~5pp로 #108 이득 보존.
  overall flat(0.9513/0.9525), PII 5종·비-EVT 불변.

### 잔여 한계 → 다음 가설
- EVT long-tail(투어·금융위기·영화제 등 ~65건) 패턴 미커버 — 의도적 보수 drop.
  PROD recall 천장(모델 약점)은 미해결.
- PII 주입이 suffix 스캐폴딩(81.7%)이라 PII F1 0.99가 과대평가일 수 있음 →
  #116(자연 주입) 측정 후 결정.
