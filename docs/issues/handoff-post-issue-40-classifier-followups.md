# 핸드오프: classifier F1≥0.95 후속 작업 (post-#40)

> 작성일: 2026-05-04
> 선행 작업: PR #41 (`feat/issue-40-classifier-restore`) — classifier CLI 복구 +
>   canonical 10종 정합 + 3-way split + SOTA 모델 sweep
> 베이스라인 결과: `docs/reports/bert-classifier-benchmark.md`
> 분석·sweep 결과: `docs/issues/issue-40-classifier-restore.md` (게이트 미달
>   분석 / 누적 변종 v2~v4 / SOTA 모델 sweep 12회 / DeBERTa family 학습 실패
>   패턴 / 데이터 천장)
> 목적: 0.95 게이트 도달이 본 측정 셋업·모델 후보로는 불가능에 가까움이
>   실측 확인됨. 후속 이슈 6개 후보를 효과·비용·의존성 기준으로 정리하고,
>   각 이슈에 진입할 때의 진입 조건·범위·성공 기준·위험을 동결한다.

---

## 0. 진입 조건

- PR #41 머지 후 develop 기준으로 새 이슈 브랜치를 분기한다 (`feat/issue-{N}-slug`).
- 본 핸드오프의 6개 후속 항목은 *상호 독립* 또는 *데이터→모델→앙상블* 의
  단방향 의존만 가지므로 병렬 진행이 가능하다. 본 문서의 권장 순서는 그
  의존성 + 효과·비용 ROI 기반.
- 각 후속 이슈는 GitHub Issue 등록 + `docs/issues/issue-{N}-{slug}.md` 단일
  파일로 진행 (`CLAUDE.md` 5단계 규칙).

## 1. 후속 작업 일람 (효과·비용 매트릭스)

| 후속 | 예상 효과 | 비용 | ROI | 의존성 |
|---|---|---|---|---|
| F1. 다중 seed 앙상블 | +1~2pp 안정 | ★ | **★★★ 최고** | 없음 |
| F2. PHONE 오류 진단 + 조건부 정규화 (VI) | VI PHONE +0~5pp | ★ | ★★ | 없음 |
| F3. DeBERTa-v3 hyperparameter sweep | ? (잠재력 미확인) | ★★ | ★★ | 없음 |
| F4. PhoBERT 통합 (VI) | VI +0~3pp | ★★★ | ★★ | 없음 |
| F5. EVT/PROD oversampling (VI) | VI EVT/PROD +5~15pp | ★★ | ★★ | 없음 |
| F6. 외부 코퍼스 통합 (데이터 보강) | **+5~10pp 양 언어** | ★★★★ | ★★★ | augmenters 측 |

**우선 권장**: F1 → F5 → F2 → (F3 ‖ F4) → F6.

핵심 근거: F1·F5·F2 는 단발성 학습/스크립트 추가만으로 검증 가능하고 즉시
효과가 보이는 반면, F6 (데이터 보강) 은 별도 augmenters 작업이 선행되어야
하므로 시간 비용이 가장 크다. F3·F4 는 잠재력은 있지만 본 sweep 에서
이미 베이스라인이 SOTA 후보를 능가했으므로 모델 측 추가 작업의 ROI 가
상대적으로 낮다.

---

## 2. 후속별 상세 가이드

### F1. 다중 seed 앙상블

**목적**: seed 별 학습 결과의 분산을 평균/투표로 흡수해 +1~2pp 안정 향상.

**범위**:
- 같은 모델·데이터·하이퍼파라미터로 seed ∈ {42, 123, 2024} 3회 학습 (또는 5 seed)
- 각 best 모델로 test 셋 inference → token-level argmax voting 또는 logit averaging
- 산출물: `results/classifier/{ja,vi}/ensemble/metrics.json`, 통합 리포트 추가

**구현 힌트**:
- `src/ner/classifier/__main__.py` 의 `--seed` 플래그 이미 존재
- 앙상블 inference 는 `train_eval.evaluate_model` 의 single-model 버전을 N-model 버전으로 확장. logit averaging 이 vote 보다 일반적으로 안정
- 추가 모듈: `src/ner/classifier/ensemble.py` (학습은 기존 `__main__` N회 실행, 평가만 새 함수)

**성공 기준**:
- 단일 seed 베이스라인 대비 +1pp 이상 평균 향상
- per-entity F1 분산 (std) 감소

**위험**:
- 학습 시간 N배 (VI 의 경우 5×800s ≈ 4000s/seed × 3 = 3.3시간)
- 효과는 검증되어 있으나 0.95 게이트 도달 단독으론 부족 — F5·F6 와 조합 필요

### F2. PHONE 오류 진단 + 조건부 정규화 (VI 한정)

**목적**: VI PHONE F1=0.81 의 실제 실패 원인을 분리 후 (boundary vs class
confusion vs miss), 그에 맞는 처방을 적용. **무조건 정규화는 위험** —
디지트만 남기면 CREDIT_CARD/ID_NUM/일반 숫자열과 구분 시그널이 사라져
precision 이 더 떨어질 수 있다.

**범위 (Step A — 진단)**:
- 기존 best 모델로 VI test 셋 inference
- PHONE 오류만 4 카테고리로 분류:
  - (a) exact-match: 정답
  - (b) class-confusion: 위치 맞지만 라벨 틀림 (어떤 라벨로 가는지 confusion matrix)
  - (c) boundary-mismatch: 라벨 맞지만 start/end 어긋남 (어긋나는 자릿수 분포)
  - (d) miss: prediction 자체 부재
- 산출: `docs/reports/vi-phone-error-analysis-2026-XX.md` (snapshot)

**범위 (Step B — 처방, Step A 결과에 따라 분기)**:
- (b) 우세: 정규화 = 역효과. surface form 부각 (특수 토큰 prefix `<PHONE>`, character feature 보강) 또는 PHONE 단독 fine-tune
- (c) 우세: 보수적 정규화 — 공백·점만 단일 하이픈으로 통일, `+`·괄호 보존. 시그널 유지하면서 fragmentation 만 줄임. 새 데이터 파일 (`pii_all_phonenorm.jsonl`) + 학습
- (d) 우세: 데이터 부족 — F5 (oversampling) 또는 F6 (외부 코퍼스) 로 합류

**구현 힌트**:
- 진단 스크립트 ~50 LOC, `src/ner/classifier/scripts/phone_error_analysis.py` 또는 `src/ner/llm_eval/error_analysis.py` 재사용 검토
- 정규화 시 char offset 재계산 알고리즘은 본 핸드오프 작성 시점 채팅 로그에 기록 (entity 단위 sort 후 prefix accumulator)

**성공 기준**:
- Step A 결과 명시 (4 카테고리 비율) → Step B 처방 정당화
- VI PHONE F1 +3pp 이상 (Step B 적용 후)

**위험**:
- 정규화는 surface form 정보 손실 → CREDIT_CARD/ID_NUM 오라벨 위험. 진단 없이 정규화 강행 금지
- (a) 가 대부분이면 PHONE 0.81 자체가 천장 — F1 (앙상블) 이나 F6 으로 합류

### F3. DeBERTa-v3 family hyperparameter sweep

**목적**: 본 sweep 에서 학습 실패 (loss 떨어지지만 평가 prediction 모두 'O')
한 DeBERTa-v3 / DeBERTa-v2 family 의 잠재력 검증.

**범위**:
- 대상 모델: `microsoft/mdeberta-v3-base`, `ku-nlp/deberta-v3-base-japanese`,
  `Fsoft-AIC/videberta-base`
- Hyperparameter 그리드:
  - `lr` ∈ {1e-5, 2e-5, 3e-5}
  - `warmup_ratio` ∈ {0.06, 0.1, 0.15}
  - `weight_decay` ∈ {0.0, 0.01, 0.1}
  - `max_grad_norm` ∈ {1.0, 0.5}
- 각 조합 1 epoch smoke + 5 epoch full → 베스트 5 조합 만 full eval
- 산출: `docs/reports/deberta-v3-hyperparam-sweep-2026-XX.md`

**구현 힌트**:
- `train_eval.fine_tune` 에 `warmup_ratio`, `weight_decay`, `max_grad_norm`
  인자 추가. 이미 있는 `--lr` 플래그 외 새 CLI 플래그 추가
- 초기 1-epoch smoke 로 collapse 여부 빠른 검증 (현재는 5 epoch 후에야 알 수 있음)

**성공 기준**:
- 적어도 한 모델·한 조합에서 베이스라인(JA 0.9058, VI 0.8985) 동급 또는 초과
- 미달 시 "DeBERTa family 는 본 데이터셋·셋업과 호환 안 됨" 으로 결론 동결

**위험**:
- 그리드 27 조합 × 모델 3 × 언어 2 = 162 회 학습. 1-epoch smoke 로 줄여도 30+ 시간
- 결과적으로 베이스라인 동급에 그칠 가능성 있음 — *결론 동결* 자체가 가치

### F4. PhoBERT 통합 (VI 단일언어 SOTA)

**목적**: VI 단일언어 모델 PhoBERT 가 fast tokenizer 미지원으로 본 sweep
제외. word_segmenter (VnCoreNLP) + slow tokenizer 의 manual greedy match 로
통합 시 잠재력 검증.

**범위**:
- `vinai/phobert-base` (135M), `vinai/phobert-large` (370M) 학습
- Tokenizer 분기 추가: `_encode_phobert` (VnCoreNLP word segmentation 후 BPE)
  → `data_utils.py` 에 신규 함수
- Char offset 보존 — VnCoreNLP 의 word boundary 와 원 문자열 매핑 필요
- 산출: PhoBERT 결과를 `docs/reports/bert-classifier-benchmark.md` 에 추가
  열로 추가 또는 별도 스냅샷 리포트

**구현 힌트**:
- VnCoreNLP 또는 `pyvi` 패키지로 word segmentation
- PhoBERT 의 권장 입력은 word-segmented (단어를 `_` 로 join). 토크나이저
  output 의 char offset 을 원본 텍스트에 매핑하는 것이 까다로움 — 별도
  단위 테스트 필수
- `__main__.py` 의 토크나이저 자동 fallback 에 PhoBERT 케이스 추가
  (`type(tokenizer) == PhobertTokenizer` 시 별도 분기)

**성공 기준**:
- PhoBERT-base/large 학습 + 평가 정상 종료, F1 산출
- 베이스라인 (xlm-roberta-base 0.8985) 대비 비교 표 추가
- 게이트 도달 여부 보고

**위험**:
- VnCoreNLP 는 Java 의존 — Python 통합 복잡도. `pyvi` 가 순수 Python 이라 우선 검토
- char offset 보존 실패 시 BIO 라벨 정렬 깨짐 — 단위 테스트로 round-trip 검증 필수
- 결과 미달 가능성 — VI 단일언어 효과는 일반 NER 에서 입증됐으나, canonical
  10종 평면 (PII 포함) 에서는 미검증

### F5. EVT/PROD class oversampling (VI 한정)

**목적**: VI EVT (test support 61) 와 PROD (208) 의 절대 학습량 부족을
oversampling 으로 보강. 본 sweep 에서 이 두 클래스가 단일 최대 병목으로
확인됨 (EVT F1=0.65, PROD F1=0.64).

**범위**:
- EVT 만 포함된 학습 sample 4-8x 복제 (또는 EVT/PROD 동시)
- 합성 EVT 생성 (LLM 기반) 옵션 검토 — augmenters/pii 모듈 패턴 참조
- 산출: oversampling 비율별 5 세트 학습, F1 비교

**구현 힌트**:
- `data_utils.split_train_valid_test` 후 train 셋에서 EVT 포함 sample 만
  필터 → N배 복제 → 원본 train 과 concat
- valid/test 셋은 변경 안 함 (분포 보존)
- `__main__.py` 에 `--oversample-evt 4` 플래그 추가

**성공 기준**:
- EVT F1 +5pp 이상 향상 (gold: 0.95 도달)
- PROD F1 +3pp 이상
- 다른 클래스 F1 회귀 < 1pp

**위험**:
- Oversampling 은 일반 클래스 (PER/LOC/ORG) 가 적게 학습되어 회귀 위험
- 합성 데이터는 silver 노이즈 추가 가능성 — augmenters/pii 의 verifier 적용 필수

### F6. 외부 코퍼스 통합 (데이터 보강)

**목적**: 본질적 천장 (PROD/EVT 어휘 다양성, EVT 절대 support, silver 노이즈)
은 모델 측 변경으로 풀리지 않음. 외부 gold 코퍼스 통합이 0.95 도달 가장
확실한 경로.

**범위 (JA)**:
- KWDLC (Kyoto University Web Document Leads Corpus) — NER 5종 풍부
- BCCWJ NER (現代日本語書き言葉均衡コーパス) — 10종+ 라벨, canonical 10종 매핑 필요
- OntoNotes JA — 다국어 OntoNotes 의 JA 부분
- Stockmark canonical 10종 평면과 라벨 정합 (entity merging) — augmenters 측 작업

**범위 (VI)**:
- VLSP 2018·2021 NER — 베트남어 NER 표준
- PhoNER — VI 단일언어 NER 코퍼스
- WikiANN-vi silver→gold 정제 — 기존 silver 라벨에 LLM verifier 재적용 (augmenters/wikiann_vi 확장) 또는 사람 검수
- ViMQ, ViNERX 등 추가 후보 검토

**구현 힌트**:
- augmenters 모듈에 새 서브패키지 추가 (`augmenters/external_corpus/`)
- 각 외부 코퍼스의 라벨 체계 → canonical 10종 매핑 표 (`docs/manual/data/`)
- 통합 학습 데이터 스펙: `data/{stockmark,wikiann_vi,external_*}/pii_all.jsonl`
  → `data/combined/{ja,vi}_pii_all.jsonl` 형태로 union 후 학습

**성공 기준**:
- 외부 코퍼스 통합 학습 데이터로 베이스라인 모델 학습 → F1 ≥ 0.95 (JA·VI 양쪽)
- 또는 미달 시 새 베이스라인 + 천장 분석 문서화

**위험**:
- 라이선스 문제 (특히 BCCWJ, VLSP) — 사전 검토 필수
- 라벨 매핑 비용 — 외부 코퍼스의 entity 정의가 canonical 10종과 1:1 매핑되지 않음. PROD/EVT 같은 본 프로젝트 정의에 외부 라벨이 부분만 해당
- 코퍼스 크기 차이 — domain shift 위험 (도메인 균형 sample weight 필요)

---

## 3. 의존성·병렬성 다이어그램

```
F1 다중 seed 앙상블 ───────┐
F2 PHONE 진단 → 정규화 ──┤
F3 DeBERTa hyperparam ─┤───→ 모두 F6 와 독립, 즉시 진행 가능
F4 PhoBERT 통합 ──────┤
F5 EVT/PROD oversample ┘

F6 외부 코퍼스 통합 ────────→ augmenters 측 선행 작업, 독립 트랙
```

F1·F2·F3·F4·F5 는 모두 같은 코드 모듈 (`src/ner/classifier/`) 변경이라
브랜치 병렬은 어려움. 순차 진행 권장. F6 는 augmenters 모듈 작업이라 별도
브랜치에서 병렬 가능.

## 4. 작성 원칙 (모든 후속 이슈 공통)

1. **베이스라인 비교 의무** — 각 후속 이슈는 PR #41 의 베이스라인
   (JA 0.9058 / VI 0.8985) 대비 Δ 를 per-entity 표로 보고. `docs/reports/bert-classifier-benchmark.md` 의 표 그대로 인용 또는 새 열로 추가.
2. **단일 변경 원칙** — 한 후속 이슈에서 *한 가지 가설* 만 검증. F1 + F5 같은
   조합은 새 후속으로 분리 (혹은 본 후속 완료 후 통합 이슈).
3. **3-way split 보존** — 80/10/10 (`seed=42`) 그대로 사용. test 셋은 어떤
   후속 이슈에서도 학습/모델 선택에 노출되지 않는다. 분할 변경이 필요하면
   별도 이슈로 분리.
4. **DeBERTa family 학습 실패 시** — F3 외 후속 (F1·F5·F4 등) 에서 DeBERTa
   family 모델을 시험할 때, 학습 loss 가 정상 감소하지만 평가 prediction
   = 'O' 인 majority-class collapse 패턴이 재현되면 본 핸드오프 §F3 의
   hyperparameter 그리드 적용 후 재시도. 단순 lr=2e-5 + bf16 으로는 부족함이
   본 sweep 에서 확정.
5. **commit/PR 컨벤션** — `CLAUDE.md` 의 입도 규칙 준수. 한 후속 이슈는
   2-3 commit 권장 (코드+테스트+docs / 실험 산출물 / 이슈 md 최초 commit).

## 5. 게이트 미달 종결 시점 결정

본 핸드오프의 6개 후속을 모두 진행해도 0.95 게이트에 도달하지 못할 가능성도
있다. 그 경우의 종결 옵션:

- **(A) 게이트 완화**: deep-interview 에서 0.95 → 0.90 또는 NER/PII 분리 게이트
  (예: NER 0.90, PII 0.95) 로 재정의. 합격선 재합의 이슈로 분리.
- **(B) 본 데이터셋 천장 동결**: PR #41 베이스라인을 production 모델로 채택.
  추가 시도는 새 데이터 추가 시점에 재개.
- **(C) 모델 외부 보완**: 추론 시 rule-based PII detector (regex 정규화) 와
  union → BERT NER 5종 + regex PII 5종 조합으로 0.95 도달 시도. classifier
  모듈은 NER 5종에만 집중.

종결 결정은 사용자 승인 후 별도 이슈로 명문화.
