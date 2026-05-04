# issue-40: classifier 깨진 CLI 복구 + canonical 10종 정합 + JA·VI 프로덕션 BERT 학습

- Issue: https://github.com/groovallstar/ner_pipeline/issues/40
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-40-classifier-restore`
- 승인일: 2026-04-30
- 완료일: <!-- 5단계에서 채움 -->

## 목적

`src/ner/classifier/` 는 PR #38(src 모놀리식 → src/ner 네임스페이스) 이후
양쪽 CLI 가 import 단계에서 즉시 실패(`ModuleNotFoundError: ner.classifier.data_utils`)
하는 dead-code 상태이며, `data_utils.py`·`train_eval.py` 는 git history 에도
존재하지 않는다(`git log --all -- '**/data_utils.py'` 빈 결과). 동시에 라벨
스키마는 라벨러·증강기 영역의 canonical 10종 평면(NER 5 + PII 5) 정착(#27,
#30, #8) 이전의 PII 7종(`NAME, ADDRESS, DOB, ID_NUMBER, ...`) 가정이 그대로
남아 있어 augmenters 출력과 정합하지 않는다.

이번 이슈의 목적은 단순한 "import 에러 해소" 가 아니라, **JA·VI 두 언어에서
canonical 10종 평면을 학습한 프로덕션 BERT 모델을 산출** 하는 것이다. CLI
복구는 그 부산물이다.

## 범위

### 포함
- `src/ner/classifier/{data_utils.py, train_eval.py}` 신규 작성 (git 복원 불가능)
- `src/ner/classifier/__main__.py` 재작성 — JA·VI 분기, canonical 10종 평면, 하드코딩 LLM 비교 테이블 제거
- `src/ner/classifier/pii_benchmark.py` 폐기 (PII 5종 단독 벤치는 본 학습 결과의 PII 슬라이스로 흡수)
- `tests/ner/classifier/` 단위 테스트 신규 (라벨 정렬·span F1·JSONL 로딩·smoke fixture)
- JA·VI 본 학습 실행 → `results/classifier/{ja,vi}/` 산출물 + `docs/reports/{ja,vi}-bert-classifier-benchmark.md` 리포트
- `src/ner/classifier/AGENTS.md` 신설 + `src/ner/CLAUDE.md`·루트 `CLAUDE.md`·`docs/manual/data/canonical-entity-schema.md:290` 보정

### 제외
- KO (KLUE 6종) BERT 학습 — 후속 이슈
- `augmenters/` 자체 코드 수정 (출력 JSONL 형식만 contract 로 검증; 부족분 발견 시 별도 이슈)
- 학습된 모델의 HF Hub 업로드 / 추론 CLI / 외부 서비스 연동 — 후속 이슈
- F1≥95% 미달 시 hyperparameter 튜닝·데이터 증강 라운드 — 미달 사유 기록 + 후속 이슈

## 성공 기준

- [ ] 테스트: `python -m pytest tests/ner/classifier/ -q` 모두 통과 (단위 + smoke 1-epoch fixture)
- [ ] CLI 동작: `python -m ner.classifier --lang ja --epochs 1 --smoke` / `--lang vi --epochs 1 --smoke` 가 ImportError 없이 종료, exit code 0
- [ ] **F1 (overall span F1, 10종 평면)**: JA ≥ 0.95, VI ≥ 0.95 — 미달 시 리포트에 사유 기록 + 후속 이슈 발행 (이슈 닫기 전 사용자 승인 필요)
- [ ] 산출물: `results/classifier/{ja,vi}/{best/, metrics.json}` + `docs/reports/{ja,vi}-bert-classifier-benchmark.md`
- [ ] 문서: `src/ner/classifier/AGENTS.md` 신설, 상위 CLAUDE.md 들 + canonical-entity-schema.md:290 의 "후속 정리 예정" 마커 제거

## 결정된 설계 (deep-interview 결과)

| 축 | 결정 | 근거 |
|---|---|---|
| 모듈 목적 | 프로덕션 NER 모델 학습 (LLM 베이스라인 비교 아님) | round 1 응답 |
| 1차 타겟 | JA + VI 동시, 둘 다 canonical 10종 평면 (NER 5 + PII 5), KO 제외 | round 2 응답 |
| 복구 전략 | 신규 작성 (B) — git 복원 (A) 불가능, history 부재 확인 완료 | git log 증거 |
| 코드 결합도 | 얕음 — augmenters 출력 JSONL 소비, `src/ner/metrics/` 만 공용 import. augmenters 와 직접 import 결합 없음 | round 3 응답 |
| 산출물 위치 | `results/classifier/{ja,vi}/` (실검증 데이터·메트릭) + `docs/reports/{ja,vi}-bert-classifier-benchmark.md` (리포트) | round 4 응답 |
| F1 합격선 | overall span F1 ≥ 0.95 (시간·모델 크기 미고려) | round 4 응답 |
| LLM 비교 | 제거. `__main__.py:124-139` 하드코딩 테이블 삭제. llm_eval 결과와의 비교 리포트 생성 안 함 | round 4 응답 |
| PII 단독 벤치 | `pii_benchmark.py` 폐기. PII 5종 메트릭은 본 학습 리포트의 per-entity 섹션에 포함 | round 4 응답 |

## JSONL contract (입력 계약 — step 1 점검 완료)

`augmenters/` 가 produce 한 실제 파일에서 확인된 스키마(추정과 다름):

```jsonl
{"text": "...",
 "entities": [{"label": "PER", "start_char": 0, "end_char": 3, "text": "..."}, ...],
 "id": "..."}
```

- `label` (key 이름이 `type` 이 아님) ∈ canonical 10종 = `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD`
- `start_char`, `end_char` (key 이름이 `span` 이 아님) — 문자 오프셋 [start, end), 반-개구간
- 1행 = 1 샘플 + `id` 필드
- JA 입력: `data/stockmark/pii_all.jsonl` — 5,307 samples (단일 파일, train/test 분할은 classifier 가 책임)
- VI 입력: `data/wikiann_vi/pii_all.jsonl` — 40,000 samples (단일 파일)

라벨 분포 (실측):
- JA: ORG 4922, PER 3638, LOC 2536, EMAIL 1023, DAT 1017, PHONE 960, PROD 939, EVT 840, ID_NUM 829, CREDIT_CARD 821 — 10종 모두 존재, support 균형
- VI: PER 20877, LOC 20221, ORG 7820, EMAIL 7373, CREDIT_CARD 7392, ID_NUM 7370, DAT 7128, PHONE 7206, PROD 2045, **EVT 492** ← 가장 낮은 support, F1≥95% 도달의 최대 위협

`compute_offset_span_f1` 은 `{type, start, end}` 형식을 기대하므로 `data_utils._bio_labels_from_offsets` 에서 augmenters 의 `label/start_char/end_char` 를 받아들이고, evaluate 단계에서 `train_eval.evaluate_model` 이 metrics 모듈 호출 직전에 변환한다. augmenters 변경은 본 이슈에서 제외 (key rename 은 augmenters 측 호환성 영향 큼 — 별도 이슈 후보).

## 구현 단계 (5)

- [x] 1. **JSONL contract 점검 완료** — augmenters 의 실제 출력은 `{"label", "start_char", "end_char"}` 형식 (계획서 추정 `{"type", "span"}` 와 다름). 라벨은 양 파일 모두 canonical 10종 내에 한정 확인. 위 "JSONL contract" 섹션에 실측 스키마 + 분포 기록. augmenters 측 key rename 은 본 이슈에서 제외.
- [x] 2. **classifier 코어 재작성 완료** — `data_utils.py`(canonical 10종 라벨맵·JSONL 로더·JA slow tokenizer 수동 정렬·VI fast tokenizer offset_mapping·BIO↔span 변환) + `train_eval.py`(HF Trainer 래퍼 + best 모델 저장 + char-offset span F1) + `__main__.py`(JA·VI 분기 CLI, `--smoke` 플래그, 하드코딩 LLM 비교 테이블 제거) + `pii_benchmark.py` **삭제 완료**. JA tokenizer 의존성 추가: `pyproject.toml` 에 `fugashi` + `unidic-lite`.
- [x] 3. **단위 테스트 완료** — `tests/ner/classifier/{__init__.py, test_data_utils.py, test_encode.py}` 11/11 pass (8 unit + 3 토크나이저 round-trip).
- [x] 4. **본 학습 + 산출물 + 변종 실험 완료** — JA·VI baseline 학습 + v2/v3/v4 누적 변종 6 회 학습. **F1≥95% 게이트 모든 변종 미달.** 최선: JA v3=0.9150 (gap 3.50pp), VI v4=0.9147 (gap 3.53pp). 리포트:
  - `docs/reports/japanese-bert-classifier-benchmark.md` (JA baseline + 변종 사후 추가)
  - `docs/reports/vietnamese-bert-classifier-benchmark.md` (VI baseline + 변종 사후 추가)
  - `docs/reports/bert-classifier-variant-experiments-2026-04.md` (4-variant ablation 통합 snapshot)
  
  핵심 발견:
  - PII 5종은 v1 부터 이미 0.95+ 포화 → class weight down-weight 효과 없음 (가설의 *PII 측* 부분은 실측에서 기각)
  - NER 5종은 v2 에서 +1.8pp(JA) / +0.6pp(VI) 상승 → 가설의 *NER 측* 부분은 부분 성립
  - large 모델은 PROD 같은 lexical-poor 클래스에 한정 효과 (JA PROD +8.2pp), VI EVT 처럼 작은 class 에서는 *역효과* (-7.9pp)
  - curriculum 은 large 의 EVT 회귀를 부분 보완 (VI EVT v3 0.545 → v4 0.660)
  - **추가 0.95 도달 경로 = 데이터 보강 (별도 이슈)**
- [x] 5. **문서 갱신 완료** — `src/ner/classifier/AGENTS.md` 신설. `src/ner/CLAUDE.md` / 루트 `CLAUDE.md` / `docs/manual/data/canonical-entity-schema.md:290` 갱신.

실측 커밋 입도: step 2+3+5(코드+테스트+docs+deps) = 1 커밋, step 4(산출물·리포트·변종 코드) = 1 커밋. **총 2 커밋 예상**.

## 게이트 미달 처리 — 사용자 승인 필요

deep-interview 에서 합의한 mitigation:
> step 4 에서 미달 시 리포트에 사유 기록 + 사용자 승인 후 후속 이슈 발행 + 본 이슈 PR 가/부 결정은 사용자 승인.

**사용자 결정 사항**:
1. 본 이슈를 그대로 PR 제출 (변종 실험 결과 포함, 0.95 미달 사유 명시) → 이슈 종결, 데이터 보강 후속 이슈로 분리
2. 본 이슈 내에서 추가 시도 (예: 데이터 보강·앙상블·hyperparameter sweep) → 후속 작업 합의 후 진행
3. 게이트 완화 (예: 0.90 또는 NER/PII 분리 게이트) → 합격선 재정의 후 본 이슈 종결

## 위험·의존성

### 🔴 높은 위험

- **F1≥95% 합격선의 현실성**: 현재 JA Stockmark 5종 NER 의 LLM 2-pass 최고
  성능은 `gemma-4-31B-it = 0.8772` Filtered F1 (`docs/reports/japanese-ner-benchmark.md`).
  BERT in-domain 지도학습은 LLM zero-shot 대비 +5~10pp 가능하나, 10종
  평면(NER 5 + PII 5) 의 micro-F1 95% 도달은 PII 5종이 regex-친화적이라
  pull-up 효과가 있어도 보장되지 않는다. 특히 PROD/EVT 엔티티의 클래스별
  F1 이 95% 미만일 가능성이 높다. **mitigation**: step 4 에서 미달 시 리포트에
  per-entity F1 표를 그대로 기록하고 후속 이슈 발행. 본 이슈 PR 가/부 결정은
  사용자 승인.

### 🟡 중간 위험

- **augmenters 출력 contract 불일치**: step 1 에서 augmenters 의 실제 JSONL
  출력이 위 contract 와 다를 수 있음 (예: `type` 대신 `label`, `span` 대신
  `start_char`/`end_char` — 실제 `pii_benchmark.py:53-71` 의 `_convert_pii_to_ner_format`
  이 그런 변환을 했었음). **mitigation**: step 1 단독으로 확인하고 불일치 시
  augmenters 변경은 본 이슈에서 제외 (별도 이슈).

### 🟢 낮은 위험

- vinai/phobert-base 는 베트남어 단일, xlm-roberta-base 는 다국어. 두 후보를
  step 4 에서 비교 후 더 나은 쪽 채택. 사전 lock-in 하지 않음.
- 학습 시간: BF16, 1 GPU 기준 step 4 가 길어질 수 있음 — 사용자 명시한 대로
  시간은 고려 대상 아님.

---

## 변경 요약

CLI 복구 + canonical 10종 정합 외에:

1. **3-way split 도입 (80/10/10)**: 기존 80/20 train/test → train/valid/test. valid 셋이 epoch best 선택을 담당해 test 셋이 학습·모델 선택 어디에도 노출되지 않는 hold-out 프로토콜 확보.
2. **토크나이저 호환성 자동화**: fast tokenizer 자동 fallback (`__main__.py`), `tokenizer.is_fast` 기준 encode 분기 (`data_utils.py`). 기존 `lang=='ja'` 강제 slow 분기 제거 — 신형 JA fast 모델 시험 가능.
3. **bf16 옵션 추가**: DeBERTa-v3 family 의 fp16 unscale 에러 회피용 `--precision {fp16,bf16}`.
4. **평가 시 float32 강제 로드**: `evaluate_model` 의 `torch_dtype=torch.float32` — bf16 학습 후 fp16 저장된 weight 의 inference NaN/underflow 회피.
5. **변종/sweep 실험**: v1 베이스 후 누적 변종(class weight, NER warmup, large 모델, curriculum) 6 회 + SOTA 모델 sweep 12 회 = 총 18 회 학습. 게이트 0.95 도달 모델 없음.

## 구현 결과 (계획 대비)
- [x] 1. JSONL contract 점검 — augmenters 출력 `{"label", "start_char", "end_char"}` 확정 (계획서 추정 `{"type", "span"}` 와 다름). canonical 10종 한정 확인. 라벨 분포 실측 기록.
- [x] 2. classifier 코어 재작성 — `data_utils.py` (canonical 10종 라벨맵·JSONL 로더·JA slow tokenizer 수동 정렬·VI fast tokenizer offset_mapping·BIO↔span 변환·3-way split) + `train_eval.py` (HF Trainer 래퍼 + best 모델 저장 + char-offset span F1 + bf16 + float32 평가) + `__main__.py` (JA·VI 분기 CLI, `--smoke`/`--precision`/`--valid-ratio` 플래그, 자동 fast/slow tokenizer fallback) + `pii_benchmark.py` 삭제. JA tokenizer 의존성 `fugashi` + `unidic-lite` 추가.
- [x] 3. 단위 테스트 — `tests/ner/classifier/{__init__.py, test_data_utils.py, test_encode.py}` 16/16 pass (10 unit + 3 토크나이저 round-trip + 2 split + 1 mask).
- [x] 4. 본 학습 + 변종 + sweep — JA·VI baseline (80/20 → 80/10/10 재분할) + 누적 변종 6회 + SOTA 모델 sweep 12회. **F1≥0.95 모든 변종/모델 미달.** 베이스라인이 최선 또는 거의 최선:
  - JA 베이스라인 0.9058 (gap 4.42pp), 모든 SOTA 후보 회귀
  - VI 베이스라인 0.8985, xlm-roberta-large 가 +0.42pp 마진 (cost 5x), 다른 후보 회귀
- [x] 5. 문서 갱신 — `docs/reports/bert-classifier-benchmark.md` (JA·VI 통합 베이스라인 결과만), 분석·sweep·변종 등 본 이슈 문서에 흡수. `src/ner/classifier/AGENTS.md` 신설 + 상위 CLAUDE.md / canonical-entity-schema.md 갱신.

실측 커밋 입도: 코드+테스트+docs+deps = 1 커밋, 실험 산출물·리포트 = 1 커밋. **총 2 커밋 예상**.

## 검증

### 테스트
```bash
python -m pytest tests/ner/classifier/ -q
# 16 passed
```

### 메트릭 (베이스라인, 80/10/10 분할)

| 언어 | Train/Valid/Test | Overall F1 | 게이트 통과 라벨 |
|---|---|---|---|
| JA | 4,247 / 530 / 530 | **0.9058** | 4 / 10 (PHONE/EMAIL/DAT/PER) |
| VI | 32,000 / 4,000 / 4,000 | **0.8985** | 1 / 10 (EMAIL) |

게이트 0.95 — **JA·VI 모두 FAIL**. 자세한 per-entity 표는 `docs/reports/bert-classifier-benchmark.md`.

### 리뷰어 확인 사항
- 3-way split 도입 후 단위 테스트 회귀 없음 (16/16)
- DeBERTa-v3 family 학습 실패는 hyperparameter 미튜닝 issue, 별도 이슈 후보
- 게이트 미달 사유 (데이터 천장) 본 문서에 기록, 후속 데이터 보강 이슈로 분리

## 게이트 미달 분석 (실측)

### 80/20 → 80/10/10 분할 변경 영향

| 언어 | 80/20 F1 | 80/10/10 F1 | Δ | 비고 |
|---|---|---|---|---|
| JA | 0.8952 | **0.9058** | +1.06pp | PROD 0.758→0.817, EVT 0.831→0.875 동시 개선. PER 게이트 추가 통과 |
| VI | 0.9108 | 0.8985 | -1.23pp | PROD test support 398→208 절반 축소로 silver 노이즈 영향 증폭 (-10.1pp), 전체 회귀 |

핵심: valid 셋이 best epoch 선택을 담당하면서 test 셋 leakage 위험 제거. JA 는 모든 NER 클래스 개선, VI 는 small-support 클래스 통계 변동성 증가.

### 누적 변종 실험 (이전 80/20 분할 기준, 변종 효과 검증용)

| 변종 | 누적 변경 | JA F1 | VI F1 |
|---|---|---|---|
| v1 baseline | 표준 CE / `eval_loss` 기준 best / base 모델 | 0.8952 | 0.9108 |
| **v2** classwt + NER-best | NER B/I weight=2.0, PII B/I weight=0.5, `metric_for_best='ner_f1'` | **0.9134** | 0.9137 |
| v3 + large | JA: `bert-large-japanese-v2` / VI: `xlm-roberta-large` | 0.9150 | 0.9135 |
| v4 + curriculum | NER warmup (PII 마스킹) 3 epoch + 21-class fine-tune | 0.9121 | **0.9147** |
| **gate** | — | 0.9500 | 0.9500 |

JA 최선 v3 = 0.9150 (gap 3.50pp), VI 최선 v4 = 0.9147 (gap 3.53pp). 어느 변종도 0.95 미달.

핵심 발견:
- **v2 가 ROI 최고** (JA +1.82pp / VI +0.29pp). v3·v4 는 학습 시간 4-5x 증가에 비해 +0.16~+0.40pp 만 추가 → 비실용적
- **PII 5종 v1 부터 포화** (≈0.95+) → class weight down-weight 효과 미미 (가설의 PII 측 부분 기각)
- **NER 5종은 +1-2pp 상승** (가설 NER 측 부분 성립)
- **large 모델은 small class 변동성 증가**: VI EVT v3 = 0.545 (-7.9pp vs v2). curriculum 으로 부분 회복 (v4 = 0.660)
- JA PROD v3 = 0.840 (+8.2pp vs v1) — 단일 최대 개선. lexical-poor 클래스에서 large + class weight 효과적
- VI EVT 는 어떤 변종도 천장 (60% 대) — 절대 support 부족 (94 → 61) 본질적 한계

### SOTA 모델 Sweep (2026.05, 신규 80/10/10 분할 기준)

#### JA Sweep (7 후보)
| 모델 | Overall F1 | Train time |
|---|---|---|
| **`tohoku-nlp/bert-base-japanese-v3`** (베이스) | **0.9058** | 99s |
| `tohoku-nlp/bert-large-japanese-v2` | 0.8894 | 676s |
| `jhu-clsp/mmBERT-base` | 0.7662 | 408s |
| `sbintuitions/modernbert-ja-130m` | 0.7649 | 259s |
| `llm-jp/llm-jp-modernbert-base` | 0.7220 | 357s |
| `ku-nlp/deberta-v3-base-japanese` (bf16, lr=2e-5) | 0.0000 | 360s — 학습 실패 |
| `microsoft/mdeberta-v3-base` (bf16, lr=2e-5) | 0.0000 | 391s — 학습 실패 |

#### VI Sweep (5 후보)
| 모델 | Overall F1 | Train time |
|---|---|---|
| **`xlm-roberta-large`** | **0.9027** | 4098s (5×) |
| `xlm-roberta-base` (베이스) | 0.8985 | 808s |
| `jhu-clsp/mmBERT-base` | 0.5278 | 1329s |
| `microsoft/mdeberta-v3-base` (bf16, lr=2e-5) | 0.0000 | 1273s — 학습 실패 |
| `Fsoft-AIC/videberta-base` (bf16, lr=2e-5) | 0.0000 | 1246s — 학습 실패 |

제외: `studio-ousia/luke-japanese-large-lite` (LukeTokenizer 호환성 문제), `vinai/phobert-base/large` (fast tokenizer 미지원, 별도 처리 필요).

#### Sweep 핵심 발견
1. **베이스라인이 최선**: 12개 SOTA 후보 중 어느 것도 의미있게 능가하지 못함. JA 는 베이스, VI 는 large 가 +0.4pp 만
2. **fast tokenizer fragmentation**: ModernBERT/mmBERT 가 일본어 PII 영숫자(`@`, 숫자열), 베트남어 tone mark 를 부정확 분해 → NER 또는 PII 클래스 collapse
   - llm-jp-modernbert-base: PII EMAIL 0.17, ID_NUM 0.12, CREDIT_CARD 0.31
   - VI mmBERT-base: NER 5종 모두 50% 미만 (PER 0.247, LOC 0.347)
3. **DeBERTa-v3/v2 family 학습 실패 패턴 (재현)**:
   - 학습 loss 정상 감소 (172 → 4.6, eval_loss 70 → 3.7)
   - 평가 prediction 모두 'O' (majority-class collapse)
   - 모델 weight NaN 없음, classifier head 학습 신호 받음 (mean ≈ 0.004)
   - 원인 추정: warmup_ratio·weight_decay·gradient_clip 추가 튜닝이 필요한 알려진 family. 단순 lr=2e-5 + bf16 으로는 불충분
   - JA·VI 모두 동일 패턴 — 별도 hyperparameter sweep 이슈 후보
4. **VI large cost-effectiveness 매우 낮음**: +0.42pp / 5x train time

## 관련 커밋
- `<hash>`: feat(classifier): CLI 복구 + canonical 10종 정합 + 3-way split + bf16/float32 평가
- `<hash>`: chore(reports): 베이스라인 통합 리포트 작성 + 이슈 문서 분석 흡수

## 후속 작업·알려진 한계

### 별도 이슈 후보 (게이트 0.95 도달 가능 경로)

1. **데이터 보강 (가장 확실)** — JA: KWDLC, BCCWJ NER, OntoNotes JA 통합 → PROD/EVT 다양성. VI: VLSP 2018·2021, PhoNER 통합 + WikiANN-vi silver→gold 부분 정제 (PROD/EVT 우선)
2. **DeBERTa-v3 family hyperparameter sweep** — warmup_ratio (0.06~0.1), weight_decay (0.01), gradient_clip (1.0) 그리드. 잠재력 미확인 상태
3. **PhoBERT 통합** — slow tokenizer 의 manual greedy match (`_encode_ja` 경로) + word_segmenter (VnCoreNLP) 결합. VI 단일언어 SOTA 잠재력
4. **PHONE 표기 정규화 전처리** — VI PHONE 0.81 의 표기 다양성 (5+ 형식). regex 기반 정규화로 토크나이저 정렬 안정화. 단, surface form 정보 손실 위험 (CREDIT_CARD/ID_NUM 과 구분 약화) — 진단 후 결정
5. **다중 seed 앙상블** — 검증된 +1~2pp 효과
6. **NER vs PII 분리 fine-tune 후 union** — PII 가 NER 학습 신호 희석하지 않음
7. **EVT class oversampling** (VI) — 4-8x 복제로 support 보강

### 알려진 한계

- **WikiANN-vi silver 노이즈** — VI PROD precision 0.54 / recall 0.80 패턴은 silver 라벨 비일관성 시사. 모델은 합리적으로 학습, gold 가 일관되지 않음
- **EVT 절대 support 천장** — VI EVT 학습 492, test 61 — class imbalance 본질적 한계
- **PROD 어휘 다양성** — 제품명은 OOV 일반화 문제 (브랜드+모델, 약어, 외래어, 숫자 포함)
- **KO 미포함** — 본 이슈 범위 외 (canonical 6종 KLUE 별도 이슈)
- **LUKE-japanese / PhoBERT** — 토크나이저 호환성 문제로 sweep 제외, 별도 통합 이슈 후보
