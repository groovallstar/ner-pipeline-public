# src/ner/classifier/ — BERT 토큰 분류 파인튜닝

JA·VI·KO canonical 10종 평면 (NER 5 + PII 5) 학습용 BERT NER 분류기. augmenters 가
produce 한 PII 주입 JSONL 을 입력으로 받아 BIO 21-class 모델을 학습하고 char-offset
span F1 을 측정한다.

## 책임 경계

- **포함**: 데이터 로딩(JSONL contract 소비) / wordpiece 정렬·BIO 변환 / HF Trainer 래퍼 / span 디코드 / 메트릭 산출
- **제외**: 데이터 증강, 라벨러, 평가용 LLM 백엔드, HF Hub 업로드, 추론 서빙
- **결합도**: 얕음 — augmenters 의 import 결합 없음. 파일(JSONL)만 소비한다. metrics 만 `src/ner/metrics` 에서 공용 import.

## 모듈

| 파일 | 역할 |
|---|---|
| `data_utils.py` | canonical 10종 라벨 맵 / JSONL 로더 / fast(offset-trim)·PhoBERT(pyvi)·JA(slow) 3-way tokenizer 분기 정렬 / BIO ↔ char-span 변환 (`decode_bio_to_spans(confs=...)` 시 span 에 conf_mean `score` 부착) |
| `train_eval.py` | HF Trainer 래퍼 (`fine_tune`) + best 모델 로드 후 평가 (`evaluate_model`, strict + relaxed span F1 동시 산출, `capture_scores=True` 시 토큰 softmax 신뢰도 포착) |
| `confidence_threshold.py` | per-class 신뢰도 임계값(confidence threshold) 운영점 — valid 에서 임계값 fit(`fit_thresholds`, greedy P·R≥target) / 적용(`apply_thresholds`) / 저장·로드(`save_thresholds`·`load_thresholds`). NER 4종(ORG/LOC/EVT/PROD)만 대상 |
| `error_analysis.py` | test-set 오답 추출 + 카테고리 분류 (BOUNDARY / TYPE_MISMATCH / MISS / HALLUCINATION) + 사람 검수용 stratified 샘플. 두 입력 경로: (1) 단일 모델 추론 (`--model-path`), (2) K-fold pooled 예측 재진단 (`--from-predictions --fold-dirs ...`, 재추론 없이 fold 별 `test_predictions.json` 소비). CLI: `python -m ner.classifier.error_analysis` |
| `kfold_pool.py` | 층화 K-fold 학습 결과의 fold 별 test 예측을 합쳐 pooled span F1 산출 + 그룹 단위 cross-fold 누출 검증(누출 시 `ValueError`, `--allow-cross-fold-leak` 으로 카운트만) + 판정 근거(`leak_check_basis`) 기록. CLI: `python -m ner.classifier.kfold_pool` |
| `__main__.py` | CLI: `python -m ner.classifier --lang {ja,vi,ko}` |

## CLI

`--group-key` 는 **모든 학습 경로에서 필수**다 (3-way·K-fold 공통). 값은 형제
행(한 원문에서 파생된 여러 행)을 묶는 필드명이거나, 형제가 없음을 명시하는
`none` 이다. 데이터셋별 올바른 값은 아래 "누출-free 분할" 참조.

```bash
# JA 본 학습 (기본 hyperparameter) — id 가 원문 파생 다행을 묶는다
python -m ner.classifier --lang ja --group-key id

# VI 본 학습 (xlm-roberta-base 기본 모델) — orig 가 원문을 묶는다
python -m ner.classifier --lang vi --group-key orig --epochs 5 --batch-size 16

# KO 본 학습 (koelectra-base-v3 기본 모델, data/klue/pii_all.jsonl)
# 원문당 1행이라 형제가 없다 — id 도 none 도 같은 분할이며 둘 다 통과한다
python -m ner.classifier --lang ko --group-key id

# 스모크 (100 train / 25 valid / 50 test / 1 epoch — CI·dev 검증용)
python -m ner.classifier --lang ja --group-key id --smoke

# 모델·데이터 override
python -m ner.classifier --lang vi --group-key orig \
    --model-name vinai/phobert-base \
    --data data/wikiann_vi/origin.jsonl \
    --max-length 256 --lr 5e-5

# 층화 K-fold 교차 검증 (fold 별 1회 학습 후 pooled 평가)
for fold in 0 1 2 3 4 5 6 7 8 9; do
  python -m ner.classifier --lang ja \
      --data data/stockmark/pii_all_phonediv.jsonl \
      --kfold 10 --fold-index ${fold} --group-key id \
      --output-dir results/classifier/ja_sweep/<실험명>/fold${fold}
done
# pooled 평가: 그룹 단위 cross-fold 누출이 있으면 ValueError 로 중단
# 그룹 키는 test_predictions.json 에 기록된 값을 자동으로 읽는다
python -m ner.classifier.kfold_pool \
    --fold-dirs results/classifier/ja_sweep/<실험명>/fold{0..9}

# 신뢰도 임계값 운영점: valid 에서 per-class 신뢰도 임계값 fit → test 적용·저장
python -m ner.classifier --lang ja --fit-threshold
# 저장된 thresholds.json 을 다른 run 에 적용
python -m ner.classifier --lang ja --confidence-thresholds path/to/thresholds.json

# stratification 비활성화 (PROD/EVT 유무와 무관한 fold 구성 — before/after 통제 비교용)
python -m ner.classifier --lang ja --no-stratify

# 2단계 NER 커리큘럼 warmup (stage1: PII 를 O 로 마스킹 후 워밍업, stage2: 21-class 본학습)
python -m ner.classifier --lang ja --curriculum --curriculum-stage1-epochs 3

# 콘솔 출력 span-F1 모드 선택 (strict|relaxed|both, 기본 both; metrics.json 은 항상 둘 다 저장)
python -m ner.classifier --lang ja --metric-mode strict

# B-/I- 토큰 per-token loss 가중 (경계 인식 강화, 기본 1.0 = 미적용)
python -m ner.classifier --lang ja --boundary-b-weight 2.0 --boundary-i-weight 1.5
```

### 신뢰도 임계값 운영점 (per-class confidence threshold)

`--fit-threshold` 는 모델 학습 후 valid split 에서 NER 4종(ORG/LOC/EVT/PROD)
클래스별 신뢰도 임계값을 greedy(overall P·R≥`--threshold-target`, 기본
0.93)로 찾아 `thresholds.json` 에 저장하고, test 에 적용한 운영점을
`metrics.json` 의 `confidence_threshold` 블록에 기록한다. **opt-in·default
OFF** — 플래그 미지정 시 기존 동작과 완전 동일(BC), `overall`/`overall_strict`
는 항상 임계값 미적용 baseline 으로 보존된다.

- 신뢰도 = span 토큰 max-softmax 의 평균(conf_mean). 저신뢰 예측이 FP 에
  편중돼 있어, 임계값으로 그 꼬리를 잘라 리콜 여유를 정밀도로 바꾼다.
- **모델 개선이 아니라 운영점** — baseline F1 천장은 불변. 임계값은 모델
  종속이라 재학습 시 반드시 재-fit(`--fit-threshold`). 하드코딩 금지.
- micro overall R≥0.93 은 포화 PII 가 떠받치는 값 — NER-4 자체 recall 은
  희생된다(per-class). 0.93-동시충족은 **pooled(다중 fold)** 에서 안정적.

`--data-extra-train-jsonl PATH` 는 별 JSONL 을 train split 에만 합치고
valid/test 는 `--data` 의 원본 split 그대로 유지한다 (학습 데이터
oversampling 보강 시 valid/test leak 방지). BC 유지 — 옵션 미지정 시
기존 단일 jsonl 학습과 동일.

기본값:
- `--lang ja`: 모델 `tohoku-nlp/bert-base-japanese-v3`, 데이터 `data/stockmark/pii_all.jsonl`
- `--lang vi`: 모델 `xlm-roberta-base`, 데이터 `data/wikiann_vi/origin.jsonl`
- `--lang ko`: 모델 `monologg/koelectra-base-v3-discriminator`, 데이터 `data/klue/pii_all.jsonl` (KLUE 유래 NER 5종+DAT + 합성 PII 4종). ELECTRA 계열이라 `--precision fp16`(기본) 사용
- `--valid-ratio 0.1`, `--test-ratio 0.1` (3-way split), `--seed 42`, `--max-length 256`, `--epochs 5`, `--batch-size 16`, `--lr 5e-5`
- 3-way 분할: train/valid/test = 80/10/10. valid 셋은 epoch best 모델 선택용 (`metric_for_best_model='eval_loss'`), test 셋은 최종 char-offset span F1 측정 단독. test 셋은 학습/모델 선택 어디에도 노출되지 않음.
- 재현성 (`--train-seed`): 기본 None 은 헤드 init 을 시드하지 않는 기존 동작(BC). 값을 주면 헤드 init·dropout·셔플을 고정해 재현 가능한 run 이 된다. GPU FP 비결합에 따른 seed-내 잔여 비결정성(loss ~1e-4)은 effect size 대비 무시 가능 — 비교 측정은 양 팔을 같은 `--train-seed` 로 고정하거나 multi-seed paired 로 본다. `metrics.json` 에 `train_seed`·`precision` 기록.
- fold 붕괴 (희귀·분할의존): 10-fold 일부 분할에서 koelectra 가 드물게(~0.3~3%) 학습 붕괴(F1≈0)한다. 검증된 근본 수정은 없음 — F1≈0 fold 만 `--train-seed` 를 바꿔 재실행한다(full-determinism 은 붕괴를 막지 못하고 재현만 하며 ~1.9× 비용이라 비채택). 상세: `docs/reports/korean-bert-classifier-fold-collapse.md`.
- **층화 K-fold 모드** (`--kfold N --fold-index K`): PROD/EVT 보유 여부로 층화하여 N개 fold 에 배정. test = fold K, valid = fold (K+1)%N, train = 나머지. `--kfold 10` 이면 분할 크기가 80/10/10 과 동일. fold 모드에서는 test 예측이 `test_predictions.json` 으로 저장되어 `kfold_pool` 의 pooled 평가 입력이 된다. N ≥ 3 필수. 평가 프로토콜 상세: `docs/reports/japanese-bert-classifier-per-entity-diagnosis.md`

### 누출-free 분할 (`--group-key`, 필수)

합성 PII 주입과 재라벨은 한 원문에서 여러 행을 파생시킨다(**형제 행**).
형제가 train·test 로 갈리면 모델이 학습에서 본 정답을 시험에서 다시 만나
span F1 이 부푼다. `--group-key <필드>` 를 주면 같은 값을 공유하는 행을 한
unit 으로 묶어 통째로 한 split 에 배정한다 — 누출이 구조적으로 불가능해진다.

**두 분할 경로 모두 적용된다** — `split_kfold_stratified`(K-fold)와
`split_train_valid_test`(기본 3-way). 후자는 오래도록 인자가 없어 무방비였다.

#### 데이터셋별 그룹 키

`id` 의 뜻은 데이터셋마다 다르다. 코드가 못 박을 수 없고 데이터가 선언해야 한다.

| 데이터 | 행 수 | `id` 그룹 | `orig` 그룹 | 그룹 키 |
|---|---|---|---|---|
| `data/klue/pii_all.jsonl` (KO) | 25,989 | 25,989 | 필드 없음 | `id` 또는 `none` (형제 없음) |
| `data/wikiann_vi/origin.jsonl` (VI) | 37,706 | 37,706 | 29,337 | **`orig`** |
| `data/stockmark/origin.jsonl` (JA) | 5,270 | **5,166** | 필드 없음 | **`id`** |

KO 의 `id` 는 KLUE 원본 인덱스(train 21,008 + dev 5,000), VI 의 `id` 는 행
일련번호, JA 의 `id` 는 원문 파생 다행이 공유하는 원문 번호다.

#### 고유한 필드는 그룹 키가 될 수 없다

값이 전부 다르면 unit 이 전부 싱글턴이라 그룹 보호가 **no-op** 이 된다.
그런데 같은 필드로 누출을 세면 중복이 0 이라 "누출 없음" 으로 읽힌다 —
보호는 없는데 게이트는 초록이다. `validate_group_key` 가 학습 **전에** 이를
거부한다: 선언한 키보다 행을 더 강하게 묶는(그룹 수가 적은) 후보 필드가
있으면 `ValueError`. VI 에 `--group-key id` 를 주면 `orig` 가 후보로 잡혀
중단된다.

`--group-key none` 은 형제가 없다는 **명시적 선언**이다. 이때 누출 카운터는
`0` 이 아니라 `null`(미측정)로 기록되며, `validity.compare` 는 이를
`INVALID`(누출 미검증)로 판정한다 — 크래시하지 않는다. 정직한 opt-out 이
벌받고 거짓 0 이 통과하면 규칙이 편법을 보상하게 된다.

#### 이중 가드와 판정 근거

split 단계 그룹 분할 + pooling 단계 `kfold_pool` 사후 검증. `__main__` 이
`test_predictions.json` 에 `group`(그룹 키의 값)과 `group_key`(필드명)를 싣고,
`pool_fold_predictions` 가 같은 그룹 값이 두 fold 의 test 에 걸치면
`cross_fold_group_dups` 로 세며 `require_no_leak=True`(기본)면 `ValueError`.
누출 baseline 을 일부러 측정할 때만 `--allow-cross-fold-leak`.

판정 근거는 `leak_check_basis` 로 남는다.

| 근거 | 뜻 | 신뢰 |
|---|---|---|
| `group` | 선언된 그룹 키의 값으로 셌다 | ✅ |
| `orig` | 그룹 키 기록이 없는 옛 예측 파일, `orig` 로 셌다 | ✅ |
| `text` | 문장 전체 비교로 셌다 | ❌ 재작성 코퍼스에선 형제를 못 본다 |
| `none` | 행 단위 분할을 명시했다 | ❌ 애초에 측정하지 않았다 |

`text` 는 특히 위험하다. KO·VI 는 PII 주입 시 문장을 새로 짓기 때문에 형제끼리
글자가 전부 다르고, 문장 비교는 **하나도 잡지 못한다**(KO 실측: 문장 전체
중복 0건). 카운트 0 이 "이상 없음" 이 아니라 "볼 수단이 없었음" 인 경우다.

- 발견·진단·정량 방법론(왜 dedup 으로 못 막았나, 측정 설계, 인플레 수치):
  `docs/manual/pipeline/3-verification.md` §3B

## JSONL contract (입력 계약)

augmenters/pii, augmenters/wikiann_vi 가 출력하는 형식 그대로 소비.

```jsonl
{"text": "...",
 "entities": [{"label": "PER", "start_char": 0, "end_char": 3, "text": "..."}],
 "id": "..."}
```

- `label` ∈ canonical 10종 = `PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD`
- `start_char`, `end_char` — 문자 오프셋 [start, end), 반-개구간
- 검증: `data_utils.load_jsonl` 가 라벨이 canonical 10종 안에 있는지 확인하고, 위반 시 `ValueError`

## 라벨 셋

```
O                              # outside
B-PER, I-PER                   # NER 5종
B-LOC, I-LOC
B-ORG, I-ORG
B-PROD, I-PROD
B-EVT, I-EVT
B-DAT, I-DAT                   # PII 5종
B-EMAIL, I-EMAIL
B-PHONE, I-PHONE
B-ID_NUM, I-ID_NUM
B-CREDIT_CARD, I-CREDIT_CARD
```

총 21 labels (`O` + 10*B + 10*I). 단일 출처: `docs/manual/data/canonical-entity-schema.md`.

## 토크나이저 분기 (3-way)

`encode_row` 가 토크나이저 종류로 분기한다 (`data_utils`):

| 분기 | 조건 | 정렬 방식 |
|---|---|---|
| fast | `tokenizer.is_fast` (XLM-R·CafeBERT·mmBERT·DeBERTa-V3 등) | `return_offsets_mapping=True` + `_trim_offset`. SentencePiece 계열이 `▁` 토큰에 선행 공백을, 숫자형 entity 끝에 문장부호를 흡착해 char-offset 이 어긋나는 것을 **선행 공백·후행 `.`/`,` trim** 으로 교정 (`_encode_vi`) |
| PhoBERT | `_is_phobert` (slow, 단어분절 전제) | `pyvi` 단어분절 후 단어별 BPE, 단어 char-span 정렬 (`_encode_phobert`). 의존성: `pyvi` |
| JA slow | 그 외 slow (`BertJapaneseTokenizer`) | `tokenize()` → `text.find(surface, pos)` greedy. `##` strip. UNK 시 0-length. 의존성: `fugashi` + `unidic-lite` |

- slow tokenizer 는 `offset_mapping` 미지원 — `_encode_ja`/`_encode_phobert` 가 수동 정렬.
- `--legacy-no-offset-trim` 으로 fast 경로 trim 을 끌 수 있다(진단·구 동작 재현용).
- DeBERTa-V3 는 정렬은 정상이나 표준 레시피 학습 실패로 벤치 제외(리포트 참조).

## 메트릭

- 주 메트릭: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`)
- 매칭: `(start, end, type)` 정확 매칭 (llm_eval 과 동일 정의 — 직접 비교 가능)
- per-entity: 10종 각각 micro-F1·precision·recall·support 산출

## 산출물

```
results/classifier/{ja,vi,ko}/
├── best/                       # best 체크포인트 (HF model dir)
├── checkpoint-*/               # 중간 체크포인트 (save_total_limit=1 로 정리)
├── metrics.json                # 학습 설정 + overall/per-entity strict·relaxed F1 (+--fit-threshold 시 confidence_threshold 블록)
└── thresholds.json             # --fit-threshold 시: per-class 임계값 + meta (conf_key/target/fit_set)

docs/reports/japanese-bert-classifier-benchmark.md      # JA 요약 (현 상태·교훈)
docs/reports/japanese-bert-classifier-history.md         # JA 히스토리 1편 (Phase 0~8, 동결)
docs/reports/japanese-bert-classifier-per-entity-diagnosis.md  # JA 엔티티별 성능 진단 (1편 후속, 층화 K-fold)
docs/reports/vietnamese-bert-classifier-benchmark.md    # VI 리포트
docs/reports/korean-bert-classifier-fold-collapse.md     # KO fold 붕괴 조사 (재현성·안정성)
```

## 출하·배포 (JA deploy)

JA 출하 아티팩트·배포 추론은 본 패키지 밖(`scripts/`·`data/`·`docs/`)에
둔다 — 학습은 CLI(`python -m ner.classifier`, `--fit-threshold` 포함)에
흡수하고 배포 추론만 분리했다 (별도 `train_*` 스크립트 없음).

| 아티팩트 | 위치 | 역할 |
|---|---|---|
| 배포 추론 | `src/ner/scripts/eval_ja_ner_test.py` (`.sh` = uv 래퍼) | 학습 없이 고정 test + 저장된 `thresholds.json` 으로 추론·태깅·P/R/F1 출력. 절대경로만 허용. 기본 배포 레이아웃 `/data/ner/ja/{model,data/test.jsonl,thresholds.json}` |
| 출하 모델 번들 | `data/stockmark/ja_ner_prod_seed1/` | `model/` + `data/{train,valid,test}.jsonl` + `metrics.json` + `thresholds.json` + `MODEL_CARD.md` (배포 시 `/data/ner/ja/` 로 복사) |
| 최종 출하 스펙 | `docs/reports/japanese-bert-classifier-spec.md` | 모델·데이터·엔티티·평가지표 단일 출처 (10-fold pooled 0.9361 / raw 0.9273) |

> 배포 추론은 위 "책임 경계 제외(추론 서빙)" 와 직교 — `scripts/` 의 독립
> 도구이며 classifier 패키지를 import 만 한다 (패키지에 서빙 코드 없음).

## 테스트

```bash
python -m pytest tests/ner/classifier/ -q
```

- `test_data_utils.py` — 라벨 맵 / BIO 정렬 / span 디코드 / split 결정성 / 층화 K-fold 무결성·층화 균등성 / group K-fold(원문 단위 묶음·누출-free)·BC(group_key=None ≡ 행 단위) / 3-way 형제 묶기·BC / 후보 키 검사(고유 키·필드 부재·null 거부, 형제 없는 코퍼스 통과) / curriculum mask
- `test_encode.py` — 실제 토크나이저(JA·VI·DeBERTa-V3·PhoBERT)로 round-trip 검증
- `test_error_analysis.py` — span 오류 분류·집계·검수 샘플링 (10 테스트)
- `test_kfold_pool.py` — pooled F1 손계산 일치 / 그룹 단위 cross-fold 누출 검출(`ValueError`)·`--allow-cross-fold-leak` 카운트 / 재작성 문장에서도 group 근거로 검출 / `none` → 카운터 `null`(미측정) / 레거시 orig→text fallback·근거 기록 / 섞인 근거는 가장 약한 것으로 보고
- `test_confidence_threshold.py` — scored decode(conf_mean)·apply·fit(greedy P·R≥target)·save/load 라운드트립

## 주의

- `PYTHONPATH` **설정·주입 금지** — uv editable install 이 자동 등록
- import: `from ner.classifier.xxx import Yyy` (src 접두어 없이)
- print/log/argparse help: 영문 / docstring·주석: 한국어 (코딩 컨벤션 `docs/specs/coding-conventions.md`)
