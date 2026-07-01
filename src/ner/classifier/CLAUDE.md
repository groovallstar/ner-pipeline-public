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
| `kfold_pool.py` | 층화 K-fold 학습 결과의 fold 별 test 예측을 합쳐 pooled span F1 산출 + 원문(`orig`) 단위 cross-fold 누출 검증(누출 시 `ValueError`, `--allow-cross-fold-leak` 으로 카운트만). CLI: `python -m ner.classifier.kfold_pool` |
| `__main__.py` | CLI: `python -m ner.classifier --lang {ja,vi,ko}` |

## CLI

```bash
# JA 본 학습 (기본 hyperparameter)
python -m ner.classifier --lang ja

# VI 본 학습 (xlm-roberta-base 기본 모델)
python -m ner.classifier --lang vi --epochs 5 --batch-size 16

# KO 본 학습 (kf-deberta-base 기본 모델, data/klue/pii_all.jsonl)
python -m ner.classifier --lang ko --precision bf16

# 스모크 (100 train / 25 valid / 50 test / 1 epoch — CI·dev 검증용)
python -m ner.classifier --lang ja --smoke

# 모델·데이터 override
python -m ner.classifier --lang vi \
    --model-name vinai/phobert-base \
    --data data/wikiann_vi/origin.jsonl \
    --max-length 256 --lr 5e-5

# 층화 K-fold 교차 검증 (fold 별 1회 학습 후 pooled 평가)
# --group-key orig: 같은 원문 파생 행을 한 fold 로 묶어 cross-fold 누출 차단
for fold in 0 1 2 3 4 5 6 7 8 9; do
  python -m ner.classifier --lang ja \
      --data data/stockmark/pii_all_phonediv.jsonl \
      --kfold 10 --fold-index ${fold} --group-key orig \
      --output-dir results/classifier/ja_sweep/<실험명>/fold${fold}
done
# pooled 평가: 원문 단위 cross-fold 누출이 있으면 ValueError 로 중단
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
- `--lang ko`: 모델 `kakaobank/kf-deberta-base`, 데이터 `data/klue/pii_all.jsonl` (KLUE 유래 NER 5종+DAT + 합성 PII 4종). DeBERTa 계열이라 `--precision bf16` 권장
- `--valid-ratio 0.1`, `--test-ratio 0.1` (3-way split), `--seed 42`, `--max-length 256`, `--epochs 5`, `--batch-size 16`, `--lr 5e-5`
- 3-way 분할: train/valid/test = 80/10/10. valid 셋은 epoch best 모델 선택용 (`metric_for_best_model='eval_loss'`), test 셋은 최종 char-offset span F1 측정 단독. test 셋은 학습/모델 선택 어디에도 노출되지 않음.
- 재현성 (`--train-seed`): 기본 None 은 헤드 init 을 시드하지 않는 기존 동작(BC). 값을 주면 헤드 init·dropout·셔플을 고정해 재현 가능한 run 이 된다. GPU FP 비결합에 따른 seed-내 잔여 비결정성(loss ~1e-4)은 effect size 대비 무시 가능 — 비교 측정은 양 팔을 같은 `--train-seed` 로 고정하거나 multi-seed paired 로 본다. `metrics.json` 에 `train_seed`·`precision` 기록.
- fold 붕괴 (희귀·분할의존): 10-fold 일부 분할에서 koelectra 가 드물게(~0.3~3%) 학습 붕괴(F1≈0)한다. 검증된 근본 수정은 없음 — F1≈0 fold 만 `--train-seed` 를 바꿔 재실행한다(full-determinism 은 붕괴를 막지 못하고 재현만 하며 ~1.9× 비용이라 비채택). 상세: `docs/reports/korean-bert-classifier-fold-collapse.md`.
- **층화 K-fold 모드** (`--kfold N --fold-index K`): PROD/EVT 보유 여부로 층화하여 N개 fold 에 배정. test = fold K, valid = fold (K+1)%N, train = 나머지. `--kfold 10` 이면 분할 크기가 80/10/10 과 동일. fold 모드에서는 test 예측이 `test_predictions.json` 으로 저장되어 `kfold_pool` 의 pooled 평가 입력이 된다. N ≥ 3 필수. 평가 프로토콜 상세: `docs/reports/japanese-bert-classifier-per-entity-diagnosis.md`

### 누출-free 분할 (`--group-key`)

합성 PII 주입은 한 원문에서 여러 행을 파생시킨다(같은 문장 + 서로 다른
PII). 행 단위 분할은 이 파생 행들이 train·test 로 갈려 **cross-fold 원문
누출**이 생기고 span F1 이 부풀 수 있다. `--group-key orig` 를 주면 같은
`row['orig']`(주입 전 원문) 값을 공유하는 행들을 한 unit 으로 묶어 통째로
한 fold 에 배정한다 — split 단계에서 누출이 구조적으로 불가능해진다.

- `split_kfold_stratified(..., group_key='orig')` 의 unit = 원문 그룹.
  층화 기준(PROD/EVT 합집합)·라운드로빈·결정성은 행 단위와 동일.
- `--group-key` 미지정(기본) 시 기존 행 단위 분할과 **완전히 동일**(같은
  seed → 동일 결과). BC 유지.
- **이중 가드**: split 단계 그룹 분할 + pooling 단계 `kfold_pool` 의 사후
  검증. `test_predictions.json` 에 `orig` 가 실리고, `pool_fold_predictions`
  가 같은 원문이 두 fold 의 test 에 걸치면 `cross_fold_orig_dups` 로 세고
  `require_no_leak=True`(기본)면 `ValueError`. 누출 baseline 을 일부러
  측정할 때만 `--allow-cross-fold-leak` 로 카운트만 받는다.
- `orig` 가 없는 레거시 코퍼스는 `text` 전체 문장 중복으로 fallback 검증.
  `id` 는 비고유(한 원문 파생 다행 공유)라 검증 기준으로 쓰지 않는다.
- 발견·진단·정량 방법론(왜 dedup 으로 못 막았나, 측정 설계, 인플레 수치):
  `docs/manual/data/vi-bert-crossfold-leak.md`

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

- `test_data_utils.py` — 라벨 맵 / BIO 정렬 / span 디코드 / split 결정성 / 층화 K-fold 무결성·층화 균등성 / group K-fold(원문 단위 묶음·누출-free)·BC(group_key=None ≡ 행 단위) / curriculum mask
- `test_encode.py` — 실제 토크나이저(JA·VI·DeBERTa-V3·PhoBERT)로 round-trip 검증
- `test_error_analysis.py` — span 오류 분류·집계·검수 샘플링 (10 테스트)
- `test_kfold_pool.py` — pooled F1 손계산 일치 / 원문(`orig`) 단위 cross-fold 누출 검출(`ValueError`)·`--allow-cross-fold-leak` 카운트 / 레거시 text 중복 fallback / 비고유 id 허용
- `test_confidence_threshold.py` — scored decode(conf_mean)·apply·fit(greedy P·R≥target)·save/load 라운드트립

## 주의

- `PYTHONPATH` **설정·주입 금지** — uv editable install 이 자동 등록
- import: `from ner.classifier.xxx import Yyy` (src 접두어 없이)
- print/log/argparse help: 영문 / docstring·주석: 한국어 (코딩 컨벤션 `docs/specs/coding-conventions.md`)
