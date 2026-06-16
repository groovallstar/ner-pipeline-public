# 베트남어 BERT NER 분류기 벤치마크 (canonical 10종 평면)

- 측정일: 2026-06-15
- 대상: WikiANN-vi silver + 합성 PII 주입 데이터에 인코더 family 파인튜닝
- 데이터: `data/wikiann_vi/pii_all.jsonl` (**중복 제거 후 38,371 행**, canonical 10종 = NER 5 + PII 5)
- 평가: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`), **stratified 5-fold pooled micro-average**
- 출처: `docs/issues/issue-103-vi-classifier-canonical-benchmark.md`
- 일본어 동일 셋업 결과는 `docs/reports/japanese-bert-classifier-benchmark.md` 참조

> **이전 표(production 0.8985, sweep, v1~v4 ablation)는 폐기·대체됨.** 두 가지
> 혼입(confound)이 있었다 — ① fast-tokenizer offset 정렬 버그(아래 §코드 수정),
> ② WikiANN-vi 내재 train/test 누출(아래 §누출). 본 표는 둘을 모두 제거한
> 캐노니컬 단일 출처다.

## 요약

- **상위 무승부**: `vinai/phobert-base-v2` (0.9461) ≈ `xlm-roberta-base`
  (0.9459) — 0.02pp 차로 통계적 동률.
- **동률의 비용 결판(아래 §배포 비용)**: phobert가 메모리 41%↓·단문 레이턴시
  20%↓·params 절반, pyvi 오버헤드는 +1ms로 무시 가능. 남는 건 운영축
  (pyvi 의존성·정렬 0.996)뿐.
- **CafeBERT(VI continued-pretrain)는 base를 못 넘음** (0.9367 < 0.9459) —
  베트남어 추가 사전학습이 이 태스크엔 이득 없음.
- **`xlm-roberta-large` 불안정** — 5-fold 중 fold0이 all-O로 완전 붕괴(F1=0),
  정상 4-fold는 0.9414. large 모델의 확률적 학습 붕괴(아래 §large 불안정).
- **DeBERTa-V3(mdeberta·videberta) 제외** — 표준 레시피로 학습 불가(아래).
- 천장은 여전히 **PROD(F1~0.71)·EVT(~0.80)** — WikiANN-vi silver 노이즈 +
  상대적 support 부족. 게이트 수치는 미설정.

## 조건

| 항목 | 값 |
|---|---|
| 데이터 | WikiANN-vi (silver) + PII 주입, **원문 기준 중복 제거 후 38,371** (아래 §누출) |
| 분할 | **stratified 5-fold** (PROD/EVT 층화, `seed=42`), pooled micro-avg |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, lr=5e-5, max_length=256, fp16, batch 16 (large/CafeBERT 8) |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid fold) |
| 하드웨어 | NVIDIA RTX A6000 49GB ×3, seed=42 |

## 누출(data leakage) 발견과 제거

raw `unimelb-nlp/wikiann`(vi)에는 데이터 품질 문제가 내재한다 — **원본
test∩train = 1,778 문장**, train 내부 중복 3,932행(19.7%). PII 주입이 행마다 다른 개인정보를 넣어
exact 문자열을 바꾸며 이 누출을 **가렸다**(겉보기 중복은 줄지만 base 문장은 여전히
겹침). 40,000 행을 random 재셔플하면 test의 **4.17%(글자 그대로 일치)/
6.33%(원문 일치)** 가 train과 겹쳐 모델이 라벨을 암기 → F1 부풀림.

**조치**: PII를 제외한 원문 문장(이하 '원문 키')이 같은 행을 하나만 남기는
**중복 제거(deduplication)** → 38,371 행(중복 1,629행 제거). 재검증: 중복 제거 후
어떤 split·fold 조합에서도 train·test 간 같은 문장 출현(cross-split 중복) **0**. 실증으로
CafeBERT가 leaky single-split 0.9504 → clean fold0 0.9259로 떨어져 부풀림을 확인.

> **잔여 한계(제거 불가)**: 전 인코더가 위키백과(VI)로 사전학습됨 → WikiANN
> 문장 *텍스트*는 사전학습에 노출(라벨은 아님). 전 모델 공통이라 *상대 비교는
> 유효*하나 *절대 F1은 낙관적*. 단, 사전학습 코퍼스 편중(PhoBERT/CafeBERT는
> VI 위키·뉴스 집중)이 위키 출처 벤치에 유리할 수 있음은 해석 시 유의.

## 5-fold 결과 (pooled, 중복 제거 후)

| 모델 | strict F1 | Precision | Recall | relaxed F1 | 비고 |
|---|---:|---:|---:|---:|---|
| **`vinai/phobert-base-v2`** | **0.9461** | 0.9317 | 0.9610 | 0.9508 | pyvi 단어분절 |
| **`xlm-roberta-base`** | **0.9459** | 0.9286 | 0.9639 | 0.9505 | |
| `jhu-clsp/mmBERT-base` | 0.9395 | 0.9250 | 0.9543 | 0.9464 | |
| `uitnlp/CafeBERT` | 0.9367 | 0.9168 | 0.9574 | 0.9426 | XLM-R-large 계열, 550M |
| `xlm-roberta-large` | 0.8364† | 0.9270 | 0.7620 | 0.8413 | †fold0 붕괴 포함 |

n_sentences = 38,371 (각 모델 전수 1회 평가).

### per-fold strict F1 (투명 보고)

| 모델 | fold0 | fold1 | fold2 | fold3 | fold4 | fold평균‡ | std‡ |
|---|---:|---:|---:|---:|---:|---:|---:|
| phobert-base-v2 | 0.9500 | 0.9423 | 0.9453 | 0.9498 | 0.9430 | 0.9461 | 0.003 |
| xlm-roberta-base | 0.9459 | 0.9447 | 0.9481 | 0.9457 | 0.9454 | 0.9460 | 0.001 |
| mmbert-base | 0.9406 | 0.9347 | 0.9435 | 0.9361 | 0.9424 | 0.9395 | 0.004 |
| cafebert | 0.9259 | 0.9494 | 0.9420 | 0.9330 | 0.9332 | 0.9367 | 0.008 |
| **xlm-roberta-large** | **0.0000** | 0.9404 | 0.9408 | 0.9384 | 0.9460 | 0.7531 | 0.377 |

‡ `fold평균`은 5개 fold F1의 **산술평균**(위 overall 표의 pooled micro-avg와
별개 — pooled는 fold 크기로 가중). `std`는 모집단 표준편차(n=5). xlm-r-large는
fold0 붕괴로 산술평균(0.7531)·pooled(0.8364)·정상4fold평균(0.9414)이 모두 다름.

대표 학습시간(평균/fold): phobert 8.6분 · xlm-r-base 10.2 · mmbert 16.9 ·
cafebert 32.8 · xlm-r-large 32.8.

### per-entity strict F1 (pooled)

| Entity | support | phobert | xlm-r-base | mmbert | cafebert | xlm-r-large† |
|---|---:|---:|---:|---:|---:|---:|
| ID_NUM | 7,225 | 0.9971 | 0.9971 | 0.9965 | 0.9941 | 0.8840 |
| EMAIL | 7,242 | 0.9966 | 0.9957 | 0.9956 | 0.9802 | 0.8809 |
| PHONE | 7,090 | 0.9963 | 0.9959 | 0.9940 | 0.9873 | 0.8842 |
| CREDIT_CARD | 7,246 | 0.9865 | 0.9862 | 0.9846 | 0.9791 | 0.8748 |
| DAT | 7,003 | 0.9845 | 0.9849 | 0.9771 | 0.9772 | 0.8767 |
| LOC | 19,922 | 0.9349 | 0.9291 | 0.9225 | 0.9220 | 0.8223 |
| PER | 20,485 | 0.9272 | 0.9317 | 0.9247 | 0.9252 | 0.8239 |
| ORG | 7,306 | 0.8814 | 0.8888 | 0.8724 | 0.8677 | 0.7710 |
| EVT | 474 | 0.7920 | 0.8031 | 0.6796 | 0.7310 | 0.6911 |
| PROD | 2,005 | 0.7171 | 0.7097 | 0.6867 | 0.6874 | 0.6142 |

†xlm-r-large는 붕괴 fold0 포함 pooled — per-entity 전반이 낮은 것은 fold0의
all-O 때문이며 정상 4-fold는 타 모델과 동급(아래).

## 배포 비용 (추론) — accuracy 동률의 결판

accuracy가 phobert ≈ xlm-r-base 동률이라 선택은 **비용 축**에서 갈린다. fold0
test 7,675문장, float32, RTX A6000 1장(`CUDA_VISIBLE_DEVICES=0`)에서 측정
(batch=32 throughput / batch=1 latency 200문장 median). 측정: `python
src/ner/scripts/bench_vi_inference_cost.py`. 상세 출처: `docs/issues/
issue-106-vi-classifier-inference-cost.md`.

| 모델 | params | peak GPU mem | encode/sent | latency b=1 | throughput b=32 | strict F1 |
|---|---:|---:|---:|---:|---:|---:|
| **`phobert-base-v2`** | 134M | **789MB** | 1.21ms (pyvi) | **6.03ms** | 345/s | 0.9461 |
| `xlm-roberta-base` | 277M | 1,335MB | 0.22ms | 7.54ms | 343/s | 0.9459 |
| `mmbert-base` | 307M | 1,470MB | 0.22ms | 20.1ms | 209/s | 0.9395 |
| `cafebert` | 559M | 2,496MB | 0.22ms | 15.3ms | 101/s | 0.9367 |
| `xlm-roberta-large` | 559M | 2,496MB | 0.21ms | 15.3ms | 101/s | 0.8364† |

- **하드웨어 비용은 전부 phobert 우위**: 메모리 41%↓(789 vs 1,335MB)·단문
  레이턴시 20%↓(6.03 vs 7.54ms)·params·disk 절반. throughput만 동률(345≈343).
- **pyvi 오버헤드는 실재하나 무시 가능**: encode 1.21 vs 0.22ms(~5.5×)지만 절대
  +1ms. end-to-end(encode+forward) phobert 7.24ms < xlm-r-base 7.76ms —
  pyvi를 더해도 phobert가 더 빠르다.
- **남는 trade-off는 순수 운영축**(성능 아님): xlm-r-base의 유일한 이점은 pyvi
  의존성 없음 + fast(Rust) 토크나이저 + char 정렬 1.0(phobert 0.996 = 0.4%
  구조적 손실 + inference 시 분절 드리프트라는 라이브 실패 모드). phobert는 그
  0.4% 핸디캡을 안고도 F1 동률 → 내재 모델은 오히려 약간 우수.
- **권고**: 메모리/지연 제약이면 phobert(절반 자원). pyvi 의존·정렬 리스크 회피 +
  2× 자원 감수 가능이면 xlm-r-base. cafebert·xlm-r-large(550M)는 정확도 열위/불안정
  + 최고 비용이라 비채택. mmbert는 정확도도 낮고 b=1 레이턴시 최악(20ms)이라 비채택.

## `xlm-roberta-large` 불안정 (large 모델 확률적 붕괴)

fold0만 eval_loss ~2.17에 고착(all-O, F1=0)하고 나머지 4-fold는 정상
(eval_loss ~0.14, F1 0.938~0.946). **정상 4-fold 평균 = 0.9414**로 base와 동급.
다른 모델은 동일 fold0에서 정상 학습하므로(데이터 문제 아님) large 모델 특유의
확률적 all-O 붕괴다. seed=42 고정이라 fold0 재실행은 동일 붕괴 재현 가능성이
높고, "성공할 때까지 재시도"는 측정 cherry-picking이라 **재실행하지 않고 그대로
보고**한다. 비용(33분/fold) 대비 base를 못 넘고 1/5 붕괴 위험까지 있어 채택
가치 낮음.

## DeBERTa-V3 제외 (`microsoft/mdeberta-v3-base`, `Fsoft-AIC/videberta-base`)

두 가지 **별개** 문제로 표준 레시피 학습 불가:

1. **발산** — lr=5e-5 no-warmup에서 첫 스텝부터 loss 폭증(126) + grad_norm
   NaN → 즉시 발산. warmup 또는 저LR(1e-5/2e-5)로 **해소 가능**.
2. **비학습** — 발산을 막아도(warmup 有, lr 1e-5·2e-5) eval_loss가 ~2.0에
   고착(all-O), F1=0. warmup·LR과 **무관**하게 재현. 테스트한 4개 설정이 모두
   bf16이었고 정밀도(fp16/fp32) 변수는 미분리.

토크나이저 정렬(아래 offset-trim)은 해결됐으나(라벨 왕복 복원율 1.0) 최적화가
표준 레시피로 안 됨. 모델별 튜닝은 본 벤치 범위("모델만, 설정 sweep 금지") 밖이라
**제외**. (videberta는 베트남어 단일언어 DeBERTa-V3이나 동일 실패.)

## 코드 수정 (이번 벤치에서 도입)

- **fast-tokenizer offset trim** (`data_utils._encode_vi`): SentencePiece 계열이
  `▁` 토큰에 선행 공백을, 숫자형 entity 끝에 문장부호를 흡착해 char-offset이
  어긋나던 문제를 공백·후행 `.`/`,` trim으로 교정. **mmBERT를 0.53→0.94로 구제**,
  XLM-R/CafeBERT 정렬도 0.966→1.0으로 동반 개선.
- **PhoBERT 통합** (`data_utils._encode_phobert` + `pyvi`): 단어분절 후 단어별
  BPE, 단어 char-span 정렬. fast-tokenizer·offset_mapping 없는 PhoBERT를 char
  span 계약에 편입(왕복 복원율 0.996). **최상위 모델로 진입.**

## 천장 원인 (본 리포트 범위 밖)

- **PROD(~0.71)·EVT(~0.80)** 가 단일 최대 천장. WikiANN-vi silver 노이즈(PROD
  precision 낮음) + 상대적 support 부족(EVT 474, PROD 2,005 vs PER/LOC 2만대).
- PII 5종은 0.98~1.00 포화. NER 5종 중 PER/LOC/ORG는 0.88~0.93.
- 개선 레버(범위 밖, 착수 시 이슈 분리): silver→gold 부분 정제(PROD/EVT),
  외부 코퍼스(VLSP/PhoNER, 단 오염 위험), EVT oversampling.

## 재현

```bash
ls data/wikiann_vi/pii_all.jsonl   # 중복 제거 후 38,371 (생성은 augmenters, gitignore)
uv sync                            # pyvi 포함
# 단일 모델 5-fold (fold 0..4 반복 후 pooling)
for f in 0 1 2 3 4; do
  python -m ner.classifier --lang vi --model-name xlm-roberta-base \
    --kfold 5 --fold-index $f --output-dir results/classifier/vi/canonical5fold/xlm-roberta-base/fold$f
done
python -m pytest tests/ner/classifier/ -q
```

## 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/classifier/vi/canonical5fold/<모델>/fold{0..4}/metrics.json` | fold별 메트릭 |
| `results/classifier/vi/canonical5fold/pooled_summary.json` | 5모델 pooled 집계 |

`results/`는 gitignore — 본 리포트 표가 영구 인용 단일 출처.
