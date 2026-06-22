# 베트남어 BERT NER 분류기 — 최종 스펙

베트남어 NER classifier (`src/ner/classifier/`, `--lang vi`) 의 출하 스펙.
현 시점 스크립트(`python -m ner.classifier`) 기준의 사용 모델·데이터·
엔티티·평가지표만 기록한다. 실험 경위·후보 비교·천장 진단은 시리즈 문서
참조: `vietnamese-bert-classifier-benchmark.md`(모델 선정 요약·누출 정정) /
`vietnamese-bert-classifier-history.md`(phase별 gold 계보·히스토리).
일본어 동일 셋업 스펙은 `japanese-bert-classifier-spec.md`.

## 모델

| 항목 | 값 |
|---|---|
| 사전학습 모델 | `vinai/phobert-base-v2` (134M, pyvi 단어분절 + word-level BPE) |
| 토크나이저 정렬 | 단어 char-span 정렬, 왕복 복원율 0.996 (`data_utils._encode_phobert`) |
| 손실 | 표준 cross-entropy (boundary 가중 미적용) |
| 학습 | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 |
| 체크포인트 선택 | `metric_for_best_model='eval_loss'` (valid) |
| seed | 42 |

## 데이터

| 항목 | 값 |
|---|---|
| 출처 | WikiANN-vi (silver) + 합성 PII 자연 주입 |
| 파일 | `data/wikiann_vi/origin.jsonl` (evtfix, gold 계보 §history) |
| 규모 | 37,706 문장 / 91,639 entity span / 유니크 원문 29,337 |
| 분할 | leak-free **group 10-fold** (`--group-key orig`, `seed=42`). fold 당 train ~30,090 / valid ~3,791 / test ~3,825. cross-fold 원문중복 0 |

> **누출-free 분할이 핵심**: WikiANN-vi 원문은 train/test 로 갈리면 모델이
> 일반화가 아닌 암기로 맞춰 NER 절대값을 부풀린다(원문 cross-fold 누출).
> 원문(`orig`) 단위 group-kfold 로 영구 차단 — pooled 가드가
> `cross_fold_orig_dups=0` 을 강제한다. 경위·정량: benchmark.md §원문 누출
> 정정 (#124).

## 엔티티 — canonical 10종 평면

NER 5종 + PII 5종. 단일 출처: `docs/manual/data/canonical-entity-schema.md`.

| 군 | 라벨 | 내용 |
|---|---|---|
| NER | `PER` | 인명 |
| NER | `LOC` | 지명·주소 |
| NER | `ORG` | 조직·시설 |
| NER | `PROD` | 상품·창작물(판매되는 제조물·작품 제목) |
| NER | `EVT` | 사건·행사 |
| PII | `DAT` | 날짜·시각 |
| PII | `EMAIL` | 이메일 주소 |
| PII | `PHONE` | 전화번호 |
| PII | `ID_NUM` | 식별번호(CCCD 등) |
| PII | `CREDIT_CARD` | 카드번호(13~19자리) |

## 평가지표

- 평가: char-offset span F1. strict = (시작, 끝, 종류) 셋 다 정확히
  일치해야 정답.
- 집계: leak-free group 10-fold pooled micro-average (전 fold test 예측을
  합쳐 한 번에 계산. 모든 문장이 정확히 1회 test → 유효 test = 37,706 문장
  전체).

### per-entity (strict, group 10-fold pooled)

| entity | F1 | Precision | Recall | support |
|---|---:|---:|---:|---:|
| **overall** | **0.9478** | 0.9440 | 0.9516 | 91,639 |
| EMAIL | 0.9998 | 0.9996 | 1.0000 | 7,182 |
| CREDIT_CARD | 0.9990 | 0.9993 | 0.9987 | 7,065 |
| PHONE | 0.9984 | 0.9980 | 0.9987 | 7,059 |
| DAT | 0.9982 | 0.9972 | 0.9992 | 7,104 |
| ID_NUM | 0.9979 | 0.9976 | 0.9982 | 7,101 |
| LOC | 0.9479 | 0.9377 | 0.9584 | 22,141 |
| PER | 0.9198 | 0.9182 | 0.9215 | 21,848 |
| ORG | 0.8714 | 0.8683 | 0.8744 | 8,033 |
| EVT | 0.8159 | 0.7793 | 0.8562 | 598 |
| PROD | 0.8070 | 0.8034 | 0.8107 | 3,508 |

PII 5종 포화(F1 ≥ 0.998). NER 5종 중 LOC·PER 0.92~0.95. 잔여 헤드룸은
ORG·EVT·PROD 에 집중 — WikiANN-vi silver 노이즈 + 상대적 support 부족
(EVT 598 = 10종 최저). PROD·EVT 천장 분해·게이트 경위는 history.md
Phase 4·5 참조.

## 재현

```bash
# 의존성 (VI tokenizer)
uv sync                            # pyvi 포함

# 10-fold 학습 (leak-free group-kfold)
for fold in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0 uv run python -m ner.classifier --lang vi \
    --model-name vinai/phobert-base-v2 \
    --data data/wikiann_vi/origin.jsonl \
    --kfold 10 --fold-index ${fold} --group-key orig --seed 42 \
    --output-dir results/classifier/vi/evtfix/phobert-base-v2/fold${fold}
done

# pooled 평가
uv run python -m ner.classifier.kfold_pool \
  --fold-dirs results/classifier/vi/evtfix/phobert-base-v2/fold{0..9} \
  --output results/classifier/vi/evtfix/phobert-base-v2/pooled_metrics.json
```

> `results/`·`data/` 는 gitignore. 위 수치는 현 시점 로컬 gold(evtfix,
> EVT 598) 재측정 — leak-free group 10-fold(`cross_fold_orig_dups=0`).
> #133 canonical 기록(benchmark.md/history.md)과 정합: overall
> 0.9474→0.9478(+0.04pp 동일). EVT 0.8249→0.8159·PROD 0.7992→0.8070 등
> per-entity 미세차는 GPU 비결정성 재학습 노이즈 내(EVT support 598,
> per-fold std ~0.06). 모델-선정 baseline·phase 진화·누출 정정은 시리즈
> 문서가 단일 출처.
