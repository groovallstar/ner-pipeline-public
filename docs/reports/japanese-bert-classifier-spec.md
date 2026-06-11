# 일본어 BERT NER 분류기 — 최종 스펙

일본어 NER classifier (`src/ner/classifier/`, `--lang ja`) 의 출하 스펙.
현 시점 스크립트(`python -m ner.classifier`) 기준의 사용 모델·데이터·
엔티티·평가지표만 기록한다. 실험 경위·후보 비교·진단은 시리즈 문서
참조: `japanese-bert-classifier-benchmark.md`(요약) /
`japanese-bert-classifier-history.md`(히스토리) /
`japanese-bert-classifier-per-entity-diagnosis.md`(엔티티 진단).

## 모델

| 항목 | 값 |
|---|---|
| 사전학습 모델 | `tohoku-nlp/bert-base-japanese-v3` (110M, slow tokenizer + MeCab) |
| 손실 | boundary-aware loss (B 토큰 ×1.5 / I 토큰 ×1.2 / O ×1.0) |
| 학습 | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 |
| 체크포인트 선택 | `metric_for_best_model='eval_loss'` (valid) |
| seed | 42 |

## 데이터

| 항목 | 값 |
|---|---|
| 출처 | Stockmark JA Wikipedia NER + 합성 PII 주입 |
| 파일 | `data/stockmark/pii_all_phonediv.jsonl` |
| 규모 | 5,270 문장 / 18,790 entity span |
| 분할 | 층화 10-fold (희소 entity PROD·EVT 보유 여부로 층화). fold 당 train 4,216 / valid 527 / test 527 |

## 엔티티 — canonical 10종 평면

NER 5종 + PII 5종. 단일 출처: `docs/manual/data/canonical-entity-schema.md`.

| 군 | 라벨 | 내용 |
|---|---|---|
| NER | `PER` | 인명 |
| NER | `LOC` | 지명·주소 |
| NER | `ORG` | 조직·시설(역·공항·병원·학교 등) |
| NER | `PROD` | 상품(판매되는 제조물) |
| NER | `EVT` | 사건·행사 |
| PII | `DAT` | 날짜·시각 |
| PII | `EMAIL` | 이메일 주소 |
| PII | `PHONE` | 전화번호 |
| PII | `ID_NUM` | 식별번호(마이넘버 등) |
| PII | `CREDIT_CARD` | 카드번호(13~19자리) |

## 평가지표

- 평가: char-offset span F1. strict = (시작, 끝, 종류) 셋 다 정확히
  일치해야 정답.
- 집계: 층화 10-fold pooled micro-average (전 fold test 예측을 합쳐 한
  번에 계산. 모든 문장이 정확히 1회 test → 유효 test = 5,270 문장 전체).

### per-entity (strict, 10-fold pooled — 임계값 적용)

per-class 신뢰도 임계값으로 NER 4종 저신뢰 예측을 걸러낸 출하 구성(임계값
적용, opt-in, default OFF). raw 모델(임계값 미적용)의 overall 은 0.9273 —
raw 대비 차이·임계값 분포 등 상세는 진단 문서
`japanese-bert-classifier-per-entity-diagnosis.md` Part 3(임계값 적용) 참조.

| entity | F1 | Precision | Recall | support |
|---|---:|---:|---:|---:|
| **overall** | **0.9361** | 0.9420 | 0.9302 | 18,790 |
| EMAIL | 0.9995 | 0.9990 | 1.0000 | 1,013 |
| PHONE | 0.9928 | 0.9887 | 0.9969 | 964 |
| CREDIT_CARD | 0.9922 | 0.9927 | 0.9917 | 961 |
| ID_NUM | 0.9819 | 0.9824 | 0.9813 | 965 |
| DAT | 0.9701 | 0.9744 | 0.9659 | 1,025 |
| PER | 0.9693 | 0.9566 | 0.9823 | 3,726 |
| LOC | 0.9073 | 0.9113 | 0.9034 | 2,899 |
| ORG | 0.9092 | 0.9239 | 0.8950 | 5,303 |
| EVT | 0.8661 | 0.9000 | 0.8347 | 992 |
| PROD | 0.8401 | 0.8739 | 0.8089 | 942 |

overall 전 지표 ≥ 0.93. PII 5종 + PER 포화권(F1 ≥ 0.969). 잔여 헤드룸은
NER 4종(LOC·ORG·EVT·PROD)에 집중.

## 재현

```bash
# 의존성 (JA tokenizer)
uv add fugashi unidic-lite && uv sync

# 10-fold 학습 + 임계값 fit/apply
for fold in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0 uv run python -m ner.classifier --lang ja \
    --data data/stockmark/pii_all_phonediv.jsonl \
    --boundary-b-weight 1.5 --boundary-i-weight 1.2 \
    --kfold 10 --fold-index ${fold} --seed 42 \
    --fit-abstain \
    --output-dir results/classifier/ja_sweep/<실험명>/fold${fold}
done

# pooled 평가 (baseline)
uv run python -m ner.classifier.kfold_pool \
  --fold-dirs results/classifier/ja_sweep/<실험명>/fold{0..9}
```

> `results/`·`data/` 는 gitignore. 위 수치는 현 시점 로컬 gold 기준 실측
> (`reach_threshold_sweep/kfold_prod`). gold 보정 경위는 엔티티 진단 문서,
> 영구 재현 기준 baseline 은 base gold strict F1 0.9195 참조.
