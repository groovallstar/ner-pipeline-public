# 일본어 BERT NER 분류기 — 벤치마크 요약 (canonical 10종 평면)

일본어 NER classifier (`src/ner/classifier/`, `--lang ja`) 의 **현재 상태
요약** — 현 production 결론, F1 추이, 라운드를 관통하는 핵심 교훈, 산출물
위치, 재현 명령.

Phase 0~8 각 단계의 가설·시도·결과 상세 (실측 수치 표 포함) 는
**`japanese-bert-classifier-history.md`** (1편, 동결) 에 있다. 본
문서에서 인용하는 Phase 번호는 모두 그 문서의 섹션을 가리킨다.

측정 프로토콜 개편 (#69) 이후의 새 실험은
**`japanese-bert-classifier-per-entity-diagnosis.md`** (2편, 층화 K-fold 프로토콜)
에 기록한다. 1편과 2편은 측정 프로토콜이 달라 수치를 직접 비교할 수
없다.

- 대상: Stockmark JA Wikipedia NER + 합성 PII 주입 데이터에 BERT family
  파인튜닝 (canonical 10종 평면 = NER 5 + PII 5)
- 평가: char-offset span F1. (시작, 끝, 종류)가 셋 다 정확히 일치해야만
  정답. + 보조 SemEval'13 Partial F1 (경계가 살짝 어긋난 건 부분 점수)
- 원시 수치 출처: `japanese-bert-classifier-history.md` +
  `docs/issues/issue-{40,45,48,51,56,58,61,64,66}-*.md`
- 베트남어 동일 평가 셋업의 분류기 결과는
  `docs/reports/vietnamese-bert-classifier-benchmark.md` 참조

## 한눈에 보는 결론 (현 production)

> ⚠️ 측정 프로토콜 개편 (#69) 에서 아래 측정값 (0.9644) 에 extra train
> 데이터의 train/test 중복 인플레이션 (~+1.7pp 추정) 이 포함된 것이
> 확인됐다. 중복 없는 test 부분 기준으로는 0.9478 (게이트 미달).
> 청정 재기준선 (rebaseline) 은 후속 이슈에서 결정 예정 — 상세:
> `japanese-bert-classifier-per-entity-diagnosis.md` §배경.

- 모델: `tohoku-nlp/bert-base-japanese-v3` (110M) + boundary-aware loss
- 데이터: `data/stockmark/pii_all_phonediv.jsonl` 5,270 행 (Stockmark
  JA Wikipedia NER + 합성 PII 주입) + S7N2 negative extra + S8 PROD
  도메인 seed (N=5) extra
  - **extra (보조 학습 데이터)**: 원본 5,270 행에 *더해* train 에만
    합치는 추가 문장. valid/test 에는 넣지 않아 평가가 부풀려지지
    않는다 (leak-free). 두 종류를 합쳐 `pii_extra_s8_prod_domain_N5.jsonl`
    (2,574 행) 로 제공
  - **S7N2 negative extra** (Phase 7, #58): *환각 줄이기용* 부정 예시.
    모델이 자꾸 entity 로 잘못 부르는 표현 (짧은 영어 약어·단일 한자·
    일반 명사) 을 "이건 entity 아님 (O)" 으로 표시한 문장을 N=2배
    복제해 추가. 학습 셋에 진짜 entity 로도 등장하는 표현 (ambiguous)
    은 자동 제외
  - **S8 PROD 도메인 seed (N=5) extra** (Phase 8, #61): *PROD recall
    높이기용* 보강. test 에서 자주 놓치는 PROD 도메인 (법안·서적·식품·
    교통·음악) 의 키워드 패턴으로 train+valid 의 해당 PROD 문장을 정밀
    선별해 N=5배 복제. 무작위 복제와 달리 *놓치는 도메인* 에 분포를 맞춤
  - **phonediv** = PHONE diversification. 합성 PHONE 번호 생성기를
    mobile 한 종류 → mobile/landline/tollfree/IP 4종 + 구분자 변형으로
    확장해 실제 전화번호 분포에 가깝게 만든 버전 (Phase 8)
- 라벨: canonical 10종 평면 = NER 5 (PER/LOC/ORG/PROD/EVT) + PII 5
  (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD)
- 학습: 80/10/10 split (seed=42), 5 epoch, BS 16, LR 5e-5, max_len 256
- 체크포인트: `results/classifier/ja_sweep/s8_prod_domain_N5_seed45/best/`
- 메트릭 (test 527 문장) — *출하 체크포인트* 와 *측정 baseline* 두 층위:
  - **출하 체크포인트 (seed 45, best-of-5)**: strict F1 = 0.9644
    (P 0.962 / R 0.967), PROD R = 0.9362, relaxed F1 = 0.9692
    (SemEval'13 Partial). *overall F1* 기준 최선 seed 로 선택 (PROD R
    기준이면 seed 42 의 0.9368 이 더 높다)
  - **측정 baseline (5-seed median, 공정 비교 기준)**: strict F1 =
    0.9634 (std 0.006) / PROD R = 0.9053 (std 0.033, range
    [0.857, 0.937]). best seed 의 0.9362 는 분산 안의 단일 draw

## 점진적 개선의 흐름 (F1 추이)

각 단계는 직전 단계의 진단·잔여 오류 분석을 입력으로 받아 다음 가설을
설계한다. 단일 시도가 아닌 누적 사이클이라는 점이 핵심.

각 행의 상세 (가설·시도·결과·per-entity 표) 는
`japanese-bert-classifier-history.md` 의 해당 Phase 섹션 참조.

| 단계 | strict F1 | 직전 한계 → 본 단계 시도 |
|---|---:|---|
| #40 baseline 첫 학습 | 0.9058 | (시작점) canonical 10종 평면 위 첫 학습 + SOTA 모델 sweep |
| #45 외부 코퍼스/증강 28회 시도 | 0.9077 | 어휘 다양성 부족 → 외부 코퍼스·LLM 증강. 대부분 회귀, 데이터 다양성 한계 확정 |
| #48 Tier 3 v2 (gold cleanup) | **0.9249** | gold 라벨 결함 의심 → test+valid+train 오답 전수 검수 → 996건 정답 보정 |
| #51 v3 (data 보정 라운드 2) | 0.9346 | 잔여 라벨 모호함 → schema 4영역 명문화 + 추가 40 corrections |
| #51 v3+CRF | 0.9311 | BIO 일관성 보강 시도 → 일부 NER 회귀로 채택 보류 |
| **#51 v3+boundary** | **0.9368** | BOUNDARY 오류 다수 → B/I weight 차등 손실, NER 5종 안정 향상 |
| #56 잔여 오류 정밀 진단 (S0) | (측정 X) | 어디서 잃고 있나 → confusion matrix, **환각 60건 중 ORG 28건 (47%)** 단일 최대 leak 식별 |
| #57 S1 threshold | — | 사후 calibration 으로 정밀도 회복 가능한지 → P/R 동시 만족 구조적 한계 확인, 미시도 결정 |
| **#58 v3+boundary+S7N2** | **0.9624** | 환각 surface 학습 부족 → 부정 예시 oversample (N=2, ambiguous 자동 제외) |
| **#61 S8 PROD domain N=5 (seed 45)** | **0.9644** | PROD recall 보강 → 도메인 휴리스틱 seed 정밀 선별 + PHONE 다양화 |

총 학습 횟수: 70회 이상 (sweep 18회 + 변종 6회 + #45 augmentation 28회
+ #51 4 variants + #58 5 variants + #61 multi-seed 10회 + 기타).

각 단계의 +0.01~+0.03pp 단위 누적이 합쳐져 baseline 대비 **+5.86pp**
순개선을 만들었다.

## 공통 측정 조건

| 항목 | 값 |
|---|---|
| 데이터 | Stockmark JA Wikipedia NER + 합성 PII 주입 |
| 분할 | 80/10/10 (`seed=42`). 5,270 행 기준 train 4,216 / valid 527 / test 527 |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid) |
| 하드웨어 | NVIDIA RTX A6000 49GB. Phase 8 은 multi-seed (42~46) |

> 행 수 변천: Phase 1 sweep 은 5,307 행 (train 4,247 / valid 530 /
> test 530). Phase 3 (#48) 에서 검수 불가 34 문장 삭제 + 3 collateral →
> **5,270 행**으로 확정, 이후 모든 측정은 5,270-row 기준. Phase 0 직후
> 변종 ablation (v1~v4) 은 80/20 분할 시점 측정이라 80/10/10 baseline
> (0.9058) 과 직접 비교하지 않는다 (Phase 1 §참고 표).

---

## 핵심 교훈 (모든 라운드를 관통하는 패턴)

> 다음 NER 작업에서 그대로 재사용할 수 있는, 라운드들을 거치며 반복
> 확인된 원칙들이다.

1. **누적 개선이 한 방을 이긴다.** 단일 기법으로 +1pp 넘게 올린 적은
   없다. 라운드마다 +0.2~1.0pp 짜리 시도를 7~8단계 쌓아 baseline 대비
   **+5.86pp** 순개선. "한 방" 보다 꾸준한 누적이 정답.

2. **데이터 정합성 개선 > 모델 측 변경.** gold cleanup (라벨 오류 996건
   보정) + 데이터 정리만으로 **+3.24pp**. 반면 SOTA 모델 sweep 18회는
   **+0.0pp**. 같은 시간이면 backbone 탐색보다 데이터 품질에 투자하라.

3. **augmentation 노이즈는 silver/gold 무관하게 독이 된다.** 외부
   코퍼스 KWDLC 통합 시 PROD F1 -9.96pp — 그쪽 ARTIFACT 정의가 우리
   PROD 와 미묘하게 어긋난 type drift (라벨 정의 불일치) 가 직격. 전문가
   라벨 (gold) 이라도 *스키마가 맞는지* 가 양보다 먼저. PROPOR 2026 의
   동일 caveat 재현.

4. **사후 calibration 은 P/R 동시 개선과 구조적으로 안 맞는다.** 출력
   임계값 (threshold) 을 학습 후에 조정하는 식의 처방은 precision↔recall
   맞교환이라 한쪽만 오른다. 양쪽을 같이 올리려면 *학습 단계* 처방뿐
   (Phase 7~8 이 실증).

5. **부정 예시 oversample 에는 ripple (파급) 효과가 있다.** PROD 환각
   seed 를 일부러 제거했더니 (v5) 오히려 PROD 가 악화. 환각 surface 를
   "이건 entity 아님 (O)" 으로 학습시키면 그 calibration 이 *다른 라벨
   판단에도* 기여한다 → 부정 예시는 통째로 주입하는 게 낫다.

6. **random oversample 의 실패 원인은 도메인 mismatch.** PROD 를
   무작위로 복제했더니 test FN 8건과 substring overlap 0/8 — 분포가
   안 맞으면 양을 늘려도 무효. 못 잡는 도메인을 키워드로 정밀 선별
   (휴리스틱 seed) 해야 분포가 일치한다.

7. **학습 자체가 non-deterministic (CUDA/cuDNN).** 데이터·seed·
   하이퍼파라미터를 고정해도 GPU 연산 순서 탓에 PROD R 이 ±0.025pp
   흔들린다. 단일 seed 측정값은 운빨이므로 **multi-seed median 비교가
   평가의 최소 조건.**

8. **boundary loss > CRF.** CRF 는 BIO 태그 전이를 강제로 일관되게
   (linear-chain) 만드는데, 그 제약이 일부 NER 클래스를 회귀시켰다.
   토큰의 B/I (시작/안쪽) 에 weight 를 차등 부여하는 단순 방식이 더
   안전했다.

9. **broad fix → 진단 → 표적 처방 순서가 효과적.** Phase 1~4 의 넓은
   수정은 어느 시점에 둔화됐다. Phase 5 진단으로 단일 최대 leak (ORG
   환각 28건) 를 특정한 뒤, 그 지점을 정조준한 Phase 7~8 표적 처방이
   가장 큰 +pp 를 만들었다. 막히면 더 넓히지 말고 멈춰서 진단하라.

---

## 산출물 위치

`results/` 와 `data/` 는 gitignore 대상. 옛 변종 (baseline / baseline_
corrected / v3_boundary / v3_crf / sweep 12 후보 등) 의 metrics.json 은
실험 종결 후 정리되어 더 이상 남아있지 않다. 옛 수치의 영구 인용 단일
출처는 `japanese-bert-classifier-history.md` 의 Phase 상세 표다.

문서 일단락 시 추가 정리: 비-production seed 의 학습 체크포인트 (best/·
checkpoint-*) 는 모두 제거하고 metrics.json 만 보존했다 (재현 명령으로
근사 복원 가능 — 학습이 non-deterministic 이라 동일 복원은 아님). 출하
체크포인트는 seed 45 (`s8_prod_domain_N5_seed45/best/`) 하나만 유지.

### 현재 남아있는 산출물

| 경로 | 내용 |
|---|---|
| `data/stockmark/pii_all_phonediv.jsonl` | 5,270 행 production 데이터 (PHONE 다양화 후) |
| `data/stockmark/pii_extra_s8_prod_domain_N5.jsonl` | S7N2 negative + PROD domain seed extra |
| `results/classifier/ja_sweep/s8_prod_domain_N5_seed45/best/` | **현 production checkpoint** |
| `results/classifier/ja_sweep/s8_prod_domain_N5_seed{42..46}/metrics.json` | S8 multi-seed (5 seeds) |
| `results/classifier/ja_sweep/baseline_phonediv_s7n2_seed{42..46}/metrics.json` | S8 fair 비교용 S7N2 control |
| `results/classifier/ja_sweep/s7_neg_n2_leakfree/metrics.json` | #58 S7N2 채택본 |
| `results/classifier/ja_sweep/phonediv_s7n2_prod_pos_N{1,2}/` | #61 random oversample 비교 측정 |
| `docs/issues/issue-{40,45,48,51,56,58,61}-*.md` | 라운드별 계획·결과 |

### 정리되어 제거된 옛 산출물 (인용은 본 문서 표를 참조)

- `results/classifier/ja_sweep/baseline/` (옛 5,307 행 baseline)
- `results/classifier/ja_sweep/baseline_corrected/` (Tier 3 v2, gold cleanup)
- `results/classifier/ja_sweep/baseline_v3/`, `v3_crf/`, `v3_boundary/`,
  `v3_combined/` (라운드 2 ablation)
- `results/classifier/ja/sweep/<모델>/` (SOTA sweep 7 후보)
- `src/ner/augmenters/entity_replacement/`, `external_corpus/` (#45 종결
  후 제거)
- `--oversample LABEL:N`, `--aug-train-data PATH` CLI 플래그 (#45 cleanup)
- `results/classifier/ja/` (옛 단일 run), `ja_sweep/phonediv_s7n2`·
  `s8_prod_domain_N5` (seed 미부여 단일 run) — multi-seed 표로 대체
- `ja_sweep/phonediv_s7n2_prodfnaware_comb_N2`·`prodfnaware_heur_N5`
  (S8 확정 전 중간 변종; heur_N5 산출물은 채택본 s8 과 동일) /
  `repro_check_adhoc_v1` (재현 확인용 일회성)
- `data/stockmark/pii_extra_s7n2_prodfnaware_{comb_N2,heur_N5}.jsonl`
  (중간 변종 입력; heur_N5 는 `pii_extra_s8_prod_domain_N5.jsonl` 와
  md5 동일한 중복)

## 재현 명령

```bash
# 의존성 (JA tokenizer)
uv add fugashi unidic-lite && uv sync

# production 모델 학습 (S8, seed 45)
CUDA_VISIBLE_DEVICES=0 uv run python -m ner.classifier --lang ja \
  --epochs 5 --batch-size 16 --max-length 256 \
  --data-path data/stockmark/pii_all_phonediv.jsonl \
  --data-extra-train-jsonl data/stockmark/pii_extra_s8_prod_domain_N5.jsonl \
  --boundary-b-weight 1.5 --boundary-i-weight 1.2 \
  --output-dir results/classifier/ja_sweep/s8_prod_domain_N5_seed45 \
  --seed 45 --metric-mode both

# multi-seed 안정성 평가
for seed in 42 43 44 45 46; do
  ... --seed $seed --output-dir .../seed${seed}
done

# 테스트
uv run python -m pytest tests/ner/classifier/ tests/ner/augmenters/ja/ -q
```
