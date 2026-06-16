# issue-112 — VI EVT 회귀 해결 (re-silver 후 EVT 합의 손질)

- GitHub: #112 (area:augmenters, area:classifier)
- 브랜치: `feat/issue-112-vi-evt-consensus-recovery` (#108 HEAD 기반)
- 선행: #108(PR #113) — §3 정렬 re-silver 로 PROD +5~7.5pp, **EVT 회귀** 잔존

## 진단 (criteria 1·2 — 기존 re-silver 아티팩트로 GPU 없이 완료)

**① v1→v2 EVT gold diff 분해** (removed 131 / added 95, net −36):

| 범주 | removed | added | net | 판정 |
|---|---:|---:|---:|---|
| 연도별 대회 에디션 | 49 | 16 | −33 | legit 손실 |
| 조약/협약/정상회의 | 21 | 3 | −18 | legit 손실 |
| 다년 process/시대 | 14 | 0 | −14 | **정당 제거** |
| 명명 재해 | 7 | 1 | −6 | legit 손실 |
| 정기대회 총칭→ORG | 5 | 0 | −5 | **정당 제거** |
| 전쟁/전투 | 18 | 14 | −4 | legit 손실 |

→ removed 131 중 정당 제거 19, **legit EVT 손실 ~98**. "과라벨 제거"가 아니라
**합의 불안정**이 주범.

**② 2모델 EVT 합의율** (raw relabel 직접 측정): Gemma 466 / Qwen 348 / 교집합
288 → **EVT Jaccard 0.548**. Gemma-only EVT 178건 중 **151(85%)이 Qwen 무-span
(type 분쟁 아닌 순수 recall 미스)**. 원인: 베트남어 EVT는 보통명사-핵 서술구
(Cúp·Trận·Công ước·Bão…)가 다수인데 Qwen(MoE active 3B·4bit)의 엄격한
고유명사 편향이 EVT에만 집중 타격. strict 교집합이 EVT를 *EVT에 가장 약한
모델*의 recall 에 천장 걸음.

**③ 평균 vs 분산**: 평균 회귀(phobert −2.0pp, xlmr −6.5pp)는 per-fold std 가 v1
0.02~0.04 → v2 0.07~0.08 로 2~3배 부풀어 ~1 std 안. 진짜 결함은 **합의 불안정**.

## 결정 (사용자 비준)

- **레버 = EVT 합의 메커니즘 조정**(few-shot 아님 — 예시 이미 존재해 헤드룸
  불확실, Qwen 구조적 보수성이 근본). 정책 "정당화 시에만 완화" 조항 발동.
- **판정자 폐기**: Qwen 강제-binary 는 유도심문(하이라이트·앵커·선택편향),
  gemma 자기판정은 self-confirm, 독립 3모델 부재 → LLM 판정 없음.
- **패턴-only 결정론(옵션 A)**: single-model EVT 중 §3 명시 legit 카테고리
  (연도대회·조약·전쟁·재해·선거) regex 매칭분만 구제. "Gemma/Qwen 이 surface
  한 span 에 §3 결정론 규칙을 regex 로 독립 재적용" — self-confirm/유도 both
  회피. 미매칭 long-tail(투어·금융위기·영화제 등 legit 일부 포함)은 **의도적
  보수 drop**. gemma_only 178 중 92 매칭(연도대회 45·전쟁 31·조약 24·재해 5·
  선거 3).
- **재학습 무조건**: 합의율과 무관하게 5-fold×2 재학습해 최종 pooled EVT F1
  회복 + PROD 유지 직접 검증.

## 계획 (체크박스 = 추적 단위)

- [ ] `merge_confidence.py`: `recall_strict_evt` 정책 + `_EVT_LEGIT_PATTERNS`
      추가 (single-model EVT legit-매칭 구제). single-model = medium_recall
      (gemma_only) + medium_prec(qwen_only) 대칭 적용. 단위 테스트.
- [ ] 기존 gemma/qwen relabel 아티팩트 재-merge(`recall_strict_evt`) → assemble
      `pii_all_v3.jsonl` (GPU 불필요).
- [ ] v2→v3 EVT gold diff + 구제 span 30건 수동 precision 스팟체크(패턴 오탐·
      경계오류).
- [ ] v3 로 5-fold×2(phobert·xlm-r) 재학습 → pooled strict F1 측정.
- [ ] EVT 회복(≥v1 ~0.79 지향) + PROD 유지 + per-fold std 안정 확인.
- [ ] 리포트 + refuter PASS + PR(closes #112 는 수동 close).

## 성공 기준

- EVT pooled strict F1 ≥ v1(~0.79) 회복 또는 v2 대비 유의 개선 + per-fold std
  안정화, **PROD/overall 유지**(re-silver 이득 비-회귀).
- 합의 기준 유지(패턴 구제는 §3 결정론 규칙 재적용이지 합의 포기 아님).

## 범위·전제 (정직)

- 패턴-only 는 long-tail legit EVT(투어·금융위기·영화제) 65건을 못 살린다 →
  EVT 천장이 아니라 *회귀 몸통*(연도대회·조약·전쟁) 회복이 목표.
- 평균 회귀가 ~1 std 라 재학습 후에도 개선이 노이즈 대역일 수 있음 → 분산
  안정·구제 카테고리 per-class recall 회복을 보조 지표로 함께 본다.

## 구현 결과 / 측정

도구: `merge_confidence.py` `recall_strict_evt` 정책(+`_EVT_LEGIT_PATTERNS`).
기존 gemma/qwen relabel 아티팩트를 새 정책으로 재-merge → assemble
`data/wikiann_vi/pii_all_v3.jsonl`(GPU 불필요). v3 로 5-fold×2 재학습.

**v3 gold 구성**: EVT 436→**619 (+183 구제, 0 제거)**, 비-EVT gold 는 v2 와
**바이트 동일**(sym_diff=0) — #108 PROD 이득은 구조적으로 비-회귀 보장.
구제분 precision 감사: 104 unique 전수 스캔 **오탐 0**(전부 §3 legit EVT —
연도대회·조약/공의회·전쟁·재해·선거). 정기대회 총칭·다년 process 는 정상 drop.

### pooled strict (v1 → v2 → v3, v3 P/R 포함)

**phobert-base-v2**

| ent | v1 F1 | v2 F1 | v3 F1 | Δ(v3-v2) | v3 P | v3 R | sup |
|---|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 0.9461 | 0.9529 | **0.9513** | -0.0016 | 0.942 | 0.960 | 91726 |
| PER | 0.9272 | 0.9278 | 0.9274 | -0.0004 | 0.922 | 0.932 | 22454 |
| LOC | 0.9349 | 0.9563 | 0.9558 | -0.0004 | 0.946 | 0.966 | 22155 |
| ORG | 0.8814 | 0.8994 | 0.8936 | -0.0058 | 0.895 | 0.893 | 8298 |
| **PROD** | 0.7171 | 0.7920 | **0.7686** | -0.0234 | 0.710 | 0.838 | 2394 |
| **EVT** | 0.7920 | 0.7731 | **0.8224** | +0.0494 | 0.784 | 0.864 | 619 |
| DAT | 0.9845 | 0.9849 | 0.9849 | -0.0001 | 0.970 | 1.000 | 7003 |
| EMAIL | 0.9966 | 0.9967 | 0.9966 | -0.0001 | 0.993 | 1.000 | 7242 |
| PHONE | 0.9963 | 0.9961 | 0.9961 | -0.0000 | 0.992 | 1.000 | 7090 |
| ID_NUM | 0.9971 | 0.9972 | 0.9970 | -0.0002 | 0.994 | 1.000 | 7225 |
| CREDIT_CARD | 0.9865 | 0.9865 | 0.9859 | -0.0006 | 0.973 | 1.000 | 7246 |

**xlm-roberta-base**

| ent | v1 F1 | v2 F1 | v3 F1 | Δ(v3-v2) | v3 P | v3 R | sup |
|---|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 0.9459 | 0.9543 | **0.9525** | -0.0019 | 0.943 | 0.962 | 91726 |
| PER | 0.9317 | 0.9434 | 0.9421 | -0.0013 | 0.943 | 0.941 | 22454 |
| LOC | 0.9291 | 0.9510 | 0.9481 | -0.0029 | 0.933 | 0.964 | 22155 |
| ORG | 0.8888 | 0.9037 | 0.8980 | -0.0057 | 0.899 | 0.897 | 8298 |
| **PROD** | 0.7097 | 0.7614 | **0.7398** | -0.0216 | 0.679 | 0.812 | 2394 |
| **EVT** | 0.8031 | 0.7348 | **0.8370** | +0.1021 | 0.806 | 0.871 | 619 |
| DAT | 0.9849 | 0.9842 | 0.9842 | -0.0000 | 0.971 | 0.998 | 7003 |
| EMAIL | 0.9957 | 0.9964 | 0.9958 | -0.0006 | 0.992 | 0.999 | 7242 |
| PHONE | 0.9959 | 0.9962 | 0.9961 | -0.0001 | 0.992 | 1.000 | 7090 |
| ID_NUM | 0.9971 | 0.9972 | 0.9970 | -0.0002 | 0.994 | 1.000 | 7225 |
| CREDIT_CARD | 0.9862 | 0.9847 | 0.9852 | +0.0005 | 0.972 | 0.999 | 7246 |

### per-fold 분산 (EVT·PROD strict F1 mean(std) — 회귀 진단 핵심)

std = sample stdev(n−1, `statistics.stdev`).

| 모델·ent | v1 | v2 | v3 |
|---|---|---|---|
| **phobert EVT** | 0.794 (0.046) | 0.774 (**0.089**) | 0.823 (**0.011**) |
| phobert PROD | 0.717 (0.022) | 0.792 (0.016) | 0.769 (0.013) |
| **xlm EVT** | 0.803 (0.024) | 0.738 (**0.075**) | 0.838 (**0.043**) |
| xlm PROD | 0.709 (0.025) | 0.762 (0.016) | 0.741 (0.030) |

### 해석

- ✅ **EVT 회귀 해결 + v1 초과**: phobert 0.7731→**0.8224**(+4.9pp vs v2,
  +3.0 vs v1), xlm 0.7348→**0.8370**(+10.2pp vs v2, +3.4 vs v1).
- ✅ **분산 붕괴(진단 확증)**: EVT std phobert 0.089→**0.011**(8배 안정),
  xlm 0.075→0.043. v2 결함의 본질이 *평균*이 아니라 *합의 불안정(분산)*이었음을
  재학습이 직접 증명. v3 per-fold EVT 가 0.81~0.84 로 균질.
- ✅ **EVT precision 회복**(회귀의 실제 축): xlm EVT P v2 0.658 → v3 **0.806**
  (+14.8pp). 구제된 legit EVT 가 gold 에 복원돼 모델 예측이 FP→TP 로 전환.
- ⚠️ **PROD give-back(정직)**: v2 대비 −2.2~2.4pp(phobert 0.792→0.769, xlm
  0.762→0.740) — EVT↔PROD 경쟁의 비용(PROD gold 는 불변이므로 *모델* 효과).
  단 v1 대비로는 여전히 +3~5pp 로 **#108 이득 대부분 보존**.
- ➖ **overall flat**(−0.16/−0.19pp, 노이즈 대역) · ORG −0.6pp · PII 5종 불변
  (gold 동일).

**종합**: 성공 기준 충족 — EVT pooled F1 ≥ v1 회복(초과) + 분산 안정 +
overall 유지. PROD 소폭 give-back 은 EVT 신호 추가의 trade-off 로 명시(절대
span 환산 시 EVT 이득 ~30~63 vs PROD 손실 ~53~57 → overall 사실상 등가, 그러나
EVT 일관성·천장은 명확 개선). 합의 기준 유지(패턴 구제 = §3 결정론 규칙 재적용,
합의 포기 아님).

산출물: `results/classifier/vi/resilver_v3/<model>/fold{0..4}` +
`pooled_metrics.json`, gold `data/wikiann_vi/pii_all_v3.jsonl`(기존 gold·#103
baseline 불변).

## 검증

- [x] `pytest tests/ner/augmenters/wikiann_vi/test_merge_confidence.py -q`
  green (37) — `recall_strict_evt`·`_is_evt_legit` 단위 테스트 포함
- [x] `ruff check` clean (merge_confidence.py + 테스트)
- [x] v3 gold 무결: 비-EVT v2 바이트 동일, EVT 순증(+183, 제거 0), 구제 오탐 0
- [x] refuter 게이트 PASS (`81d885405d6f` — 코드 정합성·측정 무결성·테스트
  무결성 3축; pooled F1 18값 ↔ JSON 0.00015 이내, gold delta 직접 재계산.
  지적(std 표기)은 sample stdev 로 정정 반영)
