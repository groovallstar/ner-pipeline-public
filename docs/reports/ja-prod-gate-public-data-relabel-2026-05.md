# JA PROD 게이트 — 공개데이터 재라벨 확장 test 실험 (2026-05)

이슈 #64 (`docs/issues/issue-64-prod-pr-gate-public-data-relabel.md`)의
S9-A~E 실행 결과. 목표: JA NER 분류기 PROD 의 P·R 동시 ≥ 0.92 를
*측정 가능하고 안정적으로* 달성 가능한지 타당성 게이트로 검증.

## 한 줄 결론 (최종, 검증 후)

**게이트 미달 + treatment 채택 불가.** 도메인 보강은 *도메인 편중 test*
에선 PROD P 를 크게 올렸으나, **운영 정식 test(원본 527/95 PROD)에선
overall F1 -0.63pp·PROD F1 -1.6pp 로 전반 회귀** — "정밀도 win" 은 측정
착시였다(과적합). 따라서 (C) production 반영은 *반영할 win 이 없어 불가*.
부차 발견: 리포트의 "PROD R 천장 0.905" 는 support 95 작은 test 의 과대치
이고, recall 한계의 본질은 **test PROD surface 의 80% 가 미학습 고유명사**
(open-vocab) 라는 구조적 한계. 권고: **production 은 S8 유지, treatment
미채택.** PROD 가 학습형 NER 에 부적합(유일 sub-0.92, 이질 grab-bag,
KLUE 는 product 미포함)하므로 게이트 재정의 또는 catalog-linking 별도 서빙.

## (구) 한 줄 결론 — 도메인 편중 test 기준 (착시, 위 검증으로 정정됨)

도메인 보강(treatment)이 도메인 편중 test 에서 PROD 정밀도 +21pp 로 보였
으나, 원본 test 재평가 결과 전반 회귀로 드러나 무효.

## 셋업

- baseline = 현 production 레시피 (S8): `pii_all_phonediv` +
  `pii_extra_s8_prod_domain_N5` + boundary loss(B1.5/I1.2)
- treatment = baseline + 신규 도메인 train extra(`train_s9c`, anchor_only
  111문장) ×5 oversample
- 공통 평가: **확장 test** = 기존 test 527문장(95 PROD) + 공개데이터 마이닝
  silver 379문장(566 PROD) = 906문장 / **661 PROD** (leak-free, test∩train=0)
- 5-seed(42~46) median, char-offset strict span F1

## 결과 (5-seed median, 확장 test)

| 그룹 | PROD P | PROD R | overall F1 | DAT R |
|---|---:|---:|---:|---:|
| baseline (S8) | 0.636 | 0.791 | 0.867 | 0.347 |
| **treatment (S8+신규×5)** | **0.849** | 0.782 | **0.914** | 0.600 |
| 게이트 | ❌ <0.92 | ❌ <0.92 | | |

per-seed PROD (P/R): baseline [0.57/0.70, 0.64/0.76, 0.65/0.85, 0.64/0.79,
0.62/0.84], treatment [0.86/0.80, 0.85/0.78, 0.87/0.79, 0.82/0.77,
0.84/0.77]. treatment 분산도 작아 안정적.

## 발견

1. **정밀도는 공개데이터로 크게 개선 가능.** 도메인 PROD 문장 보강 +
   2모델 교차검증 라벨이 PROD P 를 0.64→0.85(+21pp), F1 +4.7pp 끌어올림.
   파이프라인(명칭수집→마이닝→교차검증 재라벨)이 정밀도엔 확실히 유효.
2. **recall 이 벽.** 신규 도메인 train(111문장×5)을 넣어도 recall 0.79→
   0.78 로 불변. 모델이 *학습에서 못 본 long-tail 고유명사*(법령·작품명)를
   인식 못 하는 일반화 한계. train surface 와 test surface 가 달라 전이 안 됨.
3. **기존 천장(0.905)은 측정 아티팩트.** 동일 S8 모델이 95-PROD test 에선
   R 0.905, 661-PROD 대표 test 에선 R 0.791. **작은 test 가 성능을 과대평가**
   했다. n=95 의 CI ±6pp 문제(이슈 전제)가 실제로 확인됨.
4. **DAT recall 붕괴(0.35)도 같은 원인.** 확장 test 의 silver DAT 는
   `昭和22年`·`明治38年3月13日` 등 정상 날짜이나, 모델이 **和暦(연호) 날짜를
   학습에서 거의 못 봐서** 못 잡음. 라벨 품질 문제 아닌 일반화 갭 — PROD 와
   동형. treatment 에서 0.35→0.60 오른 건 신규 train 에 和暦이 섞여서.

## 측정 타당성 caveat

확장 test 는 *유효하지만 long-tail 편중* 이다 (마이닝이 의도적으로
약점 도메인의 희소 엔티티를 모음). 따라서 절대 수치는 production 분포보다
보수적(낮음)일 수 있다. 진실은 기존 test(0.905, 과대) 와 확장 test(0.78,
stress) 사이. 그래도 *recall 이 binding constraint* 이고 *0.92 가 현
방법으로 도달 불가* 라는 결론은 견고.

## 검증 — 원본(운영) test 재평가 (결정적)

train injection(v2 frac=0.5)으로 확장 test recall 이 0.78→0.875 로 올라
(사용자 지적대로 v1 은 학습이 굶주린 confounded 결과였음), 규모 2배(v3)
까지 진행. 그러나 **production 채택 전 필수 검증 = 원본 test 회귀 확인**
에서 treatment 가 전반 회귀 (재학습 없이 s9d3 체크포인트 원본 test 재평가):

| 원본 test 5-seed median | baseline(S8) | treatment | Δ |
|---|---:|---:|---:|
| overall F1 | 0.9607 | 0.9544 | **-0.63** |
| PROD F1 (P/R) | 0.910 (.91/.91) | 0.894 (.87/.88) | **-1.6** |
| LOC / EVT / DAT / CC | 0.952/0.933/0.986/0.976 | 0.936/0.921/0.968/0.952 | -1.6/-1.1/-1.7/-2.4 |

→ 확장 test 의 "정밀도 +21pp" 는 **도메인 편중 test 에서만의 착시**.
모델이 도메인 surface 에 과적합되어 *운영 일반 분포에선 PROD 포함 거의
모든 엔티티가 회귀.* (#45 Phase 2 "augmentation 회귀" 재현.)

## 결정 (이슈 #64)

- **production = S8 유지, treatment 미채택** (원본 test 회귀로 (C) 무효).
- PROD 0.92-both 는 본 셋업에서 도달 불가 — 게이트에서 PROD 제외/재정의가
  타당 (PROD 는 유일 sub-0.92·이질 grab-bag·KLUE 미포함 = 학습형 NER 부적합).
- 제품 추출이 다운스트림에 필요하면 **학습 NER 이 아닌 catalog gazetteer/
  entity-linking 으로 별도 서빙** (단 평가셋은 카탈로그와 독립이어야 함).
- 본 이슈의 영구 가치 = *지식*: 천장 0.905 는 작은 test 의 과대치, 도메인
  augmentation 은 과적합, PROD 는 open-vocab 구조 한계(test surface 80%
  미학습). 모델 변경 없음.

## 산출물

- 코드: `src/ner/augmenters/ja/domain_mine/`(c77ac09), classifier
  `--data-extra-test-jsonl`(9f97f1c), build_extra 분할(8ee7394)
- 데이터(gitignore): `data/domain_mine/`(names/mined/relabeled/extra),
  `results/classifier/ja_sweep/s9*`

## 데이터셋 병합 (사용자 지시, eval-오염 수용)

도메인 데이터 1022문장(confirmed+anchor_only, PROD 1419)을 운영 학습데이터
`pii_all_phonediv.jsonl` 에 직접 병합 (5270→6292행, PROD 2413).
- 백업: `data/stockmark/pii_all_phonediv_pre_domain.jsonl` (5270, 되돌리기용)
- ⚠️ canonical split 변경 → 본 리포트/벤치마크의 S8 절대수치는 더 이상
  재현 불가. **향후 측정 전 재-baseline 필수**, 과거 수치와 직접 비교 금지.
- 재현: `domain_mine` 파이프라인으로 도메인 데이터 재생성 후 append.
