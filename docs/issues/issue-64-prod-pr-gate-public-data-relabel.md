# issue-64: JA classifier PROD 게이트 — 공개데이터 canonical 재라벨로 P·R≥0.92 타당성 검증+시도

- Issue: https://github.com/groovallstar/ner_pipeline/issues/64
- PR: https://github.com/groovallstar/ner_pipeline/pull/65 (merged)
- 브랜치: `feat/issue-64-prod-pr-gate-public-data-relabel`
- 승인일: 2026-05-21

## 목적

PROD(P·R 동시 <0.92인 유일 엔티티)을 *측정 가능하고 안정적으로* P·R ≥
0.92로. "무조건 달성"이 아닌 **타당성 게이트(investigate→decide)**.

## 범위

- 포함: 약점 4도메인(법률/서적/교통카드/음악) 공개 JA 데이터셋 →
  canonical 재라벨(다중 LLM 교차검증) → train+test 양쪽 증강
- 제외: 식품(楽天 license 분기로 후속 라운드 연기) / 외부 corpus 무검증
  통합 / 일본어 사람 전수 검수 / focal·threshold(#57)

## 성공 기준

- [x] PROD test 95→661 PROD 확장(공개데이터)으로 측정 support 확보
- [ ] 확장 test 5-seed median PROD P·R ≥ 0.92 — **미달**(treatment
      P 0.85/R 0.78). 단 운영 원본 test 엔 S8 가 이미 ~0.915(±6pp CI)
- [x] 문서 갱신: `japanese-bert-classifier-benchmark.md` Phase 9 추가
- [x] 회귀 검증 — treatment 는 원본 test 회귀 → 미채택(S8 유지)로 운영 무회귀

## 위험·의존성

- 교차검증 공통 맹점 / silver test 순환성 / type-drift 재발(KWDLC -9.96pp) /
  P↔R 동시성(S8: R↑P↓) / 분포 불일치(#61 random oversample 0/8)

## 현 상태 (fact)

- S8 5-seed median: PROD P=0.889 / R=0.905. best seed45 R=0.936(분산 내 draw)
- PROD test support=95 → 0.92는 단일 측정 합격판정 불가(CI ±6pp). 동시개선은
  사후 calibration·기존데이터 재배치로 불가 실증(#57/#58/#61)
- canonical PROD = 제품·작품(상품·서비스·SW·작품·방송). book/music은 정당 PROD
- S9-A: 5도메인 공개 소스 확보. 법령=e-Gov법령API(CC BY호환·상용OK), 서적=
  openBD/NDL, 음악·교통카드=Wikidata(CC0), 식품=楽天데이터셋(⚠️학술기관限定).
  완제품 span-NER=J-NER(HF sergicalsix/Japanese_NER_Data_Hub, MIT, ENE157종)
  단 종당 5예 소규모 → test seed용. 대부분 명칭리스트라 문장마이닝+재라벨 필요

## 결정 로그 (append-only)

- 2026-05-21: deep-interview로 스코프를 "달성"이 아닌 "타당성 게이트"로 확정
- 2026-05-21: 새 데이터 = 공개 데이터셋 한정(직접 라벨링 불가), 검수는 다중
  LLM 교차검증으로 대체(일본어 사람 검수 불가 환경)
- 2026-05-21: S9-A 완료 — 공개 소스로 5도메인 커버 가능 확인. 단 J-NER 외엔
  명칭리스트라 문장마이닝+재라벨 자체 구축 필요. 식품은 license 분기 발생
- 2026-05-21: 식품 도메인 제외 결정(楽天 학술限定) → 4도메인으로 진행, 식품
  FN(玉すだれ·食べ比べセット)은 본 라운드 미해결로 후속 연기
- 2026-05-21: S9-B를 B1~B5로 분해(분해는 사람 주도), 마이닝 우선 채택.
  모듈 `augmenters/ja/domain_mine/`(schema/name_sources/sentence_miner/
  relabel/build_extra) 완료 — 4도메인 명칭(e-Gov+Wikidata+curated)→ja.wiki
  마이닝→Gemma+Qwen 합의→status 분배. 46 tests, 라이브 검증 완료
- 2026-05-21: B4 결함→수정(결정) — anchor를 gold로 덮으면 거짓 PROD 생성
  (株式会社シベール=ORG를 PROD 위조). apply_anchor로 교체해 2모델 합의가
  중재(confirmed→test/anchor_only→train/conflict 드롭). 사용자 선택 정합
- 2026-05-21: S9-C-1 classifier `--data-extra-test-jsonl` 추가(검증) +
  S9-C-2 스케일 실행 완료 — 4도메인×150명칭 마이닝 527문장→2모델 재라벨→
  분배: **test(confirmed) 379문장/566 PROD**, train(anchor_only) 111/139.
  leak-free 0. test PROD 200+ 목표 초과 → 규모 확대 불요
- 2026-05-21: S9-D 실험 착수(자율) — baseline=S8 extra만 vs treatment=S8+
  train_s9c×5(N=5, 기존 도메인-seed 레시피 정합). 둘 다 확장 test(기존95+
  566=661 PROD) 위 5-seed 학습
- 2026-05-21: S9-E 게이트 판정 = **미달**. treatment PROD median P 0.849
  (+21pp)/R 0.782, baseline P 0.636/R 0.791. 정밀도는 공개데이터로 크게
  개선되나 recall 이 벽(long-tail 고유명사 일반화 한계). 추가 발견: 기존
  천장 0.905 는 95-PROD 작은 test 의 과대치(661-PROD 대표 test 에선 S8 R
  0.79). 상세·권고 → 아래 §결과·검증 (상세). 결정 분기(목표 재정의/recall
  전용 접근/정밀도 win 수용/측정 표준화)는 사용자 판단 대기
- 2026-05-22: 사용자 지적 — finding(recall 무효)은 confounded. v1 treatment
  학습 주입이 anchor_only 139 PROD뿐, confirmed 566은 전부 test행이라 학습이
  굶음. build_extra에 confirmed test/train 분할(`--confirmed-test-frac`) 추가.
  v2: frac=0.5 → test 283 PROD(+기존95=378), train 422 PROD(3배↑). baseline
  vs treatment(×5) 5-seed 재학습으로 "학습 주입" 효과 재검증 중
- 2026-05-22: v2 결과 — 가설 확인. treatment PROD R 0.782→**0.875**(+9.3pp),
  P 0.848 유지, F1 0.941, DAT R 0.60→0.87. 게이트는 여전히 미달이나 둘 다
  0.92 근접(P 0.848/R 0.875). recall 은 데이터 주입으로 움직임 = 정밀도가
  남은 주 갭. 규모 확대(300/도메인, frac=0.5) 재학습 진행 중. 정밀도는
  추후 negative(S7) 레버 필요 가능성
- 2026-05-22: v3(규모 2배, train 861 PROD) — recall 0.875→0.883 천장(positive
  한계수익), P 0.834. test PROD surface 80%가 미학습 = open-vocab 구조 한계.
  recall<0.92이고 negative는 recall을 더 낮춰 게이트 통과 불가 확정
- 2026-05-22: (C) 검증 실패: treatment를 *원본 test* 재평가 시 overall F1
  -0.63pp·PROD -1.6pp 전반 회귀(확장 test의 정밀도 win은 도메인 과적합 착시,
  #45 재현).
- 2026-05-22: (3) 라이트 도즈 — 원본 test 5-seed median: x1 overall F1 중립
  (-0.0002)이나 PROD F1 -1.8pp(P -5.3/R +1.1), DAT -2.8; x2 전반 회귀. **어떤
  도즈도 운영 PROD 개선 못 함 = 데이터 증강 경로 소진.**
- 2026-05-22: **최종 — production 모델(체크포인트) 변경 없음.** 재프레이밍:
  *운영 test에선 S8 PROD가 이미 P 0.915/R 0.915* (support 95, ±6pp CI라
  0.92와 구분 불가 = 사실상 도달). "0.92 미달"은 long-tail stress test 한정.
  PROD 유지. 상세 → 아래 §결과·검증 (상세)
- 2026-05-22: **데이터셋 병합(사용자 지시, eval-오염 경고 2회 후 수용).**
  도메인 데이터 1022문장(confirmed+anchor_only, PROD 1419)을
  `pii_all_phonediv.jsonl` 에 직접 append → 5270→6292행(PROD 2413). 백업
  `pii_all_phonediv_pre_domain.jsonl`(5270). **결과: canonical split 변경 →
  벤치마크 S8 0.9644/PROD 수치 재현 불가, 향후 평가는 도메인 포함 분포 →
  재-baseline 필요.** 재현: domain_mine 파이프라인 재생성 후 append

## 미해결 질문 (open question)

- compute 예산(seed×run 횟수) 상한은?
- 게이트 미달 시 결정 분기 우선순위(완화 vs 추가 라운드 vs best-seed 출하)?

## 구현 단계 (가변)

- [x] S9-A 공개 JA 데이터셋 후보 5도메인 커버리지·라이선스 조사 (완료)
- [x] S9-B canonical 교차검증 재라벨 + offset 검증 (domain_mine B1~B5, 46 tests)
- [x] S9-C test 95→661 PROD 확장(공개데이터) + classifier extra-test 옵션
- [x] S9-D baseline vs treatment 5-seed median 학습 (확장 test)
- [x] S9-E 판정 = 미달(P 0.85/R 0.78), 결정 = production S8 유지

---

## 결과·검증 (상세)

> 구 `docs/reports/ja-prod-gate-public-data-relabel-2026-05.md` 를 본 이슈
> 문서로 병합.

### 한 줄 결론

게이트 미달 + treatment 채택 불가. 도메인 보강은 *도메인 편중 test* 에선
PROD P 를 크게 올렸으나 운영 원본 test 에선 전반 회귀(정밀도 win 은 과적합
착시). "PROD R 천장 0.905" 는 support 95 작은 test 의 과대치이고, recall
한계 본질은 test PROD surface 80% 가 미학습 고유명사(open-vocab). →
production S8 유지, treatment 미채택.

### 셋업

- baseline = 현 production 레시피(S8): `pii_all_phonediv` +
  `pii_extra_s8_prod_domain_N5` + boundary loss(B1.5/I1.2)
- treatment = baseline + 신규 도메인 train extra ×N oversample
- 확장 test = 기존 527문장(95 PROD) + 공개데이터 silver(566 PROD) = 661 PROD
  (leak-free, test∩train=0), 5-seed(42~46) median, char-offset strict span F1
- 라운드: v1(confirmed 전부 test) → v2(frac=0.5, confirmed 절반 train) →
  v3(규모 2배). recall 추이 0.782→0.875→0.883 (decision log 참조)

### 결과 — 확장 test, v1 (5-seed median)

| 그룹 | PROD P | PROD R | overall F1 | DAT R |
|---|---:|---:|---:|---:|
| baseline (S8) | 0.636 | 0.791 | 0.867 | 0.347 |
| **treatment (S8+신규×5)** | **0.849** | 0.782 | **0.914** | 0.600 |
| 게이트 | ❌ <0.92 | ❌ <0.92 | | |

### 발견

1. **정밀도는 공개데이터로 개선 가능하나 *도메인 매칭 test 한정*** =
   과적합. 원본 test 재평가서 회귀(아래 §검증). #45 Phase 2 "augmentation
   회귀" 재현.
2. **recall 이 벽** — test PROD surface 의 **80% 가 미학습 고유명사**.
   positive 추가는 train 에만 쌓이고 test(leak-free)는 늘 새 surface →
   전이 안 됨. v2→v3(2배)에도 recall +0.8pp 한계수익.
3. **기존 천장 0.905 는 측정 아티팩트** — 동일 S8 가 원본 test(95 PROD)엔
   R 0.91, long-tail 확장 test(661)엔 0.79. 운영 분포에선 이미 0.92 와
   통계적으로 구분 불가.
4. **DAT recall 0.35 도 동형**(和暦 날짜 미학습). silver 라벨 자체는 정상.

### 측정 타당성 caveat

확장 test 는 *유효하나 long-tail 편중*(마이닝이 약점 도메인 희소 엔티티를
모음) → 절대 수치는 운영 분포보다 보수적. 진실은 기존 test(0.905, 과대)와
확장 test(0.79, stress) 사이. 그래도 *recall 이 binding constraint* 라는
결론은 견고.

### 검증 — 원본(운영) test 재평가 (결정적)

production 채택 전 필수 검증. treatment(s9d3) 체크포인트를 *원본 test* 로
재평가(5-seed median):

| 원본 test | baseline(S8) | treatment | Δ |
|---|---:|---:|---:|
| overall F1 | 0.9607 | 0.9544 | **-0.63** |
| PROD F1 (P/R) | 0.910(.91/.91) | 0.894(.87/.88) | **-1.6** |
| LOC/EVT/DAT/CC | 0.952/0.933/0.986/0.976 | 0.936/0.921/0.968/0.952 | -1.6/-1.1/-1.7/-2.4 |

라이트 도즈(원본 test 최적화) 추가 검증: x1 overall F1 중립(-0.0002)이나
PROD F1 -1.8pp, x2 전반 회귀. **어떤 도즈도 운영 PROD 개선 못 함.**

### 결정

- **production S8 유지, treatment 미채택**(원본 test 회귀).
- PROD 0.92-both 는 본 셋업서 도달 불가 — PROD 는 유일 sub-0.92·이질
  grab-bag·KLUE 미포함 = 학습형 NER 부적합. 제품 추출 필요 시 학습 NER 이
  아닌 **catalog gazetteer/entity-linking 별도 서빙**(평가셋은 카탈로그와
  독립이어야).

### 데이터셋 병합 (사용자 지시, eval-오염 수용)

도메인 데이터 1,022문장(confirmed+anchor_only, PROD 1,419)을
`pii_all_phonediv.jsonl` 에 병합(5,270→6,292행, PROD 2,413). 백업
`pii_all_phonediv_pre_domain.jsonl`(5,270). ⚠️ canonical split 변경 →
벤치마크 S8 절대수치 재현 불가, **향후 측정 전 재-baseline 필요**. 재현:
domain_mine 파이프라인 재생성 후 append.

### 산출물

- 코드: `augmenters/ja/domain_mine/`(c77ac09), classifier
  `--data-extra-test-jsonl`(9f97f1c), build_extra 분할(8ee7394)
- 데이터(gitignore): `data/domain_mine/`, `results/classifier/ja_sweep/s9*`
