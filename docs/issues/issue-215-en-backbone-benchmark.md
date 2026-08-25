# issue-215: 영문(en) BERT 백본 NER 벤치마크 — base 급 6종 baseline 확립

- Issue: https://github.com/groovallstar/ner-pipeline/issues/215
- PR: https://github.com/groovallstar/ner-pipeline/pull/217
- 브랜치: `feat/issue-215-en-backbone-benchmark`
- 승인일: 2026-08-21
- **측정 상세·수치·해석: `docs/reports/english-bert-classifier-benchmark.md`**

## 배경 (왜)

**영문 데이터는 있는데 그것을 학습할 백본이 정해져 있지 않았다.** #209 가
OntoNotes5 를 canonical 10종 평면으로 옮겨 `data/ontonotes_en/pii/` 까지
만들었지만 그 이슈는 "모델은 만들지 않는다"를 범위에 명시했고, `classifier`
는 `--lang {ja,vi,ko}` 만 받았다.

ko·ja·vi 는 각각 백본 벤치마크를 거쳐 출발점을 못 박았다(ko #122 →
`koelectra-base-v3`). en 만 그 출발점이 없어서, 아무 모델이나 골라 학습하면
그 수치가 좋은 것인지 나쁜 것인지 견줄 자리가 없었다.

## 목적

영문 canonical 10종 데이터에 base 급 인코더 백본 6종을 같은 레시피로
파인튜닝해 ko·ja·vi 와 같은 자(char-offset strict span F1)로 재고, 영문
baseline 백본을 확정한다. 임계값 게이트가 아니라 **baseline 수치 확립**이다.

## 범위

- 포함: `--lang en` 배선 · split 병합 · base 급 6종 비교 · 3-seed median 판정
  · 리포트 · `certified/` 승격
- 제외: large 급 백본 · 기성 OntoNotes 파인튜닝 모델 비교 · 공식 분할 경로
  신설 · 모델별 하이퍼파라미터 sweep · 서빙 배선 · 게이트 수치 설정

## 검증 (수락 기준)

| # | 기준 | 집행 | 상태 |
|---|---|---|---|
| 1 | `--lang en --smoke --group-key orig` 가 끝까지 돈다 | 스모크 로그 + pytest | ✅ |
| 2 | 병합본 76,378 행 · span 자기정합 불일치 0 · `orig` group-key 통과 | pytest (데이터 없으면 skipif) | ✅ |
| 3 | 6종 결과표 완성, 학습 불가 후보는 실패 양상·시도 LR 명시 제외 | metric JSON + 리포트 표 | ✅ |
| 4 | 1위를 2pp 안 후보 3-seed median 으로 확정, std 병기 | 후보별 metric JSON 3개 | ✅ |
| 5 | 학습시간은 1 epoch 순차 재측값만 인용 | 재측 로그 + 사람 | ✅ |
| 6 | 리포트 + `certified/` 승격 + 인용 표 출처 선언 | `commit_gate.py` 인용 대조 | ✅ |

전체 테스트 스위트 통과(927 passed · 3 skipped) · `ruff check` 통과.

## 결과 (요약)

**baseline = `roberta-base`** — NER-5 strict micro-F1 median 0.8795 / macro
0.8106. 수치·해석·후보별 표는 리포트에 있다.

핵심만 세 줄로:

- **micro 로는 1위를 못 가른다** — 상위 3종이 0.23pp 안인데 seed std 가
  0.40~0.51pp 다. 갈림은 macro·PROD·std 에서 났다.
- **단일 seed 였다면 결론이 달랐다** — seed 42 기준 1위·2위가 median 에서
  4위·5위로 내려간다.
- **`deberta-v3-base` 는 세 번째 언어에서도 학습 실패** — 500 스텝 안에 발산해
  NaN 고착.

## 결정 로그

- **2026-08-21**: 분할을 공식 3분할이 아니라 병합 후 재분할로 정함. 공식
  분할을 쓰려면 `data_utils.py` 에 새 경로를 뚫어야 하는데 그 파일은 기준
  파일이라 2단계 게이트를 타고, ko #122 와 조건도 갈린다. 대가로 외부 수치와
  직접 비교는 포기.
- **2026-08-21**: 병합 시 `split` 필드를 떨어뜨리기로 함. 남기면
  `validate_group_key` 가 그것을 "더 강한 그룹 키"로 잡아 `--group-key orig`
  를 막는다. 설계 시점에는 예상하지 못했고 실제로 합쳐본 뒤 드러났다.
- **2026-08-21**: 후보 범위를 base 급으로 한정. 영문 인코더 SOTA 는 large
  급에 있지만 ko·ja·vi 가 모두 base 급이라 네 언어 리포트를 같은 파라미터
  층위에서 비교하는 쪽을 택했다.
- **2026-08-21**: 순위를 단일 seed 가 아니라 3-seed median 으로 정하기로 함
  (정의 시점 결정). ja 리포트가 자기비판하듯 출하 체크포인트의 PROD R 0.9362
  는 5-seed 분산 안의 단일 draw 였고, 백본 간 차가 1~2pp 로 붙으면 단일 seed
  순위는 실력이 아니라 추첨을 적는 것이 된다. **결과적으로 이 결정이 결론을
  바꿨다.**
- **2026-08-21**: 학습시간 지표를 병렬 wall-clock 이 아니라 1 epoch 순차 독점
  재측으로 정함. 여러 후보가 한 GPU 를 나눠 쓰면 그 숫자는 모델의 성질이
  아니라 스케줄링의 부산물이 된다.
- **2026-08-21**: 같은 워크트리에서 claude 세션 둘이 같은 명령을 각자 실행해
  같은 출력 디렉토리에 두 프로세스가 쓰는 사고 발생. `electra_base_seed43`
  (gap −1,846초)과 `xlmr_base_seed44`(중복 이력)를 폐기하고 재실행. 시간
  재측값도 오염돼 전량 폐기.
- **2026-08-21**: 폐기했던 학습시간을 호스트가 빈 뒤 5종 1 epoch 독점 순차로
  재측해 기준 5 를 채움. 재측이 앞선 병렬 관측을 반증했다 — `roberta-base`
  가 `bert-base-cased` 의 2.1배로 보였던 것이 독점에서는 252초 대 254초로
  차이가 사라졌다. 오염된 값을 폐기한 판단이 결과적으로 옳았다. 재발 방지로
  실행 스크립트가 후보마다 시작 직전 GPU 점유 프로세스 수를 확인하고 0 이
  아니면 측정하지 않고 중단하게 했다.

## 관련 커밋

- `c1b7a58` feat(classifier): 영문 en 배선 + split 병합 도구
- `e63cd8d` feat(classifier): 영문 백본 벤치마크 결과 + 집계 도구
- `617a776` docs(reports): 영문 백본 벤치마크 리포트 + 이슈 문서 축약
- `9b48973` docs(classifier): 영문 백본 학습시간 1 epoch 독점 재측
