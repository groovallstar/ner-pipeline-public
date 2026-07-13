# issue-172 · variance 게이트를 validity 최상위 패키지로 승격

`feat` 이슈 · `area:metrics` · behavior-preserving(자 불변)

## 배경 — 이름-거짓 + 위치-거짓

직전 리팩터가 507줄 `variance.py`를 `metrics/variance/`(comparability·leakage·noise·gate·_common) 서브패키지로 쪼갰다. 두 결함이 남았다:

- **이름-거짓**: `variance`(분산)는 다섯 중 `noise`(σ_fold·σ_repro·밴드) 하나만 정직하게 덮는다. comparability(비교 유효성)·leakage(누출)·gate(판정)는 분산이 아니다.
- **위치-거짓**: 이 게이트는 `metrics/`(bio_metrics·span_metrics)를 **한 줄도 import 하지 않는다**. 하는 일은 끝난 두 실험의 `fold*/metrics.json`·`pooled_metrics.json`을 읽어 비교 유효성·개선 실측을 판정하는 것 — "한 run의 F1 계산"과 다른 고도의 관심사이고 자체 CLI를 가진다.

## 설계 결정

| 결정 | 내용 |
|---|---|
| 최상위 승격 | `metrics/variance/` → `src/ner/validity/` (metrics 밖, `llm_eval`·`classifier`처럼 CLI 가진 최상위 패키지) |
| 정직한 개명 | `noise.py` → `variance.py` — 이제 `variance`가 분산(σ)만 가리킨다 |
| 세 검증 독립 유지 | comparability·leakage·variance는 이미 독립 함수(`check_comparable`·`leakage`·`fold_std`/`repro_std`). 새 verdict 껍데기 안 씌움 |
| `compare()` 동결 | `gate.py`의 verdict 사다리 verbatim. 골든 특성화 테스트가 전체 출력 byte-동일을 잠금 |

## 범위 경계 — 오늘 refactor / 나중 하네스는 자

인터뷰에서 "완료 판단 오케스트레이터(실험별 검증 부분집합 조합)"라는 향후 용도가 드러났다. 정의-시점 반박자가 **"per-block 균일 verdict를 지금 정의하는 것 = 미룬 조합/채점규칙(자)을 정의하는 것"**임을 지적 — `compare()`의 verdict 사다리는 세 검사가 얽혀 behavior-preserving하게 per-block으로 쪼갤 수 없다(INVALID가 FAIL의 위·아래 양쪽). 그래서 오늘은 **이동·개명·동결까지**, 균일 verdict contract·부분집합 조합·완료 판단은 **나중 하네스**(그것이 자를 바꾸므로 그때 정의-시점 반박자 재적용)로 미뤘다.

## 정의-시점 반박자 (REVISE → 반영)

초안(각 검증이 균일 verdict를 내고 조합기가 재조립)이 **REVISE**로 반려됐다. 잡은 것:

- `compare()` 사다리의 INVALID/FAIL **우선순위 역전**(target-missing INVALID가 leakage FAIL보다 아래) → 단일 전역 우선순위로 재현 불가
- "각 검증 단독 verdict가 compare()와 일치"는 **대조할 oracle 없음** → 반증 불가·gameable
- leakage의 **2-run 비단조 축약**(observed+미측정 → FAIL, INVALID 아님)이 per-run 시그니처와 충돌
- "동일"이 미정의라 전체-dict 표류가 부분키 테스트를 통과 → **골든 특성화 테스트가 유일한 실질 보증**

반영: 범위를 이동·개명으로 축소, `compare()` 동결, 이동 전 캡처한 전체-dict 골든으로 behavior-preserving을 검증 가능하게 못 박음.

## 구현

- `git mv`로 이력 보존 이동 + `noise.py`→`variance.py`, 내부 import 전부 `ner.validity`로
- `tests/ner/golden/validity/*.json` — 이동 **전** 캡처한 11개 시나리오(전 verdict 분기 + 반박자가 짚은 미검증 조합: 누출 observed+미측정→FAIL, 누출 FAIL+타깃 부재→FAIL, 타깃 부재→INVALID)
- `tests/ner/test_validity.py`(← `test_variance.py` rename) — `TestGoldenSnapshot`이 골든을 재구성해 `compare()` 전체 출력 == expected 단언
- stale 참조 갱신: classifier 주석·`docs/manual/pipeline/**`·`src/ner/CLAUDE.md`(metrics 표에서 variance/ 제거 + validity/ 섹션 신설)
- validity docstring에 나중 하네스 제약("variance paired Δ는 comparability 전제") 명시

## 검증

- `tests/ner` **473 passed**(기존 33 + 골든 12 포함), ruff clean
- CLI `python -m ner.validity {std,repro,compare}` 동작
- 구 경로 `import ner.metrics.variance` → `ModuleNotFoundError`(dangling 없음)
- **결과-시점 반박자 PASS** — 골든 hollow 아님(정밀 float 일치), `compare()` 본문 byte-동일(`git diff -M` 0줄), 테스트 무결성 유지

## 미룬 것 / follow-up

- 루트 `CLAUDE.md`의 `variance.py` 참조 3건(자 비유, line 94/100/102)은 별개 refuter-gate 작업이 편집 중이라 얽힘 회피 위해 보류 — 그 작업 커밋 시 함께 갱신
- `docs/issues/issue-169-*.md`의 `metrics/variance.py`는 종료 이슈 아카이브(당시 스냅샷)라 불변
- **나중 하네스**: 균일 verdict contract + 실험별 검증 부분집합 조합 + 완료 판단 → 채점규칙 변경이므로 별도 이슈 + 정의-시점 반박자
