# 실험 비교 유효성 지침

## 책임과 입력

이 패키지는 재학습 없이 K-fold 산출물의 비교 가능성, cross-fold 누출,
분산 대비 변화량을 판정한다. 입력은 `fold*/metrics.json`과
`pooled_metrics.json`이며 `ner.metrics`를 import하지 않는다.

## 판정 계약

- 평가 기준은 `lang`, `data_fingerprint`, `kfold`, `group_key`, `seed`,
  `stratify` 여섯 필드이다. 하나라도 다르거나 없으면 비교를 유효하다고
  간주하지 않는다.
- 누출은 신뢰 가능한 `group` 또는 `orig` 근거로 측정한 0만 통과한다.
  미측정과 약한 근거의 0은 `INVALID`, 관측된 누출은 `FAIL`이다.
- noise band는 entity별 `sigma_repro`가 있으면 사용하고 없으면
  `sigma_fold`로 fallback한다.
- 방향 일관성은 gain을 낮출 수만 있고 regression을 올리거나 false PASS를
  만들 수 없다.
- verdict 우선순위와 golden snapshot을 함께 유지한다.

## 금지사항

- 서로 다른 평가 기준의 fold를 paired delta로 비교하지 않는다.
- 누출 미측정을 0으로 채우지 않는다.
- 이 패키지에서 학습, gold 생성, split 생성, 한 run의 F1 계산을 수행하지 않는다.
- 결과를 본 뒤 threshold나 verdict 우선순위를 유리하게 바꾸지 않는다.

## 검증

- `uv run pytest tests/ner/test_validity.py -q`
- CLI는 `uv run python -m ner.validity --help`와 각 subcommand help로 검증한다.
- golden 변경은 의도한 판정 변화와 평가 기준 변경 근거를 함께 기록한다.
