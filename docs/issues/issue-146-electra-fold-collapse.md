# issue-146 — ELECTRA fold 붕괴 조사 (재현성·안정성)

## 배경

#140 koelectra 10-fold 재측정 중 fold 하나가 학습에 실패했다(train loss
~0.76 정체, F1 0). 같은 seed 로 다시 돌리니 수렴해, 데이터·토크나이저
버그는 아니었다. 이슈는 원인을 fp16·warmup 부재·cuDNN 비결정성으로 보고
bf16 과 warmup 으로 "근본 제거"하자고 제안했다.

## 수락 기준 (원안 → 결과)

원안은 "bf16+warmup 으로 KO 10-fold × N-seed 를 재실행해 fold 붕괴 0 을
입증"하는 것이었다. 조사 결과 **이 기준은 달성 불가로 판명**됐다 — 검증된
근본 수정이 존재하지 않는다. 따라서 작업을 "근본 제거"에서 **하드닝 + 기록 +
운영 완화**로 재정의했다(사람 승인).

## 조사 요약

koelectra 약 920회 학습으로 데이터·헤드 init·정밀도·결정성·분할 5축을
스윕했다. 핵심:

- 런 간 분산의 주범은 시드 안 된 분류 헤드 init(~1pp)이고 cuDNN 비결정성
  기여는 ~0.03pp 로 미미했다.
- 붕괴는 데이터 분할에 의존한다 — 단일 split 240회는 0건, 10-fold 는 80회
  중 3건(folds 0·4·9 집중).
- fix 귀속: warmup 은 반증(0.06 에서도 2/60 붕괴), seed 고정은 불충분
  (1/60), bf16 은 빈도를 낮추나 제거 못 하고 깨끗한 사전등록 검증이 비유의
  (fp16 3/150 vs bf16 1/150, Fisher p=0.62).

수치 영구 인용: `docs/reports/korean-bert-classifier-fold-collapse.md`.

## 결정 — 코드

채택(가산적·기본 학습 동작 불변):

- `--train-seed`(기본 None = 무시드, 기존 동작 BC; 값 지정 시 헤드 init·셔플
  고정으로 재현 가능한 run).
- `--deterministic`(train-seed 필수; cuDNN·CUBLAS 까지 결정화해 바이트 단위
  재현, 느림).
- `metrics.json` 에 `train_seed`·`deterministic`·`precision` 기록(측정
  무결성 — #140 의 precision 을 사후 확인할 수 없던 갭을 닫음).

폐기:

- `--warmup-ratio`(반증됨), `--legacy-unseeded-init`(양성대조 스캐폴딩).

기본 학습 동작을 바꾸지 않았으므로 기존 KO baseline 은 그대로 유효하고
재측정이 필요 없다.

## 운영 완화 (근본 수정 부재 대응)

- canonical 측정은 `--train-seed` + `--deterministic` 로 안정성과 재현성을
  확보한다(~1.7배 느림).
- 또는 어느 fold 가 F1≈0 으로 무너지면 `--train-seed` 만 바꿔 그 fold 를
  재실행한다. 정상 분포가 0.84±0.003 으로 좁고 붕괴는 F1≈0 의 명확한
  outlier 라, 파국만 골라 재실행하는 것은 선택 편향이 아니다.

## 검증

- `pytest tests/ner/classifier` 76 passed, `ruff` clean.
- BC: `--train-seed` 미지정 시 `train_seed=null`(무시드) — 기존 학습 경로와
  동일.
- 가드: `--deterministic` 단독 지정 시 `--deterministic requires
  --train-seed` 에러.
