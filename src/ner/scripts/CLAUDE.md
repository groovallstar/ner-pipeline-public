# src/ner/scripts/ — 보조 스크립트 가이드

## 폴더 정의

`scripts/`는 **프로젝트의 여러 패키지를 가로질러(import) 엮어 실행하는
cross-package 기능 스크립트**를 둔다. 한 패키지에 속하는 도구가 아니라
`ner.classifier`·`ner.metrics`·`ner.labelers`·`ner.augmenters` 등 **둘 이상**을
끌어와 배포 추론·평가·점검 같은 작업을 수행하는 실행 진입점이 여기 산다.

- **단일 패키지·단일 언어 전용** 도구는 여기 두지 않고 성격에 맞는 패키지로
  보낸다 — KO 라벨러/gold 도구는 `labelers/ko/`, VI silver·자연주입 도구는
  `augmenters/wikiann_vi/`, 공용 PII 주입은 `augmenters/pii/`.
- 재사용 라이브러리 로직은 각 패키지에 두고, 여기엔 그것들을 조합해 돌리는
  얇은 오케스트레이션만 둔다.

## shell 래퍼 규약

자주 실행하거나 인자가 많은 python 스크립트는 **`uv run` shell 래퍼(`.sh`)**로
감싸 호출 부담을 줄인다(긴 절대경로·다중 인자를 매번 타이핑하지 않도록).
래퍼는 스크립트 위치 기준으로 리포지토리 루트로 이동한 뒤
`exec uv run python ... "$@"` 로 본체에 인자를 그대로 넘긴다.

## 인벤토리

| 파일 | 역할 |
|------|------|
| `eval_ja_ner_test.py` | 배포된 JA NER 모델 test 추론 — `ner.classifier`(confidence_threshold·data_utils·train_eval) + `ner.metrics` 를 가로질러 임계값 적용 후 P/R/F1·단계별 타이밍 출력(절대경로만 허용) |
| `eval_ja_ner_test.sh` | 위 `.py`를 `uv run`으로 감싸 프로젝트 `.venv`에서 실행하는 래퍼(shell 래퍼 규약의 표준 예시) |
| `eval_vi_ner_test.py` | 배포된 VI NER 모델 test 추론 — `ner.classifier`(confidence_threshold·data_utils·train_eval) + `ner.metrics` 를 가로질러 임계값 적용 후 P/R/F1(strict)·단계별 타이밍 출력(thresholds.json 없으면 raw 폴백, 절대경로만 허용). 앞 100문장만 평가(`N_TEST=100`) — 배포 test 는 orig 그룹 단위 홀드아웃이라 ≥100(현재 101)이라 오버슈트분을 잘라 100에 맞춘다 |
| `eval_vi_ner_test.sh` | 위 `.py`를 `uv run`으로 감싸 실행하는 래퍼 |
| `eval_en_ner_test.py` | 배포된 EN NER 모델 test 추론 — VI 판과 같은 구성이되 평가 대상(`--limit`, 기본 0=전수)과 화면 표시(`--show`, 기본 100)를 따로 받는다. EN 배포 test 는 7,637행이라 하나로 묶으면 지표를 전수로 재는 일과 태깅을 눈으로 보는 일이 서로를 막는다 |
| `eval_en_ner_test.sh` | 위 `.py`를 `uv run`으로 감싸 실행하는 래퍼 |
| `audit_offset_alignment.py` | 토큰 char-offset 정렬 감사 — `ner.classifier`(data_utils) + `ner.metrics` 를 가로질러 ① 엔티티 라벨을 받은 토큰이 엔티티 밖 char 를 물거나 종료 경계의 길이 0 조각인 건수 ② gold 라벨 → decode → gold 엔티티 strict 복원율(모델이 완벽해도 못 넘는 채점 상한)을 학습 없이 결정적으로 잰다. `--model-dir` 를 주면 고정 체크포인트를 재평가하며 토큰별 예측 label id 시퀀스 해시를 함께 남겨, 정렬을 바꾼 전후로 점수 차이가 정렬에서만 왔는지 확인할 수 있다 |
| `build_en_ner_prod.py` | EN 배포 패키지 포장 — `python -m ner.classifier` run 을 `/data/ner/en/{model,data,metrics.json,MODEL_CARD.md}` 로 옮긴다. **학습은 하지 않는다**: 학습 경로를 복제하면 배포 체크포인트가 CLI 아닌 이 스크립트의 산물이 돼 벤치마크 원장과의 수치 대조가 서로 다른 코드 경로 비교가 된다. 분할 JSONL 은 run 이 저장하지 않아 같은 인자로 다시 유도하고, 유도한 크기가 run 기록과 어긋나면 중단한다 |
