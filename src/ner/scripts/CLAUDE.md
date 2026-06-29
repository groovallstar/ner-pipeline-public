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
| `build_vi_ner_prod.py` | VI NER 배포 패키지 빌드 — test 100문장을 누출-free(orig 그룹)로 홀드아웃 + 나머지 train/valid 8:2 로 phobert 단일 학습, `data/stockmark/*_ner_prod_seed*` 포맷(`model/`+`data/`+`metrics.json`+`MODEL_CARD.md`)으로 저장(VI 는 신뢰도 임계 미적용) |
| `eval_vi_ner_test.py` | 배포된 VI NER 모델 test 추론 — `ner.classifier`(confidence_threshold·data_utils·train_eval) + `ner.metrics` 를 가로질러 임계값 적용 후 P/R/F1(strict)·단계별 타이밍 출력(thresholds.json 없으면 raw 폴백, 절대경로만 허용). 앞 100문장만 평가(`N_TEST=100`) — 배포 test 는 orig 그룹 단위 홀드아웃이라 ≥100(현재 101)이라 오버슈트분을 잘라 100에 맞춘다 |
| `eval_vi_ner_test.sh` | 위 `.py`를 `uv run`으로 감싸 실행하는 래퍼 |
