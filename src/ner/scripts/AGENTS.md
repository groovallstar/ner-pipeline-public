# NER 보조 스크립트 지침

## 책임

이 디렉터리는 `ner.classifier`, `ner.metrics`, `ner.labelers`, `ner.augmenters`
중 둘 이상을 조합하는 얇은 실행 진입점만 둔다. 재사용 로직은 소유 패키지에
남긴다.

## 계약

- 자주 쓰는 Python 진입점의 shell wrapper는 저장소 루트로 이동한 뒤
  `exec uv run python ... "$@"`로 모든 인자를 전달한다.
- 배포 평가 스크립트는 학습하지 않고 고정 test와 저장된 threshold로 추론한다.
- `build_ner_prod.py`는 기존 run을 검증해 패키징하며 새 모델을 학습하지 않는다.
- 추출 스크립트와 사람·모델 판정 스크립트를 분리해 candidate 모집단이 판정
  규칙에 의해 좁아지지 않게 한다.

## 금지사항

- 단일 패키지 또는 단일 언어 로직을 이 디렉터리로 옮기지 않는다.
- 언어별 배포 포장 로직을 복제하지 않는다.
- audit 기대값을 같은 주입·변환 코드로 다시 만들어 항진명제 검사를 만들지 않는다.

## 검증

- `uv run pytest tests/ner/scripts -q`
- shell 변경은 `bash -n src/ner/scripts/*.sh`
- 각 Python CLI는 `uv run python <script> --help`로 import와 인자 배선을 확인한다.
