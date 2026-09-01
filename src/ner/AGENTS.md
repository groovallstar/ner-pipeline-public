# NER 라이브러리 지침

## 책임

`src/ner`는 라벨링, 평가, 증강, 토큰 분류 학습, 공용 metric, 실험 유효성,
배포 보조 스크립트를 제공한다. import는 항상 `ner` top-level 패키지에서
시작한다.

## 입력과 출력

- 입력은 canonical JSONL, LLM 응답, HuggingFace 모델·tokenizer, 학습·평가
  산출물이다.
- 출력은 정렬된 entity span, 증강 JSONL, 학습 checkpoint와 metric JSON,
  비교 verdict, 배포 패키지이다.

## 도메인 계약

- KO·JA·VI의 공통 canonical 평면은 NER 5종 `PER LOC ORG PROD EVT`와 PII
  5종 `DAT EMAIL PHONE ID_NUM CREDIT_CARD`이다.
- KO의 `DAT`는 KLUE에서 유래하고 나머지 PII 4종은 합성 주입한다.
- KO의 LOC/ORG 외연은 JA·VI와 다르므로 언어 간 ORG 수치를 직접 비교하지 않는다.
- 라벨 정의를 복제하지 않고 `docs/manual/data/canonical-entity-schema.md`를
  단일 출처로 사용한다.

## 구현 경계

- 공용 span/BIO metric은 `ner.metrics`, 비교 가능성·분산·누출 verdict는
  `ner.validity`에 둔다.
- 언어별 규칙은 `labelers/<lang>/`, 데이터 변환과 주입은 `augmenters/`,
  학습은 `classifier/`에 둔다.
- 여러 패키지를 조합하는 얇은 실행 진입점만 `scripts/`에 둔다.
- 타입 힌트와 표준 `logging`을 사용한다. 로그·예외·CLI help는 영문,
  docstring과 인라인 주석은 한국어로 작성한다.

## 금지사항

- canonical label 정의나 공용 metric 구현을 하위 모듈에 복제하지 않는다.
- 증강, 학습, 평가, 서비스 책임을 한 모듈에 섞지 않는다.
- import에 `src` 접두어를 붙이지 않는다.

## 검증

- 변경한 하위 모듈의 가장 가까운 테스트를 먼저 실행한다.
- 공용 경계 변경은 `uv run pytest tests/ner -q`와
  `uv run ruff check src/ner tests/ner`로 검증한다.
