# NER 파이프라인 작업 지침

## 프로젝트 범위

이 저장소는 한국어·일본어·베트남어 NER 파이프라인을 제공한다. 주요 범위는
OpenAI 호환 LLM 라벨링, BERT 토큰 분류 학습, PII 증강, 평가·유효성 검사,
FastAPI 추론 서비스와 Docker 배포이다.

- canonical 엔티티 스키마의 단일 출처는
  `docs/manual/data/canonical-entity-schema.md`이다.
- `src/ner/`는 라이브러리, `src/server/`는 별도 top-level 서비스 패키지이다.
- `results/`는 휘발성 실험 산출물이고, `certified/`는 문서에 인용하는 결과의
  커밋된 원장이다.
- 디렉터리별 세부 계약은 작업 경로에서 가장 가까운 `AGENTS.md`를 함께 따른다.

## 개발 환경

- 호스트에서 직접 개발한다. Docker는 외부 추론 서비스와 서버 배포에만 쓴다.
- UV로 환경을 관리하고 `uv sync` 또는 `uv pip install -e .`로 설치한다.
- `PYTHONPATH` 환경변수를 설정하거나 주입하지 않는다. editable install과
  pytest 설정이 `src/`를 등록한다.
- 라이브러리는 `from ner...`, 서비스는 `from server...` 형태로 import한다.
- 실행 예시는 `uv run python -m ner.llm_eval --help`,
  `uv run pytest tests/ner -q`, `uv run ruff check src tests`처럼 작성한다.

## 안전 경계

- gold 스키마, metric, 유효성 판정, split 로직은 실험의 평가 기준이다. 이를
  변경할 때는 전후 점수를 직접 비교하지 말고 같은 기준으로 다시 평가한다.
- `certified/`에는 채택한 실험의 원본 metric JSON만 승격한다. 수치를 손으로
  만들거나 기존 증거를 덮어쓰지 않는다.
- 테스트를 삭제하거나 무조건 skip·xfail로 바꾸어 실패를 숨기지 않는다.
- `docs/reports/`와 `docs/issues/` 표에 새 metric을 인용할 때는 해당
  `certified/` 출처를 문서에 명시한다.
- 외부 서비스 기동, GPU 할당, 대용량 모델 다운로드 전에 해당 하위 지침과
  현재 자원 상태를 확인한다.

## 작업 절차

1. 요청을 검증 가능한 최소 변경으로 정리하고 관련 코드, 호출자, 테스트,
   문서를 읽는다.
2. 기능이나 버그 수정은 실패하는 테스트를 먼저 확인한 뒤 최소 구현으로
   통과시킨다.
3. 기존 유틸리티와 패턴을 우선 사용하며 관련 없는 리팩터링과 새 의존성을
   추가하지 않는다.
4. 구현에 직접 영향을 받는 테스트와 문서를 같은 변경에서 갱신한다.
5. 변경 범위의 diff를 검토하고 임시 파일, 디버그 출력, 생성 산출물을 제거한다.

## 테스트와 문서 검증

- 먼저 가장 가까운 대상 테스트를 실행하고, 영향 범위에 따라 전체 pytest,
  Ruff, 빌드 또는 smoke test를 확장한다.
- 기본 검증은 `uv run pytest ...`와 `uv run ruff check ...`를 사용한다.
- 코드 변경 후 `docs/manual/` 구현 맵과 `docs/manual/data/` 스키마의 갱신
  필요성을 확인한다.
- 검증하지 못한 GPU, 모델, 네트워크 의존 경로는 성공으로 간주하지 않고
  정확한 미검증 사유를 보고한다.

## 독립 검토

- 기준 파일이나 주요 런타임을 변경하면 완료 전에
  `requesting-code-review`를 사용해 읽기 전용 reviewer가 요구사항, 회귀,
  측정 타당성을 검토하게 한다.
- reviewer는 제품 파일이나 증거를 수정하지 않으며, 구현자는 지적을 직접
  검증한 뒤 반영한다.
- 최종 완료 보고 전에는 `verification-before-completion`을 사용해 모든 성공
  주장에 대응하는 최신 명령 출력과 diff를 확인한다.

## 커밋 규칙

- 제목과 본문은 한국어로 작성하고 `feat`, `fix`, `docs`, `test`, `refactor`,
  `build`, `chore` 같은 type 접두사만 영문으로 쓴다.
- 구현, 직접 관련된 테스트, 문서 갱신은 같은 원자 커밋에 둔다.
- 파일 수가 아니라 관심사와 독립 revert 가능성으로만 커밋을 나눈다.
- 커밋 전에 `git diff --check`, 대상 테스트, Ruff, `git status --short`를
  확인하고 사용자 소유의 관련 없는 변경을 포함하지 않는다.
