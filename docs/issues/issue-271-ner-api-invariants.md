# issue-271: NER API 기본 동작의 정합성 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/271
- 브랜치: `feat/issue-271-ner-api-invariants`

## 배경

장문에서 한도를 넘는 단어가 그대로 인코더에 전달되어 뒷부분이 잘리며,
PhoBERT의 청크 길이 계산은 실제 인코딩의 pyvi 분절과 다르다. 반복 offset의
BIO 예측은 겹치는 span을 만들 수 있다. 연결 종료 후에도 큐 대기 요청을
추론한다.

## 목적과 범위

승인된 방향은 API 호출 방법과 기존 처리 구조를 유지하면서 정합성을 바로잡는
것이다. 새 설정, 동적 배치, 큐 감시, 추론 중단, 응답 후처리기를 추가하지 않는다.
한도 초과 단어의 추가 분할, PhoBERT 길이 계산 정합성,
반복 offset 디코딩, 슬롯 획득 직후 연결 확인, 429의 Retry-After를 포함한다.
문장 경계 확장, 겹치는 청크와 재병합, 서러게이트 처리는 보류한다.
동시 추론 기본값 8, 대기 큐 32, 대기 제한 10초는 기존대로 유지한다.

외부 연동 보완으로 `NERClient`의 선택적 429 재시도와 OpenAPI의 429 응답·
Retry-After 헤더 선언을 포함한다. 기존 클라이언트의 기본 호출은 한 번으로
유지하며 서버 설정은 추가하지 않는다.

## 성공 기준

- 기존 한도 이내 입력의 청크는 유지되며, 반환하는 모든 청크는 토큰 예산 이내다.
  공백 없는 장문의 내용과 원문 offset을 보존한다.
- PhoBERT 길이는 실제 인코더와 같은 분절 표면의 BPE 수로 계산한다.
- 반복 offset은 첫 서브워드의 라벨을 따른다. 동일 라벨은 span에 포함하고,
  충돌 라벨은 버리며, 서로 다른 offset의 정상 BIO 경계와 신뢰도 계산은 유지한다.
- 연결이 끊긴 대기 요청은 슬롯 획득 후 추론하지 않고 슬롯을 반환한다.
- 기존 429 본문을 유지하고 Retry-After: 1을 추가한다. 번역 guard에도 적용한다.
- 기존 설정의 명시적 값과 동시 추론 기본값 8을 유지하고 배포·문서도 일치시킨다.
- 클라이언트는 `max_retries`를 지정한 경우에만 429를 재시도하고 최종 응답을
  그대로 반환한다. 그 외 상태와 통신 예외는 자동 재시도하지 않는다.
- OpenAPI의 429 JSON·헤더 선언은 실제 응답과 일치하며 외부 연동 가이드에
  재시도 사용법과 상한 도달 시 처리 방법을 설명한다.

## 구현·검증 계획

1. 청크·PhoBERT 인코딩과 반복 offset의 실패 회귀 테스트를 작성하고 확인한다.
2. 기존 분할·디코딩 함수 안에서 수정한 뒤 해당 테스트를 실행한다.
3. 실제 ASGI 요청으로 대기 중 연결 종료·슬롯 회수·429 헤더를 검증한다.
4. 서버 기본 검사, classifier와 NER 검사, Ruff, diff 검사를 실행한다.
   tokenizer 검사는 로컬 파일만 사용하고 GPU·실서버 부하는 재기동하지 않는다.
5. 구현 맵과 REST 명세를 갱신하고 Superpowers 리뷰 서브에이전트로 검토한다.

## 결정 로그

- 2026-09-11: 사용자가 기본 동작의 복잡성이 늘지 않는 위 수정 방향을 승인했다.
  ASCII 마침표 추가와 엔티티 경계 개선은 입력 누락 수정과 분리하여 보류했다.
- 2026-09-11: 반복 offset의 충돌은 첫 서브워드 우선으로 결정한다. 별도 점수
  순위나 병합 단계를 만들지 않는다. 공유 디코더 수정으로 평가 결과가 달라질 수
  있으므로 과거 점수와 직접 비교하지 않으며 비교 시 같은 디코더로 재평가한다.
- 2026-09-11: 디코더의 다른 라벨 I 전환에서 직전 span이 유실되는 것도 회귀
  테스트로 확인하여 기존 span을 확정한 뒤 새 span을 시작하도록 수정한다.
  연결 종료가 확인된 요청은 추론 없이 499로 기록한다. 연결이 유지된 요청의
  응답 계약은 유지하고 실행 중 추론 취소는 도입하지 않는다.
- 2026-09-11: 사용자 정정에 따라 작업 기록을 `docs/specs/`에서 `docs/issues/`로
  옮기고 브랜치 접두사를 `fix/`에서 기본 규칙인 `feat/`로 바로잡았다.
  등록된 이슈 번호는 만들지 않으며 수정 범위와 성공 기준은 유지한다.
- 2026-09-11: 사용자가 이슈 문서명 규칙 적용을 요청하여 GitHub 이슈 #271을
  등록했다. 문서명을 `issue-271-ner-api-invariants.md`, 브랜치명을
  `feat/issue-271-ner-api-invariants`로 맞췄다. 번호 없는 로컬 기록에서 실제
  이슈와 연결된 기록으로 전환했으며 구현 범위와 성공 기준은 유지한다.
- 2026-09-11: 외부 연동 안내에 429의 두 발생 조건, 요청 단위 슬롯 사용,
  기존 JSON 구조 유지와 Retry-After 대기를 명시했다. 명세 설명만 보완하며
  API 구현이나 클라이언트에 새 기능·설정을 추가하지 않는다.
- 2026-09-11: 사용자가 추가로 예제 클라이언트의 429 자동 재시도와 OpenAPI
  선언을 모두 요청했다. 문서만 보완하던 범위에서 두 연동 기능을 추가한다.
  기본 재시도는 0회이며 지정한 횟수만 추가 요청한다. 현재 서버의 초 단위
  Retry-After를 사용하고 헤더 누락·잘못된 값에는 1초를 사용한다.
  단건·배치 재전송, 시도 상한, 다른 상태·예외의 비재시도, 실제 429 응답과
  OpenAPI의 정합성을 테스트한다. 외부 연동 가이드·구현 맵도 함께 갱신한다.
- 2026-09-11: 사용자 요청으로 외부 연동 가이드의 8번 Python 예제 절과
  해당 절 참조를 제거했다. 외부 문서에는 HTTP 계약·재시도 안내를 유지하고,
  구현한 선택적 재시도 기능과 OpenAPI 선언은 유지한다.
- 2026-09-11: 사용자가 실서버의 429 발생 요청 수 측정과 기존 동시성 8의
  비교를 요청했다. GPU 0과 포트 8018의 유휴 상태를 확인한 뒤 임시 서버로
  동시성 1과 8을 순차 측정한다. 같은 현재 코드·모델·입력을 사용하고 설정
  상한만 바꾸며, 원본 응답과 모델·코드 지문을 results/ner-api-overload/에
  별도 실행 디렉터리로 보관한다. 이전 코드 전체와의 비교는 하지 않는다.
- 2026-09-11: 비교 후 사용자가 요청 수용을 우선하여 기존 동시성 8을 선택했다.
  기본값 1로 낮춘 변경과 그에 따른 테스트·배포 설정·문서를 모두 되돌린다.
  처리량·지연 개선보다 기존 수용 동작을 유지하는 결정이며, 별도 비교 보고서는
  사용자 요청에 따라 삭제한다. 장문·span 수정, 연결 종료 확인, 429 헤더·
  재시도 기능과 OpenAPI 선언은 유지한다. 비교 원본은 Git 비추적 경로인
  `results/ner-api-overload/`에 남기며 별도 보고서를 추가하지 않는다.
- 2026-09-11: 사용자 요청에 따라 외부 연동 가이드와 REST 명세의 429 설명에서
  기존 호환성·추가 변경 표현을 제거하고 응답 코드·JSON·헤더·재시도 방법을
  현행 계약으로 명시한다. 변경 이력은 작업 이슈에 유지하며 기능은 바꾸지 않는다.

## 검증 결과

아래 구현 당시 검사와 이후 외부 연동·동시성 롤백 검사를 구분하여 기록한다.

장문·PhoBERT 회귀는 수정 전 4개 실패 후 12개 통과, 디코더 회귀는 수정 전
6개 실패 후 디코딩·신뢰도 검사 57개 통과를 확인했다. ASGI 연결 종료·429
회귀 4개의 실패 후 API·번역 검사 69개 통과를 확인했다.

- 서버 기본 검사: `uv run pytest tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py` → 186 passed, 기존 deprecation 경고 1개.
- `uv run ruff check src/server src/ner tests/server tests/ner`와
  `git diff --check`가 통과했다.
- `uv run python src/ner/scripts/audit_offset_alignment.py --help`가 통과했다.
- 로컬 ja·ko·vi·en 토크나이저로 원문 슬라이스 보존, 계산·인코딩 토큰 수 일치,
  256/4096 인코딩의 유효 토큰 일치를 확인했다. ja·ko는 공백 없는 20,000자,
  vi·en은 반복 문장 입력을 사용했다. ko 사례는 UNK로 축약될 수 있으므로
  모델의 문자별 인식이나 정확도를 입증하지 않는다.

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest tests/ner -q -rs
  --tb=short` → 891 passed, 18 failed. 실패는 EN 코퍼스 지문, EN·KO 배포
  패키지 지문, KO DAT·EVT·LOC/ORG 감사 원장과 현재 데이터의 불일치다.
  실행 로그: `/tmp/ner-api-invariants-pytest.log`.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest tests/ner/classifier
  -q -rs --tb=short` → 136 passed, 2 failed. 두 실패는 위 EN·KO 배포 지문
  불일치와 동일하다. 로컬 tokenizer를 사용하는 `test_encode.py`도 포함했다.
  실행 로그: `/tmp/ner-api-invariants-classifier.log`.
- 별도 Python 프로세스에서 `git show HEAD:src/ner/classifier/data_utils.py`의
  수정 전 모듈을 로드하고 위 실패 18개를 다시 실행하여 동일한 실패를 확인했다.
  기준 HEAD는 `c99d47fe8157e9fb28d638813aefd2b50942a038`이다. 다른 NER
  구현은 이번 작업에서 수정하지 않았다. 로그: `/tmp/ner-api-invariants-baseline.log`.
- Python 설정 생성과 `docker compose config --format json` 해석으로 기본값 1,
  명시적 동시성 8의 보존을 확인했다. 컨테이너는 기동하지 않았다.
- Superpowers `requesting-code-review`의 리뷰 서브에이전트가 최종 코드·테스트·
  문서를 읽고 대상 테스트 67개를 실행했다. Critical·Important·Minor 지적 없이
  통과했다. 기존 실패와 미검증 경로를 기록하라는 권고를 이 절에 반영했다.

기존 부하 수치는 재측정하거나 성능 개선 수치로 인용하지 않는다. 실모델 GPU·
live 서버·번역 backend와 변경 후 NER 정확도는 미검증이다. canonical 엔티티
스키마의 라벨·필드는 변경하지 않았다. 기존 데이터 불일치는 수정 범위에 포함하지
않으며 실패를 숨기기 위한 테스트·원장 변경도 하지 않았다. 위 임시 실행 로그는
장기 보존되지 않는다. 원격 이슈 #271을 등록했으며 커밋·배포는 수행하지 않았다.

### 외부 연동 기능 추가 검증

- 클라이언트·OpenAPI 회귀를 먼저 실행하여 17 failed, 3 passed를 확인했다.
  구현 후 API 검사를 포함한 대상 검사는 50 passed, 기존 deprecation 경고 1개였다.
- `uv run pytest tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py` → 202 passed, 기존 경고 1개.
- `uv run ruff check src/server tests/server`, `git diff --check`,
  `uv run python -m server.scripts.example_client --help`가 통과했다.
- Superpowers 리뷰 서브에이전트가 선택적 재시도·OpenAPI·연동 문서를 검토하고
  대상 검사 20개를 실행했다. Critical·Important·Minor 지적 없이 통과했다.
- `docs/manual/rest-api-integration-guide.md`의 Retry-After 미제공 설명을
  정정하고 발생 조건과 Python 예제를 추가했다. 원격 이슈에도 추가 범위를 반영했다.
- 네트워크 전송과 대기 경계를 대체한 클라이언트 검사 및 실제 ASGI 앱의
  429 검사로 검증했다. 외부 서버 통신, 실모델 추론과 브라우저의 Swagger 렌더링은
  실행하지 않았다. 공용 디코더에는 추가 변경이 없어 기존 NER 검증 결과를 유지한다.

### 동시성 변경 롤백 검증

- `ServerConfig` 직접 생성·환경변수 로드와 실제 Compose 해석에서 기본값 8,
  명시적 설정값 보존을 확인했다. 설정과 동시성 통합 테스트의 HEAD 대비 diff는 없다.
- `uv run pytest tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py` → 202 passed, 기존 deprecation 경고 1개.
- `uv run ruff check src/server tests/server`와 `git diff --check`가 통과했다.
- Superpowers 리뷰에서 롤백 누락·기존 사용자 변경 침범·삭제된 보고서의 참조가
  없음을 확인했고 수정 지적 없이 통과했다. 이번 롤백에서는 실서버를 재기동하거나
  배포하지 않았다.

### 최종 커밋 검증

- 서버 기본 검사 명령을 다시 실행하여 202 passed, 기존 deprecation 경고
  1개를 확인했다. 모델 통합·live 파일은 앞서 명시한 두 경로를 제외했다.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest
  tests/ner/classifier/test_data_utils.py tests/ner/classifier/test_encode.py
  tests/ner/classifier/test_confidence_threshold.py -q -rs --tb=short` → 67 passed.
- `uv run ruff check src/server src/ner tests/server tests/ner`와
  `git diff --check`가 통과했다. 기존 NER 전체의 데이터 불일치 실패는 남아 있으며
  이번 변경과 무관함을 확인한 앞선 비교 결과를 재사용한다.
- 최종 기능·문서와 롤백에 대한 Superpowers 리뷰 결과를 재사용한다. 현재 기능에
  미반영된 리뷰 지적은 없으며 사용자 소유의 다른 변경은 커밋에 포함하지 않는다.
- 로컬 브랜치에 작업 범위만 커밋으로 보관하며, 원격 푸시·병합·배포는 하지 않는다.
  커밋 식별자는 Git 이력으로 확인한다.
