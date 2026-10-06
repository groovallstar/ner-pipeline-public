# 평가 입력의 문장 수 불일치를 감지하여 F1 과대평가 방지

## 배경·목표

정답과 예측의 문장 수가 다른 입력을 평가하면 일반 `zip`이 남은 문장을
버려 F1과 support를 잘못 산출할 수 있다. 두 offset span 평가 함수가
이 입력을 명시적으로 거부하게 한다.

## 범위·제약

- strict·relaxed offset F1 함수의 문장 수 검증과 회귀 테스트를 추가한다.
- 문장 수 불일치는 양쪽 개수를 포함한 영어 `ValueError`로 거부한다.
- 문장 수가 같으면 엔티티 수 차이, 빈 문장과 전체 빈 입력을 허용하고
  기존 점수·반환 형식을 유지한다.
- BIO 평가, span 매칭 알고리즘, 데이터셋·실험 수치는 변경하지 않는다.
- 사용자가 현재 지침의 동작을 시험하도록 요청했으므로 기존 이슈 검색·재사용과
  기존 작업 문서·매뉴얼 참조를 이번 작업에 한해 생략한다. 현재 지침과
  코드·테스트를 근거로 삼고 입력 계약은 함수 docstring에 기록한다.

## 완료 기준

1. strict·relaxed 모두 정답 또는 예측의 문장이 누락되면 거부한다.
2. 한쪽 전체 입력만 비어 있는 경우와 빈 문장의 누락도 거부한다.
3. 같은 길이의 빈 입력, 엔티티 수가 다른 문장과 정상 점수는 보존한다.
4. 실패 테스트를 먼저 확인하고 대상 테스트, NER 전체 검사와 Ruff를 실행한다.
   실패·skip·실행 불가는 구분하여 기록한다.
5. 독립 코드 리뷰를 수행하고 필요한 지적을 처리한다.

## 현재 상태

- 이슈: https://github.com/groovallstar/ner-pipeline/issues/297
- 브랜치: `feat/issue-297-eval-sentence-count`.
- 분기 기준: fetch로 확인한 `origin/develop`,
  `2ca5b701fc92b8183823f86884fd51417935d896`. PR 대상은 `develop`이다.
- 분기 직전 미커밋 변경은 없고 로컬·원격 develop의 커밋 차이는 0이었다.
  현재 작업 공간을 사용한다.
- 입력 검증·회귀 테스트·검증·독립 리뷰를 마쳤다.
  기본 GPU에서 실패한 전체 검사는 GPU 2에서 550개 모두 통과했다.
- 로컬 구현 완료 후 통합 게시 요청을 받았다. 커밋·푸시·초안 PR 게시를
  진행하며 원격 게시 결과는 이슈에 연결한다.

## 결정 이력

- 사용자의 재시작 요청에 따라 앞선 시도는 취소했고 새 이슈로 진행한다.
  앞선 시도의 변경은 작업 공간에서 제거했다. 취소된 변경은
  `/tmp/ner-eval-restart-gnghyA/`에 보관했으며 구현 근거로 사용하지 않는다.
- `AGENTS.md`, `docs/AGENTS.md`, `docs/issues/AGENTS.md`,
  `src/ner/AGENTS.md`, `tests/ner/AGENTS.md`와
  `docs/manual/specs/coding-conventions.md`를 적용한다.
- 패딩이나 잘라내기는 누락을 숨길 수 있으므로 채점 전에 거부한다.
  이 검증으로 같은 길이의 문장 순서 오류까지 감지할 수는 없다.

## 검증 방법

- `uv run pytest tests/ner/test_span_metrics_input.py -q`로 실패를 확인한다.
- 기존 metric·호출자 검사를 실행한 뒤 `uv run pytest tests/ner -q`로 확장한다.
- `uv run ruff check src/ner tests/ner`와 `git diff --check`를 실행한다.
- 현재 지침의 `requesting-code-review` 절차로 독립 리뷰를 수행한다.

## 관련 자료

- [평가 함수](../../src/ner/metrics/span_metrics.py)
- [입력 계약 테스트](../../tests/ner/test_span_metrics_input.py)

## 결과·검증

- 두 공개 함수의 입구에서 공통 검증 함수를 호출한다. 문장 수 불일치 시
  `gold`·`pred` 개수가 담긴 `ValueError`를 발생시키고 채점을 진행하지 않는다.
  각 함수의 docstring에 입력 계약과 예외를 추가했다.
- 현재 코드의 `benchmark_runner`와 `train_eval`은 같은 처리 단위에서
  gold·pred 목록을 만든다. 호출자에 별도 가드를 복제하지 않았다.
- 검증 원본은 `results/issue-297-validation/`에 보관한다. Git 비추적 경로이며
  커밋·원격 보관된 근거는 아니다. 아래 결과는 이 작업의 변경에 대한 실행이다.
- RED: `uv run pytest tests/ner/test_span_metrics_input.py -q`는
  16 failed, 6 passed였다. 불일치 사례 16개 모두 `DID NOT RAISE ValueError`로
  실패했다. 원본은 `red.log`다.
- GREEN: `uv run pytest tests/ner/test_span_metrics_input.py
  tests/ner/test_span_metrics.py tests/ner/classifier/test_confidence_threshold.py
  tests/ner/classifier/test_kfold_pool.py -q`는 59 passed였다.
  원본은 `targeted.log`다.
- `uv run ruff check src/ner tests/ner`와 `git diff --check`는 통과했다.
  Ruff 원본은 `ruff.log`다.
- 첫 `uv run pytest tests/ner -q`는 1 failed, 549 passed였다.
  `test_evaluate_model_scores_a_fragmented_address_as_one_hit`가
  `train_eval.py`의 tensor GPU 이동에서 CUDA OOM으로 실패했다.
  변경된 metric 호출 전의 실행 실패이며 기본 GPU 0의 여유 메모리는
  `nvidia-smi` 조회 시 21MiB였다. 원본 `ner-suite.log`를 보존한다.
- 테스트를 변경하거나 skip하지 않고 여유 메모리가 있는 GPU 2를 지정해
  `CUDA_VISIBLE_DEVICES=2 uv run pytest tests/ner -q`를 재실행했으며
  550 passed, skip 없음이다. 원본은 `ner-suite-gpu2.log`다.
- 작업 문서의 코드·테스트 상대 링크는 pathlib로 실제 파일 존재를 확인했다.
- 리뷰 방식은 Superpowers `requesting-code-review`의 독립 서브에이전트다.
  당시 코드·새 테스트·작업 문서에 Critical·Important·Minor 지적은 없었다.
  이후 추가된 검증 결과와 완료 판단은 그 리뷰에 포함되지 않았다.
  상대 링크 지적은 실제 경로 확인 후 리뷰어가 철회했으며 수정할 결함이 없었다.
  리뷰어는 대상 검사·Ruff 결과를 전달받고 최초 전체 검사 로그를 직접 확인했다.
  GPU 2 재실행 성공은 주 작업자가 명령 종료 코드와 최종 출력을 확인했다.
- 리뷰의 판정 제외 항목인 동일 길이의 문장 순서 오류·양쪽 동시 누락과
  relaxed 매칭 방식 변경은 이번 입력 길이 검증의 범위 밖으로 유지한다.
  CUDA OOM은 실행 환경 실패로 기록하며 코드 수정의 성공 근거로 쓰지 않는다.

## 완료 기준 대조·남은 작업

- 기준 1·2: 두 함수의 양방향 불일치·빈 문장 누락 16개 사례가
  수정 전 실패하고 수정 후 통과했다.
- 기준 3: 정상 입력 신규 검사 6개와 기존 metric 검사가 통과했다.
- 기준 4: 대상 59개와 GPU 2 조건의 전체 550개, Ruff·diff 검사가 통과했다.
  최초 기본 GPU 실패는 위에 별도로 남겼다.
- 기준 5: 독립 리뷰를 마쳤으며 남은 지적은 없다.
- 로컬 구현은 완료했다. 원격 게시 전이라는 이유로 이슈를 열어 두었던 판단은
  게시 여부를 추가 완료 조건으로 삼은 오류였다. 기능 완료와 게시 상태를 구분한다.
- 과거 평가 수치 영향 조사, 서버 실모델·live 검증은 수행하지 않았다.
  기존 문서 검토·갱신은 사용자의 이번 시험 제약에 따라 수행하지 않았다.

## 통합 게시 결정

- 사용자가 현재 변경을 함께 커밋·푸시하고 PR을 만들도록 요청했다. 기존
  `feat/issue-298-readme-refresh`에서 이 변경과 README·문서 지침 변경을
  목적별 커밋으로 나누어 게시하며 PR 대상은 `develop`이다.
- 게시 전 fetch 결과 HEAD와 `origin/develop`은 모두
  `2ca5b701fc92b8183823f86884fd51417935d896`이며 포함된 기존 전용 커밋은 없다.
- 신규 테스트의 함수 간 빈 줄을 현재 규약에 맞추었다. 이전 맥락이 남은 같은
  대화에서 재시작했으므로, 기존 문서를 재열람하지 않았다는 사실만 확인할 수
  있으며 과거 정보의 영향 없는 독립 하네스 시험이었다고 입증할 수는 없다.
- 이번 검증은 새 로그 파일을 만들지 않고 콘솔 결과를 기록한다. 기존 검증
  원본은 보존한다. 최종 기록을 포함하여 변경된 부분을 독립 재검토한다.

## 통합 게시 전 최종 검증

- `uv run pytest tests/ner/test_span_metrics_input.py
  tests/ner/test_span_metrics.py tests/ner/labelers/test_ko_ner_prompts.py -q`:
  66 passed. `uv run ruff check`와 `git diff --check`도 통과했다.
- GPU 2와 `/data/ner/{ja,ko,vi,en}/model` 존재를 확인하고
  `CUDA_VISIBLE_DEVICES=2 uv run pytest tests/ -q -rs`를 실행했다.
  결과는 849 passed, 1 failed, 2 skipped, Starlette 사용 중단 경고 1건이다.
- 실패는 `tests/server/test_en_email_probe.py::test_en_email_probe_strict_recall`의
  단독 EMAIL 재현율 0.889가 기준 0.90보다 낮은 것이다. 누락 입력은
  `r.patel@bluepine.io`, `sam_lee99@acme.co.uk`였다.
- 기준 커밋 `2ca5b70`의 `span_metrics.py`를 `git show`로 읽어 메모리 모듈에
  로드한 뒤 위 테스트를 동일 GPU에서 실행해 같은 실패를 확인했다. 서버 코드와
  해당 probe에는 기준 커밋 대비 diff가 없고, probe는 변경된 F1 함수를 호출하지
  않는다. 이번 수정의 회귀로 판단하지 않으며 테스트나 기준은 바꾸지 않았다.
- skip 2건은 `NER_SERVER_TEST_LIVE_TRANSLATE=1` 미설정으로 번역 live 검사가
  실행되지 않은 것이다. 번역 성공으로 보고하지 않는다.
- 전체 검사가 모두 통과한 상태는 아니므로 기존 실모델 실패를 명시한 초안 PR로
  게시한다. 이 실패의 모델 수정은 이번 요청 범위에 포함하지 않는다.
