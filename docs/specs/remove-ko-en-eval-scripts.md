# KO·EN 배포 평가 스크립트 제거

## 문제와 목표

사용자가 불필요하다고 지정한 KO·EN 전용 평가 실행 파일을 제거한다.
대화에서 승인된 범위는 `eval_ko_ner_test.py/.sh`와
`eval_en_ner_test.py/.sh`이며, JA·VI 실행 파일과 나머지 도구는 유지한다.

## 성공 기준과 문서 영향

- 지정된 실행 파일 네 개와 EN 실행 파일만을 검증하는 테스트를 제거한다.
- 현재 매뉴얼에서 제거된 실행 경로를 안내하지 않는다. 과거 이슈 기록은 보존한다.
- 배포 모델 카드에서도 제거된 실행 경로를 안내하지 않고 JA·VI 안내는 유지한다.
- KO·EN 라이브러리, REST 서비스, 배포 패키지 검증은 유지한다.
- 엔티티 스키마에는 영향이 없으며, 구현 맵은 분류 파이프라인 매뉴얼을 갱신한다.

## 검증 계획

남은 `tests/ner/scripts` 테스트, 해당 디렉터리 Ruff, JA·VI CLI의
`--help` 및 shell 구문 검사, 삭제 경로 참조 검색, `git diff --check`를 실행한다.
실제 모델 추론은 이번 삭제 작업의 검증 범위에 포함하지 않는다.

## 결정 기록

- 기존 KO·EN 전용 평가 진입점 유지에서 제거로 변경한다. 사용자가 해당 두
  언어의 스크립트만 제거하도록 명시했으며, 전용 CLI 사용 경로를 폐기한다.

## 검증 결과

- 변경 전 `uv run pytest tests/ner/scripts -q`: 22개 통과.
- 최종 변경 후 같은 명령: 19개 통과. 폐기한 EN CLI 전용 테스트 5개를 제거하고
  모델 카드 테스트에 JA·VI 사례 2개를 추가했다.
- 모델 카드 테스트 수정 후 구현 전에는 KO·EN 2개 실패, JA·VI 2개 통과로
  삭제된 경로가 안내되는 결함을 재현했다.
- `uv run ruff check src/ner/scripts tests/ner/scripts`: 통과.
- JA·VI Python CLI의 `uv run python <script> --help`와 각 shell wrapper의
  `bash -n <script>`: 모두 통과.
- 과거 이슈와 이 spec을 제외한 저장소 검색에서 제거된 실행 파일 참조가 없다.
- 변경 경로에 한정한 `git diff --check`: 통과. 전체 검사에서는 기존 사용자
  변경인 `docs/manual/rest-api-integration-guide.md:237`의 EOF 빈 줄이 발견되었다.
- 실제 모델 추론은 실행하지 않았다.
- Superpowers 리뷰에서 모델 카드의 동적 경로 참조 1건을 지적받아 수정했다.
  최종 변경의 재리뷰에서 추가 지적 없이 승인되었다.
