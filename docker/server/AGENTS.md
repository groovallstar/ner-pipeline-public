# NER 서버 컨테이너 지침

## 책임과 입출력

이 디렉터리는 `src/server`를 CUDA 이미지에 설치하고 `uv run --frozen python
-m server`로 기동한다. 입력은 `NER_SERVER_*` 설정과 호스트
`/data/ner/{ja,ko,vi,en}/model`, 출력은 기본 8008 포트의 내부망 REST API이다.

## 계약

- Docker build context는 저장소 루트이며 모델과 HF 캐시는 런타임에 마운트한다.
- 번역을 켜면 원격 LLM의 `NER_SERVER_TRANSLATE_MODEL`과
  `NER_SERVER_TRANSLATE_BASE_URL`을 명시한다.
- 원격 번역은 컨테이너에서 `host.docker.internal`을 통해 호스트 서비스를 호출한다.
- `/health`가 `status: "ok"`를 반환해야 준비 완료이다. 포트 바인딩만으로
  모델 준비를 판단하지 않는다.

## 금지사항

- 인증 없는 서비스를 공개망에 노출하지 않는다.
- 번역 설정에 암묵적 모델이나 base URL 기본값을 추가하지 않는다.
- 코드 변경 없이 빠르게 재기동하는 경우 외에는 `--no-build`로 새 코드를
  건너뛰지 않는다.

## 검증

- `bash -n docker/server/start.sh docker/server/stop.sh docker/server/logs.sh`
- 기동 시 선택한 compose 프로젝트와 컨테이너의 실제 게시 포트를 확인하고,
  해당 주소의 `/health` 응답과 컨테이너 health 상태를 함께 확인한다. 기본 포트
  8008이나 현재 셸 변수만으로 `.env`가 반영된 실행 대상을 추정하지 않는다.
- 런타임 계약 변경은 `uv run pytest tests/server -q -rs`로 검증한다. 모델 없는
  기본 검사와 전체 검사의 선택·자원 조건은 [서버 검증 지침](../../src/server/AGENTS.md#검증)을
  따른다. 전체 검사에서 실모델 로드와 서버 기동이 가능함을 고려한다.
