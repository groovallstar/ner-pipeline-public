# vLLM 서비스 지침

## 책임과 배치

이 디렉터리는 OpenAI 호환 vLLM 서버를 모델별 전용 GPU 한 장으로 기동한다.
Gemma 4는 GPU1/8081, Qwen3.8은 GPU2/8082를 사용하며 compose 기본 포트는
8000이다. GPU0은 다른 용도로 비워 둔다.

## 입력과 출력

- 입력은 모델 ID, GPU, 포트, dtype, 메모리 설정과 HF cache mount이다.
- 출력은 `/models`와 chat completions를 제공하는 OpenAI 호환 HTTP 서비스이다.

## 계약

- 모델별 start 스크립트는 공용 `docker-compose.yml`에 환경변수와 고유 compose
  프로젝트 이름을 전달한다.
- Qwen3.8 hybrid attention 경로에는 prefix caching을 명시한다. 모델 성능이나
  속도를 비교할 때 양쪽의 cache 설정을 맞춘다.
- 준비 완료는 `/models` 응답에 기대한 모델 이름이 나타나는지로 확인한다.
- JSON 값을 갖는 복잡한 vLLM 플래그는 compose 치환 과정에서 따옴표가 깨질 수
  있으므로 동작을 실제로 검증한다.

## 금지사항

- GPU 점유를 확인하지 않고 서버를 시작하지 않는다.
- Gemma 2의 기본 GPU1/8081 설정을 Gemma 4와 동시에 사용하지 않는다.
- `stop.sh`에 컨테이너 이름을 무조건 넘기지 않는다. compose 프로젝트 이름을
  확인한다.

## 검증

- `bash -n docker/vllm/*.sh`
- 기동 후 `/models`와 로그를 확인하고, benchmark는
  `uv run python -m ner.llm_eval ...`로 실행한다.
