<!-- Generated: 2026-04-08 | Updated: 2026-08-21 -->

# vllm

## 목적

vLLM 의 OpenAI 호환 추론 서버를 띄운다. 모델마다 tensor-parallel=1 로 전용 GPU 한 장을
쓰고, 각자의 기동 스크립트를 갖는다.

| 포트 | 모델 | GPU |
|---|---|---|
| 8081 | gemma-4-31B-it-AWQ-8bit | 1 |
| 8082 | Qwen3.8-27B-W4A16-AWQ | 2 |

**GPU0 은 비워 둔다** — 다른 용도로 쓰는 자리이므로 확인 없이 서버를 올리지 않는다.
compose 자체 기본 포트는 8000 이다.

컨테이너 이름은 모델을 전부 담는다(`vllm-<모델>-<크기>-<양자화>`). `docker ps` 만 봐도
어느 체크포인트가 어느 포트를 맡는지 알 수 있게 하기 위해서다.

## 파일별 역할

| 파일 | 설명 |
|------|------|
| `docker-compose.yml` | vLLM 서비스 정의 — 모델·GPU·메모리·dtype·포트를 환경변수로 받는다. IPC host, HF 캐시 마운트 포함 |
| `start-gemma4-31b-awq-8bit.sh` | gemma-4-31B-it-AWQ-8bit 기동 (GPU1, 8081). **`gpu_mem_util` 이 0.94 로 높다** — 가중치만 33.5 GiB 라 48 GB 카드에서 KV 캐시로 갈 여유가 10 GiB 남짓이고, 이 값을 낮추면 동시에 처리할 수 있는 요청 수가 그대로 줄어든다 |
| `start-qwen3.8-27b-w4a16-awq.sh` | Qwen3.8-27B-W4A16-AWQ 기동 (GPU2, 8082) — dense 27.8B, hybrid attention + vision. `--enable-prefix-caching` 을 명시적으로 넘긴다(§prefix 캐시) |
| `stop.sh` | vLLM 컨테이너 중지. 인자가 없으면 상시 운영 2종(gemma-4·Qwen3.8)을 내린다. 그 밖의 컨테이너는 이름을 직접 넘긴다 |
| `logs.sh` | 로그 tail. 기본은 `vllm-gemma4-31b-awq-8bit` 이고, 다른 컨테이너는 인자로 넘긴다 |

**`stop.sh` 에 넘기는 것은 compose 프로젝트 이름이라 컨테이너 이름과 다를 수 있다.**
프로젝트 이름에는 점을 쓸 수 없어 Qwen3.8 은 컨테이너가 `vllm-qwen3.8-27b-w4a16-awq`,
프로젝트가 `vllm-qwen38-27b-w4a16-awq` 다.

## prefix 캐시

같은 앞부분을 공유하는 요청들의 계산을 재사용해 prefill 을 건너뛰는 기능이다. 이 파이프라인의
라벨링 프롬프트는 대부분이 모든 문장에 동일한 지시문이라 효과가 크다(ko 기준 4.65 → 1.74 초/샘플).

vLLM 은 일반 attention 모델에는 이걸 기본으로 켜므로 gemma-4 는 플래그가 필요 없다.
반면 **hybrid 구조(linear attention + mamba 상태)에서는 기본으로 꺼진다** — 그 경로 지원이
아직 실험 단계이기 때문이다. 그래서 Qwen3.8 스크립트만 `--enable-prefix-caching` 을 명시한다.
켜기 전에 ko 300샘플로 출력이 바뀌지 않는 것을 확인했다.

**두 서버를 비교할 때는 이 설정부터 양쪽을 맞춘다.** 한쪽만 켜져 있으면 서버 설정 차이가
모델 성능 차이로 보인다 — 실제로 이 때문에 한 모델이 1.8배 빠른 것처럼 나온 적이 있다.

## 에이전트용 메모

### 작업 시 유의점

- 기동 스크립트는 환경변수를 export 한 뒤 `docker compose -p <프로젝트> up` 을 부른다. compose 파일의 `${...}` 는 전부 셸 환경에서 채워진다
- 모든 기동 스크립트가 **compose 파일 하나를 공유**하고 모델별 분기가 없다. 차이는 export 하는 환경변수와 `-p <프로젝트>` 뿐이다. 그래서 `command:` 에 박힌 `--default-chat-template-kwargs '{"enable_thinking": false}'` 가 Qwen 뿐 아니라 모든 기동에 붙는다 — gemma 계열은 그냥 무시한다. Qwen 전용으로 만들려면 compose 파일을 쪼개야 한다
- 모델별 추가 플래그는 `VLLM_EXTRA_ARGS` 로 넘긴다. 단 **JSON 을 값으로 갖는 플래그는 compose 변수 치환을 못 견딘다** — 따옴표가 벗겨져 파싱이 깨지므로(예: `--speculative-config '{"method":"mtp",...}'`) 그런 기동은 `docker run` 을 직접 써야 한다
- 준비 확인은 `curl -s $URL/models | grep $MODEL_NAME` 폴링

## 의존성

### 외부
- Docker + NVIDIA Container Toolkit
- 호스트 경로: `/data`, `/work/.huggingface`

### 내부
- `ner.llm_eval` (벤치마크 CLI: `python -m ner.llm_eval`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
