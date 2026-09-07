# server — ja·ko·vi·en NER REST API 배포

`src/server` 추론 서버를 컨테이너로 빌드·기동한다. CUDA 베이스
(`pytorch:2.13.0-cuda13.0`)에서 uv 로 프로젝트(ner·server)를 설치하고
`python -m server` 를 띄운다. 모델은 이미지에 넣지 않고 런타임에 호스트
`/data` 를 마운트해 읽는다. 소비자는 **내부망 별도 프로세스**(공개 노출·인증은
범위 외 — 내부 신뢰망 전제)다.

## 기동

```bash
cp .env.example .env       # 포트·GPU·정밀도 등 조정(선택)
bash start.sh              # 빌드 + 기동 (-d --build)
bash logs.sh               # 로그 tail
curl -s localhost:8008/health
bash stop.sh               # 중지·제거
```

GPU 는 1장만 컨테이너에 노출돼 고정된다(`device_ids`; 기본 0, `NER_SERVER_GPU`
로 교체). 컨테이너 없이 호스트에서 바로 띄우는 로컬 기동은
`bash src/server/scripts/run_local.sh [--port P]`(GPU 0 고정).

## 파일

| 파일 | 역할 |
|------|------|
| `Dockerfile` | CUDA 베이스(`pytorch:2.13.0-cuda13.0`) + uv sync(`ner`·`server` 설치). `curl`(healthcheck)·`build-essential`(토크나이저 빌드)·`git` 포함. ENTRYPOINT=`uv run --frozen python -m server` — `--frozen` 이라 런타임에 `uv.lock` 을 재해결하지 않는다(이미지에 굳은 의존성 그대로 기동) |
| `docker-compose.yml` | `ner-server` 서비스 — `NER_SERVER_*` env, `/data`·HF 캐시 마운트, GPU reservation, 포트 publish, 준비성 healthcheck, restart unless-stopped, `extra_hosts`(`host.docker.internal:host-gateway`). 번역 키에는 기본값을 주지 않는다 — compose 가 값을 채우면 서버가 "안 준 것"과 구별하지 못한다 |
| `start.sh` | 빌드 + 기동. 기존 컨테이너 정리 후 올림. `--no-build` 로 기존 이미지 빠른 기동(코드 변경 없을 때 `uv sync` 레이어 재실행 회피) |
| `stop.sh` | 컨테이너 중지·제거 |
| `logs.sh` | 컨테이너 로그 tail |
| `.env.example` | 설정 표면 전체(토폴로지 + `NER_SERVER_*`). `.env` 로 복사해 사용 |

## 설정

런타임 동작은 `NER_SERVER_*` env 로 조정한다(상세 표면·기본값은 `.env.example`,
의미는 `src/server/CLAUDE.md`). compose 가 같은 디렉토리 `.env` 를 자동 로드한다.
이미지 태그는 `:latest` 금지 — 버전 pin(`ner-server:0.1.0`).

### 웹 데모 번역을 켤 때

`NER_SERVER_TRANSLATE_ENABLED=true` 만으로는 뜨지 않는다 — `TRANSLATE_BACKEND`
(`llm`|`nllb`)를 반드시 고르고, 그 백엔드의 키만 남긴다(반대편 키가 설정돼 있으면
기동 실패. 이유는 `src/server/CLAUDE.md` §번역 백엔드).

- **`llm`(원격)**: compose 가 `extra_hosts` 로 `host.docker.internal` 을
  host-gateway 에 붙여 둔다 — 컨테이너 안에서 `localhost` 는 컨테이너 자신이라
  호스트에 떠 있는 vLLM 에 닿지 못하기 때문이다. `NER_SERVER_TRANSLATE_BASE_URL`
  을 `http://host.docker.internal:8081/v1` 로 주는 것이 이 설정을 쓰는 경로다.
- **`nllb`(인프로세스)**: 원격이 필요 없는 대신 기동 때 HF 허브에서 가중치를
  내려받아(1.3B 기준 수 GB) `HF_HOME`(컨테이너 `/data/ner/_hf_cache`)에 쌓는다.
  그 경로는 호스트로 마운트돼 있어 컨테이너를 다시 만들어도 재다운로드가 없다
  (호스트 경로는 `NER_SERVER_HF_CACHE` 로 옮긴다). 첫 기동은 다운로드만큼
  느려 healthcheck 가 `starting` 에 오래 머문다. 폐쇄망이면 캐시를 미리 채워
  둔다. 번역이 NER 과 **같은 GPU** 를 쓰므로 VRAM 예산을 함께 본다 —
  동시 처리량은 `NER_SERVER_TRANSLATE_MAX_CONCURRENCY`(기본 2)로 묶인다.

## 헬스체크 / 준비성

healthcheck 는 `/health` 의 `status:"ok"`(전 언어 로드 완료)만 healthy 로
본다. 모델 로드는 uvicorn 포트 바인딩 *전*에 끝나므로 로드 중에는 포트가 닫혀
있고(connection refused), 그 뒤 healthcheck 가 `starting`→`healthy` 로 오른다.
healthcheck 자체는 오케스트레이터(compose `depends_on`·swarm·k8s readiness)가
트래픽 투입을 게이팅하는 *신호*이지 직접 포트 접근을 막는 장치는 아니다. 일부
언어 미로드면 `degraded` → unhealthy 로 정직 보고된다.

**en 이 들어간 판부터는 `/data/ner/en/model` 마운트가 healthy 의 전제다.**
지원 언어가 늘면 "전 언어 로드"의 뜻도 함께 넓어지기 때문이다. 마운트 없이
이미지만 올리면 `status` 가 `degraded` 로 굳어 healthcheck 가 계속 실패하고,
readiness 로 트래픽을 게이팅하는 배포에서는 영어뿐 아니라 **나머지 언어의
트래픽까지** 끊긴다. 서버 설정에는 언어별 opt-out 이 없어 en 만 빼고 띄울
수단도 없다.

**올리는 순서는 `/data/ner/en/model` 마운트가 먼저, 이미지 교체가 나중이다.**
반대로 하면 새 이미지가 뜨는 순간부터 마운트를 끝낼 때까지 전 언어 트래픽이
끊긴 채로 남는다.

## 의존성·주의

- NVIDIA Container Toolkit, 호스트 `/data/ner/{ja,ko,vi,en}/model` 존재
- 빌드 컨텍스트는 레포 루트(`../../`) — `.dockerignore` 가 `.venv`·산출물 제외
- 모델은 마운트(이미지 미포함) — 재학습 모델 교체는 `/data` 갱신 후 재기동
