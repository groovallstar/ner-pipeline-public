# server — ja·vi NER REST API 배포

`src/server` 추론 서버를 컨테이너로 빌드·기동한다. dev 와 동일 CUDA 베이스
(`pytorch:2.11.0-cuda13.0`)에서 uv 로 프로젝트(ner·server)를 설치하고
`python -m server` 를 띄운다. 모델은 이미지에 넣지 않고 런타임에 호스트
`/data` 를 마운트해 읽는다. 소비자는 **내부망 별도 프로세스**(공개 노출·인증은
범위 외 — 내부 신뢰망 전제)다.

## 기동

```bash
cp .env.example .env       # 포트·GPU·정밀도 등 조정(선택)
bash start.sh              # 빌드 + 기동 (-d --build)
bash logs.sh               # 로그 tail
curl -s localhost:8000/health
bash stop.sh               # 중지·제거
```

GPU 는 1장만 컨테이너에 노출돼 고정된다(`device_ids`; 기본 0, `NER_SERVER_GPU`
로 교체). 컨테이너 없이 호스트에서 바로 띄우는 로컬 기동은
`bash src/server/scripts/run_local.sh [--port P]`(GPU 0 고정).

## 파일

| 파일 | 역할 |
|------|------|
| `Dockerfile` | dev 와 동일 CUDA 베이스 + uv sync(`ner`·`server` 설치). `curl`(healthcheck)·build-essential(토크나이저 빌드) 포함. ENTRYPOINT=`uv run python -m server` |
| `docker-compose.yml` | `ner-server` 서비스 — `NER_SERVER_*` env, `/data` 마운트, GPU reservation, 포트 publish, 준비성 healthcheck, restart unless-stopped |
| `start.sh` | 빌드 + 기동. 기존 컨테이너 정리 후 올림. `--no-build` 로 기존 이미지 빠른 기동(코드 변경 없을 때 `uv sync` 레이어 재실행 회피) |
| `stop.sh` | 컨테이너 중지·제거 |
| `logs.sh` | 컨테이너 로그 tail |
| `.env.example` | 설정 표면 전체(토폴로지 + `NER_SERVER_*`). `.env` 로 복사해 사용 |

## 설정

런타임 동작은 `NER_SERVER_*` env 로 조정한다(상세 표면·기본값은 `.env.example`,
의미는 `src/server/CLAUDE.md`). compose 가 같은 디렉토리 `.env` 를 자동 로드한다.
이미지 태그는 `:latest` 금지 — 버전 pin(`ner-server:0.1.0`).

## 헬스체크 / 준비성

healthcheck 는 `/health` 의 `status:"ok"`(전 언어 로드 완료)만 healthy 로
본다. 모델 로드는 uvicorn 포트 바인딩 *전*에 끝나므로 로드 중에는 포트가 닫혀
있고(connection refused), 그 뒤 healthcheck 가 `starting`→`healthy` 로 오른다.
healthcheck 자체는 오케스트레이터(compose `depends_on`·swarm·k8s readiness)가
트래픽 투입을 게이팅하는 *신호*이지 직접 포트 접근을 막는 장치는 아니다. 일부
언어 미로드면 `degraded` → unhealthy 로 정직 보고된다.

## 의존성·주의

- NVIDIA Container Toolkit, 호스트 `/data/ner/{ja,vi}/model` 존재
- 빌드 컨텍스트는 레포 루트(`../../`) — `.dockerignore` 가 `.venv`·산출물 제외
- 모델은 마운트(이미지 미포함) — 재학습 모델 교체는 `/data` 갱신 후 재기동
