# issue-254: nllb 번역 백엔드 폐기

- Issue: https://github.com/groovallstar/ner-pipeline/issues/254
- PR: https://github.com/groovallstar/ner-pipeline/pull/255
- 브랜치: `feat/issue-254-drop-inprocess-translate-backend`
- 승인일: 2026-09-08

## 배경 (왜)

`/v1/translate` 는 백엔드가 둘이었다(`llm` | `nllb`). 인프로세스 전용 NMT
(`facebook/nllb-200-distilled-1.3B`)를 넣은 근거는 issue-197 의 둘인데, 지금
남아 있는 것이 없다.

1. **"`BASE_URL` 기본값이 조용히 `localhost:8081` 을 부른다"** — 해결됐고 이
   백엔드와 무관하다. 기본값 제거와 필수화가 집행하므로 백엔드를 지워도 남는다.
2. **"2080Ti(11GB)에서 원격 LLM 을 못 띄운다"** — issue-197 승인 3일 전에 이미
   반증됐다. `docs/reports/translate-engine-lightweight-benchmark.md` 의 2080Ti
   실측에서 llama.cpp(GGUF) Gemma-2-9B 가 Turing 에 한 번에 떴고, OpenAI 호환
   `/v1` 이라 원격 경로가 코드 변경 없이 붙는다. 카드 배치도 GPU0=NER /
   GPU1=번역으로 이미 두 장이다.

유지 비용만 남아 있었다.

- **품질을 한 번도 재지 않았다.** issue-197 이 "번역 품질은 이 이슈의 기준이
  아니다"로 범위에서 뺐고 이후에도 재지 않았다. 원격 후보들은 adequacy·음차·
  오염률까지 측정돼 있어, 화면에 나가는 출력의 질을 한쪽만 모르는 상태였다.
- **음차가 구조적으로 불가능하다.** 프롬프트가 없는 모델이라 지시할 수단이
  없는데, 번역 기능의 원래 채택 근거(issue-189)가 마스킹-복원 + 한글 음차였다.
- **조용한 실패를 막는 엔진 불변식 셋**(ASCII sentinel · 문장 단위 분할 ·
  `no_repeat_ngram` 하한)을 이 백엔드 하나 때문에 들고 있었고, 그것을 지키는
  실모델 테스트는 가중치를 요구해 기본 skip 이었다.

별도 번역 엔드포인트를 못 띄우는 배포가 없음을 확인해(2026-09-08) 마지막 유지
근거였던 자족 배포도 성립하지 않았다.

## 목적

`/v1/translate` 를 원격 OpenAI 호환 **단일 경로**로 되돌린다. 번역 계약(마스킹-
복원 · PII 회계 · 동시성 · 503/429)과 그 경로의 동작은 불변이고, 전용 코드·설정·
테스트·문서와 백엔드 이원화 표면 자체를 걷어낸다.

## 범위

**포함**

| 무엇 | 어디 |
|---|---|
| 인프로세스 백엔드 삭제 | `src/server/translate_nllb.py` |
| 전용 테스트 삭제 | `tests/server/test_translate_nllb.py` |
| 백엔드 분기·전용키 검증·sentinel 교체 제거 | `src/server/translate.py`, `tests/server/test_translate.py` |
| 설정 키 제거 | `src/server/config.py`, `docker/server/docker-compose.yml`, `.env.example` |
| 문서 갱신 | `src/server/CLAUDE.md`, `docker/CLAUDE.md`, `docker/server/CLAUDE.md`, `docs/manual/web-demo-translation.md`, `docs/manual/rest-api-spec.md` |

**제외**

- 번역 경로의 동작 변경 · 번역 품질 측정 · 번역 기본 활성화
- `certified/` 원장 변경, `docs/issues/` · `docs/reports/` 과거 기록 수정

## 성공 기준

- [x] 1. **번역 동작 불변** — 전용 케이스를 뺀 기존 `tests/server/` 가 전부
  통과. sentinel `【…】` · 프롬프트 · 타임아웃 · 503 · 429 계약 그대로이고
  `/v1/ner` 은 무영향
- [x] 2. **설정 표면이 단일 경로로 줄고, 결함은 기동 시점에 죽는다** —
  `NER_SERVER_TRANSLATE_BACKEND` · `..._DEVICE` · `..._NO_REPEAT_NGRAM` 제거.
  번역을 켰는데 `MODEL`·`BASE_URL` 이 없으면 기동 실패한다(따라서 옛 인프로세스
  전용 `.env` 는 `BASE_URL` 누락으로 죽는다). 테스트로 고정
- [x] 3. **죽은 코드·일반화 잔재 0** — `src` · `tests` · `docker` 와 살아 있는
  문서에서 폐기된 백엔드 이름 0건. 교체 소비자가 없어진 `SentinelFormat` 도
  제거하고 sentinel 표기를 `_mask`·`_restore` 에 고정
- [x] 4. **문서가 코드와 맞는다** — 살아 있는 문서에서 백엔드 이원화 서술과
  엔진 속성 절 삭제. `docs/issues/*` 스냅샷 · `docs/reports/*` ·
  `certified/translate_bench/` 원장은 손대지 않는다

## 결정 로그 (append-only)

- 2026-09-08: `NER_SERVER_TRANSLATE_BACKEND` 키를 남기지 않는다. 값이 `llm`
  하나뿐인 필수 enum 은 "고르라"가 아니라 "이 글자를 타이핑하라"가 되고, "설정만
  보고 무엇을 부르는지 안다"는 `BASE_URL` 필수화가 이미 집행한다. 대가로 옛
  `BACKEND=nllb` 설정이 조용히 무시되지만, 그 `.env` 는 `BASE_URL` 이 없어
  기동에서 죽는다.
- 2026-09-08: 살아 있는 문서에서 폐기된 백엔드 언급 자체를 지운다(폐기 흔적을
  남기지 않는다). 그래서 `SentinelFormat` 도 함께 제거했다 — 존재 이유였던
  "전용 NMT 는 lenticular bracket 이 `<unk>` 로 죽는다"를 적을 곳이 없어져,
  갈아끼울 소비자도 없는 채로 설명 없는 추상만 남기 때문이다. 경위는 issue-197
  과 이 문서에 남는다.
- 2026-09-08: HF 캐시 마운트(`HF_HOME` · `NER_SERVER_HF_CACHE`)를 compose·
  `.env.example` 에서 함께 뺐다. 그 마운트는 인프로세스 백엔드가 허브에서
  가중치를 받으려고 둔 것이고, `inference.py` 의 모델 로드는 전부 `/data` 아래
  로컬 경로라 남길 이유가 없다.
- 2026-09-08(반박자 지적 후): 살아 있는 문서 두 자리가 지운 사실을 아직 서술하고
  있었다 — `docker/CLAUDE.md` 가 ner-server 의 HF 캐시 마운트를 있다고 했고,
  `tests/server/test_translate.py` 모듈 docstring 첫 줄에 "백엔드 선택"이 남아
  있었다. 둘 다 고쳤다.
- 2026-09-08(반박자 지적 후): 같은 부류가 두 자리 더 있었다 —
  `docker/server/CLAUDE.md` 파일 표의 compose 행이 HF 캐시 마운트를 열거했고,
  `docs/manual/rest-api-spec.md` 가 참조 대상 문서의 없어진 절 이름("백엔드
  선택")을 가리켰다. 고친 뒤, 같은 부류를 지적에 맡기지 않고 살아 있는 문서
  전체에 낱말 스윕을 걸어 잔재가 없음을 확인했다.

## 구현 결과

- `translate.py` — 백엔드 분기(`_BACKEND_KEYS`·`BACKENDS`·`_resolve_backend`)를
  `_validate` 하나로 줄였다. `SentinelFormat` 을 접고 sentinel 표기를 모듈 상수
  (`_SENTINEL_LEFT`·`_SENTINEL_RIGHT`·`_SENTINEL_RE`)로 고정했으며,
  `_mask`·`_restore`·`_sentinel`·`LLMTranslator` 에서 표기 인자를 걷어냈다.
  프롬프트의 `{left}…{right}` 자리도 리터럴 `【…】` 로 인라인했다.
- `config.py` — `translate_backend`·`translate_device`·
  `translate_no_repeat_ngram` 필드와 그 env 읽기를 제거하고, 쓰이지 않게 된
  `_opt_int` 헬퍼도 함께 지웠다.
- `app.py` — 번역 전용 guard 를 두는 이유를 GPU 공유가 아니라 원격 응답 대기로
  다시 썼다. `/v1/translate/status` 의 `available` 설명도 원격 liveness 하나로
  좁혔다(동작 변경 없음).
- docker — 세 키와 HF 캐시 마운트를 compose·`.env.example` 에서 뺐다.
- 문서 — `docs/manual/web-demo-translation.md` 를 단일 경로 기준으로 다시 썼고
  (§5 엔진 속성·§6 인프로세스 절 삭제, 절 번호 재정렬), `src/server/CLAUDE.md`
  §번역 백엔드를 §번역 설정 표면으로 교체했다. `docker/CLAUDE.md` ·
  `docker/server/CLAUDE.md` · `docs/manual/rest-api-spec.md` 는 지운 사실을
  서술하던 자리를 고쳤다.

## 검증

- 테스트: `uv run pytest tests/server/ tests/hooks/ -q` → **245 passed**
  (server 194 + hooks 51). 린트: `uv run ruff check src tests` → clean.
- 테스트 수지: 전용 파일에서 20개, `test_translate.py` 에서 8개(표기 교체 3 ·
  백엔드 선택 5)를 지우고 4개(`test_sentinel_uses_lenticular_brackets` ·
  `test_model_is_required` · `test_base_url_is_required` ·
  `test_valid_config_builds_translator`)를 새로 뒀다. 수집 기준으로 224 → 194
  이며 skip 4건도 사라졌다 — 실모델 가중치를 요구하던 경로다. 순삭제는 대체가
  아니라 검사 대상의 소멸이라 게이트 예외를 사람이 승인했다.
- 기준 2 집행: 옛 인프로세스 전용 `.env`(ENABLED·MODEL·DEVICE·NO_REPEAT_NGRAM
  만 있고 `BASE_URL` 없음)를 `ServerConfig.from_env()` → `build_translator()` 로
  실제로 태워 `ValueError: NER_SERVER_TRANSLATE_BASE_URL is required when
  translation is enabled` 로 죽는 것을 확인했다(반박자 독립 재현).
- 기준 3 잔재: 살아 있는 문서·`src`·`tests`·`docker` 에 낱말 스윕을 걸어 0건.
  남은 hit 는 `docs/issues/` 스냅샷과 `docs/reports/` 의 지나간 기록뿐이다.
- 반박자: 결과-시점으로 돌려 문서 결함 넷을 잡았고(위 결정 로그), 고친 뒤 PASS.
  판정 파일은 `.omc/state/refuter/` 에 있다.

## 관련 커밋

- `803076a`: feat(server): 번역 백엔드를 원격 하나로 줄인다
