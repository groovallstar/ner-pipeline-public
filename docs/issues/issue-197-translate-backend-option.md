# issue-197: 번역 백엔드를 기동 옵션으로 선택 (llm | nllb, 기본값 없음)

- Issue: https://github.com/groovallstar/ner-pipeline/issues/197
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-197-translate-backend-option`
- 승인일: 2026-07-30

## 배경 (왜)

번역 백엔드가 `LLMTranslator` 하나로 고정돼 있었고, `NER_SERVER_TRANSLATE_BASE_URL`
이 `http://localhost:8081/v1` 을 **기본값으로 가졌다.** 여기서 둘이 문제였다.

1. **설정만 보고 무엇을 부르는지 알 수 없었다.** base_url 을 안 주고 번역을 켜도
   서버가 뜨고 `localhost:8081` 을 불렀다. 그 포트에 다른 모델이 떠 있으면 조용히
   그걸 번역기로 쓴다 — 실제로 이 박스의 8081·8082 에 서로 다른 모델(Gemma·Qwen)이
   떠 있어 헷갈리기 쉬운 배치다.
2. **소형 GPU 온프렘에서 고를 수단이 없었다.** 2080Ti(11GB, NER 추론과 VRAM 공유)
   구성에서는 원격 LLM 을 띄울 여유가 없을 수 있는데, 가볍게 도는 전용 NMT 를
   쓰려면 코드를 고쳐야 했다.

그리고 전용 NMT 를 그대로 끼우면 **조용히 깨진다.** 현행 sentinel
`【PII{nonce}_{i}】` 은 NLLB 에서 전량 소실된다 — SentencePiece 어휘에 lenticular
bracket 이 없어 양쪽 괄호가 `<unk>` 로 죽기 때문이다(근거:
`certified/translate_bench/nllb-1.3b-sentinel-ascii-2080ti/summary.json`).
마스킹-복원은 소실을 훼손이 아닌 '부재'로 처리하므로 **에러가 나지 않는다** —
화면에서 전화번호·이메일이 그냥 사라지고 아무 신호가 없다. 품질 저하가 아니라
프라이버시 경로의 조용한 실패다. 백엔드 선택을 열려면 표기 전환이 함께 가야 했다.

## 목적

번역 백엔드를 컨테이너 기동 시점 env 로 고른다(`llm` | `nllb`, **기본값 없음**).
`backend=nllb` 에서 PII verbatim 보존 100%(소실 0 · 잔여 sentinel 0), 번역
동시성이 상한으로 bound, NER + 번역 동시 부하의 단일 GPU 피크 VRAM ≤ 11264 MiB
(2080Ti 예산).

## 범위

### 포함

| 무엇 | 어디 |
|---|---|
| 백엔드 선택·설정 검증 | `src/server/config.py`, `translate.py` |
| NLLB 번역기 신규 작성 | `src/server/translate_nllb.py`(문장 분할 · 반복 억제 · ASCII sentinel) |
| 번역 전용 동시성 상한 | `src/server/app.py` |
| HF 캐시 마운트·env | `docker/server/docker-compose.yml`, `.env.example` |

### 제외

- 번역 품질 측정·엔진 간 품질 비교 — 번역은 NER 에 딸린 부가 기능이라 품질이
  선정 기준이 아니다. 엔진은 PII 보존·지연·VRAM 으로 고른다
- 벤치 하네스 재도입 — `scripts/translate_bench/` 는 제거됐고 되돌리지 않는다
- 기존 리포트·원장 변경 — `docs/reports/translate-engine-lightweight-benchmark.md`
  와 `certified/translate_bench/` 는 손대지 않는다
- 번역 기본 활성화 — `TRANSLATE_ENABLED` 는 계속 기본 false

## 성공 기준

- [x] 1. **설정 검증** — `NER_SERVER_TRANSLATE_BACKEND` 가 `llm`|`nllb` 만 받고
  기본값이 없다. 번역 활성인데 미지정·오값이면 기동 실패. `TRANSLATE_BASE_URL` 의
  기존 기본값을 제거해 `llm` 선택 시 필수가 된다. 백엔드에 안 맞는 키가 설정되면
  기동 실패(`nllb` + `BASE_URL` 등). 테스트로 고정
- [x] 2. **`backend=llm` 동작 불변** — 기존 `tests/server/` 전부 통과, sentinel
  `【…】` 유지, 프롬프트·타임아웃·503 계약 그대로
- [x] 3. **`backend=nllb` 출력이 쓸 만하다** — ASCII sentinel 자동 선택, ja·vi
  인라인 픽스처(PII 5종)가 verbatim 보존 · 소실 0 · 잔여 sentinel 0. 문장 2개
  이상 입력에서 뒷문장 유실 없음. 반복 억제 기본 켜짐이고 끌 수 있다
- [x] 4. **동시성** — 번역 동시 in-flight ≤ `TRANSLATE_MAX_CONCURRENCY`(기본 2),
  초과 시 429. NER guard 와 독립이라 서로의 예산을 잠식하지 않는다
- [x] 5. **메트릭/검증** — `backend=nllb` 로 NER + 번역 동시 부하를 걸어 단일 GPU
  피크 VRAM 을 실측·기록하고 11264 MiB 안에 든다
- [x] 6. **문서 갱신** — docker HF 캐시 마운트 + `.env.example` ·
  `src/server/CLAUDE.md` · `docker/server/CLAUDE.md` · `docs/manual/rest-api-spec.md`

## 설계

### 설정 표면 — 공용 키와 백엔드 전용 키를 가른다

```bash
NER_SERVER_TRANSLATE_ENABLED=true
NER_SERVER_TRANSLATE_BACKEND=            # llm | nllb — 기본값 없음, 활성 시 필수
NER_SERVER_TRANSLATE_MODEL=              # 두 백엔드 공용("무슨 모델")
NER_SERVER_TRANSLATE_MAX_CONCURRENCY=2

# backend=llm 일 때만
NER_SERVER_TRANSLATE_BASE_URL=
NER_SERVER_TRANSLATE_API_KEY=
NER_SERVER_TRANSLATE_TIMEOUT_S=30
# backend=nllb 일 때만
NER_SERVER_TRANSLATE_DEVICE=cuda:0
NER_SERVER_TRANSLATE_NO_REPEAT_NGRAM=      # 미설정=자동 계산(아래), 0=끔
```

`BACKEND` 가 "어떻게 돌리나", `MODEL` 이 "무슨 모델"이라 키가 겹치지 않는다.

**백엔드 전용 키는 config 에서 기본값 없이 `None` 으로 둔다.** 기본값을 채우면
"안 준 것"과 "그 값을 준 것"이 같아져, 고른 백엔드에 안 맞는 키가 왔는지를 볼 수
없다 — 그 검사가 곧 "설정만 보고 무엇을 부르는지 안다"의 집행이라 기본값을 여기서
없앤 것이다. 실효 기본값 중 timeout 30 · device `cuda:0` 은 `translate.py` 가
생성 시점에 넣고, 반복 억제 하한은 토크나이저를 쥔 `translate_nllb.py` 가
계산한다(상수로 못 박을 수 없다 — §결정 로그). compose 도 같은 이유로 번역 키에 기본값을
주지 않는다(`${VAR:-}`) — compose 가 채우면 서버 쪽 구별이 무의미해진다.

기동 실패 조건은 넷이다: 백엔드 미지정 · `llm`·`nllb` 아닌 값 · 공용 필수키
(`MODEL`, `llm` 이면 `BASE_URL`) 누락 · 고른 백엔드에 안 맞는 키.

### 두 백엔드가 대칭이 아니라서 따라오는 것

LLM 은 원격 HTTP 호출이고 NLLB 는 서버 프로세스 안에 모델을 올린다. `nllb` 를
고르면 번역이 NER 과 **같은 GPU** 에서 돌아 종전 전제("번역은 추론과 독립")가
깨진다. 그래서 셋이 함께 움직인다.

| 무엇이 깨지나 | 어떻게 받았나 |
|---|---|
| 번역이 NER 예산을 잠식한다 | `app.py` 가 번역 전용 `ConcurrencyGuard`(상한 `TRANSLATE_MAX_CONCURRENCY`, 큐 없음)를 하나 더 만든다. 초과는 대기 없이 429 — 온디맨드 버튼이라 기다리게 하는 것보다 낫다 |
| 긴 입력이 VRAM 피크를 밀어 올린다 | 한 요청의 문장 배치를 8 문장씩 나눠 태워 피크를 입력 길이에서 떼어낸다 |
| `available` = "백엔드 liveness" 가 무의미하다 | 정의를 백엔드가 쓴다. `llm` 은 엔드포인트를 찔러 보고, `nllb` 는 로드 성공이 곧 가용(로드 실패면 서버가 안 뜬다)이라 항상 true |
| 번역기 상태가 요청 사이에 공유된다 | NLLB 토크나이저는 소스 언어 태그(`src_lang`)를 인스턴스에 새겨 인코딩 때 읽는다 — 동시 요청이 겹치면 ja 가 vi 태그로 인코딩돼 **에러 없이** 엉뚱한 번역이 나온다. 토크나이저를 만지는 구간(인코딩·디코딩)을 직렬화하고 그 불변식을 테스트로 고정했다. 오래 걸리는 생성은 잠금 밖이라 동시성 이득은 남는다 |

### NLLB 에만 붙는 엔진 속성 셋

셋 다 실측에서 나온 실패를 막는 장치이며, 백엔드를 고르면 자동으로 따라온다.

1. **ASCII sentinel** — lenticular bracket 이 `<unk>` 로 죽어 복원이 전량 실패한다
   (0/186 → ASCII 186/186). 소실이 에러가 아니라 '부재'라 조용히 지나가므로
   표기 선택을 사람 손에 두지 않는다.
2. **문장 단위 분할** — 문장 모델이라 통짜 입력의 뒷문장을 통째로 버린다.
3. **반복 억제 — 단 sentinel 이 복사될 만큼만** — 끄고 beam search 를 돌리면 한
   어절을 `max_new_tokens` 까지 되풀이해 출력을 통째로 버린다. 그런데 억제를
   세게(작은 n) 걸면 sentinel 이 그 금지에 먼저 걸린다 — 아래 §결정 로그와
   §검증 참조. 그래서 하한을 sentinel 토큰 길이에서 계산하고,
   `NO_REPEAT_NGRAM=0` 으로 끌 수 있다.

## 결정 로그 (append-only)

- 2026-07-30: NLLB 번역기를 `translate.py` 안이 아니라 `translate_nllb.py` 로 뺐다
  — torch·transformers 를 끌어오므로 `backend=llm` 기동이 그 무게를 지지 않게
  임포트를 `build_translator` 안으로 미룬다.
- 2026-07-30: 번역 guard 에 대기 큐를 두지 않았다(`max_queue=0`). 번역은 웹 데모의
  온디맨드 버튼이라 몇 초~수십 초 기다리게 하는 것보다 즉시 429 로 돌려보내고 UI 가
  NER 만 보여주는 편이 낫다.
- 2026-07-30: 한 번의 `generate` 에 넣는 문장 수를 8 로 묶었다(벤치 코드엔 없던
  제약). 벤치는 짧은 문장만 다뤘지만 서비스는 `max_chars` 2만 자를 받으므로, 안
  묶으면 요청 하나가 VRAM 피크를 입력 길이에 비례해 밀어 올린다.
- 2026-07-30: `docs/manual/rest-api-spec.md` 에는 번역 env 표를 추가하지 않고, 그
  표면이 이 계약 밖이라는 한 줄만 넣었다 — `/v1/translate` 자체가 소비자 계약(§3)에
  없어서, env 만 실으면 문서가 스스로와 어긋난다.
- 2026-07-30: **반복 억제 기본값을 상수 3 에서 "sentinel 토큰 길이 + 2" 계산값으로
  바꿨다.** 실모델 검증에서 기준 3(PII 소실 0)이 기준 3의 다른 절(억제 기본 3)과
  충돌하는 것이 드러났다 — 한 문장에 sentinel 이 둘이면 두 번째가 반복 금지에 걸려
  모델이 글자를 바꿔 내놓고(`[PIIf8ca01_2]` → `[PIIF8ca001_2]`) 복원이 실패한다.
  n 을 sentinel 길이 위로 올리면 사라지는 문제라(아래 §검증 sweep), 억제를 켠다는
  기준의 뜻은 지키면서 값을 계산으로 바꿨다. nonce 가 프로세스마다 달라 토큰
  길이가 변하므로 상수로는 고정할 수 없다.
- 2026-07-30(반박자 라운드 1 FAIL 후): 인코딩 구간에 잠금을 넣었다 — 번역을 동시
  2 로 열어 놓고 토크나이저의 `src_lang` 을 호출마다 갈아끼우고 있어, ja·vi 요청이
  겹치면 한쪽이 상대 언어 태그로 인코딩되는 경합이 있었다. 잘못된 출력이 에러 없이
  나오는 형태라 이 이슈가 없애려는 실패와 같은 부류다. 오래 걸리는 생성은 잠금
  밖에 둬 동시성 이득은 유지했다. 같은 라운드에서 `no_repeat` 기본값 표기가 옛
  값(3)으로 남아 있던 문서 두 곳도 고쳤다.
- 2026-07-30(반박자 라운드 2 FAIL 후): 디코딩도 같은 잠금 안으로 넣었다 — 태그를
  읽지 않아 잠금 밖에 뒀는데, fast 토크나이저가 내부 상태를 Rust 쪽에서 빌려 쓰는
  탓에 한쪽이 디코드하는 동안 다른 쪽이 `src_lang` 을 갈면 `Already borrowed` 로
  터진다(반박자가 2스레드 200회 안에 재현). 8문장 디코드는 밀리초라 직렬화 비용이
  거의 없어, 한계로 적어 두는 대신 없앴다.

## 구현 결과

- `config.py` — 번역 설정을 공용/백엔드 전용으로 가르고, 전용 키를 `Optional`
  (미설정 = `None`)로 바꿔 "안 준 것"을 값으로 구별한다. `BASE_URL` 기본값 제거.
- `translate.py` — `build_translator` 가 백엔드를 고르고 `_resolve_backend` 가 위
  네 조건을 기동 시점에 실패시킨다. `LLMTranslator` 는 그대로다.
- `translate_nllb.py`(신규) — 인프로세스 NLLB 백엔드. ASCII sentinel · 문장 분할 ·
  반복 억제 · 문장 배치 상한. torch·transformers 는 `__init__` 에서 임포트한다.
- `app.py` — 번역 전용 guard 추가, `/v1/translate` 를 그 안에서 실행,
  `/v1/translate/status` 의 `available` 정의를 백엔드별로 다시 씀.
- docker — 번역 키에서 compose 기본값 제거, HF 캐시 호스트 마운트
  (`NER_SERVER_HF_CACHE`), `.env.example` 백엔드별 설명.

## 검증

### 테스트 (기준 1·2·3·4)

`uv run pytest tests/server/` — 151 passed, 3 skipped. 실모델 경로까지 켜면
154 passed(가중치를 가리켜야 하며, 미지정이면 그 3건만 skip):

```bash
HF_HOME=/data/ner/_hf_cache \
NER_SERVER_TEST_NLLB_MODEL=facebook/nllb-200-distilled-1.3B \
  uv run pytest tests/server/
```

기존 `tests/server/` 는 그대로 통과한다(기준 2). 새로 고정한 것은 설정 검증
10건(비활성·미지정·오값·`MODEL` 누락·`BASE_URL` 누락·양쪽 전용키 오배치·llm 경로
불변·nllb 인자 전달), 엔드포인트 동시성 2건(번역 상한 포화 + 초과 429 + NER
무영향, 그 반대 방향), 모델 무의존 NLLB 로직 18건(문장 분할·복원 배선·억제 강도
해소·문장 배치 상한·**동시 인코딩 시 언어 태그 불변**·**인코딩↔디코딩 직렬화**),
실모델 2건(ja·vi 파라미터라 3케이스 — PII 5종 verbatim, 뒷문장 유지)이다.

동시성 두 건은 잠금을 빼면 실제로 깨지는 명제다 — 같은 시나리오를 no-op 잠금으로
돌리면 인코딩 10건 중 5건이 상대 언어 태그로 인코딩되고, `_decode` 의 잠금만
지워도 직렬화 테스트가 곧바로 실패한다.

### 반복 억제 강도 sweep (기준 3)

ja·vi 인라인 픽스처(각 PII 5종·3문장)를 실모델로 돌려 `no_repeat_ngram_size` 만
바꿨다. 두 언어 결과가 같았다.

| n | 복원 | 소실 | 잔여 sentinel |
|---|---|---|---|
| 0 | 5/5 | 0 | 0 |
| 3 | 3/5 | 2 | 0 |
| 9 | 3/5 | 2 | 0 |
| 10 | 4/5 | 1 | 0 |
| 12 (= 계산된 하한) | 5/5 | 0 | 0 |
| 16 | 5/5 | 0 | 0 |

sentinel `[PIIf8ca01_0]` 은 10 토큰이고 앞 8 토큰(`▁[ P II f 8 ca 01 _`)이 모든
sentinel 에 공통이다. n 이 그보다 작으면 같은 문장의 두 번째 sentinel 을 그대로
쓰는 것이 금지돼 모델이 글자를 바꾼다 — 실제 출력이
`[PIIF8ca001_2]`·`[Piif8cas01_4]` 였다.

억제를 하한으로 올려도 붕괴 방지는 살아 있다(반복 유도 문장 9건):

| 설정 | 최대 연속 어절 반복 | 최대 출력 길이 |
|---|---|---|
| n=0 (끔) | 99회 | 399자 |
| n=12 (계산된 기본) | 6회 | 48자 |

### VRAM (기준 5)

`backend=nllb` 로 서버를 실기동(`CUDA_VISIBLE_DEVICES=0`, ja·vi NER 모델 + NLLB
1.3B fp16)하고 60초간 NER 8 워커 + 번역 4 워커를 동시에 밀며 `nvidia-smi` 로 그
프로세스 메모리를 0.2초 간격 샘플링했다.

| 무엇 | 값 |
|---|---|
| 로드 직후 | 3836 MiB |
| 동시 부하 피크 | **4778 MiB** |
| 2080Ti 예산 | 11264 MiB |
| 창 안 처리 | NER 523건 성공 · 번역 4건 성공 · 429 197건(상한 2 포화) |

문장 배치 상한(8문장)이 실제로 피크를 묶는지도 따로 쟀다 — 120문장 입력에서
torch 할당 피크가 상한 8 에서 2795 MiB, 상한 없이 한 배치면 5113 MiB 였다(대신
지연 3.6s vs 0.6s). 상한 안에서의 최악 배치(긴 문장 8개, 각 223자)는 3271 MiB.

**측정 환경 주의**: 이 박스의 GPU 는 RTX A6000 이고 2080Ti 가 아니다. 재는 값이
프로세스가 잡은 메모리라 카드가 달라도 1차적으로는 같지만, 커널·워크스페이스가
아키텍처마다 조금 다르므로 2080Ti 실측은 아니다. 피크가 예산의 42% 라 여유가
크다는 판단은 그 차이를 흡수한다.

### 남는 한계

- `backend=nllb` 에는 요청 타임아웃이 없다(원격 호출이 아니라 걸 데가 없다).
  아주 긴 입력은 번역 슬롯을 오래 잡는데, 여파는 "번역 요청이 429" 로 그치고
  `/v1/ner` 은 영향을 받지 않는다.
- 실모델 픽스처는 ja·vi 각 1건(PII 5종)이다 — 회귀 감지용 최소 집합이며, 번역
  품질은 이 이슈의 기준이 아니다(§범위 제외).

## 관련 커밋

<!-- 머지 직전 채움 -->
