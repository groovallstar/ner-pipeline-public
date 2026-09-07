# issue-250: 영문(en) NER REST API 서빙과 라틴 폴백 감지

- Issue: https://github.com/groovallstar/ner-pipeline/issues/250
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-250-en-restapi-serving` (base `origin/develop` `0f1f22b`)
- 승인일: 2026-09-07

## 배경 (왜)

en 배포 패키지는 `/data/ner/en/` 에 이미 있는데(#221 출하 · #244 재학습·재출하)
REST API 가 그것을 안 썼다. `SUPPORTED_LANGS` 가 `('ja','vi','ko')` 라
`lang:"en"` 요청은 400 이었고, 영어 문장을 `lang` 없이 던지면 `unsupported` 와
빈 엔티티가 돌아왔다. **출하된 모델이 서비스 경로에 연결돼 있지 않은 상태**다.

연결이 미뤄진 사유는 문서에 남아 있었다.
`docs/manual/pipeline/4-classification.md` 가 "EN 은 라틴 스크립트에 고유
코드포인트가 없어 양성 감지가 불가능했고, 그래서 `SUPPORTED_LANGS`·`detect.py`
를 손대는 일이 별도 결정으로 미뤄졌다" 라고 적는다. 이 이슈가 그 별도 결정이다.

`docs/issues/issue-225-zero-length-span-boundary.md:70` 이 "en 배포 패키지 …
재학습 전에는 이 코드와 함께 서비스하면 안 된다" 를 적고 있으나, 같은 줄 괄호가
해제를 기록한다 — **#244 가 재학습·재출하로 그 조건을 이미 풀었다.** grep 으로
그 줄에 먼저 닿는 사람이 금지 문장만 읽고 멈추는 것을 막으려고 여기 옮겨 적는다.

## 목적

`/v1/ner` 이 `lang:"en"` 과 라틴 자동감지를 받아 `/data/ner/en` 모델로 추론하고,
서버 `predict` 로 잰 en test strict F1 이 `/data/ner/en/metrics.json` 의
`overall_strict` 와 `abs=1e-6` 안에서 일치한다. `/v1/translate` 가 en→ko 를
`llm`·`nllb` 두 백엔드에서 지원하고 PII 5 종이 verbatim 보존된다. 복제된 언어
목록의 기계적 어긋남은 pytest 실패가 된다.

## 범위

**포함** — `src/server/**` 코어와 웹 UI · `tests/server/**` · `docker/server/*` ·
소비자 계약 문서 둘 · 언어 열거를 쥔 상위 문서 넷.

**제외(구속)** — `thresholds.json` 생성 · 비-라틴 폴백 · 응답 스키마 확장
(`detected_by` 류 필드 없음) · 감지 하네스(`server.scripts.lang_detect`) 부활 ·
백본 재선정·재학습·sweep · 기준 파일 변경.

**기준 파일 미변경** — `gate_core.py` 의 `RULER_PATHS`·`RULER_DIRS` 와 대조해
확정했다. 그래서 2 단계 승인과 정의-시점 반박자는 대상이 아니었다.

## 성공 기준

- [x] **AC-1 계약·통합** — `POST /v1/ner` 이 `lang:"en"` 단일·배치를 받는다.
- [x] **AC-2 감지 정책** — 라틴 있으면 `en`, 없으면 종전 `unsupported`. 기존
  ja·ko·vi 감지 케이스 전수 무수정 통과.
- [x] **AC-3 배포 parity** — `overall_strict` 의 f1·precision·recall 이
  `abs=1e-6`, support 는 정확 일치.
- [x] **AC-4 폴백 e2e** — 실서버에 `lang` 없이 영어 문장을 던져 `lang == 'en'`
  과 비어 있지 않은 엔티티.
- [x] **AC-5 번역 en→ko** — 두 백엔드 각각. NLLB 실모델에서 PII 5 종 verbatim.
- [x] **AC-6 UI 정합 + 문서** — `index.html` 리터럴이 서버 목록과 어긋나면
  실패하고, 훑기 범위의 모든 3 원소 이상 언어 열거가 서버 상수와 대조된다.

## 위험·의존성

**`SUPPORTED_LANGS` 에 en 을 넣는 것은 배포 계약을 바꾼다.** `/health` 는 전
언어 로드일 때만 `status:"ok"` 를 내고 compose healthcheck 가 그 문자열을
grep 하므로, `/data/ner/en/model` 이 없는 호스트는 컨테이너가 **영구
unhealthy** 가 된다. readiness 로 트래픽을 게이팅하는 배포에서는 영어뿐 아니라
나머지 언어의 트래픽까지 끊긴다. `ServerConfig.langs` 는 env opt-out 을 주지
않아 en 만 빼고 띄울 수단도 없다.

**올리는 순서는 모델 마운트가 먼저, 이미지 교체가 나중이다.** 반대로 하면 새
이미지가 뜨는 순간부터 마운트를 끝낼 때까지 전 언어 트래픽이 끊긴 채로 남는다.
이 순서를 `docker/server/CLAUDE.md` §헬스체크/준비성에 적었다.

**폴백은 번역에 오라우팅 경로를 연다.** 무부호 vi·로마자 ja 가 `lang:'en'` 으로
`/v1/translate` 에 들어가고, 마스킹이 en NER span 을 쓰므로 그 입력에서는 PII
보존 보증이 성립하지 않는다. 실측은 아래 §현 상태.

## 현 상태 (fact)

- **en 배포 패키지**: roberta-base. `thresholds.json` 은 없고 그것이 정상이다 —
  `LangModel` 이 graceful 로 흡수해 raw 로 서빙되며 vi·ko 가 이미 같은 상태다.
  수치 원장은 `certified/classifier/en/deploy-trainseed42/metrics.json`.
- **감지 결과**: `detect_lang` 이 두 단이 됐다. 배타적 스크립트 신호(가나·한글·
  vi 변별 부호)가 전부 실패한 뒤 라틴 글자가 있으면 `en`, 없으면 `unsupported`.
  라틴 판정은 코드포인트 아홉 구간으로 한다. Latin-1 Supplement 를 세 구간으로
  쪼개 `×`(U+00D7)·`÷`(U+00F7) 를 뺐고, Latin Extended Additional
  (U+1E00–U+1EFF)을 넣어 베트남어 사전조합 글자와 점 부호 라틴이 ASCII 를 한
  자도 안 낀 채 와도 폴백에 걸리게 했다.
- **언어 열거 정합 검사의 매치 총수 = 59** (훑기 36 파일, 3 원소 이상). 다음
  언어를 넣을 때 이 값과 나란히 놓고 본다 — 사슬·술어 재작성이 검사 표면을
  단조 감소시키는 것을 관측하기 위한 값이며, 테스트에 하한으로 박지 않았다
  (박으면 그 숫자를 고쳐 통과시키는 길이 열린다). 계획이 실측한 45 는
  `f97f307` 기준 좁은 범위라 이 값과 직접 비교할 수 없다.
- **en 마운트가 없을 때의 실측**: `--model-root` 에 ja·ko·vi 만 둔 서버에서
  `/health` 가 `"status": "degraded"` 를 냈고, 영어 단일 입력이 503, 라틴 항목이
  섞인 배치가 **부분 성공이 아니라 통째로** 503 이었다. compose healthcheck 는
  `"status": ?"ok"` 를 grep 하므로 이 상태에서 영구 unhealthy 다.
- **PII 보존 보증의 적용 범위**: 보증은 "넘긴 span 은 원문 그대로 남는다" 이지
  "PII 를 다 찾는다" 가 아니다. `llm` 백엔드를 켜고 잰 실측에서, 언어가 맞는
  영어 입력의 이메일 주소를 en NER 이 EMAIL 로 잡지 못해 마스킹 대상에서
  빠졌다. 출력에 그대로 남기는 했지만 그것은 번역기가 복사한 결과지 마스킹-
  복원이 지킨 것이 아니다. AC-5 의 픽스처는 다섯 종이 전부 태깅된 입력이라
  이 구멍을 보지 못한다.
- **폴백이 만든 오라우팅의 실제 출력**(V8 ⑤): 무부호 vi 입력에서 en NER 이
  `Ha Noi` 를 놓치고 한 글자 `i` 를 LOC 으로 뱉었고, 번역은 "동남아에 있는 나라"
  가 통째로 날아간 축약을 냈다. 로마자 ja 입력에서는 전화번호가 **PHONE 으로
  태깅되지 않아 마스킹 대상이 아니었고**, 출력에 살아남은 것은 마스킹-복원
  계약이 아니라 NLLB 가 숫자를 우연히 복사한 결과다. 번역 출력 자체는 그럴듯한
  거짓보다 눈에 띄는 고장에 가까웠다(음차에 `desu` 가 그대로 남음).

## 결정 로그 (append-only)

- 2026-09-07: 라틴 폴백을 `DETECTORS` 리스트 끝이 아니라 `detect_lang` 의 루프
  뒤 별도 분기로 뒀다. 리스트 끝에 두면 diff 가 한 줄이지만 "새 스크립트 언어는
  한 줄 추가" 라는 확장 불변식이 조용히 죽고, 기존 확장 테스트가 라틴 없는
  키릴만 써서 그 회귀를 못 잡는다. 차이를 기계로 판별하려고
  `test_registry_detector_wins_over_latin_fallback` 을 뒀다.
- 2026-09-07: 언어 열거 검사를 존재 금지가 아니라 **유도 대조**로 만들었다.
  금지 목록으로는 통과할 수 없다 — 교체가 전부 기존 토큰 뒤에 덧붙이는 형태라
  `ja·ko·vi` 가 `ja·ko·vi·en` 안에 그대로 남고, `test_api.py` 가 요구하는 백틱
  표기가 백틱 제거 뒤 곧바로 금지 토큰이 된다.
- 2026-09-07: 열거 정합 검사의 훑기 범위에 루트 `CLAUDE.md`·`README.md`·
  `docker/CLAUDE.md`·`pyproject.toml` 넷을 더했다. 계획 검토가 본 범위 밖인데,
  `0f1f22b` 가 그 넷의 `ja·vi` 를 손으로 `ja·ko·vi` 로 고쳐야 했다는 사실이
  근거다 — 검사 밖에 두면 en 에서 같은 누락이 반복된다.
- 2026-09-07: Q7(en 을 번역 대상에 두는 결정)을 V8 ⑤ 실측 출력을 본 뒤
  **유지**로 확정했다. 번역 출력이 그럴듯한 거짓이 아니라 눈에 띄는 고장이라
  사용자가 알아본다는 것이 근거다. 대가는 UI 정적 안내 한 줄로 노출한다.
- 2026-09-07: 결과-시점 반박자가 "검사가 자기가 지킨다고 적은 것을 안 지킨다"
  를 두 자리에서 반증했고 둘 다 받았다. 공허 방지 sentinel 의 재귀 대표가
  `src/server/` 바로 아래 파일이라 훑기를 비재귀로 좁혀도 살아남았고
  (`static/`·`scripts/` 가 통째로 빠진 채 초록), 셀렉터 대조가 뺄셈 형태라
  `auto` 옵션이 사라져도 참이었다. 각각 하위 디렉토리 대표와 합집합 비교로
  고치고, 원래 통과하고 지금 실패하는 것을 뮤테이션으로 확인했다. 함께
  지적된 `_LATIN_RANGES` 의 누락 구간과 산문 셋도 손봤다.
- 2026-09-07: 폴백의 대가를 적은 문장이 세 자리에 있는데 두 자리만 고쳐
  형태가 갈렸다는 지적을 받아 셋을 맞췄다. "라틴 글자만 있으면 영어" 는
  문자 그대로는 거짓이다 — `Hà Nội` 는 라틴 글자뿐인데 vi 로 간다. 이 이슈가
  잡으려던 "복제본이 갈라진다" 를 이슈 자신이 한 번 재현한 자리다.
- 2026-09-07: 2 원소 열거 자리가 다음 언어에서 조용히 낡는 문제를, 하한을
  낮추거나 예외 구간을 두는 대신 **언어 추가 절차의 한 줄**로 닫았다
  (`src/server/CLAUDE.md` 의 `detect.py` 행). 예외 구간은 자를 좁히는 통로가
  된다는 이유로 기각했다.

## 미해결 질문 (open question)

- **en `thresholds.json`** — PROD 타입의 precision 이 낮은 채로 폴백 종착지에서
  raw 로 나간다. 임계값 생성은 `--fit-threshold` 재학습 또는 fit-only 경로
  신설이 붙어 이 이슈 범위 밖이고, 별도 이슈로 열지 여부가 남았다.
- **`NER_SERVER_LANGS` opt-out** — 운영자가 en 만 빼고 띄울 수단이 없다. 열면
  `SUPPORTED_LANGS` 를 정본으로 삼는 유도 테스트 여럿의 전제를 건드린다.
- **훑기 범위 밖에 남은 낡은 열거** — 이 이슈는 diff 에 이름이 나온 파일과
  §2.6 훑기 범위만 봤다. 저장소 전역 스윕은 하지 않았으므로 그 밖에 남은
  자리가 있는지는 열려 있다.
- **README 의 지원 언어 표** — 학습 데이터셋·라벨 공간을 적은 표가 세 언어만
  들고 있는데 `src/ner/labelers/en/` 과 EN 코퍼스는 실재한다. 서빙이 아니라
  학습 데이터 문서의 낡음이라 이 이슈에서 손대지 않았다.
- **웹 UI 안내 문구의 한정사** — `fallbackHint` 가 "라틴 글자만 있는 문장"
  이라고 적는데 `_has_latin` 은 라틴 글자가 **하나라도** 있으면 참이다
  (`東京 Tokyo` → en). 좁게 말하는 쪽이라 읽는 사람이 과잉 기대를 갖지는
  않지만, 문자 그대로는 실제보다 좁다.
- **`test_live_server.py` 의 모델 디렉토리 가드** — `ServerConfig()` 로 경로를
  잡는데 그 아래 띄우는 서버는 `ServerConfig.from_env()` 를 쓴다.
  `NER_SERVER_MODEL_ROOT` 를 설정한 환경에서는 가드가 보는 경로와 서버가 읽는
  경로가 갈려, skip 이어야 할 자리가 503 실패가 된다. 기존 `_JA_DIR` 이 이미
  그 모양이라 이번에 새로 만든 것은 아니다.
- **`docs/reports/` 는 열거 정합 검사 훑기 범위 밖이다** — 감지 벤치마크
  리포트에 더한 절이 옛 3 클래스 세계를 옳게 서술하는지는 기계가 안 본다.
  그 문서의 세계가 지금과 달라 열거 대조를 걸면 참인 문장이 거짓 실패한다.
- **다중 글자 약어 문장 분할** — `_ABBREV_TAIL` 이 한 글자 약어만 되붙여
  `Mr.`·`Inc.` 가 문장 중간에서 잘린다. vi 가 이미 지고 있던 한계이고 en 이
  새로 만든 결함이 아니다. `_EN_FIXTURE` 에 `Inc.` 를 넣어 관찰만 한다.

## 구현 결과

`SUPPORTED_LANGS` 에 `en` 을 넣고 `detect_lang` 에 라틴 폴백 분기를 더했다.
번역은 `LANG_NAME`·`NLLB_LANG_CODE` 에 en 을 등록해 두 백엔드 모두 열었다.
웹 UI 는 셀렉터·번역 대상 목록·안내 문구를 갱신하고, 인라인 문자열이던 미지원
안내를 `UNSUPPORTED_HINT` 상수로 끌어올려 정합 테스트가 걸 앵커를 만들었다.

문서·주석·docstring 전반의 3 원소 이상 언어 열거를 전수로 훑어 서버 상수와
대조하는 `tests/server/test_docs_language_lists.py` 를 신설했다. 2 단 감지 정책을
서술하는 자리는 열거를 늘리는 대신 신호→언어 화살표 사슬로 다시 썼다 — `en` 은
신호가 아니라 폴백이라 `ja·ko·vi·en` 으로 늘리면 거짓이 되기 때문이다. 지원
집합의 참인 진부분집합은 목록이 아니라 술어로 적었다.

함께 닫은 기존 결함 하나 — `tests/server/test_api.py` 의 UI 옵션 단언이 ko 를
빠뜨린 채였다. 나머지 낡은 열거 넷은 `0f1f22b` 가 이미 닫았고, 그 커밋이 미지원
예시로 옮겨 둔 `lang:"en"` 은 en 이 지원 언어가 되면서 다시 거짓이 되므로
실재하되 지원 계획이 없는 `'th'` 로 옮겼다.

## 검증

| # | 무엇 | 결과 |
|---|---|---|
| V1 | `ruff check src/server tests/server` | All checks passed |
| V2 | `pytest tests/server/test_detect.py` | 47 passed |
| V3 | `pytest tests/server/ -m "not live"` | 202 passed · 4 skipped(NLLB 가중치) |
| V4 | `pytest tests/server/test_inference_integration.py -k test_en_` | 9 passed |
| V4.5 | `curl /health` | `status: ok` · `langs.en = {loaded: true, thresholds: false}` |
| V5 | `python -m server.scripts.example_client` | DEMO PASS · 종료코드 0 |
| V6 | `pytest tests/server/ -m live` | 4 passed (`test_en_fallback_live` 포함) |
| V7 | NLLB 실모델 `test_translate_nllb.py` | `test_real_model_preserves_pii_verbatim[en]` PASSED |
| V8 | 웹 UI HTTP·HTML 확인 | 셀렉터 en 옵션 · 영어 자동감지 `lang:en` · 한자만 미지원 안내 · 번역 경로 동작 |
| V9 | `git diff --stat` 대조 | 31 파일 전부 훑기 범위와 그 밖으로 뺀 셋에 대응 |
| V10 | 커밋 게이트 | 차단 없음 · `[gate] Some checks could not run` 미출현 |
| V11 | en 마운트 없는 호스트를 흉내낸 컨테이너 | `docker compose ps` **unhealthy**(healthcheck 연속 실패) · `/health` `degraded` · 영어 입력 503 · 라틴 섞인 배치 통째로 503 |

**V0 은 따로 돌리지 않았다.** 코드 변경 전 parity 사전 점검이라 목적이 "변경
뒤 parity 실패를 내 변경 탓으로 돌릴 수 있는가" 였는데, 변경 뒤
`test_en_parity_baseline_raw` 가 통과해 같은 사실을 더 강하게 확인한다.

**V11 은 실제 컨테이너로 쟀다.** 별도 compose 프로젝트·포트로 세 언어만
`/models` 아래 마운트하고 `NER_SERVER_MODEL_ROOT` 를 그쪽으로 돌려 en 이 없는
호스트를 흉내냈다. `start_period` 90 초가 지난 뒤 healthcheck 가 연속으로
실패해 `docker compose ps` 가 `unhealthy` 를 냈다. 통과의 모습이 "돈다" 가
아니라 "정직하게 안 뜬다" 인 유일한 항목이고, 그대로 나왔다.

**이 브랜치를 호스트에 배포했다.** `ner-server`(포트 8008)를 내리고 이
워크트리에서 이미지를 다시 빌드해 띄웠다. 지금 그 서비스는 **머지 전 브랜치
코드**로 돌고 있으며 `/health` 가 네 언어를 전부 `loaded` 로 보고한다. 되돌릴
때 쓸 직전 이미지는 `ner-server:0.1.0-pre250` 으로 태그해 뒀다. 같은 호스트의
`ner-server-en-codex`(포트 8018)는 별개 비교용이라 건드리지 않았다.

수치 표는 이 문서에 싣지 않는다. 원장은
`certified/classifier/en/deploy-trainseed42/metrics.json` 이고, parity 대조는
`test_en_parity_baseline_raw` 가 기계로 본다.

## 관련 커밋

- `12dcc9b`: feat(server): REST API 영어 서빙과 라틴 폴백 감지
