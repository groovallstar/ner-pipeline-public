# issue-115: 한국어 PII 4종 주입 — canonical 10종 평면 완성

- Issue: https://github.com/groovallstar/ner_pipeline/issues/115
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-115-ko-pii-canonical-10type`
- base: `origin/develop` (1fb85a1, PR #114 머지 반영)

## 목적

issue #111의 ko canonical NER 5종(PER/LOC/ORG/PROD/EVT) + DAT seed gold에
누락된 PII 4종(EMAIL/PHONE/ID_NUM/CREDIT_CARD)을 합성 PII 주입으로 증분해
canonical **10종 평면** gold를 완성한다. PII는 분류기가 접두로 쉽게 풀지
못하도록 **문장에 자연스럽게(llm 모드)** 삽입한다.

## 범위

- 포함:
  - PII 주입기 한국어 지역화 — `generators/ko.py`(성명·전화·주민등록번호·
    주소·날짜) + `config/base/injector`(suffix용 `_KO_CONNECTORS`) ko 분기
  - `--pii-labels` CLI(주입 라벨 제한) + llm 주입 경로에 `pii_labels` 전달
  - llm 자연삽입 ko 프롬프트(`llm_injector._INJECTION_PROMPT_KO`, 경직 접두 금지)
  - ko gold에 llm 주입(PII 4종) → 10종 gold `data/klue/pii_all.jsonl`
- 제외:
  - ko 분류기 학습·벤치마크(10종 완성 후 별도 이슈)
  - **ko 라벨러 10종 확장**(verify 전용이었으므로 제외 — ko LLM NER 10종
    벤치마크 이슈에서 측정과 함께 추가)

## 현 상태 (fact)

- ko gold `data/klue/origin.jsonl`: 26008 레코드, NER 5종 + DAT. 스키마 =
  주입기 입력과 동일(`{id, text, entities:[{label,start_char,end_char,text}]}`).
- 주입기 ja/vi → ko 지역화. llm 모드는 `generate_pii`로 값 생성 후 LLM이
  문중 자연 삽입, `extract_spans`가 string-match로 offset 추출(정확 보장).

## 성공 기준

- [ ] `--lang ko --mode llm` 주입 동작 + offset 무결(string-match 정확)
- [ ] PII가 경직 접두("이메일:" 등) 없이 자연 삽입(접두 직후 비율 ≈ 0)
- [ ] ko 10종 gold: EMAIL/PHONE/ID_NUM/CREDIT_CARD 각 > 0, 원문 엔티티 보존
- [ ] 한국 PII 포맷 spot-check(주민등록번호 `YYMMDD-Gxxxxxx` 체크섬 무효,
      `010-xxxx-xxxx` 등)
- [ ] `uv run pytest tests/ner` green(ko 주입 테스트 추가)
- [ ] docs 갱신(augmenters AGENTS, canonical-entity-schema ko PII)

## 결정 로그 (확정)

1. **주입 라벨 = PII 4종만**(EMAIL/PHONE/ID_NUM/CREDIT_CARD). KLUE 유래
   PER/LOC/DAT는 합성분 없이 순수 유지 → CLI `--pii-labels` 신규.
2. **주입 방식 = llm 자연삽입**(`--mode llm`). suffix(접두 `연락처:` + 문말
   나열)는 BERT 분류기가 접두 패턴으로 쉽게 풀어 학습 편향(`augmenters/AGENTS.md`
   "suffix BERT 적합성 낮음(패턴 편향)"). llm 모드는 접두 없이 문중 삽입.
3. **verify 미사용**. llm `extract_spans`가 PII를 string-match로 offset 정확
   추출(스모크 0 mismatch)하고 원문 엔티티도 재탐색 보존(스모크 100%). 추가
   라벨링이 불필요. `--verify` drop_span은 레코드 전체를 LLM과 대조해 사람
   KLUE gold를 ~13% 삭제(스모크 측정)하므로 ko 사람 gold에 부적합.
4. **ko 라벨러 10종 확장 제외**. verify를 먹이려고 도입했던 것이라 verify
   제거와 함께 revert. ko LLM NER을 10종으로 측정하는 건 별도 벤치 이슈.

## 구현 단계 (가변)

- [x] 1. `generators/ko.py` 신규 + `base.py`/`config.py`/`injector.py` ko 분기
- [x] 2. `__main__.py` `--lang ko`·`--pii-labels` + llm 경로 `pii_labels` 전달
- [x] 3. `llm_injector._INJECTION_PROMPT_KO`(자연삽입·경직 접두 금지)
- [x] 4. 테스트: ko 생성기·suffix 주입·주민번호 체크섬 무효·llm 프롬프트 +
      `extract_spans` 공백 허용 매칭(카드 이중공백 흡수) 회귀 테스트
- [x] 5. gold 생성 `data/klue/pii_all.jsonl`(llm, 26008→25989) + 10종 검증
- [x] 6. docs(augmenters AGENTS, canonical-entity-schema ko PII)

## 검증 (50문장 스모크, gemma-4-31B)

| 항목 | 결과 |
|---|---|
| 레코드 | 50 → 50 (드롭 0) |
| 원문 KLUE 엔티티 보존 | 95/95 (100%) |
| offset 무결 | 0 mismatch |
| 경직 접두 직후 PII | 0/70 (0%) |

(주입 PII는 문법은 자연스러우나 KLUE 뉴스 주제와 의미상 어색 — 형식 기반
인식 타입이라 분류 학습엔 무해, 접두 의존 차단이라 의도와 부합.)

## 구현 결과 (full run, gemma-4-31B)

`augmenters/pii --source jsonl --input data/klue/origin.jsonl --lang ko
--pii-labels EMAIL PHONE ID_NUM CREDIT_CARD --mode llm` → `data/klue/pii_all.jsonl`.
26008 입력 → **25989 기록**(19 스킵 0.073% — LLM이 일부 이메일·전화 재포맷).

**10종 분포**: PER 18450·LOC 7935·ORG 10410·DAT 9915·PROD 3253·EVT 1187 +
EMAIL 8320·PHONE 8448·ID_NUM 8469·CREDIT_CARD 8582.

| 검증 | 결과 |
|---|---|
| offset 무결 | **0 mismatch** (33819 PII 포함 전수) |
| 경직 접두 직후 PII | **0/33819 (0.00%)** — 자연삽입 |
| 주민등록번호 체크섬 | **8469/8469 전부 무효**(실유효 번호 비생성) |
| 원문 NER 보존 | **99.30%**(전체 입력 51509 대비; 스킵 19행 제외 시 99.40%. PER/LOC/ORG/EVT ~99.6%, DAT 98.5%·PROD 98.2%) |

원문 손실 0.7%는 LLM 재작성이 상대날짜(`7년전`)·복합 작품명을 패러프레이즈해
string-match 미발견 → drop된 것(허용 범위).

## 위험·의존성

- 워크트리 gold는 `/work/git` 심링크. 명령은 `env -u VIRTUAL_ENV -u PYTHONPATH`.
- llm 모드 = vLLM `:8081`(gemma-4-31B) 의존 + 비결정(temp 0.7). 일부 레코드는
  PII 미발견 시 drop(로그 경고).
- PII 윤리: 주민등록번호는 구조만 생성하고 체크섬 미충족(실유효 번호 비생성).
