# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트를 이해하기 위한 가이드이다.

## 프로젝트 개요

**다국어 NER(Named Entity Recognition) 파이프라인** — 멀티 백엔드 LLM 지원(OpenAI 호환 API) + BERT 토큰 분류 파인튜닝 + PII 증강 + 크롤링 기반 학습 데이터 생성.

- 한국어: canonical 10종 평면 완성 — NER 5종(PER/LOC/ORG/PROD/EVT) + DAT 은 KLUE 유래(PROD/EVT LLM 재라벨 증분, TI/QT 드롭), PII 4종(EMAIL/PHONE/ID_NUM/CREDIT_CARD)은 합성 주입
- 일본어·베트남어: canonical 10종 평면 = NER 5종(PER/LOC/ORG/PROD/EVT) + PII 5종(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD)
- 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## 개발 환경

- **호스트에서 직접 개발** (Docker는 vLLM 등 외부 서비스 전용 — 개발 컨테이너에 진입하지 않는다)
- 패키지 관리자: **UV** (`uv sync` 또는 `uv pip install -e .` 으로 editable install)
- `PYTHONPATH` **설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 `sys.path`에 자동 등록한다
- import 형태(모두 src 접두어 없이): 라이브러리는 `from ner.<모듈>.xxx import Xxx`(labelers·classifier·llm_eval·augmenters·metrics), REST API 서비스는 `from server.xxx import Xxx` (`ner` 와 분리된 top-level 패키지)
- 상세 원리·설정: `docs/wiki/concepts/src-layout-packaging.md`

## 핵심 디렉토리 구조

```
src/ner/
├── labelers/          # NER 라벨링 모듈 (상세: src/ner/labelers/AGENTS.md)
│   ├── ko/            # 한국어 (vllm, openai)
│   ├── ja/            # 일본어 (vllm, openai)
│   ├── vi/            # 베트남어 (vllm, openai)
│   ├── dataset_loader.py
│   ├── labeler_base.py
│   ├── tag_aligner.py     # BIO 태그 정렬/정규화/span 추출
│   └── hf_ner_labeler.py  # HuggingFace BERT NER 라벨러
├── llm_eval/          # 벤치마크 오케스트레이션 + 리포트 (상세: src/ner/llm_eval/AGENTS.md)
│   ├── __main__.py          # CLI (python -m ner.llm_eval --lang ko|ja|vi ...)
│   ├── benchmark_runner.py  # BenchmarkRunner (한국어 BIO / JA·VI offset-span 공용 러너)
│   ├── report.py            # ReportGenerator (span-match, seqeval, per-entity 테이블)
│   ├── error_analysis.py    # 문장별 오류 유형 분류 CLI
│   └── span_evaluator.py / span_evaluator_cli.py
├── metrics/           # span/BIO 메트릭 공용 구현 (classifier·llm_eval 공유)
│   ├── bio_metrics.py       # seqeval 기반 BIO 레벨 메트릭
│   └── span_metrics.py      # compute_offset_span_f1 등 span 레벨 메트릭
├── augmenters/        # 학습 데이터 증강 (상세: src/ner/augmenters/AGENTS.md)
│   ├── pii/           # 합성 PII 주입 (suffix/llm 모드, vLLM 교차 검증)
│   └── wikiann_vi/    # WikiANN-vi → canonical 10종 평면 재라벨 + Wikidata 검증
├── classifier/        # JA·VI·KO canonical 10종 평면 BERT 파인튜닝 (상세: src/ner/classifier/AGENTS.md)
│   ├── __main__.py         # CLI (python -m ner.classifier --lang ja|vi|ko ...)
│   ├── data_utils.py       # JSONL 로딩, fast(offset-trim)·PhoBERT(pyvi)·JA(slow) tokenizer 분기, BIO↔span 변환
│   └── train_eval.py       # HF Trainer 래퍼, char-offset span F1 (metrics 공용)
└── scripts/           # 보조 스크립트 (eval_ja_ner_test.py 등)
src/server/            # ja·vi NER REST API 서비스 (ner 라이브러리 소비; 상세: src/server/AGENTS.md)
├── __main__.py        # CLI (python -m server) — uvicorn 기동
├── app.py             # FastAPI /v1/ner(단일·배치)·/health
├── inference.py       # LangModel·ModelRegistry (모델 1회 로드·재사용, char-offset span)
├── detect.py          # 언어 자동감지 (가나→ja, 그 외→vi)
└── chunking.py        # 긴 입력 문장분할 + offset 보존
docker/
├── dev/               # 개발 컨테이너 (상세: docker/CLAUDE.md)
└── vllm/              # vLLM 서비스
results/               # 벤치마크 결과 JSON + 리포트
tests/                 # 테스트 (상세: tests/ner/CLAUDE.md)
docs/                  # 문서
│   ├── wiki/          # 프로젝트 독립적 도메인 지식 (상세: docs/wiki/schema.md)
│   ├── specs/         # 개발 규약 (코딩 컨벤션)
│   ├── manual/        # 프로젝트 구현 레퍼런스 (src/ 모듈 API·알고리즘 맵)
│   │   └── data/      # 데이터 스키마·엔티티 정의·데이터셋 스펙·데이터 검증
│   ├── reports/       # 자유 형식 벤치마크·실험 리포트 (GitHub Issue 무관)
│   └── issues/        # GitHub Issue별 plan/report 스냅샷
```

## 주요 CLI 엔트리포인트

- `python -m ner.llm_eval` — LLM NER 벤치마크 (ko/ja/vi 공용)
- `python -m ner.classifier --lang {ja,vi,ko}` — BERT 파인튜닝·평가 (canonical 10종 평면; `--group-key orig` 로 누출-free group K-fold)
- `python -m ner.augmenters.pii` — 합성 PII 주입
- `python -m ner.augmenters.wikiann_vi` — WikiANN-vi canonical 10종 평면 재라벨 (상세: `docs/manual/data/canonical-entity-schema.md`)
- `python -m server` — ja·vi NER REST API 서버 (uvicorn; 상세: `src/server/AGENTS.md`)

상세 옵션은 각 모듈의 `--help` 또는 `src/**/AGENTS.md` 참조.

## 개발 3원칙

- **원자 단위**: 한 번의 요청은 검증 가능한 최소 기능 단위로 처리한다.
- **검증 의무**: 테스트 통과 + `docs/` 반영 전까지 '완료'로 간주하지 않는다.
- **분해는 사람 주도**: 단일 단계로 검증이 어려우면 작업자에게 먼저 분해 방식을 묻는다.

## 코딩 컨벤션 (주석/문서 언어)

- `print()` / `logger.*` / 예외 메시지 / argparse `help`: **영문**
- 함수·클래스·모듈 docstring, 인라인 `#` 주석: **한국어**
- 한 줄 **79자 이내**, 함수 사이는 **한 줄만** 비운다, trailing whitespace 금지
- **코드·주석·식별자에 이슈/PR 번호·날짜·사람 이름 금지** — 영구 맥락은 커밋 메시지 `refs #N` 또는 `docs/issues/`에만
- 상세 규칙 및 예시: `docs/specs/coding-conventions.md`

## 커밋 컨벤션

- 제목·본문은 **한국어**로 작성
- `type` 접두사만 영문 허용 (`feat`, `fix`, `chore`, `docs`, `build`, `refactor`, `test`)

### 커밋 입도 (granularity)

- **기본값은 "합치기"**: 구현 + 그에 딸린 테스트 + 관련 문서 갱신은 같은 커밋이 기본이다. 분할은 아래 "분할이 정당한 경우" 중 하나에 해당할 때만 한다. 의심스러우면 합친다.
- **하위 작업 체크박스 ≠ 커밋 1개**: 이슈 계획의 체크박스는 *진행 추적 단위*이지 *커밋 단위*가 아니다. 동일 관심사·동일 모듈에서 발생한 변경은 체크박스가 여러 개여도 한 커밋으로 묶는다.
- **파일 수 휴리스틱(3파일=2커밋, 5파일=3커밋 등) 금지**: `git-master` 에이전트 또는 OMC 기본 정책의 파일 수 기반 자동 분할은 이 프로젝트에서 비활성화한다. 분할 기준은 *관심사*와 *독립 revert 가능성*뿐이다.
- **분할이 정당한 경우**(아래 중 하나라도 해당될 때만 분리):
  - 서로 다른 모듈/패키지(`labelers/` vs `classifier/` vs `augmenters/` 등)에 걸친 변경
  - 코드 변경과 무관한 대규모 문서 정리(라벨 영문화 일괄 치환 등)
  - 의존성·빌드 설정 변경(`pyproject.toml`, `uv.lock`)이 코드 변경과 분리해도 빌드가 깨지지 않을 때
  - 한 커밋이 ~500 LOC를 크게 넘고 의미 단위로 나뉠 수 있을 때
- **합치는 것이 옳은 경우**:
  - 같은 모듈의 로직 + 그에 딸린 테스트 + 그에 딸린 문서/스펙 갱신
  - 오타 수정, 링크 채움 같은 수커밋(<5줄) — 다음 의미 있는 커밋에 squash 하거나 일과 종료 시 `docs:` 한 커밋으로 묶기
  - 계획 단계의 체크박스 1·2·3이 모두 같은 함수/CLI 한 개를 만드는 데 필요했을 때
- **서브에이전트 위임 시 커밋 권한 회수**: `executor`·`git-master` 등 서브에이전트에 작업을 위임할 때는 프롬프트에 "커밋은 수행하지 말고 변경만 완료하라"를 명시한다. 커밋은 메인 세션에서 사용자 승인 후 일괄 수행하여, 에이전트 자체 판단으로 인한 잦은 자동 분할을 구조적으로 방지한다.

## docs 운영

- `docs/wiki/` 운영 규칙은 `docwiki` 스킬과 `docs/wiki/schema.md`에 위임.
- 코드 변경 후 `docs/manual/`(구현 맵)과 `docs/manual/data/`(데이터 스키마) 최신화 확인.

## 이슈 관리

- **GitHub Issues**(`groovallstar/ner_pipeline`)로 작업 단위를 관리한다. 자동 채번으로 중복을 방지한다.
- **무엇이 이슈가 되는가 — 사람이 소유, 에이전트는 default 실행.** 작업 단위 결정(이슈 등록 여부)·수락 기준 승인은 사람이 쥔다(에이전트 자기 채점 금지). 에이전트는 type+path 기반 default를 실행한다:
  - `feat`·`fix` ∧ `src/ner/**` 런타임 변경 → **이슈+브랜치+PR** (추적 가치 있는 제품 진화)
  - `docs`·`chore`·`refactor` (동작 불변·구조·문서·도구) → **develop 직접** (이슈/PR 없이, 크기 무관). PR이 별도 리뷰어를 붙이지 않아 이 부류엔 PR 오버헤드가 추적 이득보다 크다 — 실제 회귀 게이트는 refuter.
  - **type↔path 불일치**(예: `feat`인데 src 런타임 미변경, `chore`인데 src 런타임 변경) **또는 추적가치 모호** → 사람에게 에스컬레이션. 과소추적(조용한·비싼 실패) > 과다추적(시끄러운·싼 실패)이므로 모호하면 이슈 쪽으로 기운다.
- **측정-숫자 게이트(lane 무관).** `docs/reports/`·`docs/issues/`에 메트릭(수치)이 **신규·변경**되는 커밋은 직접 lane이라도 **commit 전 refuter 먼저** 실행한다 — 리퓨터 1번 축이 "리포트·docs 숫자 ↔ `results/*.json` 정합"이라, 숫자가 영구 기록에 진입하는 순간이 게이트 대상이다(트리거는 *재량*이 아니라 *사건*에 묶는다). 산문·링크·오타만 바꾸는 docs 커밋은 해당 없음.
- 마일스톤은 사용하지 않는다. 영역은 라벨로 구분한다 (`area:labelers`, `area:classifier`, `area:augmenters`, `area:llm-eval`, `area:infra`, `docs` 등).
- 이슈 등록: `gh issue create --title "제목" --body "설명" --label <area>` (`.github/ISSUE_TEMPLATE/` 템플릿 사용 권장)
- 브랜치명: `feat/issue-{번호}-{짧은-슬러그}` 예) `feat/issue-12-vi-crawler`
- 커밋 메시지: 기존 컨벤션(한국어 제목 + `type(스코프):` 접두사) 유지, 본문 끝에 `refs #12` 참조. 최종 PR 또는 마지막 커밋에는 `closes #12`로 이슈 종결.
- **계획·보고 문서는 GitHub Issue와 `docs/issues/` 양쪽에 모두 남긴다.** Issue는 실시간 협의·승인 기록, `docs/issues/`는 기능 설계·구현 히스토리의 영구 보관 용도. 이슈와 무관한 자유 형식 벤치마크·실험 리포트는 `docs/reports/`에 둔다.

### 이슈 문서 구조

이슈별로 `docs/issues/issue-{번호}-{슬러그}.md` 단일 파일을 만든다. 상세 규칙과 템플릿은 `docs/issues/README.md` 참조.

```
docs/issues/
├── README.md
└── issue-12-vi-crawler.md   # 설계 + 구현 결과 통합 단일 파일
```

- 실시간 게이트는 Issue 본문의 **수락 기준 목록**이다(3단계). `docs/issues/...md`는 산문 설계 + 구현 결과·검증을 합친 **영구 아카이브**로, 구현 완료(5단계) 후 작성·**최초 커밋**한다 — 재독용이 아니며 숫자 정합성은 refuter 게이트가 보증한다.

### 이슈 진행 절차 (경량 5단계, 승인 1회)

1. **등록**: GitHub Issue 작성 — 목적, 성공 기준(테스트/메트릭), 범위 정리
2. **브랜치**: `develop`에서 `feat/issue-{N}-slug` 분기
3. **수락 기준 확정 → 승인 요청**: 검증 가능한 acceptance criteria 3~6개를 Issue 본문에 적고 **이 기준 목록에 대해서만** 승인받는다 — 사람이 읽는 게이트는 산문 계획이 아니라 이 짧은 목록이다. test·metric이 걸린 이슈면 기준에 **목표 수치를 명시**한다(형식은 이슈마다 다름 — `results/*.json` 강제 아님). 하위 작업은 같은 본문에 체크박스로 분해. **eval·metric 이슈는 기준 확정 직후 *정의-시점* 반박자**(`refuter`)로 eval 설계가 구조적으로 누출-free·non-gameable한지 검증한 뒤 승인받는다 — 구현 전에 형식 오류를 잡는다(상세: "반박자 검증 게이트" 섹션).
4. **구현 & 원자 커밋**: 의미 단위로 커밋(체크박스 개수와 무관), 각 커밋 본문에 `refs #N`. 입도 기준은 위 "커밋 입도" 섹션 참조. 구현을 ralph 등 자율 루프로 돌리는 것은 3단계 기준이 기계검증 가능·신뢰되고 작업이 다회차/반복-shape일 때만 — 단발은 "분해는 내가 → 한 스텝만 위임 → 내가 기준으로 검증"이 기본.
5. **마무리**: 테스트 통과 + `docs/` 갱신 확인 → **산출물 확인**: `refuter` 스킬로 현재 diff를 이 이슈의 성공 기준에 대해 반증(PASS면 진행, FAIL이면 4단계로 회귀) → `docs/issues/issue-{N}-{slug}.md`에 설계 + 구현 결과·검증 작성 → 계획~구현 문서 정합성 확인 후 이슈 md **최초 커밋** → PR 생성(제목 또는 본문에 `closes #N`) → 머지 전 `git status`로 미커밋 파일 확인

> **산출물 확인(반박자)**: 5단계 *결과 시점* 반박자 — 현재 diff를 수락 기준에 대해 반증, PASS면 종료·FAIL이면 4단계 회귀. 정의·결과 2시점 정의는 아래 "반박자 검증 게이트" 섹션.

## 반박자 검증 게이트 (격리 컨텍스트, 정의·결과 2시점)

기억이 깨끗한 Sonnet/Opus 서브에이전트가 *승인이 아니라 반증*을 전담한다 — 작성자의 자기 채점을 격리된 적대자로 교차 검증. 3축: 코드 정합성·회귀 / 측정 무결성(리포트·`docs/issues` 숫자 ↔ `results/*.json`) / 테스트 삭제·약화 여부. 절차는 `refuter` 스킬.

검증을 끝에 몰지 않고 **두 시점**에 호출한다:

- **정의 시점(3단계 직후)** — eval 설계·수락 기준의 *형식*을 반증한다: falsifiable한가, gameable한가, **eval이 구조적으로 누출-free인가**(split·group-key·metric·임계). 의도("무엇을 중시하나")는 사람 고유라 반증 대상이 아니다 — 형식·방법론만 본다. *정의 오류가 가장 비싸다 — 누출된 eval은 완벽 구현·정직 기록을 전부 무효화한다. 구현 전에 잡는다.* diff가 없으니 **게이트가 아니라 리뷰** — 판정 파일 없이 findings를 사람에게 보고(절차: `refuter` §정의-시점 모드).
- **결과 시점(5단계)** — 현재 diff를 수락 기준에 대해 반증한다: 구현↔기준 정합, 숫자↔JSON 정합, 테스트 무결성.

**판정 파일(결과 시점)**: `.omc/state/refuter/<diff_hash>.json`(`verdict: PASS|FAIL`). 게이트는 모델 말이 아니라 이 파일을 직접 읽는다. PASS만 진행.

**자동화(선택)**: 루프 모드(ralph/ultrawork)에선 Stop 훅이 *결과 시점* 반박자를 자동 호출한다(`.claude/hooks/refuter_gate.py`). 단발·정의 시점은 수동 호출이고, 루프 미사용 시 hook은 휴면이다. 끄기: `OMC_SKIP_HOOKS=refuter-gate` 또는 `.claude/settings.json` `hooks.Stop`에서 제거.
