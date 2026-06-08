# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트를 이해하기 위한 가이드이다.

## 프로젝트 개요

**다국어 NER(Named Entity Recognition) 파이프라인** — 멀티 백엔드 LLM 지원(OpenAI 호환 API) + BERT 토큰 분류 파인튜닝 + PII 증강 + 크롤링 기반 학습 데이터 생성.

- 한국어: KLUE NER (6 엔티티: PS, LC, OG, DT, TI, QT)
- 일본어·베트남어: canonical 10종 평면 = NER 5종(PER/LOC/ORG/PROD/EVT) + PII 5종(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD)
- 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## 개발 환경

- **호스트에서 직접 개발** (Docker는 vLLM 등 외부 서비스 전용 — 개발 컨테이너에 진입하지 않는다)
- 패키지 관리자: **UV** (`uv sync` 또는 `uv pip install -e .` 으로 editable install)
- `PYTHONPATH` **설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 `sys.path`에 자동 등록한다
- import 형태: `from ner.labelers.xxx import Xxx`, `from ner.llm_eval.xxx import Xxx`, `from ner.classifier.xxx import Xxx` 등 (src 접두어 없이)
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
│   ├── crawlers/ko/   # 한국어 Yonhap RSS 크롤러 + NER 태깅
│   └── wikiann_vi/    # WikiANN-vi → canonical 10종 평면 재라벨 + Wikidata 검증
├── classifier/        # JA·VI canonical 10종 평면 BERT 파인튜닝 (상세: src/ner/classifier/AGENTS.md)
│   ├── __main__.py         # CLI (python -m ner.classifier --lang ja|vi ...)
│   ├── data_utils.py       # JSONL 로딩, JA(slow)·VI(fast) tokenizer 분기 정렬, BIO↔span 변환
│   └── train_eval.py       # HF Trainer 래퍼, char-offset span F1 (metrics 공용)
└── scripts/           # 보조 셸 스크립트 (eval_spans.sh 등)
docker/
├── dev/               # 개발 컨테이너 (상세: docker/CLAUDE.md)
└── vllm/              # vLLM 서비스
results/               # 벤치마크 결과 JSON + 리포트
tests/                 # 테스트 (상세: tests/ner/CLAUDE.md)
docs/                  # 문서
│   ├── wiki/          # 프로젝트 독립적 도메인 지식 (상세: docs/wiki/schema.md)
│   ├── specs/         # 개발 규약·템플릿 (코딩 컨벤션, 신규 저장소 템플릿)
│   ├── manual/        # 프로젝트 구현 레퍼런스 (src/ 모듈 API·알고리즘 맵)
│   │   └── data/      # 데이터 스키마·엔티티 정의·데이터셋 스펙·데이터 검증
│   ├── reports/       # 자유 형식 벤치마크·실험 리포트 (GitHub Issue 무관)
│   └── issues/        # GitHub Issue별 plan/report 스냅샷
```

## 주요 CLI 엔트리포인트

- `python -m ner.llm_eval` — LLM NER 벤치마크 (ko/ja/vi 공용)
- `python -m ner.classifier --lang {ja,vi}` — BERT 파인튜닝·평가 (canonical 10종 평면)
- `python -m ner.augmenters.pii` — 합성 PII 주입
- `python -m ner.augmenters.crawlers.ko` — 한국어 뉴스 크롤러 + NER 태깅
- `python -m ner.augmenters.wikiann_vi` — WikiANN-vi canonical 10종 평면 재라벨 (상세: `docs/manual/data/canonical-entity-schema.md`)

상세 옵션은 각 모듈의 `--help` 또는 `src/**/AGENTS.md` 참조.

## 개발 3원칙

- **원자 단위**: 한 번의 요청은 검증 가능한 최소 기능 단위로 처리한다.
- **검증 의무**: 테스트 통과 + `docs/` 반영 전까지 '완료'로 간주하지 않는다.
- **분해는 사람 주도**: 단일 단계로 검증이 어려우면 작업자에게 먼저 분해 방식을 묻는다.

## 코딩 컨벤션 (주석/문서 언어)

- `print()` / `logger.*` / 예외 메시지 / argparse `help`: **영문**
- 함수·클래스·모듈 docstring, 인라인 `#` 주석: **한국어**
- 문자열 리터럴: **홑따옴표(`'`) 기본**, escape 필요 시 `"` 허용
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
└── issue-12-vi-crawler.md   # 계획 + 구현 결과 통합 단일 파일
```

- 계획 단계(3단계)에서는 파일에 기록만 하고 **커밋하지 않는다.** 사용자가 로컬에서 직접 읽어가며 진행한다.
- 구현 완료(5단계) 후 같은 파일에 결과·검증 섹션을 추가하고, 계획→구현 흐름이 정합적인지 확인한 뒤 **이때 최초 커밋**한다.

### 이슈 진행 절차 (경량 5단계, 승인 1회)

1. **등록**: GitHub Issue 작성 — 목적, 성공 기준(테스트/메트릭), 범위 정리
2. **브랜치**: `develop`에서 `feat/issue-{N}-slug` 분기
3. **구현 계획 수립 → 승인 요청**: 하위 작업 3~6개를 Issue 본문의 체크박스로 분해하고, 동일 내용을 `docs/issues/issue-{N}-{slug}.md`의 계획 섹션에 기록 (**커밋하지 않음**)
4. **구현 & 원자 커밋**: 의미 단위로 커밋(체크박스 개수와 무관), 각 커밋 본문에 `refs #N`. 입도 기준은 위 "커밋 입도" 섹션 참조
5. **마무리**: 테스트 통과 + `docs/` 갱신 확인 → **산출물 확인**: `refuter` 스킬로 현재 diff를 이 이슈의 성공 기준에 대해 반증(PASS면 진행, FAIL이면 4단계로 회귀) → `docs/issues/issue-{N}-{slug}.md`에 구현 결과·검증 섹션 추가 → 계획~구현 문서 정합성 확인 후 이슈 md **최초 커밋** → PR 생성(제목 또는 본문에 `closes #N`) → 머지 전 `git status`로 미커밋 파일 확인

> **산출물 확인(반박자)**: 4단계 구현 후 이슈 종료 전, `refuter` 스킬을 호출해 기억이 깨끗한 Sonnet 서브에이전트로 현재 diff를 *승인이 아니라 반증*시킨다 — 코드 정합성·측정 무결성(리포트·`docs/issues` 숫자 ↔ `results/*.json`)·테스트 무결성 3축. PASS면 종료, FAIL이면 구현으로 회귀. 상세는 "루프 검증 게이트" 섹션 및 `.claude/skills/refuter/`.

## 루프 검증 게이트 (격리 컨텍스트 반박자)

ralph/ultrawork 등 **루프 모드일 때만** 작동하는 Stop 훅 게이트. 일상 단발 편집·대화 턴에는 뜨지 않는다.

- **발동 조건**: 루프 활성(`.omc/state/sessions/<id>/prd.json`에 미완료 story 존재, 또는 수동 토글 `.omc/state/refuter-gate.on`) **+** 미커밋 diff 존재.
- **순서**: ① 변경된 `.py`에 `ruff check`(결정적 선통과, 실패 시 완료 차단) → ② 현재 diff에 대한 반박자 판정 요구.
- **반박자**: 기억이 깨끗한 Sonnet 서브에이전트가 *승인이 아니라 반증*을 전담 — 코드 정합성·회귀, 측정 무결성(리포트·`docs/issues` 숫자 ↔ `results/*.json`), 테스트 삭제·약화 여부를 본다. 절차는 `refuter` 스킬.
- **판정 파일**: `.omc/state/refuter/<diff_hash>.json`(`verdict: PASS|FAIL`). 훅은 모델 말이 아니라 이 파일을 직접 읽는다. PASS만 완료 허용.
- **무한루프 차단**: 재진입 Stop(`stop_hook_active`)에서는 재-block하지 않는다(Claude Code 연속-block 안전장치 존중, OMC persistent-mode와 동일 패턴). 세션 누적 block 6회 초과 시에도 통과·에스컬레이션.
- **OMC 공존**: persistent-mode(계속 일하라)와 **보완적** — 게이트는 *완료 관문*이라 직교한다. 둘 다 루프 모드에서만 작동하고 재진입 비-block으로 deadlock을 피한다.
- **끄기**: 루프를 안 돌리면 자동 비활성. 강제로 끄려면 `DISABLE_OMC=1` 또는 `OMC_SKIP_HOOKS=refuter-gate`(게이트가 이 토큰을 인식), 또는 `.claude/settings.json`의 `hooks.Stop`에서 제거.
- 구현: `.claude/hooks/refuter_gate.py`, `.claude/skills/refuter/`.
