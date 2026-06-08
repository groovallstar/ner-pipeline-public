---
name: docwiki
description: docs/wiki/의 LLM-Wiki(Karpathy 모델) 운영 스킬. 새 논문·개념을 ingest하거나, 위키 기반 질문에 query로 답하거나, lint로 품질을 점검할 때 사용. 트리거 키워드 — "위키에 추가", "위키 업데이트", "위키에 넣어", "wiki에 추가", "wiki ingest", "wiki 조회", "wiki query", "wiki 린트", "wiki lint", "docwiki".
---

# docwiki

`docs/wiki/`(schema.md 기준 Karpathy LLM-Wiki)를 운영하는 스킬. 이 스킬은 3대 작업 — **Ingest / Query / Lint** — 를 정해진 절차대로 수행한다.

## 불변 원칙 (schema.md §수록 원칙)

1. **프로젝트 독립성**: `src/`, `results/`, `docker/` 등 호스트 저장소의 파일 경로·스크립트·결과 파일명을 위키 페이지 본문에 절대 쓰지 않는다. 재사용 가능한 이론·방법론·원문 요약만 담는다.
2. **경로 규칙**: 위키 내 참조는 `wiki/` 기준 상대 경로 (예: `concepts/bio-tagging.md`).
3. **페이지 유형**:
   - 이론·방법론·전략 → `docs/wiki/concepts/<slug>.md`
   - 논문·기사·블로그 1건 → `docs/wiki/sources/<slug>.md`
4. **bookkeeping은 LLM이 부담**: 사람은 방향성·검수만 한다. 교차 참조·역링크·인덱스 갱신은 매 작업마다 자동으로 처리한다.

항상 먼저 `docs/wiki/schema.md`를 읽어 최신 규칙을 확인한 뒤 작업한다.

## Operation 1 — Ingest (새 소스 추가)

**언제**: 사용자가 논문·기사·URL·개념을 위키에 반영하라고 요청할 때.

**절차**:
1. 원문을 읽고 핵심 요지 추출 (필요하면 WebFetch).
2. `docs/wiki/sources/<slug>.md` 작성 — 템플릿은 schema.md §페이지 템플릿 그대로:
   ```markdown
   # <저자> (<연도>) — <제목>

   - **저자**:
   - **연도**:
   - **매체/학회**:
   - **링크**:
   - **유형**: 1차 문헌 | 2차 자료

   ## 핵심 요지
   ## 주요 내용
   ## 인용하는 위키 페이지
   - `concepts/<file>.md`
   ```
3. 기존 `docs/wiki/concepts/` 전체를 grep해 이 소스로 갱신해야 할 페이지를 식별한다. 10~15개까지 연쇄 갱신 가능. 누락이 의심되면 더 찾는다.
4. 해당 개념이 위키에 아직 없으면 `docs/wiki/concepts/<slug>.md`를 신규 생성:
   ```markdown
   # <개념명>

   > 한 줄 요약

   ## 개요
   ## 핵심 원칙
   ## 세부 설명
   ## 출처
   - `sources/<file>.md` — <이 출처에서 가져온 핵심>
   ```
5. 양쪽 페이지에 교차 참조 추가: `sources/*.md` §인용하는 위키 페이지에 concepts 경로, `concepts/*.md` §출처에 sources 경로.
6. `docs/wiki/index.md` 갱신 — 없으면 생성. 규모 무관하게 매 ingest마다 카탈로그를 최신 상태로 둔다 (Karpathy 모델에서 index는 핵심 운영 요소).
7. `docs/wiki/log.md`에 한 줄 추가 (없으면 생성):
   ```
   ## [YYYY-MM-DD] ingest | <title>
   - <변경 요약 1줄>
   ```

## Operation 2 — Query (위키 기반 답변 + 복리 환원)

**언제**: 사용자가 위키 주제(BIO, KLUE, agent-skills, 토큰 관리, src layout 등)에 대해 질문할 때.

**절차**:
1. `docs/wiki/index.md`가 있으면 먼저 훑고, 없으면 `docs/wiki/concepts/`를 grep해 관련 페이지 선정.
2. 페이지 본문을 근거로 답변을 합성한다. 답변에 반드시 인용 경로를 명시한다 (예: *근거: `concepts/bio-tagging.md` §핵심 원칙*).
3. 위키에 정보가 없거나 낡았으면 즉시 Ingest 트리거로 전환하고 사용자에게 알린다.
4. **환원 (복리 효과)** — 답변이 재사용 가치가 있다고 판단되면:
   - 새로운 개념이면 `concepts/<slug>.md` 생성
   - 기존 페이지에 빠진 내용이면 해당 절에 병합 (예: §결정 근거, §세부 설명)
   - `log.md`에 `## [YYYY-MM-DD] query-filed | <title>` 기록
   - 이 단계가 Karpathy 모델의 핵심 — 좋은 Q&A를 휘발시키지 말 것.

## Operation 3 — Lint (품질 점검)

**언제**: 사용자가 "wiki 린트" 요청 시, 또는 대규모 ingest 직후.

**점검 항목 (schema.md §3 린트 그대로)**:
- **모순된 진술**: 같은 사실에 대해 다른 페이지가 불일치
- **낡은 주장**: 새 소스로 대체돼야 하는 내용
- **고아 페이지**: 어떤 페이지에서도 참조되지 않음 (`index.md` 포함)
- **누락된 참조**: `concepts/*.md`인데 `sources/*.md` 인용이 없음
- **끊어진 링크**: 상대 경로가 실제 파일과 일치하지 않음
- **수록 원칙 위반**: 본문에 호스트 저장소 경로·스크립트명 등장 (`src/`, `results/`, `docker/`, `.py` 파일명, 셸 명령)

**절차**:
1. `docs/wiki/` 전 파일 스캔 (Glob + Grep).
2. 항목별 위반 목록을 사용자에게 리포트.
3. 사용자 승인 후 수정하거나, 명백한 경미 수정(끊어진 경로 1건 등)은 바로 고친다.
4. `log.md`에 `## [YYYY-MM-DD] lint | <요약>` 기록, 수정 건수와 남은 이슈를 명시.

## 출력 스타일

- 작업 시작 시 어떤 operation을 수행하는지 1줄로 선언 (예: *"ingest — Karpathy LLM Wiki gist"*).
- 파일을 쓴 뒤에는 생성/수정된 경로 목록을 사용자에게 보여준다.
- Query 답변은 반드시 `concepts/` 또는 `sources/` 경로를 인용구로 포함한다.

## 스킬 밖 작업과의 경계

- `docs/api/`, `docs/guides/` — 이건 코드 종속 문서. 이 스킬의 대상이 아님.
- `.omc/wiki/` — OMC 자체 wiki 시스템. 이 스킬은 `docs/wiki/`만 다룸. 섞지 않는다.
- CLAUDE.md §Wiki 운영의 자연어 규칙은 이 스킬로 승격된 형태이며, 상충 시 이 스킬이 우선한다.
