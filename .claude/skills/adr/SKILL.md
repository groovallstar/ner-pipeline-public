---
name: adr
description: Record Architecture Decision Records for the NER pipeline. Use when choosing between competing approaches (backend, dataset spec, evaluation metric, prompt strategy), when changing public APIs in src/, or when context behind a decision would otherwise be lost.
metadata:
  source: https://github.com/addyosmani/agent-skills/tree/main/skills/documentation-and-adrs
  adaptedFor: ner_pipeline
---

# ADR (NER 파이프라인 맞춤)

코드는 *무엇*을, ADR은 *왜*를 남긴다. 되돌리기 비싼 결정을 고정한다.

## ADR을 쓸 때

- LLM 백엔드 채택/교체 (vLLM ↔ Ollama ↔ OpenAI ↔ HF)
- 데이터셋 스펙·태그 맵·BIO 정렬 전략 변경
- 평가 지표 정의 변경 (micro vs macro F1, span matching 규칙)
- 프롬프트 전략 전환 (few-shot 수, schema 방식)
- 공개 API 시그니처 변경 (`src/labelers/**`, `src/evaluators/**`)

## 위치와 네이밍

```
docs/decisions/NNNN-<kebab-slug>.md
```

번호는 순차 4자리 (`0001-...`, `0002-...`). 기존 디렉터리가 없으면 생성.

## 템플릿

```markdown
# NNNN. <결정 제목>

- Status: Proposed | Accepted | Superseded by NNNN
- Date: YYYY-MM-DD
- Deciders: <이름/역할>

## Context
무엇 때문에 이 결정을 하게 되었는가. 제약, 기존 상태, 문제.

## Decision
우리가 내린 결정은 무엇인가. 한두 문장으로 명확히.

## Consequences
긍정/부정 영향, 트레이드오프, 따라오는 후속 작업.

## Alternatives Considered
검토했으나 채택하지 않은 대안과 사유.

## References
관련 코드 경로, 이슈, 실험 결과(`results/*.json`), 위키 페이지.
```

## 운영 규칙

- 되돌림 시 기존 ADR은 `Superseded by NNNN`으로 상태만 바꾸고 본문은 유지 (이력 보존)
- 코드 변경 PR이 결정 근거가 되면 PR 설명에 ADR 링크 포함
- `docs/api/`, `docs/wiki/`와 교차 참조 (상대 경로 `docs/...`)

## CLAUDE.md 연계

- 스키마/위키 규칙은 `docs/schema.md`
- 도메인 지식은 `docs/wiki/concepts/`
- 상위 개념: `docs/wiki/concepts/agent-skills.md`
