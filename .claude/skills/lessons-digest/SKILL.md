---
name: lessons-digest
description: 반복되는 검토 지적을 모아 규칙으로 승격 — refuter 등 검증 소스의 결함성 지적을 추적 렛저에 누적하고, 테마별 재발을 사람에게 보여 선택적으로 coding-conventions·체크리스트로 승격한다. 자동 승격 없음(항상 사람 선택). 트리거 — "반복 지적 정리", "lessons digest", "규칙 승격", "승격 후보", Stop 훅의 lessons 넛지 메시지.
---

# lessons-digest — 반복 지적 → 규칙 승격

검증 소스(현재 refuter)가 매번 남기는 지적 중 **같은 종류가 여러 작업에서
반복되는 것**을 규칙으로 결정화한다. 개별 지적은 흩어져 사라지지만, 반복
패턴은 `coding-conventions.md`·검증 체크리스트로 올리면 재발을 막는다.

**핵심 원칙: 알림은 자동, 규칙 반영은 항상 사람이 손으로.** 스킬은 후보를
*제시*할 뿐, 승격은 사람이 고른 것만 실행한다(자동 승격 없음).

## 구성 요소

- **렛저**: `docs/specs/lessons-ledger.jsonl` (추적) — 결함성 지적 1건 =
  한 줄 `{id, source, ref, model, severity, text, status}`. `status` 는
  `open|promoted|declined`. git 추적이라 워크트리 정리에도 생존한다.
- **모으기 도구**: `collect.py` — 소스 스캔·중복 제거·상태 변경(부품).
- **넛지 훅**: `lessons_nudge.py` (Stop) — 미수집 지적이 임계
  (`LESSONS_DIGEST_THRESHOLD`, 기본 3) 이상이면 알림만. 차단 안 함.

## 언제

- Stop 훅의 `[lessons] N un-digested ...` 넛지를 봤을 때.
- 또는 그냥 "반복 지적 한번 정리하자" 싶을 때(수동).

## 절차

1. **모으기(ingest)** — 흩어진 소스 지적을 렛저에 누적(중복 자동 스킵):
   ```
   python3 .claude/skills/lessons-digest/collect.py ingest
   ```
   워크트리 전체의 `.omc/state/refuter/*.json` 에서 결함성(비-PASS) 지적만
   긁는다. `collect.py` 는 조악한 1차 필터일 뿐 — 진짜 판단은 다음 단계다.

2. **미검토 목록 읽기**:
   ```
   python3 .claude/skills/lessons-digest/collect.py list --status open --json
   ```

3. **테마별 클러스터(사람이 아니라 이 세션이 판단)** — `open` 항목을
   *같은 종류끼리* 묶는다. 각 클러스터에 대해:
   - **재발수** = 서로 다른 `ref`(diff) 개수. 많을수록 규칙화 가치 큼.
   - 1차 필터가 흘린 **비-결함(통과 근거·해소 완료·부정문)** 은 여기서 버린다.
   - 반복이 아니어도 *일반 규칙*으로 명백하면 후보에 올릴 수 있다(재발수는
     신호일 뿐 게이트가 아니다).

4. **후보 제시 → 사람 선택** — 클러스터를 재발수 내림차순으로 보이고,
   각 후보에 **제안 대상**을 붙인다:
   - 코드 작성 시 지킬 규칙 → `docs/specs/coding-conventions.md`
   - 검증자가 능동적으로 점검할 항목 → `refuter/SKILL.md` 체크리스트
   - 그 외(특정 모듈 문서·docs) → 해당 위치
   `AskUserQuestion`(multiSelect)으로 **승격할 것만** 고르게 한다. 자동 승격
   금지.

5. **승격 실행(선택분만)**:
   - 대상 문서에 규칙 문장을 추가(간결·일반화, 특정 이슈/숫자/이름 없이 —
     `coding-conventions.md` "이슈/PR 번호·날짜 금지" 규칙 준수).
   - 기여한 렛저 항목을 `promoted` 로:
     ```
     python3 .claude/skills/lessons-digest/collect.py status \
       --id <id1> <id2> --set promoted --target conventions
     ```
   - 제시했지만 사람이 안 고른 것은 `declined` 로 표시해 재부상을 막는다:
     ```
     python3 .claude/skills/lessons-digest/collect.py status \
       --id <idX> --set declined
     ```

## 비용 가드 (소스 무관·기본 탑재)

- **심각도 바닥**: 현재 refuter 는 고신호라 전부 수집. code-review 처럼
  다량·저신호 소스를 붙일 때 nit 티어를 버리는 지점은 `collect.py` 의
  `_classify` 다.
- **워터마크**: 3단계는 `status=open`(미검토 증분)만 읽는다 — 히스토리
  전체가 아니라 증분만 보므로 한 번 비용이 bounded.
- **배수**: 검토된 항목은 `promoted`/`declined` 로 활성 셋에서 빠진다.
  재고가 필요하면 `list --status declined` 로 되살펴본다.

## 소스 확장

`collect.py` 의 `SOURCES` 레지스트리에 어댑터 함수 하나를 추가하면 새 소스가
붙는다. 어댑터는 `(source, ref, model, text)` 튜플을 산출한다. 저신호·다량
소스는 `_classify` 에 심각도 바닥을 함께 조정한다.
