---
name: refuter
description: 격리 컨텍스트 반박자 — 현재 diff를 원기준에 대해 "승인이 아니라 반증"하고 판정을 JSON으로 기록. Stop 게이트(refuter_gate.py)가 루프 모드에서 반박자를 요청할 때, 또는 완료 직전 수동으로 산출물을 검증하고 싶을 때 사용. 트리거 — "반박", "refute", "검증 게이트", "루프 완료 검증", Stop 게이트 block 메시지.
---

# refuter — 격리 컨텍스트 반박자

완료 직전, **기억이 깨끗한 Sonnet 서브에이전트**를 띄워 현재 변경을
*승인이 아니라 반증*시킨다. 산출물을 만든 본인(메인 세션)이 자기 답을
후하게 보는 self-preferential bias를 피하는 게 목적이다.

## 언제

- Stop 게이트(`.claude/hooks/refuter_gate.py`)가 block reason으로 반박자를
  요청할 때 (루프 모드 + 미커밋 diff + ruff 통과 상태).
- 또는 완료 직전 산출물을 한 번 더 적대적으로 점검하고 싶을 때(수동).

## 절차

1. **diff_hash 계산** (게이트와 동일해야 판정 파일이 매칭된다):
   ```
   git diff HEAD | sha256sum   # 앞 12자
   ```
2. **기준 출처 확보**: 가장 가까운 `docs/issues/issue-*.md`의 계획/성공
   기준 섹션, 또는 활성 PRD(`.omc/state/sessions/<id>/prd.json`)의
   acceptance criteria. 없으면 직전 사용자 요청을 기준으로 삼는다.
3. **반박자 spawn** — Agent 툴, 격리 컨텍스트, `model=sonnet`:
   - OMC 환경이면 `subagent_type`을 `code-reviewer`(또는 더 적대적인
     `critic`)로 지정해 OMC 검증-위임 정책(approval pass는
     code-reviewer/verifier)과 정합시킨다. 미설치 시 기본 서브에이전트.
   - 역할: **반증 전담.** 승인하지 말 것. "문제 없음"을 증명하지 말고
     "기준 미충족·회귀·오염"을 찾아내려 시도하라.
   - 검증 축 3개:
     - **(a) 코드 정합성·회귀**: diff가 기준을 실제로 충족하나? 숨은
       회귀·엣지케이스 누락은?
     - **(b) 측정 무결성**(이 프로젝트의 핵심 위험): 리포트·`docs/issues`·
       커밋 메시지의 **숫자가 `results/*.json`과 일치**하나? gold/eval
       세트가 변조됐나? 시드·분할 누수는?
     - **(c) 테스트 무결성**: 테스트 삭제·`skip`·`xfail`·assert 약화로
       통과를 만든 흔적은?
   - 범위 제한(비용·5분 캐시 TTL): diff + 기준만으로 판단하고, 특정
     주장을 검증할 때만 파일을 read 하라. 저장소 전체 탐색 금지.
4. **판정 기록** — 반박자가 직접 Write 한다:
   `.omc/state/refuter/<diff_hash>.json`
   ```json
   {"verdict": "PASS" | "FAIL",
    "diff_hash": "<앞 12자>",
    "findings": ["...", "..."],
    "model": "sonnet",
    "round": 1}
   ```
5. **후속**:
   - PASS → 완료 진행(게이트가 통과시킨다).
   - FAIL → findings를 수정한다. 코드가 바뀌면 diff_hash가 달라지므로
     게이트가 새 판정을 다시 요구한다(같은 결함 재발 방지).

## 비용 메모

- 반박자는 **Sonnet** 고정(메인 Opus 대비 ~45% 저렴, 측정 오류엔 충분).
  서브에이전트는 별 컨텍스트라 메인 캐시를 깨지 않는다.
- 느리고 넓게 읽으면 메인 세션 캐시 TTL(5분)을 넘겨 재진입 비용이
  10배 튄다. **빠르고 좁게** 유지하라.
- 결정적 검사(ruff)는 게이트가 이미 0토큰으로 선통과시켰다. 반박자는
  ruff/문법이 아니라 *판단이 필요한 축*(정합성·측정·테스트)만 본다.
