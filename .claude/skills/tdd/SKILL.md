---
name: tdd
description: Test-Driven Development. Use when implementing new logic, fixing a bug, or changing behavior in src/labelers/ or src/evaluators/. Write a failing pytest test before the implementation; reproduce bugs with a test before fixing.
metadata:
  source: https://github.com/addyosmani/agent-skills/tree/main/skills/test-driven-development
  adaptedFor: ner-pipeline
---

# TDD (NER 파이프라인 맞춤)

동작을 변경하는 모든 작업은 테스트 먼저. 벤치마크/라벨링 로직은 미묘한 버그가 평가 지표를 오염시키므로 TDD를 기본 워크플로로 채택한다.

## 언제 사용

- `src/labelers/**`, `src/evaluators/**` 신규 함수 추가
- BIO 태그 정렬/span 추출 수정
- 버그 리포트 대응 (재현 테스트부터)
- 평가 지표 계산 로직 변경

**사용하지 않을 때**: 순수 설정/문서/프롬프트 문자열 변경.

## RED → GREEN → REFACTOR

1. **RED**: `tests/`에 실패하는 pytest 케이스 작성. 실패 메시지 확인.
   - 테스트 파일 위치 규칙: `tests/test_<module>.py`
   - 픽스처는 기존 `tests/conftest.py` 재사용
2. **GREEN**: 테스트를 통과시키는 최소 코드만 작성. 관련 없는 개선은 금지.
3. **REFACTOR**: 테스트 유지된 상태에서 가독성/중복 정리. 각 단계 후 `pytest tests/test_<module>.py -x` 재실행.

## 버그 수정 (Prove-It Pattern)

1. 버그를 재현하는 테스트 작성 → 실패 확인
2. 수정 → 테스트 통과 확인
3. 근처 엣지 케이스도 테스트로 고정

## 실행

```bash
PYTHONPATH=src pytest tests/test_<module>.py -x -q
PYTHONPATH=src pytest tests/ -x -q          # 전체
PYTHONPATH=src pytest tests/ -k <keyword>   # 필터
```

## 완료 기준

- 새 테스트가 존재하고 통과한다
- 기존 테스트 회귀 없음
- `docs/api/`에 해당 모듈 문서가 최신 상태 (CLAUDE.md 3대 원칙)

상위 개념: `docs/wiki/concepts/agent-skills.md` 참조.
