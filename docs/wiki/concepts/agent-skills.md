# Agent Skills — 평가 및 적용 현황

- **유형**: 개념 (methodology) · `docs/schema.md` 참조
- **출처**: https://github.com/addyosmani/agent-skills/tree/main/skills
이 페이지는 AI 에이전트 스킬 카탈로그를 NER/LLM 파이프라인 프로젝트 관점에서 평가하고, 강력 추천·보조·제외 기준을 정리한다.

---

## 강력 추천 (Strong Fit)

### 1. test-driven-development
- **요지**: 실패하는 테스트를 먼저 쓰고 최소 코드로 통과시킨 뒤 리팩터링(RED → GREEN → REFACTOR). 버그 수정 시에는 재현 테스트부터(Prove-It Pattern).
- **적용 근거**:
  - "원자적 기능 단위 / 검증 및 문서화" 원칙과 정확히 부합.
  - pytest 기반 테스트 구조가 이미 존재하는 프로젝트라면 즉시 활용 가능.
- **적용 포인트**: 신규 함수 추가 시 기본 워크플로로 채택.

### 2. debugging-and-error-recovery
- **요지**: "Stop-the-line" 규칙 — 예기치 못한 실패 발생 시 기능 추가 중단, 증거 보존, Triage 체크리스트(재현 → 격리 → 원인 → 수정 → 방어)로 근본 원인 해결.
- **적용 근거**:
  - 다중 백엔드 통합(LLM API, 모델 로딩 등)은 실패 모드가 다양 → 체계적 디버깅 프로토콜 필요.
  - 태그 정렬/span 추출 등 미묘한 버그가 평가 지표를 오염시킬 위험이 있는 NLP 파이프라인에 특히 유효.
- **적용 포인트**: 벤치마크 결과 이상치, span 평가 불일치, 라벨러 예외 발생 시 표준 절차.

### 3. performance-optimization
- **요지**: 추측 금지, 측정 먼저(MEASURE → IDENTIFY → FIX → VERIFY → GUARD). 프로파일이 지목한 병목만 손댄다.
- **적용 근거**:
  - LLM 벤치마크는 처리량·지연이 핵심이며 배치 크기, 프롬프트 길이, 샘플 수 등 튜닝 축이 많음.
  - NLP 파이프라인에서 병목은 측정 없이 예측하기 어렵다.
- **유의**: 이 스킬은 Core Web Vitals(웹 프론트엔드) 예시가 포함되어 있으나 방법론(측정 우선)은 그대로 적용 가능. 웹 지표 부분은 무시.

### 4. documentation-and-adrs
- **요지**: 코드는 *무엇*을, 문서는 *왜*를 남긴다. 중요한 기술 결정은 ADR(`docs/decisions/NNNN-*.md`)로 순번을 매겨 기록.
- **적용 근거**:
  - 문서 스키마가 확립된 프로젝트에서 ADR을 추가하면 기술 결정 맥락을 장기 보존할 수 있음.
  - "왜 이 라이브러리인가", "왜 이 평가 전략인가" 같은 결정은 코드에 드러나지 않는다.
- **적용 포인트**: 스키마 변경, 백엔드 교체, 평가 지표 변경 시 ADR 생성.

---

## 보조 (Nice to Have)

### 5. spec-driven-development
- **요지**: 코드 전에 사양 작성. SPECIFY → PLAN → TASKS → IMPLEMENT 4단계 게이트, 단계마다 사람 검토.
- **적합성**: `CLAUDE.md`의 "Human-Led Decomposition" 원칙과 맞음. 새 언어 추가(예: 베트남어 라벨러) 같은 중간 규모 기능에 유용.
- **주의**: 단일 파일 수정/오타 수정에는 과도.

### 6. planning-and-task-breakdown
- **요지**: 스펙을 검증 가능한 작은 태스크로 분해. 의존성 그래프, 인수 기준, 병렬화 가능성까지 명시.
- **적합성**: "원자적 기능 단위" 원칙의 실행 도구로 사용 가능. 데이터셋 추가 + 라벨러 + 평가 파이프라인 같은 다단계 작업에 적합.

### 7. code-review-and-quality
- **요지**: 5축 리뷰 — 정확성, 가독성, 아키텍처, 보안, 성능. 승인 기준은 "완벽"이 아니라 "코드 상태를 분명히 개선하는가".
- **적합성**: PR/머지 전 체크리스트로 사용. OMC의 `code-reviewer` 에이전트 역할과 중복되므로 기준 문서로만 참조해도 충분.

### 8. code-simplification
- **요지**: 동작을 정확히 보존하면서 가독성을 높이는 리팩터링. "줄 수 감소"가 아닌 "이해 속도 향상"이 목표.
- **적합성**: `tag_aligner.py`, 평가 러너처럼 점진적으로 복잡해진 모듈을 정리할 때 유용.

### 9. incremental-implementation
- **요지**: 얇은 수직 슬라이스로 구현 → 테스트 → 검증 → 커밋 반복. 한 번에 ~100줄 이상 작성 전 멈추기.
- **적합성**: "Atomic Functionality" 원칙의 실행 패턴. 신규 라벨러/평가 지표 추가 시 기본 리듬.

---

## 도입 방식 제안

1. **즉시 도입**: TDD, debugging-and-error-recovery — 기존 원칙과 100% 정렬, 런타임 비용 없음.
2. **다음 중간 규모 작업에 시범 적용**: spec-driven-development + planning-and-task-breakdown + incremental-implementation 세트.
3. **문서 체계 확장**: `docs/decisions/` 디렉터리 신설 후 ADR 템플릿 도입 (documentation-and-adrs).
4. **참조용 체크리스트로 보관**: code-review-and-quality, code-simplification, performance-optimization — 작업 트리거 시 스킬 원문 참조.

## 제외된 스킬 (참고)

UI/프론트엔드/배포 계열은 CLI·노트북 기반 NLP 파이프라인 성격과 맞지 않아 일반적으로 제외 대상:
`frontend-ui-engineering`, `browser-testing-with-devtools`, `api-and-interface-design`, `shipping-and-launch`, `ci-cd-and-automation`, `deprecation-and-migration`, `security-and-hardening`, `source-driven-development`, `context-engineering`, `idea-refine`, `using-agent-skills`.

필요 시 위 목록에서 개별 스킬을 재평가할 수 있다.

---

## 출처

- `sources/addyosmani-agent-skills.md` — 카탈로그 원본 및 수록 스킬 목록
