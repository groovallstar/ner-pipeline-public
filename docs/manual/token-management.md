# 토큰 관리 전략 — ner_pipeline 적용 가이드

> 출처: `docs/wiki/concepts/token-management.md`에서 추출한 프로젝트별 적용 내용.
> 일반 전략 설명은 원문 참조.

---

## 전략 1: `.claudeignore` 적용 예시

프로젝트 루트에 `.claudeignore` 파일을 생성한다:

```
# 빌드/캐시
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/

# 모델 가중치/데이터 (대용량)
*.bin
*.safetensors
*.onnx
*.pt
*.pth
*.ckpt

# 로그/DB
*.log
*.db
*.sqlite

# 환경/시크릿
.env*

# 미디어
*.png
*.jpg
*.gif
*.mp4

# Docker 볼륨/임시
docker/vllm/

# IDE
.vscode/
.idea/
```

---

## 전략 2: tasks.md 분할 — 현재 구조

현재 프로젝트에는 openspec 하위에 변경 단위별 tasks.md가 이미 분리되어 있다:

```
openspec/changes/
└── archive/
    └── 2026-04-10-add-bio-dataset-spec-registry/tasks.md
```

완료된 변경은 archive로 이동하는 패턴을 따르고 있다. 추가로 고려할 점:

- 각 tasks.md에 현재 상태 요약을 상단에 두어 전체를 읽지 않아도 진행 상황을 파악할 수 있게 한다
- archive 디렉토리를 활용하여 완료된 작업을 격리한다

---

## 전략 4: Handoff 문서 — 이 프로젝트 적용 예시

```markdown
목표: vLLM NER 라벨러 파이프라인 구현
수정한 파일: src/labelers/{ko,ja,vi}/vllm_ner_labeler.py, src/labelers/base_vllm_labeler.py, tests/test_*_labeler.py
확인한 사실: vLLM은 OpenAI 호환 API(/v1/chat/completions) 사용, AsyncOpenAI로 호출
실패한 시도: temperature>0 설정 시 출력 불안정
다음 작업: concurrency 파라미터 튜닝으로 처리 속도 최적화
완료 조건: 한국어 NER 라벨링 정확도 85% 이상
```

---

## 전략 5: Plan mode — openspec 흐름

openspec의 proposal → design → tasks 흐름이 이미 Plan mode 역할을 한다. 새 기능 구현 시 반드시 이 흐름을 따른다.

---

## 전략 6: `CLAUDE.md` 계층 구조 — 현재 구조

```
ner_pipeline/
├── CLAUDE.md              # 프로젝트 전체 (현재 존재)
├── src/
│   └── CLAUDE.md          # 소스 코드 컨벤션
├── docker/
│   └── CLAUDE.md          # Docker 환경 상세
└── tests/
    └── CLAUDE.md          # 테스트 전략/실행 방법
```

---

## 전략 7: Skills — 현재 등록된 스킬

현재 `.claude/skills/`에 이미 스킬 구조가 있다:

```
.claude/skills/
├── omc-reference/
├── openspec-apply-change/
├── openspec-archive-change/
├── openspec-explore/
└── openspec-propose/
```

**추가 스킬 후보:**
- `ner-evaluation/`: NER 평가 메트릭(seqeval, bert-score) 실행 절차
- `data-pipeline/`: 데이터 수집 → 전처리 → 라벨링 파이프라인 절차

---

## 전략 10: MCP 서버 — 이 프로젝트 적용

- 현재 작업에 필요한 MCP 서버만 활성화
- NER 파이프라인 개발 시 불필요한 서버는 비활성화
- `.claude/settings.json`에서 필요시만 토글

---

## 전략 11: 도구별 역할 분리 — 이 프로젝트에서의 역할

| 도구 | 강점 | 이 프로젝트에서의 역할 |
|------|------|------------------------|
| Claude Code | 복잡한 추론, 디버깅 | NER 파이프라인 설계/구현, 버그 수정 |
| Gemini CLI | 긴 컨텍스트 분석 | 대규모 데이터셋 구조 분석, 논문 리뷰 |
| Codex | 큰 컨텍스트 지원 | 코드베이스 전체 맵핑 |

---

## 적용 우선순위 체크리스트

### 1단계: 즉시 적용

- [x] `.claudeignore` 생성 → 빌드 산출물, 모델 가중치, 로그 차단
- [x] `/clear`와 `/compact` 습관화
- [x] HANDOFF.md 템플릿 도입

### 2단계: 이번 주 안에

- [x] `CLAUDE.md` 슬림화 및 계층 구조 도입 (루트 + src/ + docker/ + tests/)
- [ ] Plan mode (openspec 흐름) 일관 적용
- [ ] 반복 절차의 skill 승격 (NER 평가, 데이터 파이프라인)

### 3단계: 장기 구조 투자

- [ ] 서브에이전트 격리 패턴 정착
- [ ] 검색 기반 컨텍스트 주입 습관화
- [ ] MCP 서버 최소화 운영
- [ ] 도구별 역할 분리 체계화
