---
name: debug-triage
description: Systematic root-cause debugging for NER pipeline failures. Use when pytest fails, benchmarks produce unexpected metrics, LLM backends (vLLM/Ollama/OpenAI) error out, or BIO/span logic misbehaves. Follow stop-the-line + triage checklist instead of guessing.
metadata:
  source: https://github.com/addyosmani/agent-skills/tree/main/skills/debugging-and-error-recovery
  adaptedFor: ner-pipeline
---

# Debug Triage (NER 파이프라인 맞춤)

## Stop-the-Line 규칙

예기치 못한 실패가 발생하면:
1. 기능 추가 **중단**
2. 증거 **보존** (에러 전문, 스택 트레이스, 재현 커맨드, 입력 샘플 텍스트/토큰, 사용 모델명)
3. 아래 **Triage 체크리스트** 수행
4. 근본 원인 **수정**
5. 회귀 방지 **가드** (테스트 추가)
6. 검증 통과 후 **재개**

## Triage 체크리스트

### 1. 재현
- 최소 재현 커맨드 확보 (`PYTHONPATH=src python -m llm_eval --lang <x> --models ... --max-samples <N>`)
- 결정적인지 확인 (시드, `temperature=0`, `top_p`, 배치 크기)
- 재현 불가 시: 로깅/직렬화 지점 추가 후 재시도

### 2. 격리
- 최근 변경 범위 확인: `git diff`, `git log --oneline -20`
- 백엔드 구분: vLLM / Ollama / OpenAI / HF 중 어느 레이어?
- 데이터 구분: 데이터셋 로더 / 프롬프트 / 정렬 / 평가 중 어디?

### 3. 원인
- 로그·스택 트레이스 맨 아래부터 읽기
- BIO 문제면 `tag_aligner.py` 토크나이저/오프셋 매핑 우선 점검
- 성능 이상이면 입력 길이, 모델 컨텍스트, 배치 설정 확인
- LLM 응답 포맷 이상이면 프롬프트 diff와 실제 응답 raw 로그 비교

### 4. 수정
- 근본 원인만 고친다. 주변 정리는 별도 커밋.
- 회귀 테스트를 먼저 작성 (`/tdd` 스킬 연계)

### 5. 가드
- 실패 조건을 `tests/`에 고정
- 로그 메시지 개선 (다음 번 진단 시간 단축)

### 6. 검증
```bash
PYTHONPATH=src pytest tests/ -x -q
# 필요 시 소규모 벤치마크 재실행
```

## 흔한 함정

- 벤치마크 결과 JSON만 보고 원인 단정 → 반드시 원시 라벨러 출력도 확인
- 평균 F1 하락 → 데이터셋 스펙/태그 맵 변경 누락 가능성
- vLLM OOM → `max-model-len`, 배치, GPU 잔여 확인 (`nvidia-smi`)

상위 개념: `docs/wiki/concepts/agent-skills.md`.
