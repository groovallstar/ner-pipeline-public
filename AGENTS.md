# ner_pipeline

## Purpose
다국어 NER(Named Entity Recognition) 파이프라인. 멀티 백엔드 LLM 지원(OpenAI 호환
API: vLLM, OpenAI) + BERT 토큰 분류 파인튜닝 + PII 증강 + 크롤링 기반 학습 데이터
생성. 한국어(KLUE 6종), 일본어·베트남어(canonical 10종 평면) 지원.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | AI 에이전트 프로젝트 지침, 코딩 컨벤션, 커밋·이슈 관리 |
| `README.md` | 프로젝트 개요, 설치, 사용법 |
| `pyproject.toml` | Python 패키지 정의, 의존성, UV 설정 |
| `uv.lock` | UV lockfile (재현 가능한 의존성 해상도) |
| `.gitignore` | venv·캐시·모델 가중치·IDE 파일 제외 |
| `.claudeignore` | 대용량 바이너리·모델 가중치를 AI 컨텍스트에서 제외 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/ner/` | 소스 코드: labelers, llm_eval, classifier, augmenters, metrics (see `src/ner/CLAUDE.md`) |
| `docker/` | Docker 서비스 설정: dev, vLLM (see `docker/AGENTS.md`) |
| `tests/` | pytest 단위·통합 테스트 (see `tests/ner/AGENTS.md`) |
| `data/` | 데이터셋 파일 (see `data/AGENTS.md`) |
| `docs/` | 참조 문서 |

## For AI Agents

### Working In This Directory
- 패키지 관리자: UV (`uv sync` 또는 `uv pip install -e .`)
- **`PYTHONPATH` 설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 sys.path에
  자동 등록한다
- import 형태: `from ner.labelers.xxx import Xxx`
- 개발은 호스트에서 직접 (Docker는 vLLM 등 외부 서비스 전용)
- Python version: 3.13
- 활성 개발 브랜치: `develop`; `main`은 PR 대상

### Testing Requirements
- Run: `pytest tests/ -v`
- Lint: `ruff check`
- Benchmarks: `python -m ner.llm_eval --lang ko|ja|vi --models "backend:model" --max-samples N`

### Common Patterns
- LLM 라벨러 공통 인터페이스: `label(text)`, `label_spans(text)`, `label_records(records)`
- 한국어: BIO 태그 평가; 일본어·베트남어: character-offset span 평가
- LLM 백엔드: vLLM (multi-GPU tensor parallel), OpenAI API
- HuggingFace BERT NER 라벨러(`hf_ner_labeler.py`)도 벤치마크 베이스라인으로 지원

## Dependencies

### External
- `langchain-openai`, `langchain-huggingface` (의존성; 직접 import는 최소)
- `transformers`, `datasets`, `evaluate`, `seqeval`
- `torch` (CUDA), `accelerate`
- `numpy`, `pandas`, `scikit-learn`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
