# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트를 이해하기 위한 가이드이다.

## 프로젝트 개요

**다국어 NER(Named Entity Recognition) 파이프라인** — LangChain 기반 멀티 백엔드 LLM 지원.

- 한국어: KLUE NER (6 엔티티 타입)
- 일본어: Stockmark NER Wikipedia (8 엔티티 타입)

## 개발 환경

- Docker 컨테이너 내부에서 개발 (GPU 지원)
- 패키지 관리자: **UV** (`uv pip install`)
- `PYTHONPATH=/work/git/ner_pipeline/src/`
- import 형태: `from labelers.xxx import Xxx`

## 핵심 디렉토리 구조

```
src/
├── labelers/          # NER 라벨링 모듈
│   ├── ko/            # 한국어 (ollama, vllm, openai)
│   ├── ja/            # 일본어 (ollama, vllm, openai, enhanced_labeler)
│   ├── dataset_loader.py
│   └── labeler_base.py
└── evaluators/        # 벤치마크 및 평가
docker/
├── dev/               # 개발 컨테이너 (상세: docker/CLAUDE.md)
├── ollama/            # Ollama 서비스
└── vllm/              # vLLM 서비스
results/               # 벤치마크 결과 JSON + 리포트
tests/                 # 테스트 (상세: tests/CLAUDE.md)
docs/                  # 참고 문서
```

## 주요 의존성

| 레이어 | 라이브러리 |
|--------|-----------|
| LLM 백엔드 | LangChain + OpenAI, Ollama, HuggingFace |
| NLP/ML | transformers, datasets, evaluate, seqeval, bert-score |
| 데이터 | NumPy, Pandas, scikit-learn |

## 벤치마크 실행

```bash
# 일본어
PYTHONPATH=src python -m evaluators --lang ja --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200

# 한국어
PYTHONPATH=src python -m evaluators --lang ko --models "vllm:Qwen/Qwen3.5-27B" --max-samples 500
```

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
