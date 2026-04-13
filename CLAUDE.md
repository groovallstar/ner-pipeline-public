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
│   ├── labeler_base.py
│   ├── tag_aligner.py    # BIO 태그 정렬/정규화/span 추출
│   └── hf_ner_labeler.py # HuggingFace BERT NER 라벨러
└── evaluators/        # 벤치마크 및 평가 (metrics, report, runner)
docker/
├── dev/               # 개발 컨테이너 (상세: docker/CLAUDE.md)
├── ollama/            # Ollama 서비스
└── vllm/              # vLLM 서비스
results/               # 벤치마크 결과 JSON + 리포트
tests/                 # 테스트 (상세: tests/CLAUDE.md)
docs/                  # 문서 (상세: docs/schema.md)
│   ├── wiki/          # 프로젝트 독립적 도메인 지식
│   ├── api/           # 코드 종속 API 레퍼런스
│   └── guides/        # 프로젝트 종속 실행 가이드
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
PYTHONPATH=src python -m llm_eval --lang ja --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200

# 한국어
PYTHONPATH=src python -m llm_eval --lang ko --models "vllm:Qwen/Qwen3.5-27B" --max-samples 500
```

## Development Protocol (The 3 Rules)

1. **Atomic Functionality**: Every request must be handled in the smallest possible functional unit.
2. **Verification & Documentation**: A function is not "done" until:
   - A corresponding test case passes.
   - Its logic is documented in `/docs/`.
3. **Human-Led Decomposition**: The User defines the functions. Claude must ask for confirmation if a function seems too large to be verified in a single step.

## 코딩 컨벤션 (주석/문서 언어)

- `print()` / `logger.*` / 예외 메시지 / argparse `help`: **영문**
- 함수·클래스·모듈 docstring, 인라인 `#` 주석: **한국어**
- 상세 규칙 및 예시: `docs/guides/coding-conventions.md`

## Wiki 운영

`docs/wiki/`에 프로젝트 독립적 도메인 지식을 관리한다. 운영 규칙은 `docs/schema.md` 참조.

**트리거 키워드**: "위키에 추가", "위키 업데이트", "위키에 넣어", "wiki에 추가"

위키 관련 요청 시 다음 절차를 따른다:
1. 기존 위키 페이지에 통합 가능한지 먼저 확인 (`docs/wiki/` 내 파일 탐색)
2. 통합 가능하면 기존 페이지 업데이트, 아니면 새 페이지 생성
3. 관련 페이지 간 교차 참조 추가
4. `docs/schema.md`의 페이지 유형 기준에 따라 배치 (concepts / entities / sources)

**코드 변경 후**: `docs/api/` 문서가 최신인지 확인한다.

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
