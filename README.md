# NER Pipeline

LangChain 기반 다국어 Named Entity Recognition(NER) 파이프라인. 멀티 백엔드 LLM을 활용한 개체명 인식 및 벤치마크 시스템.

## 지원 언어 및 데이터셋

| 언어 | 데이터셋 | 엔티티 타입 | 최고 F1 |
|------|----------|-------------|---------|
| 한국어 | KLUE NER | PS, LC, OG, DT, TI, QT (6종) | 0.656 |
| 일본어 | Stockmark NER Wikipedia | 人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名 (8종) | 0.849 |

## 지원 백엔드

- **vLLM** — 로컬 GPU 추론 (Qwen3.5-27B, Qwen3.5-35B-A3B 등)
- **OpenAI** — API 기반 (gpt-5-mini 등)

## 프로젝트 구조

```
src/
├── labelers/              # NER 라벨링 모듈
│   ├── ko/                # 한국어 라벨러 (vllm, openai)
│   ├── ja/                # 일본어 라벨러 (vllm, openai)
│   ├── dataset_loader.py  # HuggingFace 데이터셋 로딩
│   └── labeler_base.py    # 라벨러 베이스 클래스
└── evaluators/            # 벤치마크 CLI 및 평가 모듈
docker/
├── dev/                   # 개발 컨테이너 (GPU)
└── vllm/                  # vLLM 서비스
results/                   # 벤치마크 결과 JSON + 리포트
tests/                     # pytest 테스트
```

## 설치

```bash
# UV 패키지 매니저 사용
uv pip install -r pyproject.toml

# 또는 Docker 개발 환경
cd docker/dev && docker compose up -d
```

## 사용법

### 벤치마크 실행

```bash
# 일본어 NER 벤치마크
python -m llm_eval --lang ja \
    --models "vllm:Qwen/Qwen3.5-27B" \
    --max-samples 200 \
    --vllm-url "http://localhost:8081/v1"

# 한국어 NER 벤치마크
python -m llm_eval --lang ko \
    --models "vllm:Qwen/Qwen3.5-27B" \
    --max-samples 500 \
    --vllm-url "http://localhost:8081/v1"
```

### 테스트

```bash
pytest tests/ -v
```

## 주요 의존성

- LangChain + OpenAI / HuggingFace
- transformers, datasets, evaluate, seqeval
- NumPy, Pandas, scikit-learn

## License

Private repository.
