# NER Pipeline

다국어 Named Entity Recognition(NER) 파이프라인. 멀티 백엔드 LLM 라벨링 + BERT 토큰 분류 + PII 증강.

## 지원 언어

| 언어 | 데이터셋 | 라벨 공간 |
|------|---------|----------|
| 한국어 | KLUE NER | 6종 (PS, LC, OG, DT, TI, QT) |
| 일본어 | Stockmark NER Wikipedia | canonical 5종 NER + 5종 PII = 10종 평면 |
| 베트남어 | WikiANN-vi (silver 재라벨) | canonical 10종 평면 (JA 와 공통) |

라벨 정의·매핑·경계 규칙: [`docs/manual/data/canonical-entity-schema.md`](docs/manual/data/canonical-entity-schema.md)

## 백엔드

vLLM (로컬 GPU), OpenAI, HuggingFace BERT baseline.

## 프로젝트 구조

```
src/ner/
├── labelers/{ko,ja,vi}/   # 언어별 LLM 라벨러
├── llm_eval/              # 벤치마크 오케스트레이션·리포트
├── augmenters/{pii,wikiann_vi,crawlers/ko}/  # 학습 데이터 증강
├── classifier/            # BERT 토큰 분류 파인튜닝
├── metrics/               # span/BIO 메트릭 공용 구현
└── scripts/               # 보조 셸 스크립트
docker/{dev,vllm}/         # 개발 컨테이너 + vLLM 서비스
results/                   # 벤치마크 산출 (gitignored)
tests/ner/                 # pytest 테스트
docs/                      # manual·reports·issues·wiki·specs
```

상세 가이드: [`CLAUDE.md`](CLAUDE.md), 모듈 레퍼런스: [`docs/manual/`](docs/manual/)

## 설치

```bash
uv sync                # 또는: uv pip install -e .
```

## 사용법

```bash
# LLM NER 벤치마크
python -m ner.llm_eval --lang {ko,ja,vi} \
    --models "vllm:<model>" --max-samples 200 \
    --vllm-url "http://localhost:8081/v1"

# BERT 파인튜닝·평가
python -m ner.classifier              # 일본어 NER 5종
python -m ner.classifier.pii_benchmark  # PII 7종

# 데이터 증강
python -m ner.augmenters.pii          # 합성 PII 주입
python -m ner.augmenters.wikiann_vi   # WikiANN-vi 재라벨
```

각 CLI 의 전체 옵션은 `--help` 참조.

## 테스트

```bash
pytest tests/ -v
```

## 벤치마크 리포트

[`docs/reports/`](docs/reports/) — 언어·실험별 최신 측정치.

## License

Private repository.
