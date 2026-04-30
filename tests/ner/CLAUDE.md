# tests/ — 테스트 가이드

## 실행

```bash
pytest tests/ -v
```

## 테스트 프레임워크

- **pytest** — 테스트 실행
- **ruff check** — 린트

## 테스트 전략

- 단위 테스트: 개별 labeler, span_matcher, metrics 함수
- 통합 테스트: 데이터셋 로딩 → 라벨링 → 평가 파이프라인
- 벤치마크: `python -m ner.llm_eval` CLI로 실행 (tests/ 외부)

## 주의사항

- LLM 백엔드(Ollama, vLLM, OpenAI) 연동 테스트는 서버가 실행 중이어야 함
- GPU 의존 테스트는 CI에서 스킵될 수 있음
