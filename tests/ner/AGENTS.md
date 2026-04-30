# tests/ner

## Purpose
pytest 기반 단위·통합 테스트. augmenters(pii, wikiann_vi), labelers(ja, ko, vi),
llm_eval, metrics 등 NER 파이프라인 전반을 커버한다.

## 구조

```
tests/ner/
├── CLAUDE.md
├── AGENTS.md
├── augmenters/
│   ├── pii/           # suffix·llm injector, label merger, loader, verifier
│   └── wikiann_vi/    # 재라벨 파서·offset 매칭·kappa·Wikidata anchor·confidence 병합
├── labelers/
│   └── ja/            # span_matcher (match_spans) 테스트
└── llm_eval/          # MetricsCalculator (span_f1, span_match 등) 테스트
```

## For AI Agents

### Working In This Directory
- Run: `pytest tests/ -v`
- Lint: `ruff check`
- LLM 백엔드(vLLM, OpenAI) 연동 테스트는 서버 실행 중이어야 함
- GPU 의존 테스트는 CI에서 스킵될 수 있음
- 테스트는 `PYTHONPATH` 설정 없이 동작 (uv editable install 기준)

### Testing Requirements
- 단위 테스트: LLM API 호출 mock 처리
- 통합 테스트: 데이터셋 로딩 → 라벨링 → 평가 파이프라인 검증
- 새 테스트 추가 시 기존 패턴 참고 (`mock AsyncOpenAI`, fixture 공유)

## Dependencies

### Internal
- `ner.metrics.bio_metrics` (MetricsCalculator)
- `ner.labelers.ja.span_matcher`
- `ner.augmenters.pii.*`, `ner.augmenters.wikiann_vi.*`

### External
- `pytest`, `ruff`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
