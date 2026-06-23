# tests/ner

## Purpose
pytest 기반 단위·통합 테스트. augmenters(pii, wikiann_vi), labelers(ja, ko, vi),
llm_eval, metrics 등 NER 파이프라인 전반을 커버한다.

## 구조

```
tests/ner/
├── CLAUDE.md
├── AGENTS.md
├── test_base_labelers.py
├── test_bio_dataset.py
├── test_llm_helpers.py
├── test_span_evaluator.py
├── test_span_f1.py
├── test_span_matcher.py
├── test_span_metrics.py
├── augmenters/
│   ├── pii/           # suffix·llm injector, label merger, loader, verifier
│   └── wikiann_vi/    # 재라벨 파서·kappa·Wikidata anchor·confidence 병합
├── classifier/        # confidence_threshold, boundary weights, data_utils,
│   │                  # encode, error_analysis, kfold_pool
├── labelers/
│   ├── ja/            # test_ja_dataset_loader (canonical JSONL 로딩)
│   └── vi/            # test_dataset_loader (VI canonical JSONL 로딩)
└── llm_eval/          # eval_mode CLI dispatch, vi_silver_quality,
                       # wikiann_vi_gold 테스트
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
- `ner.metrics.bio_metrics`, `ner.metrics.span_metrics`
- `ner.labelers.ja.*`, `ner.labelers.vi.*`
- `ner.classifier.*`
- `ner.llm_eval.*`
- `ner.augmenters.pii.*`, `ner.augmenters.wikiann_vi.*`

### External
- `pytest`, `ruff`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
