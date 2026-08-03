# tests/ner/ — 테스트 가이드

pytest 기반 단위·통합 테스트. augmenters(pii, wikiann_vi), labelers(ja, ko, vi),
llm_eval, metrics, validity 등 NER 파이프라인 전반을 커버한다.

## 실행

```bash
pytest tests/ -v        # 전체
ruff check              # 린트
```

## 구조

```
tests/ner/
├── CLAUDE.md
├── test_base_labelers.py
├── test_bio_dataset.py
├── test_llm_helpers.py
├── test_span_evaluator.py
├── test_span_f1.py
├── test_span_matcher.py
├── test_span_metrics.py
├── test_validity.py   # ner.validity — σ_fold·σ_repro, comparability,
│                      # 누출 근거, compare verdict 우선순위
├── augmenters/
│   ├── pii/           # suffix·llm injector, label merger, loader, verifier,
│   │                  # ko·vi 생성기·주입, __main__ verify 라벨러 lang 분기
│   └── wikiann_vi/    # 재라벨 파서·kappa·Wikidata anchor·confidence 병합,
│                      # silver-갭 additive 삽입
├── classifier/        # confidence_threshold, boundary weights, data_utils,
│   │                  # encode, error_analysis, kfold_pool
├── golden/
│   └── validity/      # verdict 별 byte-고정 스냅샷 11종 (pass·fail·invalid·
│                      # inconclusive) — compare() 출력 전체를 대조
├── labelers/
│   ├── test_ko_evt_r2_audit.py  # KO EVT canonical 규칙 감사·R2 회수 —
│   │                  # 형태소 경계, 위반 탐지(축2·축3 트림·R1)와 겹침 가드,
│   │                  # 회수 삽입·불변 타입 byte-identical, 보고 전용 bare 비대칭
│   ├── test_ko_evt_axis1_audit.py  # KO EVT 축1-복합 head 열거·자리 스캔 —
│   │                  # 열거의 결정성(span 중간에 묻힌 head 포착·빈도 문턱·
│   │                  # 고유명 이력으로 수식부 제외·재현 파라미터),
│   │                  # 고유명 가드(1글자 성씨·저비율 제외·조사 붙은 선행어절
│   │                  # 제외), gold EVT 겹침을 버리지 않고 분류
│   ├── ja/            # test_ja_dataset_loader (canonical JSONL 로딩)
│   └── vi/            # test_dataset_loader (VI canonical JSONL 로딩)
└── llm_eval/          # eval_mode CLI dispatch, vi_silver_quality,
                       # wikiann_vi_gold 테스트
```

`labelers/` 에 `ko/` 디렉토리는 없다 — 한국어 라벨러 테스트는 최상위
`test_base_labelers.py`·`test_bio_dataset.py` 와 `labelers/test_ko_evt_r2_audit.py`
에 흩어져 있다.

`tests/server/`는 별도 패키지(server 계약·통합·live 테스트)로, 본 문서는
`tests/ner/`만 다룬다. 상세 전략: `src/server/CLAUDE.md`.

## 테스트 전략

- 단위 테스트: 개별 labeler·span_matcher·metrics 함수 (LLM API 호출은 mock)
- 통합 테스트: 데이터셋 로딩 → 라벨링 → 평가 파이프라인 검증
- 벤치마크: `python -m ner.llm_eval` CLI로 실행 (tests/ 외부)
- 새 테스트 추가 시 기존 패턴 참고 (`mock AsyncOpenAI`, fixture 공유)

## 주의사항

- LLM 백엔드(vLLM, OpenAI) 연동 테스트는 서버가 실행 중이어야 함
- 조건부 skip 은 GPU 유무가 아니라 **로컬 자원 부재**로 걸린다 — HF 캐시가 없으면
  `pytest.skip`(`test_bio_dataset.py`), 선택적 패키지(`pyvi`·`fugashi` 등)가 없으면
  `importorskip`(`classifier/test_encode.py`). CI 설정은 리포에 없어(호스트 실행 전제)
  "CI 에서 스킵" 이라는 경로 자체가 없다
- 테스트는 `PYTHONPATH` 설정 없이 동작 (uv editable install 기준)

## 의존성

- 내부: `ner.metrics.{bio_metrics,span_metrics}`, `ner.validity.*`,
  `ner.labelers.{ja,vi}.*`, `ner.classifier.*`, `ner.llm_eval.*`,
  `ner.augmenters.{pii,wikiann_vi}.*`
- 외부: `pytest`, `ruff`
