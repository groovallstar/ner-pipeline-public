# labelers

## Purpose
다국어 NER 라벨링 패키지. HuggingFace 데이터셋 로더와 LLM 기반 라벨러를 제공한다.
vLLM 또는 OpenAI API 백엔드를 호출하여 텍스트에서 named entity를 추출하고,
LLM JSON span 출력을 BIO 태그 시퀀스로 변환한다.
한국어(KO), 일본어(JA), 베트남어(VI), 영어(EN) 서브패키지 포함.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 공개 API: DatasetLoader, DatasetNotFoundError, NERRecord 익스포트 |
| `labeler_base.py` | 공용 `parse_json_response()` — think-tag·markdown fence·wrapped dict 처리 |
| `base_vllm_labeler.py` | vLLM 백엔드 공용 베이스 (언어팩 주입 서브클래싱) |
| `base_openai_labeler.py` | OpenAI 호환 배치 라벨러 공용 베이스 |
| `dataset_loader.py` | `HFTokenDatasetLoader` (alias: `DatasetLoader`) — HuggingFace NER datasets (KLUE 등), ClassLabel 변환 |
| `tag_aligner.py` | `TagAligner`: BIO 정렬·span 추출(`extract_spans_from_bio`), 태그 정규화(`normalize_tag`, PER→PS 등) |
| `hf_ner_labeler.py` | `HFNERLabeler` — HuggingFace pipeline 기반 BERT NER 베이스라인 |
| `llm_helpers.py` | 공용 LLM 유틸 |
| `run_labeling.py` | 일괄 라벨링 실행 스크립트 |
| `bio_dataset.py` | `TokenUnit`·`DatasetSpec`·`REGISTRY`·`extract_spans`·`load` 등 BIO 데이터셋 로딩/스팬 추출 (`llm_eval/span_evaluator.py` 소비) |
| `span_matcher.py` | `match_spans()` 공용 스팬 매칭 (ja·vi·ko·llm_eval 공유) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `ko/` | 한국어 NER 라벨러 — canonical NER 5종 (PER/LOC/ORG/PROD/EVT) + DAT, KLUE 유래·TI/QT 드롭 (PROD/EVT는 LLM 재라벨 증분; PII는 augmenters에서 합성 주입). **LOC/ORG 는 narrow-ORG 재정의를 따라 JA·VI 와 외연이 다르다** — ORG=정부·행정·공공·정치 기관만이고 인공 시설·민간조직은 비-entity (see `ko/CLAUDE.md`) |
| `ja/` | 일본어 NER 라벨러 — canonical 10종 평면; pre-dumped Stockmark JSONL 전용 (see `ja/CLAUDE.md`) |
| `vi/` | 베트남어 NER 라벨러 — canonical 10종 평면; canonical WikiANN-vi JSONL 덤프 전용 |
| `en/` | 영어 NER 라벨러 — canonical 10종 평면, **vllm 백엔드만**. LOC/ORG 는 JA·VI 관례(인공 시설=ORG)를 따른다. 용도가 하나다: `augmenters/pii` 의 주입 결과를 LLM 이 독립적으로 다시 뽑아 대조하는 교차 검증. 원천 OntoNotes5 가 사람 gold 라 재라벨 대상이 없어 `openai_ner_labeler`·`dataset_loader` 는 두지 않았다 |

라벨 스키마 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## For AI Agents

### Working In This Directory
- **LLM 라벨러** 공통 인터페이스: `label(text)`, `label_spans(text)`, `label_records(records)`.
  `hf_ner_labeler.py`의 `HFNERLabeler`는 이 계약 밖이다 — 로컬 BERT 베이스라인이라
  `label()` 외에 `label_sentence()`·`label_syllables()`를 갖고 `label_spans()`·`label_records()`가 없다
- `parse_json_response()`는 `labeler_base.py`의 canonical JSON 추출 유틸 — 새 JSON 파싱 코드 작성 금지
- `split_sentences()`와 `spans_to_bio()`는 `llm_helpers.py`에 단일 정의되며 베이스 클래스를 통해 호출됨
- LLM 라벨러는 `total_prompt_tokens`, `total_completion_tokens`로 토큰 사용량 추적 —
  `HFNERLabeler`는 로컬 추론이라 이 속성이 없다
- `DatasetLoader`는 JSONL fallback 경로(`/data/ner/`)를 지원

### Testing Requirements
- LLM API 호출은 mock 처리
- `parse_json_response()`를 edge case(think-tag, nested dict, markdown-fenced JSON)로 테스트
- 통합 테스트는 실행 중인 LLM 서버 필요

### Common Patterns
- 프롬프트 엔지니어링이 주요 품질 레버 — 언어별로 고도 튜닝됨
- 클라이언트는 백엔드마다 다르다 — vLLM 베이스는 `AsyncOpenAI` 전용(전 문장 동시 발사),
  OpenAI 베이스는 동기 `OpenAI` 와 `AsyncOpenAI` 를 둘 다 만들어 `label()` 은 동기 클라이언트로 간다
- 모델 예시: vLLM 호환 OpenAI API 모델, OpenAI gpt 시리즈

## Dependencies

### Internal
- leaf 패키지 (라벨링 전용); `tag_aligner.py`는 `llm_eval/`이 벤치마크 평가에 소비
- `bio_dataset.py`·`span_matcher.py`도 `llm_eval/`이 소비 (`span_evaluator.py` 등)

### External
- `openai`, `datasets`, `transformers`, `torch`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
