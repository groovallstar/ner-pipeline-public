<!-- Parent: ../AGENTS.md -->

# labelers

## Purpose
다국어 NER 라벨링 패키지. HuggingFace 데이터셋 로더와 LLM 기반 라벨러를 제공한다.
vLLM 또는 OpenAI API 백엔드를 호출하여 텍스트에서 named entity를 추출하고,
LLM JSON span 출력을 BIO 태그 시퀀스로 변환한다.
한국어(KO), 일본어(JA), 베트남어(VI) 서브패키지 포함.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 공개 API: DatasetLoader, NERRecord 익스포트 |
| `labeler_base.py` | 공용 `parse_json_response()` — think-tag·markdown fence·wrapped dict 처리 |
| `dataset_loader.py` | `DatasetLoader` for HuggingFace NER datasets (KLUE 등), ClassLabel 변환 |
| `tag_aligner.py` | `TagAligner`: BIO 정렬, KLUE 음절 토크나이저, 태그 정규화(PER→PS 등), `extract_spans_from_bio()` |
| `hf_ner_labeler.py` | `HFNERLabeler` — HuggingFace pipeline 기반 BERT NER 베이스라인 |
| `llm_helpers.py` | 공용 LLM 유틸 |
| `run_labeling.py` | 일괄 라벨링 실행 스크립트 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `ko/` | 한국어 NER 라벨러 — KLUE 6종 (PS, LC, OG, DT, TI, QT) (see `ko/AGENTS.md`) |
| `ja/` | 일본어 NER 라벨러 — canonical 10종 평면; pre-dumped Stockmark JSONL 전용 (see `ja/AGENTS.md`) |
| `vi/` | 베트남어 NER 라벨러 — canonical 10종 평면; canonical WikiANN-vi JSONL 덤프 전용 (see `vi/AGENTS.md`) |

라벨 스키마 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## For AI Agents

### Working In This Directory
- 모든 라벨러 공통 인터페이스: `label(text)`, `label_spans(text)`, `label_records(records)`
- `parse_json_response()`는 `labeler_base.py`의 canonical JSON 추출 유틸 — 새 JSON 파싱 코드 작성 금지
- `_split_sentences()`와 `_spans_to_bio()`는 각 라벨러 파일에 복제됨 — 수정 시 모든 사본 갱신
- 각 라벨러는 `total_prompt_tokens`, `total_completion_tokens`로 토큰 사용량 추적
- `DatasetLoader`는 JSONL fallback 경로(`/data/ner/`)를 지원

### Testing Requirements
- LLM API 호출은 mock 처리
- `parse_json_response()`를 edge case(think-tag, nested dict, markdown-fenced JSON)로 테스트
- 통합 테스트는 실행 중인 LLM 서버 필요

### Common Patterns
- 프롬프트 엔지니어링이 주요 품질 레버 — 언어별로 고도 튜닝됨
- 라벨러는 `openai.AsyncOpenAI` 사용 (vllm/openai 공용)
- 모델 예시: vLLM 호환 OpenAI API 모델, OpenAI gpt 시리즈

## Dependencies

### Internal
- leaf 패키지 (라벨링 전용); `tag_aligner.py`는 `llm_eval/`이 벤치마크 평가에 소비

### External
- `openai`, `datasets`, `transformers`, `torch`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
