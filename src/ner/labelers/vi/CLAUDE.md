# vi

## Purpose
베트남어 NER 라벨러. WikiANN-vi 재라벨 덤프 기반, canonical 10종 평면
(NER 5종 + PII 5종) 출력. vLLM 백엔드 + WikiANN-vi 전용
데이터셋 로더 포함.
라벨 스키마 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports prompts, VllmNERLabeler |
| `ner_prompts.py` | Vietnamese NER prompts — canonical 10종 평면 엔티티 타입 정의 및 disambiguation 규칙. 단일 문장용 `SINGLE_PROMPT_TEMPLATE` 한 벌 |
| `dataset_loader.py` | `VietnameseDatasetLoader` — WikiANN-vi 재라벨 덤프(`data/wikiann_vi/{train,valid,test}.jsonl`)를 읽음. `bio_to_offset_spans`·`offset_spans_to_bio` 유틸 포함 |
| `vllm_ner_labeler.py` | Vietnamese vLLM labeler — `BaseVllmLabeler` 경량 서브클래스, `lang="vi"` 주입 |

## For AI Agents

### Working In This Directory
- Vietnamese evaluation uses character-offset spans (not BIO tags). span
  matching은 공용 `ner.labelers.span_matcher.match_spans()`를 사용하며,
  vi 전용 구현은 없다 — base 라벨러 계층이 위임함
- `VietnameseDatasetLoader`는 두 종류의 JSONL을 읽는다:
  1. **기본 split 파일** (`data/wikiann_vi/{train,valid,test}.jsonl`):
     PII 주입까지 완료된 최종 덤프(`entities` 스키마). `load(split=...)`
     로 읽으며, `load_local`을 통해 `gold_spans`로 정규화한다
  2. **recall-merge 중간 덤프** (임의 경로): `gold_spans_relabel_merged`
     필드를 가진 augmenters 산출물. `load(path=..., span_key=...)`로 읽음.
     WikiANN 원본 3종은 `gold_spans`
- `DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG", "PROD", "EVT", "EMAIL",
  "PHONE", "DAT", "ID_NUM", "CREDIT_CARD"]` — canonical 10종 평면 목록.
  상세: `docs/manual/data/canonical-entity-schema.md`
- WikiANN-vi 원본은 3종(PER/LOC/ORG)이며 시설을 LOC로 표기. canonical
  스키마는 시설을 ORG로 분류하므로 원본 gold 평가 시 LOC↔ORG 미스매치가
  발생하는 것은 설계 의도다 — 5종 gold(`data/wikiann_vi/`)를 기준으로
  평가해야 정상 비교가 된다
- 프롬프트는 베트남어 원문으로 작성되어 있다 — diacritics 보존·다중 단어
  엔티티 단일화·직함 제외 규칙이 모두 베트남어 예시로 표현됨

### Testing Requirements
- Dataset loader는 로컬 JSONL 덤프 필수; 파일 없으면 `FileNotFoundError`
  즉시 발생 — HuggingFace 또는 네트워크 접근 없음
- `bio_to_offset_spans` / `offset_spans_to_bio` 왕복 변환 테스트
- Mock `AsyncOpenAI.chat.completions.create()` (vLLM 백엔드)

### Common Patterns
- 두 라벨러 모두 `ner_prompts.py`에서 프롬프트를 import
- vLLM은 전 문장을 동시 전송(`concurrency=32`); OpenAI는
  `max_tokens_per_batch=1000` 배치 처리

## LLM 평가 경로

- `python -m ner.llm_eval --lang vi` → offset_span 모드(공용 러너)
- `ner.llm_eval.vi_silver_quality` — silver vs gold 비교
- `ner.llm_eval.wikiann_vi_gold` — WikiANN-vi gold 기준 평가

## Dependencies

### Internal
- `ner.labelers.vi.ner_prompts`, `ner.labelers.vi.dataset_loader`
- `ner.labelers.base_vllm_labeler`
- `ner.labelers.span_matcher` (공용 — base 라벨러 계층 경유)

### External
- `openai`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
