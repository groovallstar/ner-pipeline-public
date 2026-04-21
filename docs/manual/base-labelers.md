# Base Labelers (공통 NER 라벨러 베이스)

`src/labelers/` 루트에 위치한 백엔드별 추상 베이스 3종. ko/ja 라벨러는 이 베이스를
상속받아 언어팩(`entity_types`, 프롬프트 템플릿, `lang` 코드)만 주입한다.

## BaseVllmLabeler

- **파일**: `src/labelers/base_vllm_labeler.py`
- **공개 메서드**:
  - `label(text: str) -> List[dict]` — 문장 분할 후 BIO 태그 레코드 반환
  - `label_spans(text: str) -> List[dict]` — BIO 변환 없이 원시 span 리스트 반환
  - `label_records(records: List[dict]) -> List[dict]` — `text` 필드 레코드 배치 처리

**주요 `__init__` 파라미터**: `base_url`, `model`, `entity_types`, `max_tokens`,
`concurrency`, `thinking`, `lang`, `single_prompt_template`. vLLM 컨테이너의
OpenAI 호환 API를 `AsyncOpenAI`로 호출하며 `asyncio.Semaphore(concurrency)`로
동시성을 제한한다. `response.usage`에서 토큰 사용량을
`total_prompt_tokens`/`total_completion_tokens`에 누적한다.

## BaseOllamaLabeler

- **파일**: `src/labelers/base_ollama_labeler.py`
- **공개 메서드**: `label`, `label_spans`, `label_records` (동일 시그니처).

**주요 `__init__` 파라미터**: `model`, `entity_types`, `base_url`, `num_ctx`,
`batch_size`, `lang`, `single_prompt_template`, `batch_prompt_template`.
`langchain_ollama.ChatOllama`(`format="json"`)로 로컬 Ollama 서버를 호출하며
배치 실패 시 단일 호출로 폴백한다.

## BaseOpenAILabeler

- **파일**: `src/labelers/base_openai_labeler.py`
- **공개 메서드**: `label` (sync), `label_spans` (async concurrent), `label_records`.

**주요 `__init__` 파라미터**: `model`, `entity_types`, `max_tokens_per_batch`,
`base_url`, `concurrency`, `lang`, `system_prompt`, `user_prompt_template`.
`OpenAI` 와 `AsyncOpenAI` 두 클라이언트를 모두 생성하여 sync/async 경로를
병행 유지한다. `OPENAI_API_KEY` 환경변수 기반 인증(`base_url` 지정 시 생략 가능).

## 공통 헬퍼 — `src/labelers/llm_helpers.py`

- `split_sentences(text, lang)` — ko/en은 `.!?`, ja는 `。！？` 포함하여 분할
- `parse_spans(raw)` — `<think>` 제거, bare array / wrapped dict / `[...]` 정규식 폴백
- `spans_to_bio(tokens, spans)` — exact → substring 매칭 2단계 BIO 변환

## 서브클래스

| 백엔드 | ko | ja |
|-------|----|----|
| vLLM | `labelers.ko.vllm_ner_labeler.VllmNERLabeler` | `labelers.ja.vllm_ner_labeler.VllmNERLabeler` |
| Ollama | `labelers.ko.ollama_ner_labeler.OllamaNERLabeler` | `labelers.ja.ollama_ner_labeler.OllamaNERLabeler` |
| OpenAI | `labelers.ko.openai_ner_labeler.OpenAINERLabeler` | `labelers.ja.openai_ner_labeler.OpenAINERLabeler` |

모든 서브클래스는 `__init__`만 오버라이드하며 `super().__init__(..., lang=<lang>,
<prompt kwargs>)`로 언어팩을 주입한다 (AC13 드리프트 가드 참조).

> vi(베트남어) 라벨러(`labelers.vi.{vllm,ollama,openai}_ner_labeler`)는 현재 Base*를
> 상속받지 않는 독립 구현이다. 추후 공통 베이스로 통합 예정이며, 그 전까지는 위
> 표에서 제외한다.
