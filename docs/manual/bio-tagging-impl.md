# Implementation Map — BIO Tagging & Span Extraction

> 출처: `docs/wiki/concepts/bio-tagging.md`에서 추출한 코드 경로 및 프로젝트별 구현 참조.
> 프로젝트 독립적인 알고리즘 설명은 원문 참조.

## 관련 코드 파일

| 역할 | 파일 |
|------|------|
| BIO ↔ Span 변환, 태그 정규화 | `src/labelers/tag_aligner.py` |
| LLM 프롬프트 (span JSON 형식 지정) | `src/labelers/ko/ner_prompts.py` |
| HuggingFace 데이터셋 로딩 | `src/labelers/dataset_loader.py` |

---

## 왜 span 형식인가 — 프롬프트 설계 선택 (코드 근거)

`src/labelers/ko/ner_prompts.py`의 `SINGLE_PROMPT_TEMPLATE`은 다음 세 지점에서 형식을 고정한다:

- `src/labelers/ko/ner_prompts.py:15` — `"텍스트에서 개체명을 찾아 JSON 배열로만 반환하세요."`
- `src/labelers/ko/ner_prompts.py:64` — `"JSON 배열만 출력, 다른 설명 금지"`
- `src/labelers/ko/ner_prompts.py:68-84` — few-shot 예시 전부 `[{"text": ..., "type": ...}]` 형식으로 제시

---

## `extract_spans_from_bio` — 코드 위치 및 동작 증거

`src/labelers/tag_aligner.py:94-130` 의 `extract_spans_from_bio`:

```python
def extract_spans_from_bio(tokens, tags, lang="ko"):
    # Detect token level: if any token is pure whitespace, it's syllable-level
    has_space_tokens = any(t.strip() == "" for t in tokens)
    joiner = "" if has_space_tokens else " "
    ...
```

- KLUE(음절 + 공백 토큰) → `joiner=""` → `['대','한','민','국']` → `"대한민국"`
- WikiANN(어절) → `joiner=" "` → `['대한민국']` → `"대한민국"`

동일 함수가 두 형식 모두 span으로 정확히 역변환한다.

## `spans_to_syllable_bio` — 코드 위치

`src/labelers/tag_aligner.py:221-295` 의 `spans_to_syllable_bio`:
- LLM span을 원문 character offset으로 매핑하므로 gold 토큰 단위와 무관하게 동작한다.
