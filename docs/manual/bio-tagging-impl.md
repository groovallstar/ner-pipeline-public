# BIO 태깅·span 추출의 구현 맵

> 출처: `docs/wiki/concepts/bio-tagging.md`에서 추출한 코드 경로 및 프로젝트별 구현 참조.
> 프로젝트 독립적인 알고리즘 설명은 원문 참조.

## 관련 코드 파일

| 역할 | 파일 |
|------|------|
| BIO ↔ Span 변환, 태그 정규화 | `src/ner/labelers/tag_aligner.py` |
| LLM 프롬프트 (span JSON 형식 지정) | `src/ner/labelers/ko/ner_prompts.py` |
| HuggingFace 데이터셋 로딩 | `src/ner/labelers/dataset_loader.py` |

---

## span 형식을 고른 이유

`src/ner/labelers/ko/ner_prompts.py`의 `SINGLE_PROMPT_TEMPLATE`은 세 지점에서 형식을 고정한다 — 지시문 첫 줄이 JSON 배열만 반환하라고 못 박고, `## 핵심 규칙` 이 "JSON 배열만 출력, 다른 설명 금지"로 다시 닫으며, `## 예시` 의 입력/출력 페어가 모두 `[{"text": ..., "type": ...}]` 형태로 제시된다.

> 줄 번호는 적지 않는다. 프롬프트는 규칙이 늘 때마다 움직이는 파일이라 여기 적은 번호가 곧 어긋나고, 실제로 세 번호가 모두 밀린 채 남아 있었다. 절 제목(`## 핵심 규칙`·`## 예시`)으로 찾는다.

---

## 토큰 단위를 가리지 않는 span 복원

`src/ner/labelers/tag_aligner.py` 의 `extract_spans_from_bio`:

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

## gold 토큰과 무관한 offset 매핑

`src/ner/labelers/tag_aligner.py` 의 `spans_to_syllable_bio`:
- LLM span을 원문 character offset으로 매핑하므로 gold 토큰 단위와 무관하게 동작한다.
