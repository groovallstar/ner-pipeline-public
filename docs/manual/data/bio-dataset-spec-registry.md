# BIO Dataset Spec Registry

BIO 토큰 시퀀스 NER 데이터셋을 스펙 기반으로 로드하고, BIO 변종을 정규화한 뒤 엔티티 span까지 추출하는 모듈.

- 소스: `src/ner/labelers/bio_dataset.py`
- 테스트: `tests/ner/test_bio_dataset.py`
- 설계 문서: `openspec/changes/add-bio-dataset-spec-registry/`

## 배경

한국어 NER 데이터셋은 토큰화 단위(음절/어절/형태소)와 BIO 변종(표준 IOB2 vs KMOU의 `I` 단독 태그)이 다르다. 이 메타데이터를 `DatasetSpec`으로 1급 개념화하여 "로드 → BIO 정규화 → span 추출"을 한 번에 수행할 수 있게 했다. 토큰 단위별 span 추출 과정의 이론적 배경은 `docs/wiki/concepts/bio-tagging.md` §2~§4를 참조.

## 공개 API

### 타입

| 이름 | 종류 | 설명 |
|------|------|------|
| `TokenUnit` | `str` Enum | `SYLLABLE` / `WORD` / `MORPHEME` |
| `DatasetSpec` | frozen dataclass | `name`, `lang`, `unit`, `bio_variant`, `joiner`, `config` |
| `REGISTRY` | `dict[str, DatasetSpec]` | 등록된 데이터셋 스펙 |
| `DatasetNotFoundError` | Exception | 미등록 스펙 또는 HF 로드 실패 시 발생 |

### 함수

#### `normalize_bio_variant(tags, variant) -> list[str]`

데이터셋 고유 BIO 태그를 표준 IOB2로 변환한다.

| variant | 동작 |
|---------|------|
| `"iob2"` | identity (복사 반환) |
| `"kmou_i_alone"` | `B_TYPE` → `B-TYPE`, 단독 `I` → `I-{직전 B의 타입}`, 직전 B 없으면 `O` |

```python
normalize_bio_variant(["B_PS", "I", "O"], "kmou_i_alone")
# → ["B-PS", "I-PS", "O"]
```

#### `extract_spans(tokens, bio_tags, joiner) -> list[dict]`

IOB2 태그열에서 엔티티 span을 추출한다.

- `joiner=""` → 음절 토큰 붙이기 (KLUE)
- `joiner=" "` → 어절/형태소 토큰 공백 조인 (KMOU)
- 공백 토큰(`" "`)은 span 텍스트에서 제외

```python
extract_spans(["경", "찰", "은"], ["B-OG", "I-OG", "O"], joiner="")
# → [{"text": "경찰", "type": "OG"}]
```

#### `load(spec_name, split, max_samples=None) -> list[dict]`

REGISTRY 조회 → HF/JSONL 로드 → BIO 정규화 → span 추출을 한 번에 수행.

```python
from ner.labelers.bio_dataset import load

records = load("klue", "validation", max_samples=5)
# records[0] = {
#     "id": "0",
#     "tokens": ["경", "찰", "은", " ", ...],
#     "bio_tags": ["B-OG", "I-OG", "O", "O", ...],
#     "spans": [{"text": "경찰", "type": "OG"}, ...],
#     "sentence": "경찰은 ...",
# }
```

## 등록된 데이터셋 (REGISTRY)

| key | HF name | lang | 토큰 단위 | BIO 변종 | joiner | 엔티티 타입 |
|-----|---------|------|-----------|----------|--------|-------------|
| `"klue"` | `klue` (config: `ner`) | ko | 음절 (`SYLLABLE`) | `iob2` | `""` | PS, LC, OG, DT, TI, QT (6종) |
| `"nlp-kmu/kor_ner"` | `nlp-kmu/kor_ner` | ko | 형태소 (`MORPHEME`) | `kmou_i_alone` | `" "` | PS, LC, OG, DT, TI (5종, QT 없음) |

### KLUE
- 로드 경로: JSONL fallback (`/data/ner/klue/{split}.jsonl`) 우선, 없으면 HF `datasets`
- 음절 단위 토큰화, 공백 토큰 포함 (예: `["경", "찰", "은", " ", "또"]`)
- 표준 IOB2 태그 (`B-OG`, `I-OG`, `O`)

### nlp-kmu/kor_ner (KMOU NER)
- 로드 경로: HF `datasets` arrow 캐시
- 형태소 단위 토큰화 (예: `["한석규", "가", "운영", "하", "는"]`)
- KMOU 변종 BIO: `B_TYPE` + 단독 `I` (ClassLabel: `["I", "O", "B_OG", "B_TI", "B_LC", "B_DT", "B_PS"]`)
- `normalize_bio_variant`가 `B_PS` → `B-PS`, `I` → `I-PS` 등으로 자동 정규화

## 데이터 흐름

```
REGISTRY[spec_name]
    ↓
HF load_dataset / JSONL fallback
    ↓
ClassLabel int → str 디코딩
    ↓
normalize_bio_variant (KMOU: B_TYPE→B-TYPE, I→I-TYPE)
    ↓
extract_spans (IOB2 → [{text, type}])
    ↓
{"id", "tokens", "bio_tags", "spans", "sentence"}
```

## 확장

새 BIO 데이터셋을 추가하려면 `REGISTRY`에 `DatasetSpec` 한 줄을 추가한다:

```python
REGISTRY["tner/wikiann"] = DatasetSpec(
    name="tner/wikiann", lang="ko", unit=TokenUnit.WORD,
    bio_variant="iob2", joiner=" ", config="ko",
)
```

Stockmark NER(일본어)처럼 raw text + char offset span 포맷인 데이터셋은 이 registry의 대상이 아니다 — BIO 토큰 시퀀스 포맷만 지원한다.

## 테스트 케이스

실행: `pytest tests/ner/test_bio_dataset.py -v`

### 단위 테스트 (21개, 캐시 독립)

| 클래스 | 테스트 | 검증 내용 |
|--------|--------|-----------|
| `TestTokenUnit` | `test_members` | enum 멤버가 SYLLABLE/WORD/MORPHEME 3종뿐인지 |
| | `test_values_are_lowercase` | 각 멤버의 `.value`가 소문자 문자열인지 |
| | `test_str_contains_value` | str 혼합 상속으로 `== "syllable"` 비교 가능한지 |
| `TestDatasetSpec` | `test_frozen` | 필드 변경 시 `FrozenInstanceError` 발생 |
| | `test_hashable` | dict 키로 사용 가능 |
| | `test_config_defaults_to_none` | `config` 미지정 시 `None` |
| `TestRegistry` | `test_klue_entry` | KLUE 스펙 필드 값 (lang, unit, bio_variant, joiner, config) |
| | `test_kmou_entry` | KMOU 스펙 필드 값 |
| | `test_all_korean` | 모든 엔트리가 `lang == "ko"` |
| `TestNormalizeBioVariant` | `test_iob2_identity` | `iob2`는 identity + 복사본 반환(원본 불변) |
| | `test_kmou_single_entity` | `["B_PS", "I", "O"]` → `["B-PS", "I-PS", "O"]` |
| | `test_kmou_multiple_entities` | 두 엔티티 사이 O가 있을 때 타입이 정확히 갱신되는지 |
| | `test_kmou_dangling_i` | B 없이 등장하는 단독 `I`가 `O`로 fallback |
| | `test_unknown_variant_raises` | 미지원 variant에 `ValueError` 발생 |
| `TestExtractSpans` | `test_syllable_with_space_token` | 음절 토큰 + 공백 토큰에서 span 추출 (joiner="") |
| | `test_word_level` | 어절 토큰에서 span 추출 (joiner=" ") |
| | `test_type_switch_without_o` | O 없이 `B-PS` → `B-LC` 전환 시 두 span 분리 |
| | `test_whitespace_tokens_excluded` | 공백 토큰이 span 텍스트에 포함되지 않는지 |
| | `test_trailing_entity` | 시퀀스 끝에 위치한 엔티티가 누락 없이 추출되는지 |
| | `test_empty_input` | 빈 입력에 빈 리스트 반환 |
| `TestLoadErrors` | `test_unknown_spec_raises` | 미등록 이름에 `DatasetNotFoundError` 발생 |

### 스모크 테스트 (6개, HF 캐시 필요)

`@pytest.mark.integration` 마크. 캐시 디렉토리가 없으면 자동 skip.

| 클래스 | 테스트 | 검증 내용 |
|--------|--------|-----------|
| `TestSmokeKLUE` | `test_load_returns_valid_records` | `load("klue", ...)` 반환 레코드가 `{id, tokens, bio_tags, spans, sentence}` 구조이고 bio_tags가 IOB2 형식 |
| | `test_at_least_one_span` | 5개 레코드 중 최소 1개에 비어있지 않은 spans |
| | `test_entity_types_in_range` | span의 type이 KLUE 6종(PS/LC/OG/DT/TI/QT) 범위 내 |
| `TestSmokeKMOU` | `test_load_returns_iob2_normalized` | KMOU 원본의 `B_TYPE`/`I` 태그가 정규화되어 bare `"I"`나 underscore 태그가 없는지 |
| | `test_at_least_one_span` | 5개 레코드 중 최소 1개에 비어있지 않은 spans |
| | `test_entity_types_in_range` | span의 type이 KMOU 5종(PS/LC/OG/DT/TI) 범위 내 |
