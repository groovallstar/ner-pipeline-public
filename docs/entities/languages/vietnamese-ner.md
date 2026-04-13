# 베트남어 NER 라벨링 방법론

> 기준 코드 시점: 2026-04-09 (develop 브랜치)
> 대상 데이터셋: WikiANN Vietnamese (3 엔티티 타입: PER, LOC, ORG)
> 대상 코드: `src/labelers/vi/`, `src/evaluators/`, `src/labelers/dataset_loader.py`
> 벤치마크 상태: **미실행** — 코드 구현 완료, 벤치마크 결과 없음

## 목차

1. [전체 파이프라인 흐름도](#1-전체-파이프라인-흐름도)
2. [Stage 1: 데이터 준비](#2-stage-1-데이터-준비)
3. [Stage 2: NER Tag 목록 및 프롬프트 설계](#3-stage-2-ner-tag-목록-및-프롬프트-설계)
4. [Stage 3: LLM 라벨링 실행](#4-stage-3-llm-라벨링-실행)
5. [Stage 4: 결과 확인 (평가)](#5-stage-4-결과-확인-평가)
6. [한국어 파이프라인과의 차이점 요약](#6-한국어-파이프라인과의-차이점-요약)
7. [설계 의사결정 요약](#7-설계-의사결정-요약)

---

## 1. 전체 파이프라인 흐름도

```
WikiANN Dataset (unimelb-nlp/wikiann, config="vi")
    │
    ▼ DatasetLoader.load()                          [dataset_loader.py:35-62]
List[NERRecord] {tokens, ner_tags, id}
    │
    ▼ TagAligner.reconstruct_text()                  [tag_aligner.py:137-149]
원본 텍스트 문자열 (word-level 토큰 space-join)
    │
    ▼ _split_sentences() → 배치 구성
문장 리스트 (batch_size 단위)                         [vi/ollama_ner_labeler.py:33-48]
    │
    ▼ ner_prompts.py (SINGLE/BATCH 템플릿)
프롬프트 문자열 (베트남어)                             [vi/ner_prompts.py]
    │
    ▼ ChatOllama / vLLM / OpenAI (temperature=0, JSON mode)
LLM JSON 응답                                        [vi/ollama_ner_labeler.py:165-216]
    │
    ▼ JSON 파싱
[{"text": "Nguyễn Xuân Phúc", "type": "PER"}, ...]  [vi/ollama_ner_labeler.py:195-216]
    │
    ├──▶ label_spans() 경로: raw spans 직접 반환
    │
    └──▶ label() 경로: _spans_to_bio() 변환
         BIO 태그 리스트 ["B-PER", "O", "B-LOC", ...] [vi/ollama_ner_labeler.py:51-89]
    │
    ▼ BenchmarkRunner._run_single()                  [benchmark_runner.py]
    │
    ├── extract_spans_from_bio()  gold span 추출      [tag_aligner.py:94-130]
    ├── normalize_tag()           태그 정규화          [tag_aligner.py:44-59]
    │   └── _TAG_NORMALIZE_MAP_VI 사용                [tag_aligner.py:30-35]
    └── TagAligner.align()        토큰 정렬           [tag_aligner.py:152-174]
    │
    ▼ MetricsCalculator                              [metrics.py]
    ├── compute_span_match()   Span Match (exact/relaxed)
    ├── compute_seqeval()      seqeval BIO F1
    ├── compute_span_f1()      Character Span F1
    └── compute_bertscore()    BERTScore (optional)
    │
    ▼ ReportGenerator                                [report.py]
벤치마크 결과 (CLI 테이블 + JSON 파일)
```

---

## 2. Stage 1: 데이터 준비

### 입력

WikiANN Vietnamese 데이터셋. 한국어와 동일한 `DatasetLoader`를 사용하되, config를 `"vi"`로 지정한다:

| 경로 | 소스 | 조건 |
|------|------|------|
| JSONL 폴백 | `/data/ner/unimelb-nlp_wikiann/test.jsonl` | 파일이 존재하면 우선 사용 |
| HuggingFace | `load_dataset("unimelb-nlp/wikiann", "vi", split="test")` | JSONL 없을 때 |

**코드:** `_run_vietnamese()` (`src/evaluators/__main__.py:250-277`)

```python
# __main__.py:250-256
loader = DatasetLoader()
gold_records = loader.load(
    args.dataset, config="vi", split=args.split, max_samples=args.max_samples,
)
```

기본 데이터셋과 split 설정 (`__main__.py:174-179`):

```python
if args.lang == "vi":
    args.dataset = "unimelb-nlp/wikiann"
# ...
args.split = "test" if args.lang in ("ja", "vi") else "validation"
```

### 출력

`List[NERRecord]` — 각 레코드는 다음 구조:

```json
{
  "tokens": ["Nguyễn", "Văn", "A", "sinh", "ra", "tại", "Hà", "Nội"],
  "ner_tags": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "B-LOC", "I-LOC"],
  "id": "0"
}
```

### WikiANN 토큰화 특성

WikiANN은 **단어(word) 단위** 토큰화를 사용한다:

- 각 단어가 하나의 토큰: `["Nguyễn", "Văn", "A"]`
- 단어 사이 공백은 별도 토큰이 **아님** (KLUE와의 핵심 차이)
- BIO 태그도 단어 단위로 부여됨

이 특성 덕분에 한국어 KLUE의 음절→단어 정렬 문제가 발생하지 않는다.

### ClassLabel 변환

WikiANN에서도 NER 태그는 정수(`ClassLabel`)로 저장된다. `DatasetLoader`가 자동으로 문자열 태그로 변환한다 (`dataset_loader.py:83-99`):

```python
if label_feature is not None and isinstance(label_feature, ClassLabel):
    ner_tags = [label_feature.int2str(t) for t in raw_tags]  # 0 → "O", 1 → "B-PER", ...
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 한국어와 동일한 `DatasetLoader` 재사용 | WikiANN도 HuggingFace datasets 형식이며, tokens/ner_tags 컬럼 구조가 동일 |
| config="vi"로 언어 지정 | WikiANN은 다국어 데이터셋으로 config 파라미터로 언어를 선택 |
| split="test" 기본값 | WikiANN에는 validation split이 없거나 작을 수 있으므로 test 사용 |
| `sentence` 필드 없음 | WikiANN은 KLUE와 달리 원본 문장 필드를 제공하지 않음 → `TagAligner.reconstruct_text()`로 재구성 |

---

## 3. Stage 2: NER Tag 목록 및 프롬프트 설계

### 엔티티 타입 정의

**코드:** `src/labelers/vi/ner_prompts.py:10`

| 태그 | 의미 | 설명 | 예시 |
|------|------|------|------|
| PER | 인물 (Người) | 사람 이름 (성+이름 전체 또는 일부), 별명, 예명. 그룹/밴드명은 ORG | "Nguyễn Xuân Phúc", "Hồ Chí Minh", "Bác Hồ" |
| LOC | 지명 (Địa điểm) | 국가, 도시, 성(tỉnh), 군(huyện), 사(xã), 도로, 강, 산, 바다, 섬, 건축물. 복합 지명은 하나로 | "Hà Nội", "Việt Nam", "Sông Mekong", "Chùa Một Cột" |
| ORG | 기관 (Tổ chức) | 회사, 기관, 조직, 정당, 스포츠팀, 학교, 병원. 복합 기관명은 하나로 | "Đảng Cộng sản Việt Nam", "Samsung", "Đại học Quốc gia Hà Nội" |

한국어 KLUE는 6개 타입(PS, LC, OG, DT, TI, QT)을 사용하지만, 베트남어 WikiANN은 **3개 타입(PER, LOC, ORG)만** 사용한다. 날짜(DT), 시간(TI), 수량(QT)은 WikiANN 어노테이션 범위에 포함되지 않는다.

### 핵심 라벨링 규칙

프롬프트에 포함된 5가지 핵심 규칙 (`ner_prompts.py:27-32`):

1. **원문 보존**: 베트남어 성조 부호(dấu)를 정확히 유지하며 추출 — "Hà Nội"를 "Ha Noi"로 변환하면 안 됨
2. **다중 단어 엔티티 통합**: 여러 단어로 구성된 엔티티를 하나로 묶음 — "Thành phố Hồ Chí Minh" → LOC 1개
3. **직함/호칭 제외**: "Ông"(씨), "Bà"(여사), "Chủ tịch"(주석), "Thủ tướng"(총리) 등 직함은 제외하고 이름만 추출
4. **JSON 배열만 출력**: 설명 없이 JSON 배열만 반환
5. **빈 결과**: 엔티티 없으면 `[]` 반환

### 프롬프트 구조

한국어와 동일하게 3종류의 프롬프트 템플릿이 존재한다:

| 템플릿 | 변수명 | 용도 | 형식 |
|--------|--------|------|------|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 단일 문장 라벨링 | `Đầu vào: {sentence}\nĐầu ra:` |
| BATCH | `BATCH_PROMPT_TEMPLATE` | 다중 문장 배치 라벨링 | `0: {sent0}\n1: {sent1}\n...` → `{"0": [...], "1": [...]}` |
| SYSTEM+USER | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` | OpenAI 채팅 형식 | system/user 메시지 분리 |

**코드:** `src/labelers/vi/ner_prompts.py:13-99`

### Few-shot 예시의 설계 의도

프롬프트에 3개의 예시가 포함되어 있다 (`ner_prompts.py:34-42`). 각 예시가 커버하는 엣지 케이스:

| 예시 | 커버하는 엣지 케이스 |
|------|---------------------|
| 1. Chủ tịch Nguyễn Xuân Phúc đã đến thăm Đà Nẵng... | 직함 제외(Chủ tịch→제거), PER/LOC/ORG 혼합, 기업명("Tập đoàn Vingroup"=ORG) |
| 2. Sông Mekong chảy qua Campuchia và Việt Nam... | 복합 지명("Sông Mekong"=LOC), 다중 LOC 인식, 외래 지명("Campuchia") |
| 3. Đại học Quốc gia Hà Nội vừa ký kết... | 복합 기관명("Đại học Quốc gia Hà Nội"=ORG), 동일 단어의 문맥별 타입("Hà Nội"가 LOC vs ORG 내 일부) |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 3종류 프롬프트 | 한국어와 동일한 아키텍처 — 백엔드별 최적 형식이 다름 |
| 베트남어 전용 프롬프트 | 프롬프트 전체를 베트남어로 작성하여 LLM의 베트남어 NER 성능 극대화 |
| 성조 부호 보존 규칙 강조 | 베트남어의 핵심 특성 — 성조가 다르면 다른 단어가 되므로 정확한 추출 필수 |
| 직함 제외 규칙 | 베트남어에서 "Ông", "Bà" 등 호칭이 이름 앞에 붙는 것이 일반적이므로 명시적으로 제외 규칙 필요 |
| 3개 Few-shot 예시 (한국어 7개 대비 적음) | 엔티티 타입이 3개로 단순하여, 적은 예시로도 충분한 패턴 커버 가능 |

---

## 4. Stage 3: LLM 라벨링 실행

### 전체 흐름

Canonical 구현: `OllamaNERLabeler` (`src/labelers/vi/ollama_ner_labeler.py`)

```
입력 텍스트
    │
    ▼ _split_sentences()        (1) 문장 분리
문장 리스트
    │
    ▼ batch_size 단위 그룹핑     (2) 배치 구성
배치 리스트
    │
    ▼ _call_llm_batch()          (3) LLM 호출
    │   └── 단일 문장 → _call_llm() 직접 호출
    │   └── 다중 문장 → BATCH 프롬프트 사용
    │
    ▼ JSON 파싱                  (4) 응답 처리
[{"text": "...", "type": "..."}, ...]
    │
    ├── label_spans() 경로: 여기서 반환 (raw spans)
    │
    └── label() 경로:
        ▼ _spans_to_bio()        (5) BIO 변환
        NERRecord {tokens, ner_tags, id}
```

### 4.1 문장 분리

**코드:** `_split_sentences()` (`vi/ollama_ner_labeler.py:33-48`)

```python
parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
```

- 마침표(`.`), 느낌표(`!`), 물음표(`?`) 뒤 공백 또는 줄바꿈으로 분리
- **최소 10자 버퍼링**: 짧은 조각은 다음 조각과 합침
- 한국어와 동일한 로직 (베트남어도 라틴 문자 기반 문장부호 사용)

### 4.2 배치 구성

**코드:** `label()` 메서드 (`vi/ollama_ner_labeler.py:115-142`)

- `batch_size` 파라미터 (기본값: 10)로 문장을 그룹핑
- 빈 문장은 사전에 필터링 (`non_empty`)
- 각 배치마다 타이밍 계측 출력 (`[TIMER]`)

**단일 문장 최적화** (`_call_llm_batch()`, line 166-167):
```python
if len(sentences) == 1:
    return [self._call_llm(sentences[0])]  # SINGLE 프롬프트 사용
```

### 4.3 LLM 호출

**코드:** `OllamaNERLabeler.__init__()` (`vi/ollama_ner_labeler.py:92-113`)

```python
self._llm = ChatOllama(
    model=model,
    format="json",       # JSON 출력 강제
    temperature=0,       # 결정적 출력
    think=False,         # thinking 비활성화
    reasoning=False,
)
```

한국어 Ollama 라벨러와 동일한 설정을 사용한다.

| 설정 | 값 | 이유 |
|------|------|------|
| `temperature` | 0 | NER은 정확성이 중요 — 창의적 변형 불필요 |
| `format` | "json" | LLM이 반드시 유효한 JSON을 출력하도록 강제 |
| `think` / `reasoning` | False | thinking 토큰이 JSON 파싱을 방해하지 않도록 |

### 4.4 JSON 파싱

두 가지 파싱 경로가 존재한다:

**경로 A — OllamaNERLabeler 인라인 파싱** (`vi/ollama_ner_labeler.py:195-216`):
```python
data = json.loads(raw)
if isinstance(data, list):          # 일반 케이스: [{"text":..., "type":...}, ...]
    return data
if isinstance(data, dict):
    if "text" in data and "type" in data:  # 단일 엔티티 bare object
        return [data]
    for val in data.values():              # {"entities": [...]} 래퍼
        if isinstance(val, list):
            return val
```

**경로 B — VllmNERLabeler의 `_parse_spans()`** (`vi/vllm_ner_labeler.py:48-72`):
추가로 `<think>` 태그 제거, 정규식으로 `[...]` 배열 추출 등 더 견고한 파싱을 수행:
```python
raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
# ... json.loads 시도 후 실패 시 정규식으로 [...] 추출
match = re.search(r"\[.*?\]", raw, re.DOTALL)
```

**폴백 전략**: 배치 호출 실패 시 → 개별 문장 단위로 재시도 (`vi/ollama_ner_labeler.py:191-193`):
```python
except (json.JSONDecodeError, Exception) as e:
    return [self._call_llm(s) for s in sentences]  # 개별 폴백
```

### 4.5 spans → BIO 변환

**코드:** `_spans_to_bio()` (`vi/ollama_ner_labeler.py:51-89`)

LLM이 반환한 entity spans를 토큰 레벨 BIO 태그로 변환한다. **2단계 매칭 전략**:

**1단계 — Exact match** (line 63-69):
```python
# span "Nguyễn Xuân Phúc" → span_tokens ["Nguyễn", "Xuân", "Phúc"]
# tokens [..., "Nguyễn", "Xuân", "Phúc", ...] 에서 연속 일치 찾기
if tokens[i : i + n] == span_tokens:
    tags[i] = f"B-{entity_type}"
```

**2단계 — Substring match** (line 72-83):
```python
# 부분 문자열 매칭으로 약간의 토큰 불일치 처리
if all(sp in tokens[i + j] or tokens[i + j] in sp for j, sp in enumerate(span_tokens)):
```

베트남어는 한국어와 달리 조사가 단어에 부착되지 않으므로, exact match만으로 대부분의 경우가 처리된다. 그러나 substring match는 LLM이 약간 다른 형태의 텍스트를 반환하는 경우에 대비한 안전장치로 유지된다.

### 백엔드 변형

동일 인터페이스(`label()`, `label_spans()`)로 3개 백엔드를 지원한다:

| 파일 | 클래스 | 특징 |
|------|--------|------|
| `vi/ollama_ner_labeler.py` | `OllamaNERLabeler` | 동기 처리, ChatOllama, 배치 지원 |
| `vi/vllm_ner_labeler.py` | `VllmNERLabeler` | **async 동시성** (`concurrency` 파라미터, 기본 32), AsyncOpenAI 호환 |
| `vi/openai_ner_labeler.py` | `OpenAINERLabeler` | **system/user 채팅 형식**, `response_format={"type": "json_object"}` |

**코드:** 라벨러 팩토리 (`src/evaluators/__main__.py:113-138`)

```python
def _create_labeler_vi(backend: str, model_name: str, args):
    if backend == "ollama":
        from labelers.vi.ollama_ner_labeler import OllamaNERLabeler
        # ...
    elif backend == "vllm":
        from labelers.vi.vllm_ner_labeler import VllmNERLabeler
        # ...
    elif backend == "openai":
        from labelers.vi.openai_ner_labeler import OpenAINERLabeler
        # ...
    elif backend == "hf":
        from labelers.hf_ner_labeler import HFNERLabeler
        return backend, HFNERLabeler(model_name=model_name, lang="vi")
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 한국어와 동일한 문장 분리 로직 | 베트남어도 라틴 문자 기반으로 `.!?` 문장부호를 사용하므로 동일 정규식 적용 가능 |
| 2단계 매칭 유지 | 베트남어는 조사 부착이 없지만, LLM 출력의 미세한 차이(성조 부호 누락 등)에 대비 |
| 3개 백엔드 동일 지원 | 파이프라인 아키텍처 일관성 유지 — 한국어/일본어와 동일한 백엔드 선택지 제공 |
| HF 백엔드 추가 지원 | BERT 기반 모델과 LLM 성능 비교를 위한 HFNERLabeler 연동 (`lang="vi"`) |

---

## 5. Stage 4: 결과 확인 (평가)

### 평가 진입점

**코드:** `_run_vietnamese()` (`src/evaluators/__main__.py:250-277`)

```bash
PYTHONPATH=src python -m evaluators --lang vi --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200
```

CLI가 수행하는 흐름:
1. `DatasetLoader`로 WikiANN(vi) gold records 로딩
2. `BenchmarkRunner` 생성 (`lang="vi"` 지정)
3. 각 labeler에 대해 `_run_single()` 실행
4. `ReportGenerator`로 결과 출력

```python
# __main__.py:259
runner = BenchmarkRunner(gold_records, compute_bertscore=not args.no_bertscore, lang="vi")
```

### 평가 흐름 상세

한국어와 동일한 `BenchmarkRunner`를 사용한다. 각 gold record에 대해:

```
gold record {tokens, ner_tags}
    │
    ├── (1) 텍스트 재구성: reconstruct_text() (word-level space-join)
    │       → "Nguyễn Văn A sinh ra tại Hà Nội"
    │
    ├── (2) gold spans 추출: extract_spans_from_bio(gold_tokens, gold_tags, lang="vi")
    │       → [{"text": "Nguyễn Văn A", "type": "PER"}, {"text": "Hà Nội", "type": "LOC"}]
    │
    ├── (3) LLM 라벨링: labeler.label_spans(text) 또는 labeler.label(text)
    │       → predicted spans 또는 BIO 태그
    │
    ├── (4) 태그 정렬: TagAligner.align() (word-level이므로 음절 변환 불필요)
    │
    └── (5) 메트릭 계산 (전체 샘플에 대해)
```

### 태그 정규화

**코드:** `normalize_tag()` (`src/labelers/tag_aligner.py:44-59`), `_TAG_NORMALIZE_MAP_VI` (`tag_aligner.py:30-35`)

베트남어 태그 정규화 맵:

```python
_TAG_NORMALIZE_MAP_VI = {
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "MISCELLANEOUS": "MISC",
}
```

**핵심 차이**: 한국어는 국제 표준 태그를 KLUE 고유 태그로 변환(PER→PS, LOC→LC, ORG→OG)하지만, 베트남어는 **국제 표준 태그를 그대로 유지**(PER, LOC, ORG)한다. 정규화는 LLM이 풀네임("PERSON", "LOCATION")을 출력하는 경우에만 약어로 변환한다.

### 토큰 정렬의 단순화

WikiANN은 word-level 토큰을 사용하므로, KLUE의 음절→단어 정렬 문제가 발생하지 않는다:

- **한국어 (KLUE)**: 음절 단위 gold tokens → LLM word-level 출력 → `spans_to_syllable_bio()` 또는 `align_syllable_to_word()` 필요
- **베트남어 (WikiANN)**: word-level gold tokens → LLM word-level 출력 → `TagAligner.align()` 직접 사용 가능

`extract_spans_from_bio()` (`tag_aligner.py:94-130`)에서 space token 유무를 자동 감지:

```python
has_space_tokens = any(t.strip() == "" for t in tokens)
joiner = "" if has_space_tokens else " "  # WikiANN: " " (word-level)
```

### 4종 메트릭

한국어와 동일한 `MetricsCalculator`를 사용한다:

| 메트릭 | 역할 | 비고 |
|--------|------|------|
| **Span Match** | entity 단위 exact/relaxed 매칭 | **Primary** |
| **seqeval** | 단어 BIO 태그 기반 F1/Precision/Recall | Secondary |
| **Character Span F1** | 문자 수준 span 매칭 | |
| **BERTScore** | 의미적 유사도 기반 평가 | Optional (`--no-bertscore`) |

### 벤치마크 상태

> **주의**: 베트남어 NER 파이프라인은 코드 구현이 완료되었으나, 아직 벤치마크가 실행되지 않았다. 한국어 파이프라인에는 `results/` 디렉토리에 벤치마크 결과 JSON이 존재하지만, 베트남어 결과는 없다.

예상 실행 명령:

```bash
# 베트남어 벤치마크 실행 (예시)
PYTHONPATH=src python -m evaluators --lang vi --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200 --output results/vi_bench_200_27b.json
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 한국어와 동일한 `BenchmarkRunner` 사용 | BIO 기반 평가 프레임워크를 공유하여 코드 중복 제거 |
| 국제 표준 태그 유지 (PER, LOC, ORG) | WikiANN이 국제 표준 태그를 사용하므로, 불필요한 태그 변환 없이 직접 비교 가능 |
| `lang="vi"` 전달 | 태그 정규화 맵 선택 및 BERTScore 언어 모델 지정에 사용 |
| 음절 BIO 변환 불필요 | WikiANN의 word-level 토큰이 LLM 출력과 동일한 수준이므로 복잡한 정렬 로직 생략 가능 |

---

## 6. 한국어 파이프라인과의 차이점 요약

| 항목 | 한국어 (KLUE) | 베트남어 (WikiANN) |
|------|--------------|-------------------|
| **데이터셋** | KLUE NER (`klue`, config=`ner`) | WikiANN (`unimelb-nlp/wikiann`, config=`vi`) |
| **데이터 split** | `validation` | `test` |
| **엔티티 타입** | 6개: PS, LC, OG, DT, TI, QT | 3개: PER, LOC, ORG |
| **태그 표준** | KLUE 고유 (PS, LC, OG...) | 국제 표준 (PER, LOC, ORG) |
| **태그 정규화** | PER→PS, LOC→LC, ORG→OG 등 | PERSON→PER, LOCATION→LOC 등 (약어 이미 표준) |
| **토큰 수준** | 음절(syllable) 단위 + 공백 토큰 | 단어(word) 단위 |
| **`sentence` 필드** | 있음 (원본 문장 포함) | 없음 (토큰 재결합으로 복원) |
| **음절 BIO 변환** | 필요 (`spans_to_syllable_bio`) | 불필요 (word-level 직접 비교) |
| **프롬프트 언어** | 한국어 | 베트남어 |
| **프롬프트 규칙 수** | 7개 (조사 제외, 접미사 처리 등) | 5개 (성조 보존, 직함 제외 등) |
| **Few-shot 예시** | 7개 | 3개 |
| **핵심 언어 특성** | 조사 부착, 음절 토큰화 | 성조 부호(dấu), 다중 단어 이름 |
| **평가 러너** | `BenchmarkRunner` (lang="ko") | `BenchmarkRunner` (lang="vi") — 동일 클래스 |
| **벤치마크 결과** | 있음 (`results/ko_*.json`) | **없음** (미실행) |

---

## 7. 설계 의사결정 요약

전체 파이프라인에서 내려진 핵심 설계 결정들:

| # | 단계 | 결정 | 이유 | 대안 (채택하지 않은 것) |
|---|------|------|------|----------------------|
| 1 | 데이터 | DatasetLoader 재사용 | WikiANN과 KLUE 모두 tokens/ner_tags 구조가 동일 | 별도 VietnameseDatasetLoader 구현 (불필요한 코드 중복) |
| 2 | 데이터 | config="vi"로 언어 지정 | WikiANN 다국어 데이터셋의 표준 접근법 | 별도 데이터셋 사용 |
| 3 | 태그 | 국제 표준 태그 유지 | WikiANN이 PER/LOC/ORG를 사용하므로 변환 불필요 | KLUE 스타일로 변환 (PS/LC/OG → 혼란 유발) |
| 4 | 프롬프트 | 베트남어 전체 작성 | LLM의 베트남어 이해도 극대화 | 영어/한국어 프롬프트 (성능 저하 예상) |
| 5 | 프롬프트 | 3개 Few-shot | 3개 타입에 대해 충분한 엣지 케이스 커버 | 7개 (한국어 수준) — 엔티티 타입이 적어 불필요 |
| 6 | 프롬프트 | 성조 부호 보존 명시 | 베트남어에서 성조는 의미 구분의 핵심 | 별도 언급 없음 (LLM이 성조 누락 가능) |
| 7 | 프롬프트 | 직함 제외 규칙 | "Ông", "Bà" 등이 이름 앞에 붙는 베트남어 관용 반영 | 한국어식 조사 제외 규칙 (베트남어에 해당 없음) |
| 8 | 라벨링 | 한국어와 동일한 3개 백엔드 | 파이프라인 일관성 + 백엔드 선택 유연성 | 단일 백엔드만 지원 |
| 9 | 평가 | BenchmarkRunner 공유 | BIO 기반 seqeval이 word-level에서도 동일 동작 | 별도 평가 러너 (코드 중복) |
| 10 | 평가 | 음절 변환 생략 | word-level 토큰으로 직접 비교 가능 | 음절 변환 로직 적용 (불필요한 복잡도) |
