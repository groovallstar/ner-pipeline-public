# 한국어 NER 라벨링 방법론

> 기준 코드 시점: 2026-04-09 (develop 브랜치)
> 대상 데이터셋: KLUE NER (6 엔티티 타입)
> 대상 코드: `src/labelers/ko/`, `src/evaluators/`, `src/labelers/dataset_loader.py`

## 목차

1. [전체 파이프라인 흐름도](#1-전체-파이프라인-흐름도)
2. [Stage 1: 데이터 준비](#2-stage-1-데이터-준비)
3. [Stage 2: NER Tag 목록 및 프롬프트 설계](#3-stage-2-ner-tag-목록-및-프롬프트-설계)
4. [Stage 3: LLM 라벨링 실행](#4-stage-3-llm-라벨링-실행)
5. [Stage 4: 결과 확인 (평가)](#5-stage-4-결과-확인-평가)
6. [설계 의사결정 요약](#6-설계-의사결정-요약)

---

## 1. 전체 파이프라인 흐름도

```
KLUE NER Dataset (HuggingFace 또는 JSONL)
    │
    ▼ DatasetLoader.load()                          [dataset_loader.py]
List[NERRecord] {tokens, ner_tags, id, sentence?}
    │
    ▼ record["sentence"] 또는 TagAligner.reconstruct_text()
원본 텍스트 문자열                                    [benchmark_runner.py:74-78]
    │
    ▼ _split_sentences() → 배치 구성
문장 리스트 (batch_size 단위)                         [ollama_ner_labeler.py:24-39]
    │
    ▼ ner_prompts.py (SINGLE/BATCH 템플릿)
프롬프트 문자열                                       [ner_prompts.py]
    │
    ▼ ChatOllama / vLLM / OpenAI (temperature=0, JSON mode)
LLM JSON 응답                                        [ollama_ner_labeler.py:168-224]
    │
    ▼ JSON 파싱
[{"text": "경찰", "type": "OG"}, ...]               [ollama_ner_labeler.py:199-224]
    │
    ├──▶ label_spans() 경로: raw spans 직접 반환
    │
    └──▶ label() 경로: _spans_to_bio() 변환
         BIO 태그 리스트 ["B-OG", "O", "B-PS", ...]  [ollama_ner_labeler.py:42-85]
    │
    ▼ BenchmarkRunner._run_single()                  [benchmark_runner.py:58-186]
    │
    ├── spans_to_syllable_bio()   음절 BIO 변환       [tag_aligner.py:222-296]
    ├── extract_spans_from_bio()  gold span 추출      [tag_aligner.py:94-130]
    └── normalize_tag()           태그 정규화          [tag_aligner.py:44-59]
    │
    ▼ MetricsCalculator                              [metrics.py]
    ├── compute_span_match()   Span Match (exact/relaxed) — primary
    ├── compute_seqeval()      seqeval BIO F1 — secondary
    ├── compute_span_f1()      Character Span F1
    └── compute_bertscore()    BERTScore (optional)
    │
    ▼ ReportGenerator                                [report.py]
벤치마크 결과 (CLI 테이블 + JSON 파일)
```

---

## 2. Stage 1: 데이터 준비

### 입력

KLUE NER 데이터셋. 두 가지 로딩 경로가 존재한다:

| 경로 | 소스 | 조건 |
|------|------|------|
| JSONL 폴백 | `/data/ner/klue/validation.jsonl` | 파일이 존재하면 우선 사용 |
| HuggingFace | `load_dataset("klue", "ner", split="validation")` | JSONL 없을 때 |

**코드:** `DatasetLoader.load()` (`src/labelers/dataset_loader.py:35-62`)

```python
# JSONL 폴백 경로 확인 (dataset_loader.py:42-45)
jsonl_path = Path("/data/ner") / name.replace("/", "_") / f"{split}.jsonl"
if jsonl_path.exists():
    return self._load_jsonl(jsonl_path, max_samples)
```

### 출력

`List[NERRecord]` — 각 레코드는 다음 구조:

```json
{
  "tokens": ["경", "찰", "은", " ", "박", "씨", "의", " ", "딸", ...],
  "ner_tags": ["B-OG", "I-OG", "O", "O", "B-PS", "O", "O", "O", "O", ...],
  "id": "0",
  "sentence": "경찰은 <경찰:OG>박씨의 딸(32)과 ..."
}
```

### KLUE 토큰화 특성

KLUE NER은 **음절(syllable) 단위** 토큰화를 사용한다:

- 각 한글 음절이 하나의 토큰: `["경", "찰", "은"]`
- 단어 사이 공백은 빈 문자열 토큰으로 표현: `["경", "찰", "은", " ", "박", ...]`
- BIO 태그도 음절 단위로 부여됨

이 특성은 이후 Stage 3에서 LLM 출력을 음절 BIO로 변환할 때 핵심적인 제약이 된다.

### `sentence` 필드

KLUE NER 데이터에는 원본 문장이 포함되어 있다 (`dataset_loader.py:102-104`):

```python
if "sentence" in row:
    record["sentence"] = row["sentence"]
```

이 필드는 `benchmark_runner.py:74-78`에서 LLM에 전달할 텍스트의 **primary source**로 사용된다. `sentence` 필드가 없는 경우에만 `TagAligner.reconstruct_text()`로 토큰을 재결합한다.

```python
# benchmark_runner.py:74-78
sentence = record.get("sentence")
if sentence:
    text = re.sub(r'<([^:>]+):[A-Z]+>', r'\1', sentence)  # KLUE 태그 마크업 제거
else:
    text = TagAligner.reconstruct_text(gold_tokens)
```

### ClassLabel 변환

HuggingFace datasets에서 NER 태그는 정수(`ClassLabel`)로 저장된다. `DatasetLoader`가 자동으로 문자열 태그로 변환한다 (`dataset_loader.py:83-99`):

```python
if label_feature is not None and isinstance(label_feature, ClassLabel):
    ner_tags = [label_feature.int2str(t) for t in raw_tags]  # 0 → "O", 1 → "B-PS", ...
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| JSONL 폴백을 먼저 시도 | Python 3.13 환경에서 HuggingFace `datasets` 라이브러리 호환성 문제가 발생하여, 사전 export된 JSONL을 우선 사용 |
| `NERRecord`를 dict로 정의 | 경량 데이터 전달 — dataclass 대신 dict 사용으로 직렬화 편의 |
| `sentence` 필드 보존 | 토큰 재결합보다 원본 문장이 정확함 (KLUE의 경우 태그 마크업 포함 원문 제공) |

---

## 3. Stage 2: NER Tag 목록 및 프롬프트 설계

### 엔티티 타입 정의

**코드:** `src/labelers/ko/ner_prompts.py:11`

| 태그 | 의미 | 설명 | 예시 |
|------|------|------|------|
| PS | 인명 (Person) | 사람 이름, 성만도 가능. 가수/아이돌 그룹명 포함 | "김철수", "BTS", "소녀시대" |
| LC | 지명 (Location) | 장소, 지역, 국가, 도시, 건물, 시설. 복합 지명 전체를 하나로 | "서울삼성동그랜드인터콘티넨탈" |
| OG | 기관명 (Organization) | 회사, 기관, 단체, 정당, 팀. "경찰"도 기관 맥락이면 OG | "삼성전자", "경찰" |
| DT | 날짜 (Date) | 연도, 월, 일, 기간, 상대날짜, 시대, 요일 | "지난19일", "5공화국시절" |
| TI | 시간 (Time) | 시각, 시간대, 경과시간. "새벽", "오전" 단독도 TI | "오후 2시", "새벽" |
| QT | 수량 (Quantity) | 숫자+단위, 금액, 비율, 순서, 서수, 횟수 | "100명", "50억원", "첫번째" |

### 핵심 라벨링 규칙

프롬프트에 포함된 7가지 핵심 규칙 (`ner_prompts.py:50-57`):

1. **조사 제외**: 은/는/이/가/을/를/에/에서/으로/의/과/와/부터/까지 — 반드시 제거
2. **접미사 처리**: "씨"/"님" 제외. "김모" → "김"만 추출 (성+익명). "강모연" (3글자+) → 전체 추출
3. **복합 개체명 묶기**: 붙어 있는 개체명은 하나로 ("지난19일" → DT, "경남진해" → LC)
4. **괄호 내 처리**: 괄호 안 숫자 → QT, 괄호 안 날짜 → 별도 DT
5. **추출 제외 대상**: 사건명, 프로그램명, 작품명, 선박명은 개체명이 아님
6. **출력 형식**: JSON 배열만 출력, 다른 설명 금지
7. **빈 결과**: 개체명 없으면 `[]` 반환

### 프롬프트 구조

3종류의 프롬프트 템플릿이 존재한다:

| 템플릿 | 변수명 | 용도 | 형식 |
|--------|--------|------|------|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 단일 문장 라벨링 | `입력: {sentence}\n출력:` |
| BATCH | `BATCH_PROMPT_TEMPLATE` | 다중 문장 배치 라벨링 | `0: {sent0}\n1: {sent1}\n...` → `{"0": [...], "1": [...]}` |
| SYSTEM+USER | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` | OpenAI 채팅 형식 | system/user 메시지 분리 |

### Few-shot 예시의 설계 의도

프롬프트에 7개의 예시가 포함되어 있다 (`ner_prompts.py:59-79`). 각 예시가 커버하는 엣지 케이스:

| 예시 | 커버하는 엣지 케이스 |
|------|---------------------|
| 1. 경찰은 박씨의 딸(32)과... | 일반명사 기관(경찰=OG), 접미사 제거(박씨→박), 괄호 내 숫자(32=QT) |
| 2. 18번 홀(파5)에서... | 숫자+단위 묶기(18번홀=QT), 스포츠 용어(파5=QT), 서수(첫 번째=QT) |
| 3. 지난19일 오전9시30분... | 붙어쓰기 날짜(지난19일=DT), 시간(오전9시30분=TI), 복합 지명 |
| 4. 이번 주 월요일 새벽에... | 복합 날짜(이번 주 월요일=DT), 시간 단독(새벽=TI), 시대(5공화국시절=DT) |
| 5. 삼성전자는 오늘 오후 2시... | DT/TI 분리(오늘=DT, 오후 2시=TI), 복수 LC(서울, 강남구, 코엑스) |
| 6. 각 조 3위에 오른 6개국 중... | 스포츠 국가→OG(한국=OG), 단위 없는 숫자(4=QT, 0=QT) |
| 7. 어제(10월 10일) 방송된... | 괄호 안 날짜 분리(어제=DT + 10월10일=DT), 그룹명→PS(원더걸스, 소녀시대) |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 3종류 프롬프트 | 백엔드별 최적 형식이 다름: Ollama/vLLM은 단일 프롬프트, OpenAI는 system/user 분리가 성능 우수 |
| 상세한 엔티티 정의 + 규칙 | LLM의 라벨링 일관성을 높이기 위함. 특히 조사 제외, 복합 개체명 묶기는 한국어 특유의 문제 |
| 7개 Few-shot 예시 | 각 예시가 서로 다른 엣지 케이스를 커버하여, LLM이 다양한 패턴을 학습 |
| BATCH 프롬프트의 인덱스 키 | 다중 문장 처리 시 문장-결과 매핑을 명확히 하기 위함 (`{"0": [...], "1": [...]}`) |

---

## 4. Stage 3: LLM 라벨링 실행

### 전체 흐름

Canonical 구현: `OllamaNERLabeler` (`src/labelers/ko/ollama_ner_labeler.py`)

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

**코드:** `_split_sentences()` (`ollama_ner_labeler.py:24-39`)

```python
parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
```

- 마침표(`.`), 느낌표(`!`), 물음표(`?`) 뒤 공백 또는 줄바꿈으로 분리
- **최소 10자 버퍼링**: 짧은 조각은 다음 조각과 합침 (불필요한 분리 방지)
- 문장이 하나도 안 나오면 원본 텍스트 전체를 하나의 문장으로 반환

### 4.2 배치 구성

**코드:** `label()` 메서드 (`ollama_ner_labeler.py:111-141`)

- `batch_size` 파라미터 (기본값: 10)로 문장을 그룹핑
- 빈 문장은 사전에 필터링 (`non_empty`)
- 각 배치마다 타이밍 계측 출력 (`[TIMER]`)

**단일 문장 최적화** (`_call_llm_batch()`, line 170-171):
```python
if len(sentences) == 1:
    return [self._call_llm(sentences[0])]  # SINGLE 프롬프트 사용
```

### 4.3 LLM 호출

**코드:** `OllamaNERLabeler.__init__()` (`ollama_ner_labeler.py:89-109`)

```python
self._llm = ChatOllama(
    model=model,
    format="json",       # JSON 출력 강제
    temperature=0,       # 결정적 출력
    think=False,         # thinking 비활성화
    reasoning=False,
)
```

| 설정 | 값 | 이유 |
|------|------|------|
| `temperature` | 0 | NER은 정확성이 중요 — 창의적 변형 불필요 |
| `format` | "json" | LLM이 반드시 유효한 JSON을 출력하도록 강제 |
| `think` / `reasoning` | False | thinking 토큰이 JSON 파싱을 방해하지 않도록 |

### 4.4 JSON 파싱

두 가지 파싱 경로가 존재한다:

**경로 A — OllamaNERLabeler 인라인 파싱** (`ollama_ner_labeler.py:199-224`):
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

**경로 B — `parse_json_response()`** (`src/labelers/labeler_base.py:10-34`):
vllm/openai 라벨러에서 사용. 추가로 `<think>` 태그 제거, markdown fence 내 JSON 추출 등 더 견고한 파싱을 수행:
```python
raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
# ... json.loads 시도 후 실패 시 정규식으로 [...] 추출
match = re.search(r"\[.*?\]", raw, re.DOTALL)
```

**폴백 전략**: 배치 호출 실패 시 → 개별 문장 단위로 재시도 (`_call_llm_batch()`, line 196-197):
```python
except (json.JSONDecodeError, Exception) as e:
    return [self._call_llm(s) for s in sentences]  # 개별 폴백
```

### 4.5 spans → BIO 변환

**코드:** `_spans_to_bio()` (`ollama_ner_labeler.py:42-85`)

LLM이 반환한 entity spans를 토큰 레벨 BIO 태그로 변환한다. **2단계 매칭 전략**:

**1단계 — Exact match** (line 60-66):
```python
# span "김 철수" → span_tokens ["김", "철수"]
# tokens [..., "김", "철수", ...] 에서 연속 일치 찾기
if tokens[i : i + n] == span_tokens:
    tags[i] = f"B-{entity_type}"
```

**2단계 — Substring match** (line 69-79):
```python
# span "서울시" → tokens에 "서울시에서"가 있을 때
# "서울시" in "서울시에서" 또는 "서울시에서" in "서울시" 체크
if all(sp in tokens[i + j] or tokens[i + j] in sp for j, sp in enumerate(span_tokens)):
```

이 2단계 매칭은 한국어의 조사 부착 문제를 처리한다. LLM이 "서울시"를 추출하더라도, 원본 토큰이 "서울시에서"인 경우 substring 매칭으로 올바르게 태깅한다.

### 백엔드 변형

동일 인터페이스(`label()`, `label_spans()`)로 3개 백엔드를 지원한다:

| 파일 | 클래스 | 특징 |
|------|--------|------|
| `ko/ollama_ner_labeler.py` | `OllamaNERLabeler` | 동기 처리, ChatOllama, 배치 지원 |
| `ko/vllm_ner_labeler.py` | `VllmNERLabeler` | **async 동시성** (`concurrency` 파라미터), ChatOpenAI 호환 |
| `ko/openai_ner_labeler.py` | `OpenAINERLabeler` | **system/user 채팅 형식**, `parse_json_response()` 사용 |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 문장 분리 후 배치 | 긴 텍스트를 한 번에 보내면 LLM 컨텍스트 효율이 떨어지고, 엔티티 누락 증가. 문장 단위 분리 후 배치로 처리량 확보 |
| 최소 10자 버퍼링 | "네." 같은 짧은 조각은 단독으로 보내면 비효율적 |
| 2단계 매칭 (exact → substring) | 한국어 조사가 토큰에 부착되어 있어 exact match만으로는 부족. substring 폴백으로 "서울시" ↔ "서울시에서" 매칭 해결 |
| 배치 실패 시 개별 폴백 | 배치 JSON 파싱 실패(포맷 오류 등) 시에도 최대한 결과를 확보 |
| `label()` vs `label_spans()` 이중 인터페이스 | `label()`: BIO 태그가 필요한 seqeval 평가용. `label_spans()`: span 기반 평가용 (BIO 변환 없이 직접 사용) |

---

## 5. Stage 4: 결과 확인 (평가)

### 평가 진입점

**코드:** `_run_korean()` (`src/evaluators/__main__.py:191-217`)

```bash
PYTHONPATH=src python -m evaluators --lang ko --models "vllm:Qwen/Qwen3.5-27B" --max-samples 500
```

CLI가 수행하는 흐름:
1. `DatasetLoader`로 gold records 로딩
2. `BenchmarkRunner` 생성 후 labeler 추가
3. `runner.run()` → 각 labeler에 대해 `_run_single()` 실행
4. `ReportGenerator`로 결과 출력

### 평가 흐름 상세

**코드:** `BenchmarkRunner._run_single()` (`src/evaluators/benchmark_runner.py:58-186`)

각 gold record에 대해:

```
gold record {tokens, ner_tags, sentence}
    │
    ├── (1) 텍스트 재구성: sentence 필드 또는 reconstruct_text()
    │
    ├── (2) gold spans 추출: extract_spans_from_bio(gold_tokens, gold_tags)
    │       → [{"text": "경찰", "type": "OG"}, ...]
    │
    ├── (3) LLM 라벨링: labeler.label_spans(text)
    │       → [{"text": "경찰", "type": "OG"}, ...]  (predicted spans)
    │
    ├── (4) 음절 BIO 변환: spans_to_syllable_bio(text, gold_tokens, pred_spans)
    │       → ["B-OG", "I-OG", "O", ...]  (predicted BIO, gold 토큰 그리드에 정렬)
    │
    └── (5) 메트릭 계산 (전체 샘플에 대해)
```

### 태그 정규화

**코드:** `normalize_tag()` (`src/labelers/tag_aligner.py:44-59`)

LLM이 다양한 태그 형식을 출력할 수 있으므로, 모든 태그를 KLUE 표준으로 정규화한다:

```
PER → PS,  LOC → LC,  ORG → OG
DATE → DT, TIME → TI, QUANTITY → QT
```

### spans → 음절 BIO 변환

**코드:** `TagAligner.spans_to_syllable_bio()` (`src/labelers/tag_aligner.py:222-296`)

LLM이 반환한 text spans를 KLUE의 음절 단위 BIO 태그로 변환하는 핵심 로직:

1. **char→syl 매핑 구축** (line 234-251): 원본 텍스트의 각 문자 위치 → 음절 토큰 인덱스 매핑
2. **span 위치 찾기** (line 258-282): 원본 텍스트에서 entity text의 문자 범위 탐색
   - 정확 매치 → `text.find(entity_text)`
   - 공백 무시 매치 → `_find_ignore_spaces()` (예: "지난 19일" ↔ "지난19일")
   - 중복 처리: `_next_search` 딕셔너리로 이미 매칭된 위치 이후부터 탐색
3. **BIO 태그 할당** (line 284-294): 매칭된 문자 범위의 음절 토큰에 B-/I- 태그 부여

### 4종 메트릭

**코드:** `MetricsCalculator` (`src/evaluators/metrics.py`)

| 메트릭 | 메서드 | 역할 | 비고 |
|--------|--------|------|------|
| **Span Match** | `compute_span_match()` | entity 단위 exact/relaxed 매칭 | **Primary** — BIO 변환 없이 직접 span 비교 |
| **seqeval** | `compute_seqeval()` | 음절 BIO 태그 기반 F1/Precision/Recall | Secondary — BERT 모델과 비교 시 사용 |
| **Character Span F1** | `compute_span_f1()` | 문자 수준 span 매칭 (KLUE 공식 메트릭) | BIO 태그에서 span 추출 후 비교 |
| **BERTScore** | `compute_bertscore()` | 의미적 유사도 기반 평가 | Optional — `--no-bertscore` 플래그로 비활성화 |

**Span Match 상세:**
- **Exact**: gold span과 predicted span의 text와 type이 정확히 일치
- **Relaxed**: text가 부분적으로 겹치면 매칭 (type은 일치해야 함)

### 결과 출력

**코드:** `ReportGenerator` (`src/evaluators/report.py`)

- CLI 테이블: entity type별 F1/Precision/Recall + 전체 평균
- JSON 파일: `results/ko_bench_500_27b.json` 형태로 저장

```json
{
  "timestamp": "2026-04-06T10:15:31+00:00",
  "results": [{
    "model_name": "Qwen/Qwen3.5-27B",
    "metrics": {
      "span_match": {"exact": {"overall": {"f1": 0.7789}}, "relaxed": {...}},
      "overall": {"f1": 0.6557},        // seqeval
      "span_f1": {"overall": {"f1": 0.6557}},
      "bertscore": {...}
    },
    "latency": {"total_seconds": 1745.85, "avg_per_sample": 3.5923}
  }]
}
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| Span Match를 primary 메트릭으로 | BIO 변환 과정의 정렬 오류를 우회하여, LLM 라벨링 품질을 직접 평가 |
| seqeval을 secondary로 유지 | BERT 기반 모델은 BIO 태그를 직접 출력하므로, LLM과 BERT를 동일 기준으로 비교하기 위해 |
| 음절 BIO 변환이 필요한 이유 | KLUE gold 데이터가 음절 단위 BIO이므로, seqeval 비교를 위해 LLM spans를 같은 형식으로 변환 필요 |
| 4종 메트릭 병행 | 각 메트릭이 다른 관점을 제공: span match(엔티티 품질), seqeval(토큰 정확도), BERTScore(의미 유사도) |
| BERTScore를 optional로 | 계산 비용이 높아 빠른 반복 시에는 생략 가능 |

---

## 6. 설계 의사결정 요약

전체 파이프라인에서 내려진 핵심 설계 결정들:

| # | 단계 | 결정 | 이유 | 대안 (채택하지 않은 것) |
|---|------|------|------|----------------------|
| 1 | 데이터 | JSONL 폴백 우선 | Python 3.13 호환성 | HuggingFace만 사용 (환경 제약) |
| 2 | 데이터 | sentence 필드 보존 | 원본 문장이 토큰 재결합보다 정확 | 항상 reconstruct_text() 사용 |
| 3 | 프롬프트 | 3종 템플릿 | 백엔드별 최적 형식 상이 | 단일 범용 프롬프트 |
| 4 | 프롬프트 | 7개 Few-shot | 각각 다른 엣지 케이스 커버 | Zero-shot (정확도 저하) |
| 5 | 라벨링 | 문장 분리 → 배치 | 컨텍스트 효율 + 처리량 | 전체 텍스트 한 번에 전송 |
| 6 | 라벨링 | temperature=0, JSON mode | 결정적 출력 + 파싱 안정성 | 높은 temperature (변형 발생) |
| 7 | 라벨링 | 2단계 span→BIO 매칭 | 조사 부착 처리 | exact match만 (누락 증가) |
| 8 | 라벨링 | 배치 실패 시 개별 폴백 | 결과 확보 극대화 | 실패 시 에러 반환 |
| 9 | 평가 | Span Match primary | BIO 변환 오류 우회 | seqeval만 사용 (변환 노이즈) |
| 10 | 평가 | 4종 메트릭 병행 | 다각적 품질 평가 | 단일 메트릭 |
