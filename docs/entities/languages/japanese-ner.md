# 일본어 NER 라벨링 방법론

> 기준 코드 시점: 2026-04-09 (develop 브랜치)
> 대상 데이터셋: Stockmark NER Wikipedia (8 엔티티 타입)
> 대상 코드: `src/labelers/ja/`, `src/evaluators/ja_benchmark_runner.py`, `src/evaluators/ja_metrics.py`

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
Stockmark NER Wikipedia Dataset (HuggingFace)
    |
    v JapaneseDatasetLoader.load()                 [ja/dataset_loader.py:25-60]
List[dict] {id, text, gold_spans: [{text, type, start, end}]}
    |
    v record["text"] (원본 텍스트 그대로 사용)
원본 텍스트 문자열                                    [ja_benchmark_runner.py:66]
    |
    v _split_sentences() -> 배치 구성
문장 리스트 (batch_size 단위)                         [ja/ollama_ner_labeler.py:24-39]
    |
    v ner_prompts.py (SINGLE/BATCH/SYSTEM+USER 템플릿)
프롬프트 문자열                                       [ja/ner_prompts.py]
    |
    v ChatOllama / vLLM / OpenAI (temperature=0, JSON mode)
LLM JSON 응답                                        [ja/ollama_ner_labeler.py:168-224]
    |
    v JSON 파싱
[{"text": "東京", "type": "地名"}, ...]              [ja/ollama_ner_labeler.py:199-224]
    |
    v label_spans() 경로: raw spans 직접 반환
    |
    v match_spans() -- LLM 텍스트 span -> 문자 오프셋 변환
[{"text": "東京", "type": "地名", "start": 5, "end": 7}]  [ja/span_matcher.py:19-87]
    |
    v JaBenchmarkRunner._run_single()               [ja_benchmark_runner.py:57-121]
    |
    v compute_offset_span_f1()                      [ja_metrics.py:9-54]
    |  (sent_idx, start, end, type) 튜플 기반 exact match
    |
    v JaReportGenerator                             [ja_benchmark_runner.py:124-224]
벤치마크 결과 (CLI 테이블 + JSON 파일)
```

**한국어 파이프라인과의 핵심 차이**: BIO 태그 변환 단계가 없다. LLM 출력 → `match_spans()` → 문자 오프셋 span → `compute_offset_span_f1()`로 직접 평가한다. seqeval, TagAligner, `_spans_to_bio()` 등 BIO 관련 모듈을 우회한다.

---

## 2. Stage 1: 데이터 준비

### 입력

Stockmark NER Wikipedia 데이터셋. HuggingFace Hub에서 직접 로딩한다.

| 항목 | 값 |
|------|------|
| 데이터셋 | `stockmark/ner-wikipedia-dataset` |
| 원본 split | `train`만 존재 |
| 분할 전략 | `train_test_split(test_size=0.2, seed=42)` |
| 데이터 형식 | 원본 텍스트 + 문자 오프셋 span (BIO가 아님) |

**코드:** `JapaneseDatasetLoader.load()` (`src/labelers/ja/dataset_loader.py:25-60`)

```python
# Stockmark은 train split만 존재하므로 직접 분할
hf_dataset = load_dataset(dataset_name, split="train", cache_dir=self.cache_dir)
splits = hf_dataset.train_test_split(test_size=test_size, seed=seed)
selected = splits["test"] if split == "test" else splits["train"]
```

### 출력

`List[dict]` -- 각 레코드는 다음 구조:

```json
{
  "id": "12345",
  "text": "織田信長は安土城を建て、本能寺の変で明智光秀に討たれた。",
  "gold_spans": [
    {"text": "織田信長", "type": "人名", "start": 0, "end": 4},
    {"text": "安土城", "type": "施設名", "start": 5, "end": 8},
    {"text": "本能寺の変", "type": "イベント名", "start": 11, "end": 16},
    {"text": "明智光秀", "type": "人名", "start": 17, "end": 21}
  ]
}
```

### gold_spans 변환

HuggingFace 원본 데이터의 `entities` 필드를 `gold_spans` 형식으로 변환한다 (`dataset_loader.py:63-86`):

```python
for entity in entities:
    span = entity.get("span", [0, 0])
    gold_spans.append({
        "text": entity.get("name", ""),
        "type": entity.get("type", ""),
        "start": span[0],
        "end": span[1],
    })
```

### 한국어와의 차이

| 항목 | 한국어 (KLUE) | 일본어 (Stockmark) |
|------|---------------|-------------------|
| 데이터 형식 | 음절 단위 BIO 태그 | 문자 오프셋 span |
| gold 데이터 | `{tokens, ner_tags}` | `{text, gold_spans}` |
| 텍스트 접근 | `sentence` 필드 또는 토큰 재결합 | `text` 필드 직접 사용 |
| split 제공 | validation split 존재 | train만 존재 (자체 분할) |
| 로딩 폴백 | JSONL 파일 우선 | HuggingFace 직접 로딩만 |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 문자 오프셋 span 형식 유지 | Stockmark 원본 데이터가 offset span이므로, BIO 변환 없이 직접 활용하는 것이 정보 손실 최소화 |
| `train_test_split(seed=42)` | 재현 가능한 분할 보장. Stockmark에 공식 test split이 없으므로 직접 생성 |
| JSONL 폴백 불필요 | 일본어 환경에서 HuggingFace datasets 호환성 문제 없음 |

---

## 3. Stage 2: NER Tag 목록 및 프롬프트 설계

### 엔티티 타입 정의

**코드:** `src/labelers/ja/ner_prompts.py:7`

| 태그 | 의미 | 설명 | 예시 |
|------|------|------|------|
| 人名 | 인물명 | 풀네임, 성만, 이름만 | "織田信長", "田中", "太郎" |
| 法人名 | 법인명 | 기업, 회사, 철도회사, 방송국 | "トヨタ自動車", "JR東日本" |
| 地名 | 지명 | 국가, 지역, 도시, 자연지명 | "東京", "日本", "富士山" |
| 施設名 | 시설명 | 건물, 역, 공항, 점포, 연구소, 도서관, 미술관, 병원 | "東京タワー", "東京駅", "イオン上田店" |
| 製品名 | 제품명 | 상품, 서비스, 소프트웨어, 작품명, 프로그램명 | "iPhone", "Windows", "プリウス" |
| イベント名 | 이벤트명 | 행사, 대회, 사건, 전쟁, 조약 | "オリンピック", "関ヶ原の戦い" |
| 政治的組織名 | 정치적 조직명 | 정당, 정부기관, 국제기관, 군대, 재판소, 의회 | "自民党", "国連", "財務省", "米軍" |
| その他の組織名 | 기타 조직명 | 위에 해당하지 않는 단체, 스포츠리그, 스포츠팀, 대학 | "早稲田大学", "FIFA", "FCバルセロナ" |

### 핵심 라벨링 규칙

프롬프트에 포함된 4가지 핵심 규칙 (`ner_prompts.py:40-44`):

1. **조사 제외**: は/が/を/に/で/と/の/へ/から/まで/も/や -- 반드시 제거
2. **JSON 배열만 출력**: 다른 설명 금지
3. **빈 결과**: 고유표현 없으면 `[]` 반환
4. **복합 명칭**: 분리하지 않고 하나로 묶기

### 일본어 특화 분류 규칙

프롬프트에는 경계 케이스에 대한 상세한 분류 지침이 포함되어 있다 (`ner_prompts.py:14-37`):

| 규칙 | 설명 | 예시 |
|------|------|------|
| 철도회사 = 法人名 | "○○鉄道"는 施設名이 아님 | "JR東日本" → 法人名 |
| 연구소/도서관/병원 = 施設名 | "○○研究所", "○○病院"은 법인이 아님 | "財政研究所" → 施設名 |
| 점포 = 施設名 | "○○店"은 法人名이 아님 | "イオン上田店" → 施設名 |
| 스포츠리그 = その他の組織名 | イベント名이 아님 | "ラ・リーガ", "Jリーグ" → その他の組織名 |
| 군대/군사조직 = 政治的組織名 | "○○軍", "○○師団" | "米軍", "第57狙撃師団" → 政治的組織名 |
| 정부기관 = 政治的組織名 | "○○庁", "○○省", "○○院" | "財務省" → 政治的組織名 |

### 프롬프트 구조

한국어와 동일하게 3종류의 프롬프트 템플릿을 사용한다:

| 템플릿 | 변수명 | 용도 | 사용 클래스 |
|--------|--------|------|------------|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 단일 문장 라벨링 | OllamaNERLabeler, VllmNERLabeler |
| BATCH | `BATCH_PROMPT_TEMPLATE` | 다중 문장 배치 라벨링 | OllamaNERLabeler |
| SYSTEM+USER | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` | OpenAI 채팅 형식 | OpenAINERLabeler |

### Few-shot 예시의 설계 의도

프롬프트에 7개의 예시가 포함되어 있다 (`ner_prompts.py:46-66`). 각 예시가 커버하는 케이스:

| 예시 | 커버하는 케이스 |
|------|----------------|
| 1. 織田信長は安土城を建て... | 인물(人名), 시설(施設名), 이벤트(イベント名), 조사 제외(は/を/で/に) |
| 2. トヨタ自動車は東京モーターショーで... | 법인(法人名), 이벤트(イベント名), 제품(製品名) |
| 3. 国連の安全保障理事会は... | 정치적 조직(政治的組織名) 복수, 지명(地名), 시설(施設名) |
| 4. 元々は北海道東海大学の... | 법인(法人名)으로서의 대학, 시설(施設名)으로서의 캠퍼스 |
| 5. 神栖郵便局は茨城県... | 시설(施設名)으로서의 우체국, 복합 지명(地名) |
| 6. 国旗団は第一次世界大戦... | 기타 조직(その他の組織名), 이벤트(イベント名) |
| 7. 藤岡高校から高崎鉄道管理局を経て... | 시설(施設名)으로서의 학교, 정치적 조직(政治的組織名)으로서의 관리국, 기타 조직(その他の組織名)으로서의 스포츠팀 |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 8 엔티티 타입 사용 | Stockmark 데이터셋의 원본 타입 체계를 그대로 따름 (한국어 KLUE의 6 타입과 상이) |
| 일본어 태그명 사용 | 한국어(PS/LC/OG 등 영어 약어)와 달리, 일본어 원문 태그명을 그대로 사용하여 프롬프트-골드 데이터 간 태그 정규화 불필요 |
| 상세한 경계 분류 규칙 | 法人名/施設名/その他の組織名 간 경계가 모호하므로 명시적 규칙 필요 (예: 철도회사, 점포, 스포츠리그) |
| 7개 Few-shot | 8개 엔티티 타입을 모두 커버하면서 경계 케이스를 학습시키기 위함 |

---

## 4. Stage 3: LLM 라벨링 실행

### 전체 흐름

Canonical 구현: `OllamaNERLabeler` (`src/labelers/ja/ollama_ner_labeler.py`)

```
입력 텍스트
    |
    v _split_sentences()             (1) 문장 분리
문장 리스트
    |
    v batch_size 단위 그룹핑          (2) 배치 구성
배치 리스트
    |
    v _call_llm_batch()              (3) LLM 호출
    |   +-- 단일 문장 -> _call_llm() 직접 호출
    |   +-- 다중 문장 -> BATCH 프롬프트 사용
    |
    v JSON 파싱                      (4) 응답 처리
[{"text": "東京", "type": "地名"}, ...]
    |
    v label_spans() 경로: 여기서 반환 (raw spans)
    |
    v match_spans()                  (5) 문자 오프셋 매칭 [span_matcher.py]
[{"text": "東京", "type": "地名", "start": 5, "end": 7}]
```

**한국어와의 핵심 차이**: `label()` 경로의 `_spans_to_bio()` 변환 대신, `label_spans()` 경로로 raw span을 반환한 뒤 `match_spans()`로 문자 오프셋을 부여한다. 벤치마크 평가 시 BIO 변환이 전혀 발생하지 않는다.

### 4.1 문장 분리

**코드:** `_split_sentences()` (`ja/ollama_ner_labeler.py:24-39`)

```python
parts = re.split(r'(?<=[.!?。！？])\s*|\n+', text.strip())
```

한국어 버전과의 차이:
- 일본어 문장 부호 추가: `。` (마침표), `！` (느낌표), `？` (물음표)
- 분리 후 공백 처리: `\s*` (한국어는 `\s+`)
- 최소 10자 버퍼링은 동일

### 4.2 배치 구성

**코드:** `label()` 메서드 (`ja/ollama_ner_labeler.py:111-141`)

- `batch_size` 파라미터 (기본값: 10)로 문장 그룹핑
- 빈 문장 사전 필터링
- 타이밍 계측 출력 (`[TIMER]`)

### 4.3 LLM 호출

**코드:** `OllamaNERLabeler.__init__()` (`ja/ollama_ner_labeler.py:88-109`)

```python
self._llm = ChatOllama(
    model=model,
    format="json",       # JSON 출력 강제
    temperature=0,       # 결정적 출력
    think=False,         # thinking 비활성화
    reasoning=False,
)
```

한국어 버전과 동일한 설정을 사용한다.

### 4.4 JSON 파싱

두 가지 파싱 경로가 존재한다:

**경로 A -- OllamaNERLabeler 인라인 파싱** (`ja/ollama_ner_labeler.py:199-224`):

```python
data = json.loads(raw)
if isinstance(data, list):                    # [{"text":..., "type":...}, ...]
    return data
if isinstance(data, dict):
    if "text" in data and "type" in data:     # 단일 엔티티 bare object
        return [data]
    for val in data.values():                 # {"entities": [...]} 래퍼
        if isinstance(val, list):
            return val
```

**경로 B -- VllmNERLabeler의 `_parse_spans()`** (`ja/vllm_ner_labeler.py:55-79`):

`<think>` 태그 제거 후 JSON 파싱. 실패 시 정규식으로 `[...]` 추출:

```python
raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
# json.loads 시도 후 실패 시 정규식 폴백
match = re.search(r"\[.*?\]", raw, re.DOTALL)
```

**폴백 전략**: 배치 호출 실패 시 개별 문장 단위 재시도 (`ja/ollama_ner_labeler.py:195-197`):

```python
except (json.JSONDecodeError, Exception) as e:
    return [self._call_llm(s) for s in sentences]
```

### 4.5 span_matcher.py -- 문자 오프셋 매칭

**코드:** `match_spans()` (`src/labelers/ja/span_matcher.py:19-87`)

LLM은 `{"text": "東京", "type": "地名"}` 형태로 위치 정보 없이 엔티티를 반환한다. `match_spans()`는 원본 텍스트에서 해당 문자열의 문자 오프셋을 찾아 `{"text", "type", "start", "end"}`를 반환한다.

**알고리즘:**

1. **길이 역순 정렬** (line 32-37): 긴 엔티티를 먼저 매칭하여 부분 문자열 충돌 방지
2. **3단계 매칭 시도**:
   - Exact match: `_find_all_occurrences(original_text, text)` (line 49)
   - 공백 제거 매칭: LLM이 삽입한 불필요 공백 제거 후 재시도 (line 52-56)
   - **조사 제거 매칭**: `_strip_particles(text)` 후 재시도 (line 59-65)
3. **충돌 방지** (line 69-79): `consumed` 집합으로 이미 매칭된 문자 인덱스를 추적, 겹치지 않는 최좌측 후보 선택
4. **원래 순서 복원** (line 86-87): 정렬을 해제하고 LLM 출력 순서로 복원

### 4.6 일본어 조사 제거

**코드:** `_strip_particles()` (`src/labelers/ja/span_matcher.py:95-105`)

LLM이 엔티티 끝에 조사나 경칭을 포함할 수 있다. 이를 제거하여 매칭 성공률을 높인다:

```python
_PARTICLES = ("は", "が", "を", "に", "で", "と", "の", "へ", "から", "まで", "も", "や", "より")
_SUFFIXES = ("氏", "さん", "君", "ちゃん", "様")
```

- 조사는 길이 역순으로 처리 (예: "から"를 "か"보다 먼저 시도)
- 텍스트 길이가 조사/접미사보다 긴 경우에만 제거 (1글자 엔티티 보호)
- 조사와 접미사는 각각 최대 1개씩만 제거

### 4.7 EnhancedLabeler -- 자기 일관성 투표 및 2패스 검증

**코드:** `EnhancedLabeler` (`src/labelers/ja/enhanced_labeler.py:14-152`)

일본어 파이프라인에만 존재하는 고급 라벨링 래퍼. 기본 라벨러(`label_spans()` 메서드를 가진 모든 라벨러)를 감싸서 두 가지 기법을 적용한다:

**자기 일관성 투표 (Self-Consistency Voting)** (`enhanced_labeler.py:50-78`):

```python
# voting_rounds 횟수만큼 반복 추론
for _ in range(self.voting_rounds):
    spans = self.base.label_spans(text)
    round_entities.add((text, type))  # (텍스트, 타입) 튜플로 정규화

# 과반수 이상 등장한 엔티티만 유지
threshold = self.voting_rounds / 2
winners = [e for e, count in counter.items() if count > threshold]
```

- `voting_rounds=1`이면 비활성화 (단일 추론)
- 다수결 원칙: 전체 라운드의 50% 초과 등장해야 채택

**2패스 검증 (Two-Pass Verification)** (`enhanced_labeler.py:80-116`):

```python
verify_prompt = (
    f"以下のテキストから抽出された固有表現が正しいか検証してください。\n"
    f"正しいものだけをJSON配列で返してください。\n\n"
    f"テキスト: {text}\n"
    f"抽出された固有表現: {entities_str}\n\n"
    f"正しい固有表現のみ出力（JSON配列）:"
)
```

- 1패스에서 추출된 엔티티를 2패스에서 LLM에게 검증 요청
- 올바른 엔티티만 JSON 배열로 반환하도록 요청 (false positive 필터링)
- 검증 실패 시 원래 span을 그대로 반환 (폴백)
- Ollama와 vLLM 백엔드 모두 지원 (`_call_llm` 또는 `_chat` 메서드 자동 감지, line 99-108)

### 백엔드 변형

동일 인터페이스(`label()`, `label_spans()`)로 3개 백엔드를 지원한다:

| 파일 | 클래스 | 특징 |
|------|--------|------|
| `ja/ollama_ner_labeler.py` | `OllamaNERLabeler` | 동기 처리, ChatOllama, 배치 지원 |
| `ja/vllm_ner_labeler.py` | `VllmNERLabeler` | **async 동시성** (`concurrency` 파라미터, 기본 32), AsyncOpenAI |
| `ja/openai_ner_labeler.py` | `OpenAINERLabeler` | **system/user 채팅 형식**, 토큰 기반 배치 분할 |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| `label_spans()` 경로를 주 경로로 사용 | Stockmark 데이터가 offset span이므로 BIO 변환이 불필요 |
| `match_spans()` 별도 모듈 분리 | LLM 출력(위치 없음)과 gold 데이터(위치 있음) 간 브릿지 역할. 재사용 가능한 독립 모듈 |
| 3단계 매칭 (exact -> 공백 제거 -> 조사 제거) | 일본어 조사/공백 문제를 단계적으로 해결하여 매칭률 극대화 |
| 길이 역순 매칭 | "東京タワー"와 "東京"이 모두 있을 때, 긴 엔티티를 먼저 매칭하여 부분 문자열 충돌 방지 |
| `consumed` 집합으로 중복 방지 | 동일 문자 범위에 여러 엔티티가 매칭되는 것을 원천 차단 |
| EnhancedLabeler를 래퍼로 구현 | 기존 라벨러를 수정하지 않고 투표/검증 기능을 선택적으로 추가 가능 |

---

## 5. Stage 4: 결과 확인 (평가)

### 평가 진입점

**코드:** `src/evaluators/__main__.py`

```bash
PYTHONPATH=src python -m evaluators --lang ja --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200
```

CLI가 수행하는 흐름:
1. `JapaneseDatasetLoader`로 gold records 로딩
2. `JaBenchmarkRunner` 생성 후 labeler 추가
3. `runner.run()` -> 각 labeler에 대해 `_run_single()` 실행
4. `JaReportGenerator`로 결과 출력

### 평가 흐름 상세

**코드:** `JaBenchmarkRunner._run_single()` (`src/evaluators/ja_benchmark_runner.py:57-121`)

각 gold record에 대해:

```
gold record {id, text, gold_spans}
    |
    +-- (1) 텍스트 추출: record["text"] 그대로 사용
    |
    +-- (2) LLM 라벨링: labeler.label_spans(text)
    |       -> [{"text": "東京", "type": "地名"}, ...]  (raw spans)
    |
    +-- (3) span 매칭: match_spans(text, raw_spans)
    |       -> [{"text": "東京", "type": "地名", "start": 5, "end": 7}]
    |
    +-- (4) gold/pred span 수집
    |
    +-- (5) compute_offset_span_f1(gold_spans_all, pred_spans_all)
```

### 평가 메트릭: compute_offset_span_f1()

**코드:** `src/evaluators/ja_metrics.py:9-54`

문자 오프셋 기반 span-level F1을 계산한다. seqeval을 사용하지 않는다.

**매칭 기준:**

```python
# (sent_idx, start, end, type) 4-튜플이 정확히 일치해야 TP
key = (sent_idx, s["start"], s["end"], s["type"])
```

- **Exact match only**: start, end, type이 모두 일치해야 함
- **Micro-average**: 전체 엔티티에 대한 TP/FP/FN 기반 P/R/F1
- **Per-entity breakdown**: 엔티티 타입별 개별 P/R/F1/support

```python
tp = len(all_gold & all_pred)          # 정확히 일치하는 span 수
precision = tp / len(all_pred)          # 예측 중 정확한 비율
recall = tp / len(all_gold)             # gold 중 찾은 비율
f1 = 2 * precision * recall / (precision + recall)
```

### 한국어 평가와의 차이

| 항목 | 한국어 | 일본어 |
|------|--------|--------|
| 주 평가 메트릭 | Span Match (exact/relaxed) | Offset Span F1 (exact only) |
| 보조 메트릭 | seqeval BIO F1, Character Span F1, BERTScore | 없음 (단일 메트릭) |
| 매칭 단위 | entity text + type | character offset (start, end) + type |
| BIO 변환 필요 | 예 (spans_to_syllable_bio) | 아니오 |
| TagAligner 필요 | 예 | 아니오 |
| relaxed matching | 예 (부분 겹침 허용) | 아니오 (exact only) |

### 결과 출력

**코드:** `JaReportGenerator` (`src/evaluators/ja_benchmark_runner.py:124-224`)

- CLI 테이블: 모델별 F1/Precision/Recall + 속도 + 토큰 사용량
- Per-entity breakdown: 8개 엔티티 타입별 개별 성능
- JSON 파일: `results/ja_benchmark_*.json`

```json
{
  "timestamp": "2026-04-09T...",
  "language": "ja",
  "results": [{
    "model_name": "Qwen/Qwen3.5-27B",
    "backend": "vllm",
    "metrics": {
      "span_f1": {
        "overall": {"f1": 0.72, "precision": 0.75, "recall": 0.69, "support": 1234},
        "per_entity": {
          "人名": {"f1": 0.85, ...},
          "法人名": {"f1": 0.70, ...},
          ...
        }
      }
    },
    "latency": {"total_seconds": 180.5, "samples_per_second": 1.11}
  }]
}
```

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| Offset Span F1 단일 메트릭 | gold 데이터가 offset span이므로 BIO 변환 없이 직접 비교 가능. 변환 노이즈 제거 |
| seqeval 미사용 | BIO 태그 자체가 존재하지 않으므로 seqeval 적용 불가 |
| relaxed matching 미지원 | 문자 오프셋은 명확한 경계를 가지므로, relaxed 매칭보다 exact가 더 의미 있음 |
| BERTScore 미사용 | span 기반 평가에서 의미적 유사도 평가의 필요성이 낮음 |
| 단일 메트릭의 단순성 | 평가 파이프라인 복잡도를 줄여 유지보수 용이 |

---

## 6. 한국어 파이프라인과의 차이점 요약

| 항목 | 한국어 (KLUE) | 일본어 (Stockmark) |
|------|---------------|-------------------|
| **데이터셋** | KLUE NER (klue/ner) | Stockmark NER Wikipedia (stockmark/ner-wikipedia-dataset) |
| **엔티티 수** | 6 타입 (PS, LC, OG, DT, TI, QT) | 8 타입 (人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名) |
| **태그 표기** | 영어 약어 (PS, LC, OG...) | 일본어 원문 (人名, 法人名, 地名...) |
| **gold 데이터 형식** | 음절 BIO 태그 (`tokens` + `ner_tags`) | 문자 오프셋 span (`text` + `gold_spans: [{start, end, type}]`) |
| **데이터 로딩** | JSONL 폴백 우선, HuggingFace 폴백 | HuggingFace 직접 로딩 |
| **split 전략** | validation split 사용 | `train_test_split(test_size=0.2, seed=42)` 자체 분할 |
| **조사 처리** | 프롬프트 규칙 + `_spans_to_bio()` substring match | 프롬프트 규칙 + `span_matcher._strip_particles()` 조사 제거 |
| **조사 목록** | 은/는/이/가/을/를/에/에서/으로/의/과/와/부터/까지 | は/が/を/に/で/と/の/へ/から/まで/も/や/より + 경칭(氏/さん/君/ちゃん/様) |
| **span -> 태그 변환** | `_spans_to_bio()` (BIO 태그 생성) | `match_spans()` (문자 오프셋 부여, BIO 변환 없음) |
| **태그 정규화** | 필요 (PER->PS, LOC->LC 등) | 불필요 (일본어 태그명 그대로 사용) |
| **평가 메트릭** | Span Match + seqeval + Character Span F1 + BERTScore (4종) | Offset Span F1 (1종) |
| **평가 라이브러리** | seqeval, bert-score 의존 | 자체 구현 (`ja_metrics.py`), 외부 의존 없음 |
| **TagAligner** | 필수 (음절 BIO 변환) | 미사용 |
| **EnhancedLabeler** | 없음 | 있음 (자기 일관성 투표 + 2패스 검증) |
| **주요 모듈** | `dataset_loader.py`, `labelers/tag_aligner.py`, `metrics.py`, `benchmark_runner.py` | `ja/dataset_loader.py`, `ja/span_matcher.py`, `ja_metrics.py`, `ja_benchmark_runner.py` |

---

## 7. 설계 의사결정 요약

전체 파이프라인에서 내려진 핵심 설계 결정들:

| # | 단계 | 결정 | 이유 | 대안 (채택하지 않은 것) |
|---|------|------|------|----------------------|
| 1 | 데이터 | 문자 오프셋 span 형식 유지 | Stockmark 원본 형식 활용, BIO 변환 정보 손실 방지 | BIO 태그로 변환 (불필요한 복잡도) |
| 2 | 데이터 | `train_test_split(seed=42)` | 재현 가능한 분할, 공식 test split 부재 | 전체 데이터 사용 (과적합 위험) |
| 3 | 프롬프트 | 일본어 태그명 그대로 사용 | 태그 정규화 단계 제거, 프롬프트-골드 데이터 일치 | 영어 약어 (정규화 필요) |
| 4 | 프롬프트 | 8개 엔티티 + 상세 경계 규칙 | Stockmark 타입 체계 준수, 법인/시설/조직 경계 명확화 | 6개 타입으로 축소 (정보 손실) |
| 5 | 프롬프트 | 7개 Few-shot 예시 | 8개 엔티티 타입 전체 커버 + 경계 케이스 학습 | Zero-shot (정확도 저하) |
| 6 | 라벨링 | `match_spans()` 별도 모듈 | LLM 출력(위치 없음)과 gold(위치 있음) 간 독립적 브릿지 | BIO 변환 후 비교 (변환 노이즈) |
| 7 | 라벨링 | 길이 역순 + consumed 추적 | 부분 문자열 충돌 및 중복 매칭 방지 | 출현 순서대로 매칭 (충돌 위험) |
| 8 | 라벨링 | 조사 제거 폴백 (3단계 매칭) | 일본어 조사 부착 문제를 단계적으로 해결 | exact match만 (매칭률 저하) |
| 9 | 라벨링 | EnhancedLabeler 래퍼 패턴 | 기존 라벨러 무변경으로 투표/검증 기능 추가 | 각 라벨러에 직접 구현 (코드 중복) |
| 10 | 평가 | Offset Span F1 단일 메트릭 | BIO 변환 불필요, 평가 파이프라인 단순화 | 다중 메트릭 (불필요한 복잡도) |
| 11 | 평가 | seqeval/BERTScore 미사용 | BIO 태그가 없으므로 적용 불가/불필요 | BIO 변환 후 seqeval (변환 노이즈 유입) |
