# 일본어 NER 라벨링 방법론

> 대상 데이터셋: Stockmark NER Wikipedia (canonical 10종 평면 = 5종 NER + 5종 PII/날짜)
> 대상 코드: `src/ner/labelers/ja/`, `src/ner/llm_eval/`, `src/ner/metrics/`

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
canonical JSONL 덤프 (data/stockmark/{train,test}.jsonl)
    |
    v JapaneseDatasetLoader.load()                  [ja/dataset_loader.py]
List[dict] {id, text, gold_spans:[{text, type, start, end}]}
    |
    v record["text"] (원본 텍스트 그대로 사용)
원본 텍스트 문자열                                    [llm_eval/benchmark_runner.py]
    |
    v _split_sentences() → 동시성 그룹/배치 구성     [labelers/base_vllm_labeler.py · base_openai_labeler.py]
문장 리스트
    |
    v ner_prompts.py (SINGLE / SYSTEM+USER 템플릿)
프롬프트 문자열                                       [ja/ner_prompts.py]
    |
    v vLLM AsyncOpenAI / OpenAI SDK (temperature=0, JSON mode)
LLM JSON 응답
    |
    v JSON 파싱 → raw spans
[{"text": "東京", "type": "LOC"}, ...]
    |
    v match_spans() — LLM 텍스트 span → 문자 오프셋 변환
[{"text": "東京", "type": "LOC", "start": 5, "end": 7}]   [ja/span_matcher.py]
    |
    v BenchmarkRunner._run_offset_span_async()      [llm_eval/benchmark_runner.py]
    |
    v compute_offset_span_f1()                      [metrics/span_metrics.py]
    |  (sent_idx, start, end, type) 튜플 기반 exact match
    |
    v ReportGenerator                                [llm_eval/report.py]
벤치마크 결과 (CLI 테이블 + JSON 파일)
```

**한국어 파이프라인과의 핵심 차이**: BIO 태그 변환 단계가 없다. LLM 출력 → `match_spans()` → 문자 오프셋 span → `compute_offset_span_f1()`로 직접 평가한다. seqeval, TagAligner, `_spans_to_bio()` 등 BIO 관련 모듈을 우회한다.

---

## 2. Stage 1: 데이터 준비

### 입력

canonical Stockmark JSONL 덤프를 그대로 읽는다. HF Hub 자동 로딩·라벨 매핑·폴백 모두 없다 — canonical 매핑은 1회성 마이그레이션 도구가 미리 수행하여 JSONL로 떨어뜨린다.

| 항목 | 값 |
|------|------|
| 입력 경로 | `data/stockmark/train.jsonl`, `data/stockmark/test.jsonl` (NER 5종) · `data/stockmark/pii_train.jsonl`, `data/stockmark/pii_test.jsonl` (PII 주입 10종) |
| 라인 스키마 | `{id, text, entities:[{label, start_char, end_char, text}]}` |
| 데이터 형식 | 원본 텍스트 + 문자 오프셋 span (BIO 아님) |
| 라벨 공간 | NER 5종 (`PER/LOC/ORG/PROD/EVT`) — PII 주입본은 10종 평면 (5종 + `DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`) |

**코드:** `JapaneseDatasetLoader.load()` (`src/ner/labelers/ja/dataset_loader.py`)

```python
loader = JapaneseDatasetLoader()
records = loader.load(split='test', max_samples=200)
# 또는 임의 경로
records = JapaneseDatasetLoader.load_local('data/.../pii_injected.jsonl')
```

파일이 없으면 `FileNotFoundError`. PII 주입 산출물처럼 `start/end`·`type` 키를 쓰는 변형도 동일 스키마로 흡수한다(`label|type`, `start_char|start`, `end_char|end` 양 표기 허용).

### 출력 스키마

`List[dict]` — 각 레코드는 다음 구조:

```json
{
  "id": "12345",
  "text": "織田信長は安土城を建て、本能寺の変で明智光秀に討たれた。",
  "gold_spans": [
    {"text": "織田信長", "type": "PER", "start": 0, "end": 4},
    {"text": "安土城",   "type": "ORG", "start": 5, "end": 8},
    {"text": "本能寺の変", "type": "EVT", "start": 11, "end": 16},
    {"text": "明智光秀", "type": "PER", "start": 17, "end": 21}
  ]
}
```

`gold_spans[*].type`은 canonical 10종 문자열을 그대로 담는다 — 평가·리포트 단에서 추가 정규화 없이 비교한다.

---

## 3. Stage 2: NER Tag 목록 및 프롬프트 설계

### 엔티티 타입 정의

**코드:** `src/ner/labelers/ja/ner_prompts.py` (`DEFAULT_ENTITY_TYPES`) — canonical 10종 평면 목록.

엔티티 정의·LOC/ORG 경계 규칙·HF 원본(`人名`/`法人名`/`施設名` 등) → canonical 매핑·모호 사례 결정표는 `docs/manual/data/canonical-entity-schema.md` 단일 출처.

### 핵심 라벨링 규칙

프롬프트(`ner_prompts.py:SINGLE_PROMPT_TEMPLATE`)에 명시된 5가지 규칙:

1. **조사 제외**: `は/が/を/に/で/と/の/へ/から/まで/も/や` — 반드시 제거
2. **JSON 배열만 출력**: 다른 설명 금지
3. **빈 결과**: 고유표현 없으면 `[]` 반환
4. **복합 명칭**: 분리하지 않고 하나로 묶기
5. **라벨은 영문 기호만**: `PER/LOC/ORG/PROD/EVT/EMAIL/PHONE/DAT/ID_NUM/CREDIT_CARD` — 일본어 라벨(`人名` 등)을 그대로 출력하지 않는다

### 판정 우선순위 (모호할 때 위에서 아래)

| 순위 | 판정 | 적용 대상 | 예시 |
|---|---|---|---|
| 1 | **ORG** | 조직·법인·공권력·인공 시설 모두 — 기업·대학(본체·캠퍼스·부속시설)·정당·정부·군·국제기관·스포츠팀/리그·협회 + 역·공항·병원·학교·점포·박물관·寺·神社 | `早稲田大学`, `財務省`, `米軍`, `NHK`, `FCバルセロナ`, `東京駅`, `セントメアリー病院` |
| 2 | **LOC** | 지리적 위치만 — 국가·행정구역·자연지명·주소 | `富士山`, `東京都千代田区`, `茨城県神栖市` |
| 3 | **PROD** | 상품·서비스·작품·방송 프로그램 | `iPhone`, `プリウス`, `NHKスペシャル` |
| 4 | **EVT** | 일회성 행사·전쟁·조약·대회 | `関ヶ原の戦い`, `第45回NHK紅白歌合戦` |

추가 규칙:
- **대학**: 본체·캠퍼스·부속시설 모두 ORG. `○○大学病院`도 ORG.
- **행정구역 복합체**: `東京都千代田区` 같은 복합체는 단일 LOC 한 덩어리.
- **복합 조직명**: `리그명+팀명`은 분리하지 않고 한 덩어리.
- **번지 포함 주소**: `東京都千代田区1丁目2-3 ビル7F`처럼 번지·층 포함도 LOC (별도 ADDRESS 타입 없음).
- **일본어 라벨 leakage 금지**: `人名/地名/施設名` 같은 원어 라벨을 출력에 그대로 흘리지 않는다.

### 프롬프트 구조

3종 템플릿이 백엔드별로 다르게 쓰인다:

| 템플릿 | 변수명 | 용도 | 사용 클래스 |
|--------|--------|------|------------|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 문장 단위 라벨링 (concurrency로 다중 호출) | `VllmNERLabeler` |
| SYSTEM+USER | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` | OpenAI 채팅 형식 (한 호출에 다문 묶음) | `OpenAINERLabeler` |

### Few-shot 예시 구성

`SINGLE_PROMPT_TEMPLATE`에 11개 예시가 포함되어 있다.

| 그룹 | 예시 | 커버하는 케이스 |
|------|------|----------------|
| 메인 8종 | `国連の安全保障理事会はニューヨークの国連本部で…` | ORG 복수 + LOC |
| | `元々は北海道東海大学の旭川キャンパス。` | 대학 본체·캠퍼스 모두 ORG |
| | `神栖郵便局は茨城県神栖市にある郵便局。` | 우체국 = ORG, 행정구역 = LOC |
| | `国旗団は第一次世界大戦に出征した退役軍人の連合会…` | ORG + EVT |
| | `藤岡高校から高崎鉄道管理局を経て、大映スターズへ入団。` | 학교·관리국·스포츠팀 모두 ORG |
| | `ラ・リーガのFCバルセロナはチャンピオンズリーグで優勝した。` | 리그·팀 모두 ORG |
| | `東京駅から成田空港まで電車で移動した。` | 역·공항 = ORG |
| | `富士山は日本最高峰で、東京から見える。` | 자연지명·국가·도시 = LOC |
| PII 종합 | `担当は山田太郎（1985年4月3日生）。連絡先：090-…、メール：…` | PER + DAT + PHONE + EMAIL + LOC + CREDIT_CARD + ID_NUM |
| DAT/LOC | `2016年1月29日、セルティックFCに移籍した。住所：…ビル7F。` | 문맥무관 DAT, 번지·층 포함 주소 LOC |
| | `田中氏は1985年に生まれた。福島県福島市にある美術館を運営。` | 연도 단독 DAT, 행정구역 LOC |
| EMAIL 카운터 | `連絡先はabc@example.comです。メール：hanako@yahoo.co.jp。` | TLD 누락(`@example`, `@yahoo.co`) 금지 |

OpenAI 백엔드의 `SYSTEM_PROMPT`는 합성 PII 벤치마크에서 거부 응답을 회피하기 위한 안전 도입부(인공 합성 데이터·정보 추출 전용 명시)를 추가로 포함한다.

---

## 4. Stage 3: LLM 라벨링 실행

### 전체 흐름

JA 라벨러(`VllmNERLabeler`, `OpenAINERLabeler`)는 공통 베이스(`labelers/base_vllm_labeler.py`, `base_openai_labeler.py`)의 얇은 서브클래스다. JA 별도 파일은 프롬프트 템플릿·기본 모델명·`lang='ja'` 만 주입한다.

```
입력 텍스트
    |
    v _split_sentences()             (1) 문장 분리
문장 리스트
    |
    v 백엔드별 호출 단위 구성          (2) 분할
    |   ├─ vLLM:    문장 단위 SINGLE 프롬프트 × concurrency 동시 호출
    |   └─ OpenAI:  max_tokens_per_batch 토큰 한도로 문장 묶음 →
    |               SYSTEM + USER(다문) × concurrency 동시 호출
    |
    v LLM 호출                       (3)
    |
    v JSON 파싱                      (4) 응답 처리
[{"text": "東京", "type": "LOC"}, ...]
    |
    v label_spans() / alabel_spans() 경로: raw spans 반환
    |
    v match_spans()                  (5) 문자 오프셋 매칭
[{"text": "東京", "type": "LOC", "start": 5, "end": 7}]
```

**한국어와의 핵심 차이**: `label()` 경로의 `_spans_to_bio()` 변환 대신, `label_spans()` 경로로 raw span을 반환한 뒤 `match_spans()`로 문자 오프셋을 부여한다. 벤치마크 평가 시 BIO 변환이 일어나지 않는다.

### 4.1 문장 분리

베이스 클래스의 `_split_sentences()`가 일본어 문장 부호(`。`, `！`, `？`)를 포함한 정규식으로 텍스트를 분리한다. 짧은 단편은 최소 길이 버퍼링으로 통합한다.

### 4.2 호출 단위 구성

- **vLLM**: `_split_sentences()` 결과 문장 각각을 SINGLE 프롬프트로 1회씩 호출. `concurrency`(기본 32)로 동시 실행.
- **OpenAI**: `_make_batches()`가 문장을 `max_tokens_per_batch`(기본 1000) 토큰 한도로 묶어 USER 프롬프트의 `{sentences}`에 주입. 배치 단위를 `concurrency`(기본 4)로 동시 실행.
- 빈 문장은 양쪽 모두 사전 필터링.

### 4.3 LLM 호출

| 백엔드 | 클라이언트 | 동시성 | 핵심 옵션 |
|--------|-----------|--------|----------|
| vllm | `AsyncOpenAI` (vLLM 호환 엔드포인트) | async (`concurrency`, 기본 32) | `temperature=0`, `<think>` 태그 strip (`thinking=False`) |
| openai | OpenAI SDK | async (`concurrency`, 기본 4) | system+user 채팅, 토큰 기준 배치 분할(`max_tokens_per_batch`) |

### 4.4 JSON 파싱

**vLLM 파서**: `<think>...</think>` 태그를 먼저 제거한 뒤 `json.loads(raw)` → 형태별 분기. JSON 파싱이 실패하면 정규식(`\[.*?\]`) 폴백으로 배열을 추출한다.

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

**OpenAI 파서** (`_parse_batch_response`): 응답이 `{"0": [...], "1": [...], ...}` 형태(문장 인덱스 → span 리스트)임을 가정하고 입력 문장 수만큼 분리·복원한다. 인덱스 누락·범위 초과는 빈 리스트로 채운다.

**배치 폴백**: 배치 호출 실패 시 묶음을 더 작게 쪼개거나 개별 문장 단위로 재시도한다.

### 4.5 span_matcher.py — 문자 오프셋 매칭

**코드:** `match_spans()` (`src/ner/labelers/ja/span_matcher.py`)

LLM은 `{"text": "東京", "type": "LOC"}` 형태로 위치 정보 없이 엔티티를 반환한다. `match_spans()`가 원문에서 해당 문자열의 문자 오프셋을 찾아 `{text, type, start, end}`를 부여한다.

**알고리즘:**

1. **길이 역순 정렬** — 긴 엔티티를 먼저 매칭하여 부분 문자열 충돌 방지
2. **3단계 매칭 시도**:
   - Exact match: `_find_all_occurrences(original_text, text)`
   - 공백 제거 매칭: LLM이 삽입한 불필요 공백 제거 후 재시도
   - 조사 제거 매칭: `_strip_particles(text)` 후 재시도
3. **충돌 방지**: `consumed` 집합으로 이미 매칭된 문자 인덱스를 추적, 겹치지 않는 최좌측 후보 선택
4. **원래 순서 복원**: 정렬을 해제하고 LLM 출력 순서로 복원

### 4.6 일본어 조사·경칭 제거

LLM이 엔티티 끝에 조사나 경칭을 포함할 수 있다. 매칭 성공률을 위해 `_strip_particles()`가 다음을 끝에서 1개씩 제거한다:

```python
_PARTICLES = ('は', 'が', 'を', 'に', 'で', 'と', 'の', 'へ',
              'から', 'まで', 'も', 'や', 'より')
_SUFFIXES  = ('氏', 'さん', '君', 'ちゃん', '様')
```

- 조사는 길이 역순 시도 (예: `から`를 `か`보다 먼저)
- 텍스트 길이가 조사·접미사보다 클 때만 제거 (1글자 엔티티 보호)

### 백엔드 변형

동일 인터페이스(`label()`, `label_spans()`/`alabel_spans()`)로 2개 백엔드를 지원한다:

| 파일 | 클래스 | 특징 |
|------|--------|------|
| `ja/vllm_ner_labeler.py` | `VllmNERLabeler` | AsyncOpenAI(vLLM 호환), 문장 단위 SINGLE 호출 + concurrency 동시 실행, `<think>` strip |
| `ja/openai_ner_labeler.py` | `OpenAINERLabeler` | OpenAI SDK, system/user 채팅, 토큰 기반 배치 분할, 합성 PII 안전 도입부 system prompt |

---

## 5. Stage 4: 결과 확인 (평가)

### 평가 진입점

**코드:** `src/ner/llm_eval/__main__.py`

```bash
python -m ner.llm_eval --lang ja --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200
```

CLI 흐름:
1. `JapaneseDatasetLoader`로 gold records 로딩 (`_load_gold` 분기)
2. 통합 `BenchmarkRunner`(`eval_mode='offset_span'`) 생성 후 labeler 추가
3. `runner.run()` → 각 labeler에 대해 `_run_offset_span()` 실행
   (내부적으로 `_run_offset_span_async()`로 sample 단위 비동기 처리)
4. 통합 `ReportGenerator`로 결과 출력 (`lang='ja'`)

### 평가 흐름

**코드:** `BenchmarkRunner._run_offset_span_async()` (`src/ner/llm_eval/benchmark_runner.py`)

각 gold record에 대해:

```
gold record {id, text, gold_spans}
    |
    ├─ (1) 텍스트 추출: record["text"] 그대로 사용
    |
    ├─ (2) LLM 라벨링: labeler.alabel_spans(text) 또는 label_spans
    |       → [{"text": "東京", "type": "LOC"}, ...]  (raw spans)
    |
    ├─ (3) span 매칭: match_spans(text, raw_spans)
    |       → [{"text": "東京", "type": "LOC", "start": 5, "end": 7}]
    |
    ├─ (4) gold/pred span 수집
    |
    └─ (5) compute_offset_span_f1(gold_spans_all, pred_spans_all)
```

### 평가 메트릭: compute_offset_span_f1()

**코드:** `src/ner/metrics/span_metrics.py`

문자 오프셋 기반 span-level F1. seqeval을 사용하지 않는다.

**매칭 기준:**

```python
# (sent_idx, start, end, type) 4-튜플이 정확히 일치해야 TP
key = (sent_idx, s["start"], s["end"], s["type"])
```

- **Exact match only**: start, end, type 모두 일치 시 TP
- **Micro-average**: 전체 엔티티에 대한 TP/FP/FN 기반 P/R/F1
- **Per-entity breakdown**: 엔티티 타입별 P/R/F1/support

```python
tp = len(all_gold & all_pred)           # 정확히 일치하는 span 수
precision = tp / len(all_pred)
recall    = tp / len(all_gold)
f1        = 2 * precision * recall / (precision + recall)
```

---

## 6. 한국어 파이프라인과의 차이점 요약

| 항목 | 한국어 (KLUE) | 일본어 (Stockmark) |
|------|---------------|-------------------|
| 데이터셋 | KLUE NER (`klue/ner`) | Stockmark NER Wikipedia (`stockmark/ner-wikipedia-dataset`) |
| 엔티티 수 | 6 타입 (PS, LC, OG, DT, TI, QT) | 10 타입 평면 (PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD) |
| 태그 표기 | KLUE 약어 | canonical 영문 (PER/LOC/ORG/PROD/EVT + PII) |
| gold 데이터 형식 | 음절 BIO (`tokens` + `ner_tags`) | 문자 오프셋 span (`text` + `gold_spans`) |
| 데이터 로딩 | JSONL 폴백 우선, HF 폴백 | canonical JSONL 덤프 전용 (HF 자동 로딩 없음) |
| split 제공 | validation split 사용 | canonical 덤프에 train/test 두 파일 분리 보관 |
| 조사 처리 | 프롬프트 규칙 + `_spans_to_bio()` substring match | 프롬프트 규칙 + `span_matcher._strip_particles()` |
| 조사·경칭 목록 | 은/는/이/가/을/를/에/에서/으로/의/과/와/부터/까지 | は/が/を/に/で/と/の/へ/から/まで/も/や/より + 경칭(氏/さん/君/ちゃん/様) |
| span → 태그 변환 | `_spans_to_bio()` (BIO 생성) | `match_spans()` (오프셋 부여, BIO 변환 없음) |
| 태그 정규화 | 필요 (PER→PS 등) | 불필요 (canonical 그대로) |
| 평가 메트릭 | Span Match + seqeval BIO F1 + Char Span F1 | Offset Span F1 단일 |
| 평가 라이브러리 | seqeval 의존 | 자체 구현 (`metrics/span_metrics.py`) |
| TagAligner | 필수 | 미사용 |
| 주요 모듈 | `dataset_loader.py`, `labelers/tag_aligner.py`, `metrics/bio_metrics.py`, `llm_eval/benchmark_runner.py` | `ja/dataset_loader.py`, `ja/span_matcher.py`, `metrics/span_metrics.py`, `llm_eval/benchmark_runner.py` |

---

## 7. 설계 의사결정 요약

| # | 단계 | 결정 | 이유 | 대안 (채택하지 않음) |
|---|------|------|------|----------------------|
| 1 | 데이터 | 문자 오프셋 span 형식 유지 | Stockmark 원본이 offset span. BIO 변환의 정보 손실 회피 | BIO 태그로 변환 (불필요한 복잡도) |
| 2 | 데이터 | canonical JSONL 덤프 전용 | 라벨 매핑·세이프티 정정 등 한 번 결정해야 할 변환을 1회성 도구에 격리. 런타임에서는 단순 읽기 | HF 자동 로딩 + 런타임 매핑 (변환 분산·재현성 약화) |
| 3 | 프롬프트 | canonical 영문 라벨 (10종 평면) | OntoNotes 관용 표기로 다국어 라벨 공간 일관성 확보 + NER/PII 구분 없는 평면 단순화 | 데이터셋 원어 라벨 그대로 (다국어 통합 곤란, leakage 위험) |
| 4 | 프롬프트 | LOC=지리만, ORG=인공시설 전부 | 접미사·운영체 룰로 봉합하던 LOC/FAC/CORP/POL/ORG 경계 모호성을 구조적으로 제거 | 시설/조직 분리 (경계 분쟁 누적, 평가 편차 큼) |
| 5 | 프롬프트 | 메인 8 + DAT/LOC 2 + EMAIL 카운터 1 = 11개 Few-shot | 5종 NER + 5종 PII 동시 커버 + EMAIL 누락·번지 주소 같은 함정 케이스 학습 | Zero-shot (정확도 저하) |
| 6 | 라벨링 | `match_spans()` 별도 모듈 | LLM 출력(위치 없음) ↔ gold(위치 있음) 간 독립적 브릿지 | BIO 변환 후 비교 (변환 노이즈) |
| 7 | 라벨링 | 길이 역순 + `consumed` 추적 | 부분 문자열 충돌·중복 매칭 방지 | 출현 순서대로 매칭 (충돌 위험) |
| 8 | 라벨링 | 3단계 매칭 (exact → 공백 제거 → 조사 제거) | 일본어 조사·공백 부착 문제를 단계적 해결 | exact만 (매칭률 저하) |
| 9 | 평가 | Offset Span F1 단일 메트릭 | gold가 offset span. BIO 변환 불필요, 변환 노이즈 제거 | 다중 메트릭 (불필요한 복잡도) |
| 10 | 평가 | seqeval 미사용 | BIO 태그 부재, offset span 직접 평가 | BIO 변환 후 seqeval (변환 노이즈 유입) |
