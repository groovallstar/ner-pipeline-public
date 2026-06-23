# 베트남어 NER 라벨링 방법론

> 대상 데이터셋: WikiANN Vietnamese (`unimelb-nlp/wikiann`, config=`vi`, 원본 3종 PER/LOC/ORG)
> 라벨러 출력 스키마: canonical 10종 평면 (NER 5종 + PII 5종)
> 라벨 정의의 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## 목차

1. [전체 파이프라인 흐름도](#1-전체-파이프라인-흐름도)
2. [Stage 1 — 데이터 준비](#2-stage-1--데이터-준비)
3. [Stage 2 — 엔티티 스키마와 WikiANN 매핑](#3-stage-2--엔티티-스키마와-wikiann-매핑)
4. [Stage 3 — 프롬프트 설계](#4-stage-3--프롬프트-설계)
5. [Stage 4 — LLM 라벨링 실행](#5-stage-4--llm-라벨링-실행)
6. [Stage 5 — 평가](#6-stage-5--평가)
7. [Silver 검증 전략 (재라벨 파이프라인)](#7-silver-검증-전략-재라벨-파이프라인)
8. [한국어·일본어 파이프라인과의 차이](#8-한국어일본어-파이프라인과의-차이)

---

## 1. 전체 파이프라인 흐름도

```
WikiANN Dataset (unimelb-nlp/wikiann, config=vi)
    │
    ▼ DatasetLoader.load()                          [labelers/dataset_loader.py]
List[NERRecord] {tokens, ner_tags, id}
    │
    ▼ TagAligner.reconstruct_text()                 [labelers/tag_aligner.py]
원본 텍스트 (word-level 토큰 space-join)
    │
    ▼ ner_prompts.py SINGLE/SYSTEM+USER 템플릿     [labelers/vi/ner_prompts.py]
프롬프트 (베트남어, 10종 라벨 공간)
    │
    ▼ vLLM 또는 OpenAI 호출                          [labelers/vi/{vllm,openai}_ner_labeler.py]
LLM JSON 응답 → spans 파싱
[{"text": "Nguyễn Xuân Phúc", "type": "PER"}, ...]
    │
    ▼ label_spans() 경로: raw spans 반환 (VI 평가 경로)
    │
    ▼ BenchmarkRunner._run_offset_span()            [llm_eval/benchmark_runner.py]
    ├── match_spans(text, raw_spans)  pred span 정합
    └── gold_spans (load_local/load)  gold span (entities → {type,start,end})
    │
    ▼ compute_offset_span_f1                         [metrics/span_metrics.py]
    │
    ▼ ReportGenerator                               [llm_eval/report.py]
벤치마크 결과 (CLI 테이블 + JSON)

(KO 는 label() → BIO 변환 → NERRecord → `_run_single` → seqeval 경로. VI·JA 는
위 offset_span 경로.)
```

재라벨 파이프라인 (silver 데이터셋 생성, §7) 은 별개의 흐름:

```
WikiANN split → augmenters/wikiann_vi/__main__.py (모델별 1회)
    → Gemma·Qwen 독립 재라벨 (각 gold_spans_relabel)
    → merge_confidence (recall_strict) → gold_spans_relabel_merged 단일 덤프
    → kappa + Wikidata anchor 검증
    → entities 스키마 변환 + split → data/wikiann_vi/{train,valid,test}.jsonl
    → augmenters/pii 주입 → data/wikiann_vi/pii_{train,valid,test,all}.jsonl
```

> merge_confidence 출력은 `gold_spans_relabel_merged`(type/start/end) 필드를 가진
> 덤프이며, 최종 `{train,valid,test}.jsonl`·`pii_*.jsonl` 은 `entities`
> (label/start_char/end_char) 스키마(§7.4)다. 빌드 셸 절차는
> `docs/reports/vietnamese-ner-silver-quality.md` §재현 참조.

---

## 2. Stage 1 — 데이터 준비

### 입력

| 경로 | 소스 | 조건 |
|---|---|---|
| JSONL | `data/wikiann_vi/test.jsonl` | 기본값; 파일 없으면 `FileNotFoundError` (HF 자동 폴백 없음) |

gold 로딩 진입점: `_load_gold()` in `src/ner/llm_eval/__main__.py`.
`_create_labeler_vi()`는 LLM 라벨러 생성 전용이며 데이터 로딩과 무관.
기본 split 은 `test` (KLUE 의 `validation` 과 다름 — WikiANN-vi 의 validation 은 작거나 없음).

### 출력 — `NERRecord`

```json
{
  "tokens": ["Nguyễn", "Văn", "A", "sinh", "ra", "tại", "Hà", "Nội"],
  "ner_tags": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "B-LOC", "I-LOC"],
  "id": "0"
}
```

> 위 `{tokens, ner_tags}` 는 WikiANN HF 원본(재라벨 파이프라인 입력)의 형식이다.
> **벤치마크 평가 gold** 는 `_load_gold()` 가 materialize 된 silver
> (`{train,valid,test}.jsonl`, entities) 를 `VietnameseDatasetLoader` 로 읽어
> `{text, gold_spans}`(offset_span) 로 공급한다(§6.2). 두 경로를 혼동하지 말 것.

### WikiANN 토큰화 특성

- **단어(word) 단위** 토큰화. KLUE 의 음절(syllable) 단위와 다름
- BIO 태그도 단어 단위로 부여
- 한국어 KLUE 의 음절→단어 정렬 문제 미발생

`ClassLabel` 정수 태그 자동 변환: `dataset_loader.py` 가 `ClassLabel.int2str()` 으로 0→"O", 1→"B-PER" 등 처리.

---

## 3. Stage 2 — 엔티티 스키마와 WikiANN 매핑

엔티티 정의·LOC/ORG 경계 규칙·WikiANN 3종 → canonical 5종 매핑·모호
사례 결정표는 `docs/manual/data/canonical-entity-schema.md` 단일 출처.

본 라벨러는 WikiANN 평가 외에 PII 주입본 평가와 코드를 공유하므로 10종
출력 공간(NER 5종 + PII 5종) 을 유지한다. WikiANN 3종 gold 로 평가 시
PROD/EVT/PII 5종 출력은 FP 로 잡혀 precision 이 인위적으로 하락하는데,
이는 의도된 trade-off (스키마 문서 §7.3 참조 — 본 항목만 본 문서 외부의
스키마 문서가 가진 평가 주의이므로 별도 명시).

---

## 4. Stage 3 — 프롬프트 설계

### 4.1 2종 프롬프트 템플릿

`src/ner/labelers/vi/ner_prompts.py` (BATCH 템플릿은 라벨러에 없고
`augmenters/wikiann_vi/prompts.py` 재라벨 파이프라인 전용):

| 템플릿 | 변수명 | 용도 | 형식 |
|---|---|---|---|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 단일 문장 라벨링 (vLLM) | `Đầu vào: {sentence}\nĐầu ra:` |
| SYSTEM+USER | `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` | OpenAI 채팅 형식 (다문 묶음) | system/user 메시지 분리 |

### 4.2 6가지 핵심 라벨링 규칙

1. **원문 보존** — 베트남어 성조 부호(dấu) 정확히 유지 ("Hà Nội" → "Ha Noi" 변환 금지)
2. **다중 단어 엔티티 통합** — "Thành phố Hồ Chí Minh" → 1개 LOC, "Đại học Quốc gia Hà Nội" → 1개 ORG
3. **직함·호칭 제외** — "Ông", "Bà", "Chủ tịch", "Thủ tướng", "Tướng", "GS" 등 제거 후 이름만
4. **JSON 배열만 출력** — 설명 없이 array 만
5. **빈 결과** — 엔티티 없으면 `[]`
6. **영문 태그 사용** — `PER`/`LOC`/`ORG`/`PROD`/`EVT`/`EMAIL`/`PHONE`/`DAT`/`ID_NUM`/`CREDIT_CARD`

### 4.3 PROD/EVT 분기 규칙 (silver 재라벨 프롬프트 추가)

silver 재라벨 (`augmenters/wikiann_vi/prompts.py`) 에는 `canonical-entity-schema.md` §2~3 의 경계 규칙을 프롬프트에 추가 명시:

- 정기 리그/대회 → ORG, 특정 연도판 → EVT
- 작품(음악·영화·책·만화·게임·TV) → PROD
- "Danh sách..." 글 무시
- Latin binomial 학명 무시
- 모델 번호 단독 무시 (브랜드+결합만)
- 인프라 (철도·지하철 노선): 운영=ORG, 노선=LOC

### 4.4 Few-shot 예시

`src/ner/labelers/vi/ner_prompts.py` SINGLE 템플릿에 약 10개 예시 (메인 9 + PII 종합 1):
- PER + LOC + ORG 혼합 (`Chủ tịch Nguyễn Xuân Phúc...`)
- 정당 + 정부기관 + 대학 (`Đảng Cộng sản Việt Nam và Bộ Giáo dục...`)
- 시설 (공항·병원) (`Vietnam Airlines vận hành chuyến bay...`)
- 스포츠 (`Hà Nội FC giành chức vô địch V.League...`)
- 1회성 사건 (`Trong Chiến tranh Việt Nam, Hiệp định Paris...`)
- 제품·소프트웨어 (`Samsung giới thiệu Galaxy S24...`)
- PII 종합 (`Phụ trách là Trần Minh ... CCCD: 079123456789.`)
- 자연지명 (`Vịnh Hạ Long là một kỳ quan thiên nhiên...`)
- 시설 + 자연지명 분리 (`Chùa Một Cột nằm ở quận Ba Đình, Hà Nội.`)
- 작품(음악) (`'' Diễm xưa '' ( Trịnh Công Sơn ).` → `Diễm xưa`=PROD, `Trịnh Công Sơn`=PER)

---

## 5. Stage 4 — LLM 라벨링 실행

### 5.1 백엔드 — vLLM + OpenAI 2종

| 파일 | 클래스 | 특징 |
|---|---|---|
| `src/ner/labelers/vi/vllm_ner_labeler.py` | `VllmNERLabeler` | async 동시성 (`concurrency` 파라미터, 기본 32), AsyncOpenAI 호환, vLLM 서버용 |
| `src/ner/labelers/vi/openai_ner_labeler.py` | `OpenAINERLabeler` | system/user 채팅 형식, `response_format={"type": "json_object"}` |
| `src/ner/labelers/hf_ner_labeler.py` | `HFNERLabeler` | HuggingFace BERT 베이스라인 (`lang="vi"`) |

라벨러 팩토리: `src/ner/llm_eval/__main__.py::_create_labeler_vi()`

### 5.2 단계별 처리

```
입력 텍스트
    ▼ split_sentences()        정규식 [.!?]+공백 으로 문장 분리
문장 리스트
    ▼ 백엔드별 호출 단위 구성
    │   ├─ vLLM:   문장 단위 SINGLE 프롬프트 × concurrency 동시 호출
    │   └─ OpenAI: _make_batches() 토큰 한도 묶음 → SYSTEM+USER × concurrency
    ▼ JSON 파싱                <think> 태그 제거 + json.loads + 정규식 [...] 폴백
[{"text": "...", "type": "..."}, ...]
    ├── label_spans() 경로: raw spans 반환 (VI 평가 경로)
    └── label() 경로: spans → BIO 변환 (KO 전용 경로, VI 미사용)
```

설정:
- `temperature` — vLLM `0.0`(결정성), OpenAI `1`(provider 기본값)
- `format=json` 또는 `response_format={"type": "json_object"}` — 유효 JSON 강제
- vLLM 컨테이너 `enable_thinking=false` — thinking 토큰이 JSON 파싱 방해 방지
- 폴백: 배치 호출 실패 시 개별 문장 단위 재시도

---

## 6. Stage 5 — 평가

### 6.1 CLI

```bash
python -m ner.llm_eval --lang vi --models "vllm:cyankiwi/gemma-4-31B-it-AWQ-8bit" --max-samples 200
```

### 6.2 평가 흐름

`src/ner/llm_eval/__main__.py` 단일 dispatch → `BenchmarkRunner`
(lang="vi", eval_mode="offset_span") `_run_offset_span` 실행:

```
gold record {text, gold_spans}            # _load_gold → load(split) / load_local
    ├── (1) LLM 라벨링: labeler.label_spans(text) → raw spans
    ├── (2) span 정합: match_spans(text, raw_spans) → pred_spans
    └── (3) 메트릭: compute_offset_span_f1(gold_spans, pred_spans)
```

gold 는 `data/wikiann_vi/test.jsonl`(entities) 또는 `--local-file` PII 주입본을
`VietnameseDatasetLoader` 가 `gold_spans` 스키마로 읽어 공급한다.

### 6.3 태그 정규화

`src/ner/labelers/tag_aligner.py::_TAG_NORMALIZE_MAP_VI` :

```python
{"PERSON": "PER", "LOCATION": "LOC", "ORGANIZATION": "ORG", "MISCELLANEOUS": "MISC"}
```

LLM 이 풀네임을 출력하는 경우 약어로 변환. KO 와 달리 별도 KLUE 약어 변환은 없음.

### 6.4 메트릭

VI 평가는 **offset_span 경로** (`eval_mode='offset_span'`) — JA 와 동일.
`_eval_mode_for_lang('vi') → 'offset_span'` 으로 dispatch 되며(KO 만 BIO),
gold·pred 모두 문자 오프셋 span(`{type, start, end}`)으로 직접 비교한다.

| 메트릭 | 역할 | 구현 |
|---|---|---|
| Offset Span F1 | char-offset span exact 매칭 (**Primary**) | `src/ner/metrics/span_metrics.py::compute_offset_span_f1` |

이 메트릭은 classifier(`train_eval.py`)와 **동일 함수**라 LLM 라벨러와 BERT
분류기 결과가 직접 비교 가능하다. seqeval/BIO 정렬(`TagAligner`·
`extract_spans_from_bio`)은 KO BIO 경로 전용이며 VI 에는 적용되지 않는다.

---

## 7. Silver 검증 전략 (재라벨 파이프라인)

`augmenters/wikiann_vi/` 가 WikiANN 3종 gold 를 LLM 으로 5종 silver 로 재라벨하는 별도 파이프라인. 결과는 `data/wikiann_vi/{train,valid,test}.jsonl` 로 저장 (Stockmark 포맷).

### 7.1 이중 silver 인식

- WikiANN 원본 = Wikipedia 인터링크 기반 자동 silver
- 본 5종 라벨 = WikiANN silver 를 LLM 이 재분류 → 추가 silver
- → 본 데이터셋은 이중 silver. 절대 F1 비교는 부적합. 모델 간 상대 순위·cross-model kappa·Wikidata anchor agreement 만 해석 대상.

### 7.2 3중 검증 레이어

1. **Cross-model agreement** — Gemma + Qwen 2개 LLM 독립 라벨링 → Cohen kappa + per-type 합의율 (`augmenters/wikiann_vi/kappa.py`)
2. **Wikidata anchor** — 베트남어 Wikipedia → Wikidata Q-ID → P31(instance of) 으로 canonical type 역추정 (`augmenters/wikiann_vi/wikidata_anchor.py`, 약 163개 Q-ID 매핑). 상세: `docs/manual/data/wikidata-anchor-verification.md`
3. **WikiANN 3종 gold 직접 비교** — silver PER/LOC/ORG 만 필터링해 gold 와 span F1 (`llm_eval/vi_silver_quality.py`)

### 7.3 confidence 카테고리 + 정책 (`merge_confidence.py`)

두 모델 출력 비교로 4 카테고리 분류 (`high`/`medium_recall`/`medium_prec`/`conflict`), 6 정책 중 선택(아래 5종 + EVT 구제 변형 `recall_strict_evt`). CLI `--policy` 기본값은 `recall`, silver 빌드 권장값은 `recall_strict`:

| 정책 | PER/LOC/ORG | PROD/EVT |
|---|---|---|
| `recall` (CLI 기본값) | high + medium_recall | high + medium_recall |
| `precision` | high + medium_prec | high + medium_prec |
| `high_only` | high 만 | high 만 |
| `full` | 전체 (conflict 포함) | 전체 |
| **`recall_strict`** (빌드 권장) | **high + medium_recall** | **high 만** |

상세 평가 결과·신뢰 등급은 `docs/reports/vietnamese-ner-silver-quality.md` 참조.

### 7.4 산출물 포맷 (Stockmark 호환)

```json
{
  "id": "0",
  "text": "Đồng bằng sông Cửu Long",
  "entities": [
    {"label": "LOC", "start_char": 0, "end_char": 23, "text": "Đồng bằng sông Cửu Long"}
  ]
}
```

---

## 8. 한국어·일본어 파이프라인과의 차이

| 항목 | 한국어 (KLUE) | 일본어 (Stockmark) | 베트남어 (WikiANN) |
|---|---|---|---|
| **데이터셋** | `klue` config=`ner` | `stockmark/ner-wikipedia-dataset` (canonical JSONL 덤프) | `unimelb-nlp/wikiann` config=`vi` |
| **데이터 split** | `validation` | `test` | `test` |
| **gold 엔티티 종수** | 6 (PS/LC/OG/DT/TI/QT) | 5 canonical (PER/LOC/ORG/PROD/EVT) | 3 (PER/LOC/ORG) |
| **라벨러 출력 종수** | 6 | 10 평면 (5 NER + 5 PII) | 10 평면 (5 NER + 5 PII) |
| **태그 표준** | KLUE 고유 (PS/LC/OG…) | canonical 영문 약어 | canonical 영문 약어 |
| **태그 정규화** | PER→PS, LOC→LC | (정규화 불필요, 이미 표준) | PERSON→PER, LOCATION→LOC |
| **토큰 수준** | 음절 + 공백 토큰 | 형태소 (sentencepiece) | 단어 (word) |
| **`sentence` 필드** | 있음 | 있음 (canonical 덤프) | 없음 (토큰 재결합) |
| **음절 BIO 변환** | 필요 | 불필요 | 불필요 |
| **프롬프트 언어** | 한국어 | 일본어 | 베트남어 |
| **silver 재라벨 파이프라인** | — | — | `augmenters/wikiann_vi/` (Gemma+Qwen+Wikidata 3중 검증) |
| **PII 주입** | — | `augmenters/pii/` (LLMInjector + PIIVerifier) | `augmenters/pii/` (동일) |
| **평가 러너** | `BenchmarkRunner` (lang="ko") | 동일 클래스 (lang="ja") | 동일 클래스 (lang="vi") |
