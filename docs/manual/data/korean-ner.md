# 한국어 NER 라벨링 방법론

> 대상 데이터셋: KLUE NER → canonical 6종(PER/LOC/ORG/DAT/PROD/EVT)
> 대상 코드: `src/ner/labelers/ko/`, `src/ner/llm_eval/`, `src/ner/metrics/`, `src/ner/labelers/{dataset_loader,tag_aligner,llm_helpers}.py`

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
    ▼ DatasetLoader.load()                          [labelers/dataset_loader.py]
List[NERRecord] {tokens, ner_tags, id, sentence?}
    │
    ▼ record["sentence"] 또는 TagAligner.reconstruct_text()
원본 텍스트 문자열                                    [llm_eval/benchmark_runner.py]
    │
    ▼ split_sentences(text, lang='ko')              [labelers/llm_helpers.py]
문장 리스트 → concurrency 단위 동시 호출
    │
    ▼ ner_prompts.py (SINGLE 템플릿)
프롬프트 문자열                                       [labelers/ko/ner_prompts.py]
    │
    ▼ vLLM(AsyncOpenAI 호환, temperature=0, JSON mode)
LLM JSON 응답
    │
    ▼ parse_spans(raw)                              [labelers/llm_helpers.py]
[{"text": "경찰", "type": "ORG"}, ...]
    │
    ├──▶ label_spans() 경로: raw spans 직접 반환
    │
    └──▶ label() 경로: spans_to_bio() 변환
         BIO 태그 리스트 ["B-ORG", "O", "B-PER", ...] [labelers/llm_helpers.py]
    │
    ▼ BenchmarkRunner._run_single()                 [llm_eval/benchmark_runner.py]
    │   eval_mode='bio' (ko/vi 공통)
    │
    ├── spans_to_syllable_bio()   음절 BIO 변환       [labelers/tag_aligner.py]
    ├── extract_spans_from_bio()  gold span 추출      [labelers/tag_aligner.py]
    └── normalize_tag(lang='ko')  태그 정규화          [labelers/tag_aligner.py]
    │
    ▼ MetricsCalculator + span_metrics              [metrics/{bio_metrics,span_metrics}.py]
    ├── compute_span_match()   Span Match (exact/relaxed) — primary
    ├── compute_seqeval()      seqeval BIO F1 — secondary
    └── compute_span_f1()      Character Span F1
    │
    ▼ ReportGenerator                                [llm_eval/report.py]
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

**코드:** `DatasetLoader.load()` (= `HFTokenDatasetLoader` 의 별칭, `src/ner/labelers/dataset_loader.py`)


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

KLUE NER 데이터에는 원본 문장이 포함되어 있다. `BenchmarkRunner`는 LLM에 전달할 텍스트의 **primary source**로 `sentence` 필드를 사용하며, 없으면 `TagAligner.reconstruct_text()`로 토큰을 재결합한다. KLUE 원문에 포함된 `<경찰:OG>` 형태의 마크업은 정규식으로 제거한다.

```python
sentence = record.get("sentence")
if sentence:
    text = re.sub(r'<([^:>]+):[A-Z]+>', r'\1', sentence)  # KLUE 태그 마크업 제거
else:
    text = TagAligner.reconstruct_text(gold_tokens)
```

### ClassLabel 변환

HuggingFace datasets에서 NER 태그는 정수(`ClassLabel`)로 저장된다. `DatasetLoader`가 자동으로 문자열 태그로 변환한다:

```python
if label_feature is not None and isinstance(label_feature, ClassLabel):
    ner_tags = [label_feature.int2str(t) for t in raw_tags]  # 0 → "O", 1 → "B-PS", ...
```

> 로딩 직후 `ner_tags`는 **KLUE 원본 약어**(`PS/LC/OG/DT/TI/QT`)를 그대로 담는다.
> 이후 평가 단계의 `normalize_tag(lang='ko')`가 canonical(`PER/LOC/ORG/DAT`)로
> 정규화하며, KLUE `TI/QT`는 매핑 없이 통과해 gold span 단계에서 드롭된다.
> 따라서 §3 이후 본 문서의 태그 표기는 모두 canonical이다.

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| JSONL 폴백을 먼저 시도 | Python 3.13 환경에서 HuggingFace `datasets` 라이브러리 호환성 문제가 발생하여, 사전 export된 JSONL을 우선 사용 |
| `NERRecord`를 dict로 정의 | 경량 데이터 전달 — dataclass 대신 dict 사용으로 직렬화 편의 |
| `sentence` 필드 보존 | 토큰 재결합보다 원본 문장이 정확함 (KLUE의 경우 태그 마크업 포함 원문 제공) |

---

## 3. Stage 2: NER Tag 목록 및 프롬프트 설계

### 엔티티 타입 정의

**코드:** `src/ner/labelers/ko/ner_prompts.py`의 `DEFAULT_ENTITY_TYPES`
(`["PER", "LOC", "ORG", "DAT", "PROD", "EVT"]` — canonical 6종)

| 태그 | 의미 | 설명 | 예시 |
|------|------|------|------|
| PER | 인명 (Person) | 사람 이름, 성만 나오면 성만 추출. 익명 알파벳(A·B)·가수/아이돌 그룹명·외국인·등장인물명 포함 | "김철수", "박"(씨), "원더걸스", "카게무샤" |
| LOC | 지명 (Location) | 장소·지역·국가·도시·건물·시설. 복합 지명은 전체를 하나로 묶음. 국가명은 기본 LOC | "서울삼성동그랜드인터콘티넨탈", "경남진해", "몽골" |
| ORG | 기관 (Organization) | 회사·기관·단체·정당·팀·대표팀·학교. "경찰"·"정부" 등 일반명사도 기관 맥락이면 ORG. 운영 리그·구단도 ORG | "삼성전자", "경찰", "프리미어리그", "엠비씨" |
| DAT | 날짜 (Date) | 연도·월·일·기간·상대날짜·시대·시절·요일. 숫자+날짜단위를 하나로 묶음 | "지난19일", "5공화국시절", "14년" |
| PROD | 제품·작품 (Product) | 시판 물품·창작 작품(영화·드라마·노래·서적·만화·게임·방송 프로그램)·패키지 SW/OS·형식·모델명 제조물 | "갤럭시S24", "기생충", "황금어장" |
| EVT | 행사·사건 (Event) | 1회성 행사·대회·경기·전쟁·조약·사건·자연재해·named 위기·주기적 선거/투표 | "임진왜란", "외환위기", "총선", "한일월드컵" |

> KLUE 원본의 `TI`(시간)·`QT`(수량)는 canonical에서 **드롭**(개체명 비추출).
> `PROD`/`EVT` 경계는 canonical 스키마 §3·§3.1~§3.3 기준(운영 리그=ORG vs
> 경기·대회=EVT, 법령·시대구분·추상 쟁점·상/훈장·무형 서비스는 비-entity).

### 핵심 라벨링 규칙

`SINGLE_PROMPT_TEMPLATE`의 "핵심 규칙" 섹션 7가지:

1. **조사 제외**: 은/는/이/가/을/를/에/에서/으로/의/과/와/부터/까지/도/만 — 반드시 제거
2. **접미사 처리**: "씨"/"님" 제외. "김모"/"이모" → "김"/"이"만 추출 (성+익명). "강모연" (3글자+) → 전체 추출
3. **복합 개체명 묶기**: 붙어 있는 개체명은 하나로 ("지난19일" → DAT, "경남진해" → LOC)
4. **괄호 내 날짜**: 괄호 안 날짜는 별도 DAT로 분리 ("어제(10월 10일)" → "어제"=DAT, "10월10일"=DAT)
5. **타입 판정·제외**: 사건·전쟁·대회 = EVT, 작품·프로그램·상품·선박 = PROD. 법령·시대구분·추상 쟁점·상/훈장·무형 서비스는 개체명 아님(추출 금지). 모호하면 우선순위 ORG→LOC→PROD→EVT
6. **출력 형식**: JSON 배열만 출력, 다른 설명 금지
7. **빈 결과**: 개체명 없으면 `[]` 반환

### 프롬프트 구조

프롬프트 템플릿은 한 벌뿐이다 (BATCH_PROMPT_TEMPLATE는 라벨러에 없음 —
`augmenters/wikiann_vi/` 재라벨 파이프라인 전용):

| 템플릿 | 변수명 | 용도 | 형식 |
|--------|--------|------|------|
| SINGLE | `SINGLE_PROMPT_TEMPLATE` | 단일 문장 라벨링 (vLLM) | `입력: {sentence}\n출력:` |

전에는 OpenAI 채팅 형식용 `SYSTEM_PROMPT`+`USER_PROMPT_TEMPLATE` 가 같은 규칙을 따로
한 벌 더 들고 있었다. 그 경로를 실제로 돌린 기록이 없는데도 규칙이 바뀔 때마다 두 번씩
고쳐야 했고, 그러고도 두 벌이 갈라졌으므로(같은 문장에 다른 라벨이 나올 수 있고 갈라짐을
잡을 검사가 없었다) 라벨러와 함께 걷어냈다.

**프롬프트 길이에는 상한이 있다.** vLLM 은 `프롬프트 + max_tokens(출력 예약)` 이
`max-model-len` 을 넘으면 요청을 받자마자 400 으로 거절한다. 실패가 리포트에서는 "모델
성능이 나쁘다" 로 보이므로, `tests/ner/labelers/test_prompt_token_budget.py` 가 고정
지시문의 토큰 수를 재서 예산을 넘으면 커밋 전에 실패한다.

### Few-shot 예시의 설계 의도

`SINGLE_PROMPT_TEMPLATE` 본문에 13개의 입력/출력 예시가 포함된다. 각 예시가 커버하는 엣지 케이스:

| 예시 | 커버하는 엣지 케이스 |
|------|---------------------|
| 1. 경찰은 박씨의 딸(32)과 김모(33)씨... | 일반명사 기관(경찰=ORG), 접미사·익명 제거(박씨→박, 김모→김), 괄호 내 숫자는 비추출(QT 드롭) |
| 2. 18번 홀(파5)에서... 최운정 장하나 LPGA | 인명 2명=PER, ORG(LPGA), 스포츠 수량(18번홀·파5)은 비추출 |
| 3. 지난19일 오전9시30분 서울삼성동그랜드인터콘티넨탈... | 붙어쓰기 날짜(지난19일=DAT), 복합 LOC 한 덩어리, 시간(오전9시30분)은 비추출(TI 드롭) |
| 4. 이번 주 월요일 새벽에... 5공화국시절 | 복합 날짜(이번 주 월요일=DAT), 시대(5공화국시절=DAT), 시간 단독(새벽)은 비추출 |
| 5. 삼성전자는 오늘 오후 2시 서울 강남구 코엑스... | ORG, DAT(오늘), 복수 LOC(서울·강남구·코엑스), 시각·금액은 비추출 |
| 6. 각 조 3위 6개국 중 한국이... | 스포츠 국가대표팀(한국=ORG), 숫자는 비추출 |
| 7. 어제(10월 10일) 방송된 원더걸스... 소녀시대 | 괄호 안 날짜 분리(어제=DAT + 10월10일=DAT), 그룹명→PER(원더걸스·소녀시대) |
| 8. 경남 진해에서 국악예술단... 최씨 지난 2010년 | 복합 LOC 묶기(경남진해=LOC), ORG, 성만 추출(최), 붙은 날짜(지난2010년=DAT) |
| 9. 김 선수 소속팀 두산 베어스는... 한국시리즈 | 성만 추출(김), 팀명 ORG(두산 베어스·삼성), 기간 DAT(14년), 대회=EVT(한국시리즈) |
| 10. 부산시 사하구 하단동... 몽골 선적 한수원 | 행정구역 한 덩어리 LOC, 국가 LOC(몽골), ORG(한수원), 수량은 비추출 |
| 11. 봉준호 감독의 기생충은 임진왜란을... 명량 | 감독=PER, 작품=PROD(기생충·명량), 전쟁=EVT(임진왜란) |
| 12. 외환위기 직후 총선에서 엠비씨 황금어장... | named 위기=EVT(외환위기), 선거=EVT(총선), 방송사=ORG(엠비씨), 프로그램=PROD(황금어장), 법령(도로교통법)은 비추출 |
| 13. 프리미어리그 토트넘은 챔피언스리그... 갤럭시S24 | 운영 리그·팀=ORG(프리미어리그·토트넘), 대회=EVT(챔피언스리그), 제품=PROD(갤럭시S24) |

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 프롬프트 한 벌 | 규칙을 두 곳에 적으면 한쪽만 고쳐져 갈라진다 — 쓰지 않는 두 번째 벌은 유지비만 남는다 |
| 상세한 엔티티 정의 + 규칙 | LLM의 라벨링 일관성을 높이기 위함. 특히 조사 제외, 복합 개체명 묶기는 한국어 특유의 문제 |
| 다수의 Few-shot 예시 | 각 예시가 서로 다른 엣지 케이스를 커버하여, LLM이 다양한 패턴을 학습 |

---

## 4. Stage 3: LLM 라벨링 실행

### 베이스 클래스 구조

KO 라벨러는 공통 베이스(`src/ner/labelers/base_vllm_labeler.py`)의 얇은 서브클래스다.
KO 별도 파일은 프롬프트 템플릿·기본 모델명·`lang='ko'` 만 주입한다.

| 파일 | 클래스 | 백엔드 | 동시성 |
|------|--------|--------|--------|
| `src/ner/labelers/ko/vllm_ner_labeler.py` | `VllmNERLabeler` | vLLM(AsyncOpenAI 호환) | `concurrency` (기본 32) |

**동기 API 는 라벨러 전용 이벤트 루프를 재사용한다.** `label()`·`label_spans()` 는
비동기 경로의 동기 래퍼인데, 호출마다 루프를 새로 만들면 클라이언트의 연결 풀과 세마포어가
첫 루프에 묶인 채 남아 두 번째 호출부터 깨진다(`Event loop is closed`,
`bound to a different event loop`). 벤치마크는 라벨러 하나로 샘플을 순차 처리하므로 정확히
이 경로를 탄다. 루프를 라벨러 수명 동안 하나로 유지하고, 정리는 `close()` 로 한다.

### 전체 흐름

```
입력 텍스트
    │
    ▼ split_sentences(text, lang='ko')   (1) 문장 분리
문장 리스트 (llm_helpers.py)
    │
    ▼ 호출 단위 구성                      (2) 분할
    │   └─ 문장 단위 SINGLE × concurrency 동시 호출
    │
    ▼ LLM 호출                            (3) temperature=0, JSON mode
    │
    ▼ parse_spans(raw)                    (4) <think> strip + json.loads + 정규식 [...] 폴백
[{"text": "...", "type": "..."}, ...]
    │
    ├── label_spans() / alabel_spans() 경로: raw spans 반환
    │
    └── label() 경로:
        ▼ spans_to_bio(tokens, spans)     (5) 2단계 매칭으로 BIO 변환
        NERRecord {tokens, ner_tags, id}
```

### 4.1 문장 분리

**코드:** `split_sentences(text, lang='ko')` (`src/ner/labelers/llm_helpers.py`)

- 한국어/영어: `.!?` 뒤 공백 또는 줄바꿈으로 분리
- 일본어: `。！？` 포함하여 분리
- 짧은 단편은 최소 길이 버퍼링으로 통합
- 문장이 하나도 안 나오면 원본 텍스트 전체를 하나의 문장으로 반환

### 4.2 호출 단위 구성

- **vLLM**: `split_sentences()` 결과 문장 각각을 SINGLE 프롬프트로 1회씩 호출. `concurrency`(기본 32)로 동시 실행.
- 빈 문장은 사전 필터링.

### 4.3 LLM 호출

| 백엔드 | 클라이언트 | 동시성 | 핵심 옵션 |
|--------|-----------|--------|----------|
| vLLM | `AsyncOpenAI` (vLLM 호환 엔드포인트) | async (`concurrency`, 기본 32) | `temperature=0`, `<think>` 태그 strip (`thinking=False`) |

| 설정 | 값 | 이유 |
|------|------|------|
| `temperature` | 0 | NER은 정확성이 중요 — 창의적 변형 불필요 |
| `response_format`/`format` | JSON | LLM이 반드시 유효한 JSON을 출력하도록 강제 |
| `thinking` | False | thinking 토큰이 JSON 파싱을 방해하지 않도록 |
| `max_tokens` | 1024 | 출력 예약분은 실제 사용량과 무관하게 컨텍스트에서 먼저 빠진다. 실측 출력이 77 토큰이라 1024 로도 넉넉하고, 남는 자리는 프롬프트가 쓴다 |

### 4.4 JSON 파싱

**코드:** `parse_spans(raw)` (`src/ner/labelers/llm_helpers.py`)

```python
raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
data = json.loads(raw)
if isinstance(data, list):                    # [{"text":..., "type":...}, ...]
    return data
if isinstance(data, dict):
    if "text" in data and "type" in data:     # 단일 엔티티 bare object
        return [data]
    for val in data.values():                 # {"entities": [...]} 래퍼
        if isinstance(val, list):
            return val
# 정규식 폴백: 본문에서 [...] 추출
match = re.search(r"\[.*?\]", raw, re.DOTALL)
```

### 4.5 spans → BIO 변환

**코드:** `spans_to_bio(tokens, spans)` (`src/ner/labelers/llm_helpers.py`)

LLM이 반환한 entity spans를 토큰 레벨 BIO 태그로 변환한다. **2단계 매칭 전략**:

**1단계 — Exact match**:
```python
# span "김 철수" → span_tokens ["김", "철수"]
# tokens [..., "김", "철수", ...] 에서 연속 일치 찾기
if tokens[i : i + n] == span_tokens:
    tags[i] = f"B-{entity_type}"
```

**2단계 — Substring match**:
```python
# span "서울시" → tokens에 "서울시에서"가 있을 때
# "서울시" in "서울시에서" 또는 "서울시에서" in "서울시" 체크
if all(sp in tokens[i + j] or tokens[i + j] in sp for j, sp in enumerate(span_tokens)):
```

이 2단계 매칭은 한국어의 조사 부착 문제를 처리한다. LLM이 "서울시"를 추출하더라도, 원본 토큰이 "서울시에서"인 경우 substring 매칭으로 올바르게 태깅한다.

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| 문장 분리 후 동시 호출 | 긴 텍스트를 한 번에 보내면 LLM 컨텍스트 효율이 떨어지고, 엔티티 누락 증가. 문장 단위 분리 후 concurrency로 처리량 확보 |
| 최소 길이 버퍼링 | "네." 같은 짧은 조각은 단독으로 보내면 비효율적 |
| 2단계 매칭 (exact → substring) | 한국어 조사가 토큰에 부착되어 있어 exact match만으로는 부족. substring 폴백으로 "서울시" ↔ "서울시에서" 매칭 해결 |
| 배치 실패 시 개별 폴백 | 배치 JSON 파싱 실패(포맷 오류 등) 시에도 최대한 결과를 확보 |
| `label()` vs `label_spans()` 이중 인터페이스 | `label()`: BIO 태그가 필요한 seqeval 평가용. `label_spans()`: span 기반 평가용 (BIO 변환 없이 직접 사용) |

---

## 5. Stage 4: 결과 확인 (평가)

### 평가 진입점

**코드:** `src/ner/llm_eval/__main__.py` (한·일·베 공통 dispatch)

```bash
python -m ner.llm_eval --lang ko --models "vllm:Qwen/Qwen3-32B" --max-samples 500
```

CLI 흐름:
1. `_eval_mode_for_lang('ko') → 'bio'`
2. `DatasetLoader`로 gold records 로딩
3. 통합 `BenchmarkRunner(eval_mode='bio')` 생성 후 labeler 추가
4. `runner.run()` → 각 labeler에 대해 `_run_single()` 실행
5. `ReportGenerator`(`src/ner/llm_eval/report.py`)로 결과 출력

### 평가 흐름 상세

**코드:** `BenchmarkRunner._run_single()` (`src/ner/llm_eval/benchmark_runner.py`)

각 gold record에 대해:

```
gold record {tokens, ner_tags, sentence}
    │
    ├── (1) 텍스트 재구성: sentence 필드 또는 reconstruct_text()
    │
    ├── (2) gold spans 추출: extract_spans_from_bio(gold_tokens, gold_tags, lang='ko')
    │       → [{"text": "경찰", "type": "ORG"}, ...]  (KLUE OG → canonical ORG 정규화됨)
    │
    ├── (3) LLM 라벨링: labeler.label_spans(text)
    │       → [{"text": "경찰", "type": "ORG"}, ...]  (predicted spans)
    │
    ├── (4) 음절 BIO 변환: spans_to_syllable_bio(text, gold_tokens, pred_spans)
    │       → ["B-ORG", "I-ORG", "O", ...]  (predicted BIO, gold 토큰 그리드에 정렬)
    │
    └── (5) 메트릭 계산 (전체 샘플에 대해)
```

### 태그 정규화

**코드:** `normalize_tag(tag, lang='ko')` (`src/ner/labelers/tag_aligner.py`)

LLM·gold가 KLUE 약어를 출력할 수 있으므로, 모든 태그를 canonical 표준(PER/LOC/ORG/DAT)으로 정규화한다 (`_TAG_NORMALIZE_MAP_KO`):

```
PS → PER,  LC → LOC,  OG → ORG,  DT → DAT
PERSON → PER, LOCATION → LOC, ORGANIZATION → ORG, DATE → DAT
(TI·QT는 매핑하지 않고 통과 → 후속 gold 단계에서 드롭)
```

### spans → 음절 BIO 변환

**코드:** `TagAligner.spans_to_syllable_bio()` (`src/ner/labelers/tag_aligner.py`)

LLM이 반환한 text spans를 KLUE의 음절 단위 BIO 태그로 변환하는 핵심 로직:

1. **char→syl 매핑 구축**: 원본 텍스트의 각 문자 위치 → 음절 토큰 인덱스 매핑
2. **span 위치 찾기**: 원본 텍스트에서 entity text의 문자 범위 탐색
   - 정확 매치 → `text.find(entity_text)`
   - 공백 무시 매치 → `_find_ignore_spaces()` (예: "지난 19일" ↔ "지난19일")
   - 중복 처리: `_next_search` 딕셔너리로 이미 매칭된 위치 이후부터 탐색
3. **BIO 태그 할당**: 매칭된 문자 범위의 음절 토큰에 B-/I- 태그 부여

### 3종 메트릭

**코드:** `MetricsCalculator` (`src/ner/metrics/bio_metrics.py`)의 아래 3개 메서드
(KO `bio` 경로는 `compute_offset_span_f1`를 쓰지 않는다 — 그것은 JA·VI offset-span 경로 전용)

| 메트릭 | 메서드 | 역할 | 비고 |
|--------|--------|------|------|
| **Span Match** | `compute_span_match()` | entity 단위 exact/relaxed 매칭 | **Primary** — BIO 변환 없이 직접 span 비교 |
| **seqeval** | `compute_seqeval()` | 음절 BIO 태그 기반 F1/Precision/Recall | Secondary — BERT 모델과 비교 시 사용 |
| **Character Span F1** | `compute_span_f1()` | 문자 수준 span 매칭 (KLUE 공식 메트릭) | BIO 태그에서 span 추출 후 비교 |

**Span Match 상세:**
- **Exact**: gold span과 predicted span의 text와 type이 정확히 일치
- **Relaxed**: text가 부분적으로 겹치면 매칭 (type은 일치해야 함)

### 왜 이렇게 설계했는가

| 결정 | 이유 |
|------|------|
| Span Match를 primary 메트릭으로 | BIO 변환 과정의 정렬 오류를 우회하여, LLM 라벨링 품질을 직접 평가 |
| seqeval을 secondary로 유지 | BERT 기반 모델은 BIO 태그를 직접 출력하므로, LLM과 BERT를 동일 기준으로 비교하기 위해 |
| 음절 BIO 변환이 필요한 이유 | KLUE gold 데이터가 음절 단위 BIO이므로, seqeval 비교를 위해 LLM spans를 같은 형식으로 변환 필요 |
| 3종 메트릭 병행 | 각 메트릭이 다른 관점을 제공: span match(엔티티 품질), seqeval(토큰 정확도), character span F1(문자 수준 매칭) |

---

## 6. 설계 의사결정 요약

전체 파이프라인에서 내려진 핵심 설계 결정들:

| # | 단계 | 결정 | 이유 | 대안 (채택하지 않은 것) |
|---|------|------|------|----------------------|
| 1 | 데이터 | JSONL 폴백 우선 | Python 3.13 호환성 | HuggingFace만 사용 (환경 제약) |
| 2 | 데이터 | sentence 필드 보존 | 원본 문장이 토큰 재결합보다 정확 | 항상 reconstruct_text() 사용 |
| 3 | 프롬프트 | 3종 템플릿 | 백엔드별 최적 형식 상이 | 단일 범용 프롬프트 |
| 4 | 프롬프트 | 다수 Few-shot | 각각 다른 엣지 케이스 커버 | Zero-shot (정확도 저하) |
| 5 | 라벨링 | 문장 분리 → concurrency 동시 호출 | 컨텍스트 효율 + 처리량 | 전체 텍스트 한 번에 전송 |
| 6 | 라벨링 | temperature=0, JSON mode | 결정적 출력 + 파싱 안정성 | 높은 temperature (변형 발생) |
| 7 | 라벨링 | 2단계 span→BIO 매칭 | 조사 부착 처리 | exact match만 (누락 증가) |
| 8 | 라벨링 | 배치 실패 시 개별 폴백 | 결과 확보 극대화 | 실패 시 에러 반환 |
| 9 | 평가 | Span Match primary | BIO 변환 오류 우회 | seqeval만 사용 (변환 노이즈) |
| 10 | 평가 | 3종 메트릭 병행 | 다각적 품질 평가 | 단일 메트릭 |

## 7. gold 재생성·무회귀 비교 프로토콜

PROD/EVT 등 NER 레이어를 LLM 재라벨로 갈아끼울 때(경계 룰 개정 등),
재생성 전후 무회귀를 측정하는 표준 절차.

### PII 주입은 비가역 — graft 금지

KO PII 4종은 **llm 자연삽입**(`ner.augmenters.pii --mode llm`)으로 주입되며,
PII 문자열만 끼우는 게 아니라 connective 절("관련 문의는 …으로")까지 더해
**원문 일부를 재작성**한다. 따라서 PII-주입 텍스트에서 clean 본문을 복원하거나,
clean gold 의 NER span 을 offset 으로 PII-텍스트에 얹는 graft 는 불가능하다
(실측: span 의 ~1.7% 가 PII-텍스트에 그대로 없어 누락 → baseline 온전·신규
누락의 **비대칭 편향**). PII 를 보존하려고 graft 하지 말 것.

### clean NER-only 로 비교

재생성 전후 비교는 **PII 없는 clean gold** 로 한다. 같은 KLUE 본문
(`klue_to_canonical_gold` 는 결정적 → baseline·신규 text 100% 동일)에 NER 라벨만
달리하고, `--no-stratify`(PROD/EVT 층화 비활성 = label-불변 split)로 학습해
fold 멤버십을 재라벨 전후 동일하게 고정한다. PII 는 재생성과 직교하므로
(주입 로직 불변) NER 무회귀 측정에서 빼도 무방하다.

### 측정 정직성

- gold 가 바뀐 타입(PROD/EVT, eponymy 로 라벨이 바뀐 PER span)은 **보고만** —
  옛 gold 기준 F1 은 순환이라 개선/회귀를 주장하지 않는다.
- 무회귀 게이트는 **gold 불변 타입**만: 미교체 LOC/ORG/DAT + eponymy 미접촉 PER
  (merge contains-replace 가 교체한 LOC/ORG/DAT span 은 재라벨마다 달라 제외).
- 출하 10종(PII 포함) 모델은 게이트 통과 후 **확정된 clean gold 에 PII 재주입**해
  별도 빌드한다.
