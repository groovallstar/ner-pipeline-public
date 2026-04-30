# 핸드오프: `docs/manual/data/` drift cleanup (post-#27)

> 작성일: 2026-04-28
> 선행 작업: `docs/manual/data/japanese-ner.md` 전면 재작성 (현 브랜치)
> 목적: `docs/manual/data/` 9개 파일 중 5개에 남은 코드-문서 drift를
>   현재 develop 코드 기준으로 정리. 1개(`japanese-ner.md`)는 본
>   세션에서 처리 완료.

---

## 0. 진입 조건

- develop 브랜치 최신 상태 (`54a3d01 chore(labelers): Ollama 백엔드 제거
  · langchain-ollama 의존성 정리` 포함) 기준으로 문서를 갱신한다.
- 작업 브랜치는 `feat/issue-27-ja-5type-bench-and-pii-rule45`처럼 다른
  목적의 feat 브랜치에서 진행하지 말고 **별도 docs 브랜치**로 분리한다.
  현재 #27 브랜치는 develop의 ollama 제거 커밋을 아직 반영하지 않았다 —
  머지·리베이스로 ollama 파일이 정리된 뒤 시작하면 코드와 문서가 자연
  스럽게 정합한다.
- `docs/manual/data/japanese-ner.md`는 이미 develop 기준 최종본으로 재작성
  완료(현 브랜치). 본 핸드오프의 5개 후속 파일도 동일한 원칙을 따른다.

## 1. 작성 원칙 (모든 후속 파일 공통)

1. **최종본만 유지** — 변경 이력·이슈 번호(`이슈 #N에서 …`)·"이전엔 X
   였으나" 같은 시점 기록을 본문에 두지 않는다. 이력은 git log·
   `docs/issues/`·`docs/reports/`가 보관한다. 작업 시작 시 기존 `## 변경
   이력` 섹션·"축소 전 N종" 류 문구·"이슈 #N에서 재정의됨" 같은
   각주를 함께 제거한다.
2. **모듈 경로는 모듈 단위** — `src/labelers/ja/ner_prompts.py:40-44`
   같은 라인 번호 인용은 곧바로 stale 된다. 모듈 경로(`src/...`)와 함수
   이름까지만 적고, 라인 번호는 알고리즘 설명에 한해 코드 블록 안에서만
   언급한다.
3. **백엔드는 vLLM + OpenAI 2종** — develop의 ollama 제거 이후 기준.
   `OllamaNERLabeler`, `BaseOllamaLabeler`, `ChatOllama`, `BATCH_PROMPT_TEMPLATE`,
   `--ollama-url` / `--num-ctx` / `--batch-size` 모두 본문에서 제거한다.
4. **canonical 정의의 단일 출처는 `japanese-canonical-entity-schema.md`** — 다른
   파일은 항목별 한 줄 요약 + 이 파일로 링크. 동일 정의 중복 금지.

## 2. 권장 진행 순서

```
[병렬 가능 — 즉시]
  D1  vietnamese-ner.md         (~0.5일, JA와 같은 패턴)
  D2  korean-ner.md             (~0.5일, evaluators→llm_eval sweep + ollama 제거)
  D3  wikidata-anchor-verification.md (~0.5일, 분량 작음)

[D1·D2·D3 정리 후]
  D4  ner-dataset-formats-comparison.md (~0.3일, Stockmark 8종 framing 갱신)
  D5  vietnamese-ner-8types.md          (~0.3일, deprecated 마커 또는 흡수)
```

D1~D3는 서로 독립이므로 병렬 가능. D4·D5는 D1·D2·D3가 합의해 놓은
최종 라벨 체계를 그대로 인용하므로 후행이 안전하다.

---

## 3. 파일별 작업 가이드

### D1. `vietnamese-ner.md`

#### 현재 drift

- **모듈 경로 sweep**: `src/evaluators/__main__.py`, `_run_vietnamese()` 같은
  더 이상 존재하지 않는 함수·경로 잔존. 실제 코드: `src/llm_eval/__main__.py`,
  통합 `BenchmarkRunner`, `_create_labeler_vi()` (L113).
- **내부 불일치**: 헤더는 "canonical 10종 평면"이라고 적었지만 §3 entity
  표에는 PER/LOC/ORG 3종만 나열. `src/labelers/vi/ner_prompts.py:DEFAULT_ENTITY_TYPES`
  은 10종 평면.
- **규칙 수 불일치**: 본문 "5가지 핵심 규칙" → 실제 프롬프트 6규칙.
- **Few-shot 수 불일치**: 본문 "3개" → 실제 9개(메인 8 + PII 1).
- **#27 LOC/ORG 재정의 미반영**: LOC 정의에 여전히 "건축물" 포함. 실제
  `vi/ner_prompts.py`는 LOC=지리, ORG=시설 전부 (JA와 동일 원칙).
- **ollama 잔재**: backend 표·CLI 예시에 ollama 항목 잔존.

#### 작업

1. `## 변경 이력` 섹션이 있다면 제거.
2. §1 흐름도: 모듈 경로를 develop 기준으로 sweep.
3. §3 entity 표: `vi/ner_prompts.py`의 10종 평면을 그대로 옮기되, 정의는
   `japanese-canonical-entity-schema.md` 링크 한 줄로 위임 (전체 정의 복붙 금지).
4. §3 라벨링 규칙: 현재 프롬프트의 6규칙을 그대로 옮긴다.
5. §3 Few-shot 표: 현재 프롬프트의 9개 예시 중 대표 케이스 위주로 정리.
   JA 문서의 그룹핑(메인 / PII 종합 / DAT-LOC / EMAIL counter) 방식 참고.
6. §4 백엔드 표·코드 인용: vLLM + OpenAI 2종으로 정리.
7. §6 KO 비교 표(있다면) `metrics/span_metrics.py`, `llm_eval/benchmark_runner.py`로
   경로 갱신.

#### 참조 src

- `src/labelers/vi/ner_prompts.py`
- `src/labelers/vi/dataset_loader.py`
- `src/llm_eval/__main__.py` (`_create_labeler_vi`, `_load_gold` 의 vi 분기)
- `src/llm_eval/benchmark_runner.py` (vi는 `eval_mode='bio'` 경로)
- `src/augmenters/wikiann_vi/` (canonical 매핑·Wikidata 검증)

### D2. `korean-ner.md`

#### 현재 drift

- **모듈 경로 sweep**: `src/evaluators/__main__.py:191-217 _run_korean()`,
  `src/evaluators/metrics.py::MetricsCalculator` 같은 stale 인용. 실제:
  `src/llm_eval/__main__.py` 단일 dispatch (`_run_benchmark`),
  `src/metrics/bio_metrics.py::MetricsCalculator`.
- **ollama 잔재**: backend 표·CLI 예시·`ChatOllama` 설명 잔존.

#### 작업

1. `## 변경 이력` 제거 (있다면).
2. 모듈 경로 sweep: `evaluators` → `llm_eval` + `metrics`.
3. backend 표·CLI: vLLM + OpenAI + HF (KO는 HF baseline 모델도 지원).
4. 프롬프트 템플릿 표 사용 클래스 컬럼: `OllamaNERLabeler` 제거,
   `VllmNERLabeler`, `OpenAINERLabeler`만.
5. KO는 BIO 평가 경로(`eval_mode='bio'`)이므로 §5 평가 흐름은 JA와
   다름 — `_run_single()` 경로 그대로 (`benchmark_runner.py:137-265`).
   다만 함수 이름·라인은 모듈명만 인용.

#### 참조 src

- `src/labelers/ko/ner_prompts.py`
- `src/labelers/ko/dataset_loader.py` (또는 공용 `labelers/dataset_loader.py`)
- `src/llm_eval/__main__.py` (KO는 `_create_labeler` 기본 분기)
- `src/llm_eval/benchmark_runner.py` (`_run_single`)
- `src/labelers/tag_aligner.py`, `src/metrics/bio_metrics.py`

### D3. `wikidata-anchor-verification.md`

#### 현재 drift

- **Q-ID 개수**: 본문 "134개" → 실제 약 90개 (`wikidata_anchor.py:46-201`).
- **출력 타입 수**: 본문 "8-type" → 실제 5-type (FAC/CORP/POL → ORG로 fold).
- **`_UA` user-agent 헤더**: 본문 `ner_pipeline/issue-10` → 실제
  `ner_pipeline/issue-13`.
- **CLI 출력 문자열**: 본문 "Mapped to 8-type" → 실제 "Mapped to 5-type"
  (`wikidata_anchor.py::main()` print).
- **`--span-key` 기본값**: 본문 `gold_spans_8type`이라는 문자열은 코드와
  일치하지만 의미가 stale (5-type 시점에 8type이라는 키 명이 혼란).
  코드 그대로 적되 "내부 키 이름은 호환 보존" 정도의 한 줄로 정리.

#### 작업

1. `## 변경 이력` 제거 (있다면).
2. Q-ID 개수·출력 타입 수·UA·CLI 문자열 4지점을 코드 grep으로 실시간
   확인하고 갱신. 분량 작아 1~2시간이면 끝.
3. canonical 매핑 표는 `wikidata_anchor.py::WIKIDATA_TO_CANONICAL`을 직접
   요약. PER/LOC/ORG/PROD/EVT 5섹션 + FAC/CORP/POL 옛 섹션 헤더가
   유지되는 경우 헤더만 보이고 값은 모두 ORG임을 한 줄 주석으로 명시.

#### 참조 src

- `src/augmenters/wikiann_vi/wikidata_anchor.py`
- 호출되는 CLI: `python -m augmenters.wikiann_vi.wikidata_anchor` (모듈
  단위 실행. argparse 본체 위치는 `__main__.py` 또는 같은 파일 하단).

### D4. `ner-dataset-formats-comparison.md`

#### 현재 drift

- Stockmark을 여전히 "8종"으로 framing. 실제 운영은 canonical 5종 NER +
  5종 PII = 10종 평면(JA·VI 공통). 표·§3.2·§3.4 모두 8종 가정으로 작성.
- 8종 원본 라벨 자체는 HF 데이터셋의 사실(`人名/法人名/...`)이므로 *원본
  라벨 = 8종, 운영 라벨 = canonical 10종 평면* 임을 분리해 기술.

#### 작업

1. `## 변경 이력`·"#21 축소 후" 류 시점 표현 제거.
2. KLUE 6종 / WikiANN-vi 3종 / Stockmark 원본 8종 / canonical 10종 평면 —
   네 라벨 공간을 한 표에서 비교.
3. canonical로의 매핑은 한 줄로 `japanese-canonical-entity-schema.md` 링크.
4. BIO formalism, 음절·서브워드 정렬 차이 등 형식 비교 본문은 그대로
   유지(이건 history가 아니라 spec).

#### 참조

- `docs/manual/data/japanese-canonical-entity-schema.md` (링크 대상)
- `src/labelers/dataset_loader.py` (KLUE 로딩 어댑터)
- `src/labelers/ja/dataset_loader.py` (Stockmark canonical 덤프 로더)
- `src/labelers/vi/dataset_loader.py`

### D5. `vietnamese-ner-8types.md`

#### 현재 상태

- 8-type 재라벨 절차서. 현재 운영은 5-type silver(`gemma_*.jsonl` 등) +
  Wikidata 앵커 기준이라 본 문서 본문은 deprecated.
- 헤더 부근에 #21 banner는 이미 있으나 본문은 8종 절차를 현역처럼 기술.

#### 작업 — 두 가지 선택지

**A. 흡수·삭제** (권장):
- 핵심 절차(WikiANN-vi 재라벨 워크플로 자체)를 `vietnamese-ner.md` 또는
  `wikidata-anchor-verification.md`에 흡수.
- 흡수 후 본 파일을 git에서 삭제. 이력은 git log + `docs/issues/`에 충분.

**B. 보존**:
- 파일 최상단에 한 줄 추가: `> Deprecated. 현재 운영은 5-type silver. 새
  작업은 vietnamese-ner.md를 참조한다.`
- 본문은 그대로 두고 새 변경 금지. (이 경우 long-tail의 사실 왜곡 위험은
  남는다.)

A를 권장. B는 곧바로 또 stale 된다.

---

## 4. 검증 — 모든 파일 공통

작업 종료 시 다음 grep이 0건이어야 한다(파일별 적용):

```bash
grep -nE 'src/evaluators|JaBenchmarkRunner|JaReportGenerator|ja_benchmark_runner|ja_metrics\.py|FAC → LOC|FAC→LOC|地名∪施設名|施設名→LOC|train_test_split|이슈 #[0-9]|축소 전 [0-9]종|변경 이력|2단 판정|대학 [0-9]단|병원 단일|ollama|chatollama|BATCH_PROMPT|--num-ctx|--batch-size|--ollama-url' docs/manual/data/<file>.md
```

또한 작업 후 `python -m llm_eval --help`, `python -m augmenters.wikiann_vi
--help` 출력이 문서의 CLI 예시와 일치하는지 1회 점검.

## 5. 진행 추적

본 핸드오프 문서는 5개 파일이 모두 정리되면 삭제(또는 `closed` 표기)
한다. 부분 삭제 금지 — 5개 모두 끝난 뒤 일괄 정리.

각 파일은 별도 GitHub Issue 또는 1개 통합 이슈로 묶어 진행 가능. 통합
이슈 권장(파일 단위 분할이면 5개 PR이 발생, 본 작업 성격상 비용 비대).

## 6. 본 세션에서 처리 완료된 항목 (참고)

- `docs/manual/data/japanese-ner.md` 전면 재작성 (521 → 400줄):
  - 변경 이력 섹션 제거, 이슈 번호·시점 표현 모두 제거
  - `src/evaluators/` → `src/llm_eval/` + `src/metrics/` sweep
  - `JaBenchmarkRunner`/`JaReportGenerator`/`ja_metrics.py` →
    통합 `BenchmarkRunner`/`ReportGenerator`/`metrics/span_metrics.py`
  - HF Hub 자동 로딩 서술 → canonical JSONL 덤프 전용
  - LOC vs ORG 정의·대학·병원 룰을 develop 코드 기준으로 갱신
  - Few-shot 11개(메인 8 + DAT/LOC 2 + EMAIL counter 1) 표
  - ollama 백엔드·`BATCH_PROMPT_TEMPLATE`·`base_ollama_labeler.py` 제거
  - 백엔드 표 vLLM + OpenAI 2종으로 정리
- 검증: 위 §4 grep 모두 0건.
