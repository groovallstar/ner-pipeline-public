# WikiANN-vi 5종 silver 재라벨 + 10종 PII 주입본 생성 (2026-04-27)

이슈 #23 산출물 검증 리포트.

## 배경

- #21에서 NER 8종이 5종(`PER · LOC · ORG · PROD · EVT`)으로 축소되고
  PII 5종과 합쳐 **10종 평면 목록**(`PER · LOC · ORG · PROD · EVT ·
  DATE · EMAIL · PHONE · ID_NUM · CREDIT_CARD`)이 단일 출처가 됨.
- 1a39401에서 VI 라벨러는 10종 canonical 덤프 전용으로 리팩터됐으나
  실제 WikiANN-vi 데이터는 과거 8종 silver 산출물만 있었다.
- Stockmark 측은 #21에서 이미 5종 gold + 10종 PII 주입본
  (`stockmark_pii_1000.jsonl`)이 정리되어 있어 재생성 범위 외.

## 산출물

| 카테고리 | 파일 | 라인 수 |
|---|---|---|
| WikiANN-vi 5종 silver (Gemma) | `data/wikiann_vi/gemma_{train,validation,test}.jsonl` | 20,000 / 10,000 / 10,000 |
| WikiANN-vi 5종 silver (Qwen) | `data/wikiann_vi/qwen_{train,validation,test}.jsonl` | 20,000 / 10,000 / 10,000 |
| Cross-model kappa | `data/wikiann_vi/kappa_{train,validation,test}.json` | — |
| Recall-merge (high+medium_recall) | `data/wikiann_vi/vi_wikiann_recall_{train,validation,test}.jsonl` | 22,872 / 11,300 / 11,484 spans |
| Wikidata anchor 검증 | `data/wikiann_vi/wikidata_anchor_{train,validation,test}.json` | — |
| **10종 PII 주입본** | `data/pii/wikiann_vi_pii.jsonl` | 39,979 |
| PII stats | `data/pii/wikiann_vi_pii.stats.json` | — |

구 `*_8type_*.jsonl` 9개 파일(약 43MB)은 본 이슈로 일괄 삭제됨. 데이터
파일명만 변경됐고 JSONL 안 필드명(`gold_spans_8type`,
`gold_spans_8type_merged`)·모듈 파일명·스펙 문서명은 #21 호환성 결정에
따라 그대로 유지.

## 백엔드·실행 환경

- **5종 silver 재라벨**: vLLM Gemma `cyankiwi/gemma-4-31B-it-AWQ-8bit`
  (localhost:8081) + Qwen `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
  (localhost:8082). 3-스플릿 전량, wall time 87분.
- **PII 주입(LLM 모드)**: Gemma 단일 모델, llm-concurrency=16, 단일
  event loop + `asyncio.gather`. 40,000 record → 39,979 record (21건은
  PII 값이 생성 텍스트에 포함되지 않아 ValueError로 drop). HTTP 요청
  31,189건, retry 0건. wall time 약 122분.

## 검증

### 1. Cross-model kappa (Gemma vs Qwen)

| split | paired spans | kappa | po (관측 일치율) |
|---|---:|---:|---:|
| train      | 24,646 | **0.6453** | 73.10% |
| validation | 12,245 | **0.6317** | 72.02% |
| test       | 12,451 | **0.6333** | 72.18% |

per-type 일치율: PER 78–80%, LOC 80–82%, ORG 81–82%, EVT 62–66%, PROD
52–55%. PROD 가 가장 의견 분기 큼 — 이 클래스는 후속 BERT 학습에서도
주의 대상.

### 2. Wikidata 인터링크 앵커 일치율 (recall-merge 결과 기준)

| split | total ents | with mapped type | matches | **agreement** |
|---|---:|---:|---:|---:|
| train      | 22,872 | 13,523 | 13,169 | **97.38%** |
| validation | 11,300 |  6,724 |  6,529 | **97.10%** |
| test       | 11,484 |  6,790 |  6,587 | **97.01%** |

per-type (train 기준): PER 98.9% · LOC 99.5% · ORG 90.1% · PROD 95.6%
· EVT 83.9%. #10의 8종 historical 트랙(91% 수준) 대비 명백한 향상.

### 3. Stockmark #16 1건 read-only spot-check

`data/stockmark/test.jsonl:183` (id=2942700) `セントメアリー病院` =
`LOC` 확인 (#16 7건 중 5종 매핑 후 분별 가능한 1건 정합). 부수: 같은
라인 `コロラド・メサ大学` = `ORG` (대학=CORP→ORG 매핑 정상).

### 4. PII 주입본 (`data/pii/wikiann_vi_pii.jsonl`) 검증 4종

전 39,979 레코드 / 97,306 entity 대상.

#### 4-1. 라벨 셋 ⊂ 10종

```
{CREDIT_CARD, DAT, EMAIL, EVT, ID_NUM, LOC, ORG, PER, PHONE, PROD}
```

10종 정확히 일치. 외부 라벨 누설 0건.

#### 4-2. Offset 정합성

`text[start:end] == ent.text` 위반: **0 / 97,306** entity.

#### 4-3. Prefix-동반률 + 영문 라벨 leakage

| 패턴 | 매치 수 | 비율 |
|---|---:|---:|
| 베트남어 단서어(`Liên hệ:` / `SĐT:` / `Email:` / `CCCD:` / `Địa chỉ:` / `Ngày sinh:` / `Thẻ:` / `ID:` / `CMND:`) | **0** | 0.00% |
| 영문 라벨 leakage(`PHONE:` / `EMAIL:` / `ID_NUM:` / `ID_NUMBER:` / `CREDIT_CARD:` / `DAT:` / `ADDRESS:` / `NAME:`) | **0** | 0.00% |

비교: 일본어 Stockmark 측정에서 단서어 동반 27.5%, 영문 라벨 leakage
170건. VI 프롬프트에 부정 예시(Rule 4·5) 선행 추가가 결정적.

#### 4-4. PII 밀도 분포

`{0:0.2, 1:0.4, 2:0.3, 3:0.1}` (`pii_max=3` 트림 후 동일).

라벨 7종 중 NAME→PER, ADDRESS→LOC 병합으로 5종만 직접 관측 가능.
**관측 가능 PII 카운트(EMAIL+PHONE+ID_NUM+CREDIT_CARD+DAT)** 분포:

| 관측 n | 실측 (count / %) | **이론값**\* |
|---:|---|---:|
| 0 | 13,146 (32.88%) | 32.8% |
| 1 | 17,682 (44.23%) | 44.3% |
| 2 |  8,051 (20.14%) | 20.0% |
| 3 |  1,100 ( 2.75%) |  2.86% |

\* DEFAULT_DENSITY 와 7-라벨 풀에서 5/7 가 관측 가능하다는 가정으로
초기-누락 후 NAME/ADDRESS 마스킹이 적용된 분포의 해석적 기댓값.

**해석**: 관측치가 이론값과 0.1pp 단위로 일치 → raw N 샘플링은
DEFAULT_DENSITY 와 정확히 부합한다. JA 측에서 보고됐던 1.8%
`samples_no_pii_ratio` 괴리는 `src/augmenters/pii/stats.py:21` 의
측정 결함(NER을 포함한 entities 전체 부재만 카운트)에서 비롯된 것으로
확정. **stats.py 수정은 본 이슈 범위 외, 별도 후속 이슈로 분리.**

## 코드 변경

| 파일 | 변경 |
|---|---|
| `src/augmenters/pii/llm_injector.py` | `_INJECTION_PROMPT_VI` 신규(부정 예시 포함), `build_injection_prompt(..., lang)` 매개변수, `LLMInjector` 비동기 일괄 처리(`_inject_async` + `_gather_inject` + 단일 `asyncio.run`) |
| `tests/augmenters/pii/test_llm_injector.py` | VI 단위 테스트 2건(`test_vi_prompt_contains_negative_examples`, `test_vi_empty_pii`) 추가 — 15/15 pass |
| `src/augmenters/wikiann_vi/__main__.py` | docstring 예시 경로 `gemma_8type.jsonl → gemma_test.jsonl` |
| `src/labelers/vi/dataset_loader.py` | docstring 경로 + `DEFAULT_PATH` 3건 `vi_wikiann_8type_recall_*.jsonl → vi_wikiann_recall_*.jsonl` |

`LLMInjector` 비동기 리팩터는 본 배치 1차 시도에서 record당 `asyncio.run`
이 새 event loop 을 닫을 때 `AsyncOpenAI` connection pool 이 cleanup
단계에서 `RuntimeError('Event loop is closed')` 를 발생시켜 openai
client 가 매 요청을 retry → 실효 처리량이 17h+ 로 부풀려진 문제를
해소하기 위함. 재현·진단 후 단일 event loop + `asyncio.gather` 설계로
교체. 결과: retry 0건, 전체 wall time 약 122분 (4.4 req/s × 31K).

## 후속 작업·알려진 한계

- **stats.py 측정 결함**: `samples_no_pii_ratio` 가 PII만 부재인 케이스를
  잡지 않음. 본 이슈 검증에서는 별도 분석 스크립트로 측정. 측정 정의
  변경은 호환성 영향이 있어 별도 이슈로 분리.
- **이중 silver**: WikiANN 원본(silver, 3종) × LLM 재라벨(silver, 5종).
  절대 F1 비교는 의미 없으며 스키마 일관성·kappa·Wikidata 앵커 일치율
  로만 품질을 보증한다.
- **PROD 클래스 합의 약함**: kappa per-type ≈ 0.52–0.55. 후속 BERT
  학습 시 클래스 불균형·경계 모호성 별도 점검 필요.
- **카테고리 B/C/D/E 명칭 정리**: JSONL 필드명·모듈명·스펙 문서명·
  과거 docs는 #21 호환성 결정에 따라 본 이슈에서 변경하지 않음. 향후
  메이저 마이그레이션에서 다룰 후보.
