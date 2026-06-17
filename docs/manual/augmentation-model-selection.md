# 일본어·베트남어 NER 데이터 증강 모델 선정 메타 (2026-04)

> JA(PII 주입)·VI(WikiANN 재라벨) 양쪽 파이프라인의 공통 배경·각 언어의 증강 방식·모델 선정 기준을 한 곳에 모은 메타 문서.
> 실제 측정 숫자는 각 리포트(`japanese-ner-benchmark.md`, `japanese-ner-pii-benchmark.md`, `vietnamese-ner-silver-quality.md`)를 참조.
> 라벨 정의의 단일 출처: `docs/manual/data/canonical-entity-schema.md` (JA·VI 공통).

## 0. 두 파이프라인의 공통 배경

본 프로젝트의 다국어 NER은 공개 데이터셋만으로는 학습 목표를 달성할 수 없다.

- **일본어 (Stockmark NER Wikipedia)** — 8종 NER 라벨은 있으나 **PII가 전혀 포함되지 않음**. PII 검출·마스킹이 목표이므로 합성 PII 주입이 필수.
- **베트남어 (WikiANN-vi)** — **3종(PER/LOC/ORG)만 BIO 태그됨**. 프로젝트 통합 라벨 공간(canonical 5종 NER `PER/LOC/ORG/PROD/EVT`)에 부합하려면 PROD/EVT 추가 + 시설=ORG 경계 재정의가 필요.

두 경우 모두 인간 어노테이션을 새로 생성하는 비용을 회피하기 위해 **LLM 기반 증강**을 채택한다. LLM이 단일 출력을 그대로 학습 데이터로 쓰면 환각(hallucination)·shortcut learning 위험이 크므로, 두 파이프라인 모두 **두 번째 LLM(또는 외부 앵커)으로 교차 검증**하는 2단계 구조를 공유한다.

이 문서가 측정·비교하는 대상은 그 2단계에 투입할 후보 LLM이다.

## 1. 일본어 — 합성 PII 주입 증강

### 1.1 증강 방식

가장 단순한 방법은 규칙 기반 주입기(`src/ner/augmenters/pii/injector.py::PIIInjector`)로 문장의 앞·뒤·랜덤 위치에 PII 토큰을 삽입하는 것이다. 이 방식은 구현이 간단하지만 두 가지 위험이 있다.

1. **문맥 부자연** — "…である。 電話 090-1234-5678" 처럼 문장과 무관하게 끼워 넣으면, BERT의 positional embedding과 `[SEP]`·구두점 경계 토큰에 대한 self-attention이 결합해 "문장 처음과 끝 근처의 숫자열·@ = PII"라는 위치 기반 shortcut을 학습한다. 결과적으로 정규식 수준의 표면 패턴만 잡고, 문장 중간에 자연스럽게 등장하는 PII에는 일반화하지 못하는 shortcut learning 문제가 발생한다.
2. **주변 단서 결여** — 자연 문장에서 PII는 보통 「担当の○○」「連絡先は○○」처럼 trigger phrase와 함께 등장하는데, 랜덤 삽입에서는 이 단서가 사라져 실제 문서에서는 PII가 등장해도 감지하지 못할 가능성이 있다.

이를 회피하기 위해 **LLM에게 "원문에 PII를 자연스러운 문맥으로 삽입하라"고 요청**해 학습 데이터를 만든다 (`src/ner/augmenters/pii/llm_injector.py::LLMInjector`). 프롬프트는 언어별 `_INJECTION_PROMPT_JA`·`_INJECTION_PROMPT_VI`로 분리되어 있고, `_INJECTION_PROMPTS` dict에서 `lang` 키로 선택된다. `_INJECTION_PROMPT`는 `_INJECTION_PROMPT_JA`의 하위 호환 alias다.

### 1.2 2단계 구조 (Inject + Verify)

| 단계 | 클래스 | 역할 |
|---|---|---|
| 1. Inject | `LLMInjector` | 원문에 PII를 자연스럽게 삽입하고 생성 텍스트에서 span offset을 추출 |
| 2. Verify | `PIIVerifier` | 주입 결과를 **독립적으로 재라벨링**, 주입 LLM이 만든 span이 자연 문맥에 녹아들어 PII·원본 엔티티 양쪽에서 재검출되는지 교차 검증. `policy=drop_span`으로 검증자가 미검출한 `missed` span과 타입 불일치 `conflicts` span을 모두 제거(레코드 자체는 보존) |

CLI(`src/ner/augmenters/pii/__main__.py`)는 `--inject-url/--inject-model`과 `--verify-url/--verify-model`을 분리 지정할 수 있어, **성능이 가장 높은 상위 2개 모델을 각각 주입·검증 역할로 배치**하는 실제 워크플로우와 일치한다.

### 1.3 라벨 공간 · 산출물

- 라벨 공간: **10종 평면 canonical** = 5종 NER (`PER/LOC/ORG/PROD/EVT`) + 5종 PII/날짜 (`EMAIL/PHONE/DAT/ID_NUM/CREDIT_CARD`)
- LOC/ORG 경계: 시설=ORG (#27 Phase 4)
- 입력: `data/stockmark/{train,test}.jsonl` (5종 NER 원본, PII 미주입)
- 출력: `data/stockmark/pii_{train,test}.jsonl` + `.stats.json` + `.verify.json` (10종 평면, train 4,243 / test 1,064 records)

### 1.4 두 측정 리포트의 역할 분담

| 벤치 | 데이터셋 | 라벨 공간 | 확인하는 능력 |
|---|---|---|---|
| japanese-ner-benchmark | `data/stockmark/test.jsonl` (1,069 samples, PII 미주입) | gold 5종, 프롬프트는 10종 평면 (Raw/Filtered F1 분리 보고) | PII 주입 대상인 **원본 5종 NER**에 대한 기본 품질 — 주입 과정에서 기존 span을 깨뜨리지 않을 기반선 |
| japanese-ner-pii-benchmark | `data/stockmark/pii_{train,test}.jsonl` | **10종 평면 canonical** | **주입 대상 태스크 자체의 최종 품질** — 5종 NER + 5종 PII에서 주입·검증 파이프라인이 실제로 유효한 span을 뽑아낼 수 있는지 |

10종 통합 프롬프트를 5종 gold에 매기면 PII 출력은 모두 FP가 된다. 이를 분리하기 위해 NER 벤치는 **Raw F1**(전체 출력 그대로)과 **Filtered F1**(예측에서 PII 5종 제외)을 모두 보고하며, silver 라벨러 정당성 결론은 Filtered F1으로 내린다.

## 2. 베트남어 — WikiANN 재라벨 확장

### 2.1 증강 방식

WikiANN-vi는 BIO 태그 + 3종(PER/LOC/ORG)만 제공한다. 프로젝트 canonical은 5종 NER이므로:

- **PROD/EVT** 두 타입은 원본에 부재 — 새로 어노테이션해야 함
- **시설(역·공항·병원·학교 등)** 의 LOC/ORG 분류가 #27 경계 재정의로 ORG로 이동 — 기존 LOC 일부를 ORG로 재할당해야 함

이를 인간 어노테이션 없이 해결하는 방법으로 **LLM 재라벨**을 채택한다 (`src/ner/augmenters/wikiann_vi/relabel_8type.py::Relabeler`). LLM은 원문 전체를 받아 5종 canonical로 span을 출력하고, 매칭 단계에서 원문 문자 오프셋으로 변환한다 (`match_offsets`).

> 파일·필드명의 `8type` 리터럴은 이슈 #21 축소 이후에도 데이터 호환성을 위해 유지한다(의미는 canonical 5종).

### 2.2 2단계 구조 (재라벨 × 2 + 외부 앵커)

JA가 inject→verify 직렬이라면 VI는 **두 모델의 독립 재라벨 + 외부 앵커**를 병렬로 합쳐 신뢰도를 추정한다.

| 단계 | 모듈 | 역할 |
|---|---|---|
| 1. 재라벨 A | `relabel_8type.py` | Gemma로 5종 재라벨 (recall 우선) |
| 2. 재라벨 B | `relabel_8type.py` | Qwen3.6 MoE로 5종 재라벨 (precision 우선) |
| 3. Cross-model κ | `kappa.py` | 두 모델 합의율(Cohen's kappa) — coverage·type 합의 분리 측정 |
| 4. Wikidata 앵커 | `wikidata_anchor.py` | 표면형 → vi.wikipedia 페이지 → Wikidata Q-ID → P31 → canonical 매핑(`WIKIDATA_TO_CANONICAL`)으로 외부 정답 추정 |
| 5. 신뢰도 병합 | `merge_confidence.py` | 두 결과를 span별 confidence(`high/medium_recall/medium_prec/conflict`)로 병합. 정책: `recall`/`precision`/`high_only`/`full`/`recall_strict`(PROD·EVT만 `high`로 제한) |

이슈 #30(`vi_silver_quality.py`, 본 시점 진행 중)에서 silver 재생성 후 **WikiANN 3-gold(PER/LOC/ORG)**로 silver의 절대 F1을 측정하여 production 채택 기준을 확정한다.

### 2.3 라벨 공간 · 산출물

- 라벨 공간: **canonical 5종 NER** (`PER/LOC/ORG/PROD/EVT`) — 재라벨
  단계. 이슈 #8 의 PII 주입 단계에서는 5종 PII (`EMAIL/PHONE/DAT/
  ID_NUM/CREDIT_CARD`) 가 추가되어 **10종 평면** 으로 확장.
- LOC/ORG 경계: 시설=ORG (#27, `WIKIDATA_TO_CANONICAL`에서 시설 Q-ID는 ORG로 재배치)
- 입력: HF `unimelb-nlp/wikiann` config=`vi` (BIO 3종)
- 출력 (재라벨): `data/wikiann_vi/{gemma,qwen}_{train,validation,test}.jsonl`
  + `vi_wikiann_recall_*.jsonl` + `kappa_*.json` + `wikidata_anchor_*.json`
  (모두 `.gitignore`)
- 출력 (PII silver, #8): `data/wikiann_vi/{train,valid,test}.jsonl` (5종
  canonical, recall_strict policy) + `pii_{train,valid,test}.jsonl` (10종)
- 레코드 스키마: `{id, text, gold_spans, gold_spans_8type, relabel_model}`
  — 원본 `gold_spans`(WikiANN 3종)과 silver `gold_spans_8type`(canonical
  5종)을 한 레코드에 병행 보관

### 2.4 측정 리포트

| 리포트 | 단계 | 라벨 공간 | 확인하는 능력 |
|---|---|---|---|
| `vietnamese-ner-silver-quality.md` | 재라벨 (#30) | 5종 NER | 두 재라벨 모델의 합의율·Wikidata 앵커·WikiANN 3-gold F1. PROD/EVT 의 학습 가능 임계 검증 |
| `vietnamese-ner-benchmark.md` | 재라벨 후 NER 벤치 (#8) | 5종 NER (silver gold) | PII 주입 전 라벨러 정당성 — 10종 통합 프롬프트 운영 시점 동작 |
| `vietnamese-ner-pii-benchmark.md` | PII 주입 후 (#8) | 10종 평면 | suffix-mode 합성 PII silver + Qwen verifier 신뢰율 (NER 88~94% / PII 98~99%), 모델별 10종 F1 |

### 2.5 PII 증강 (이슈 #8 추가)

VI 도 JA 와 동일 캐노니컬 라벨 공간 (10종 평면) 을 갖도록, #30 silver
위에 합성 PII 5종을 부착한다. JA 의 LLM-inject 방식 대신 **suffix-mode
규칙 주입** 을 채택 — 문장 끝에 단서어 + 합성 값(`SĐT: 090...`,
`Email: a@b.com`) 을 부착하므로 inject 단계 LLM 호출이 불필요하다.

| 단계 | 모듈 | 역할 |
|---|---|---|
| 1. Inject | `pii/__main__.py --mode suffix` | 단서어 + 합성 값 부착 (LLM 호출 0) |
| 2. Verify | `PIIVerifier` (Qwen3.6 단독) | 부착 결과의 span 합치성 검증, `drop_span` policy |

JA 와의 차이:
- inject LLM 미사용 — VI 라벨러의 자연도 prompting 부담 회피
- verify 단일 모델 (Qwen3.6) — JA 의 dual (Gemma inject + Qwen verify) 대비
  cross-bias 측정은 별도. 실측 confirmed=93.7% 로 운영 충분.

## 3. 모델 선정 기준

JA·VI 모두 후보 모델은 다음 조건을 동시에 만족해야 한다.

### 3.1 공통 기준 (우선순위)

1. **베이스 NER Span F1** — 두 파이프라인 모두 5종 NER 위에 작동. 기본 NER 품질이 떨어지면 PII 주입에서는 원본 span이 깨지고, VI 재라벨에서는 신규 타입을 잘못 만들어낸다.
2. **샘플당 처리 시간 (sec/sample)** — JA는 inject 1회 + verify 1회 = **샘플당 2회 LLM 호출**, VI는 모델 A 1회 + 모델 B 1회 = **샘플당 2회 LLM 호출**. 두 파이프라인 모두 데이터셋 전체(JA train 4.3K / VI train 20K)를 반복 처리하므로 처리량이 곧 운영 비용.
3. **VRAM·GPU 수** — 두 모델을 동시 부팅(주입·검증 또는 재라벨 A·B)하므로 단일 호스트에서 TP=1 듀얼 부팅 가능 여부가 실용성 분기점. AWQ-8/AWQ-4 양자화가 우선 후보.
4. **안전 필터·거부율** — JA는 합성 PII에 대한 거부 응답(특히 외부 API GPT 계열) 회피를 위해 system prompt에 disclaimer 도입부 필요. VI는 일반 NER이라 영향 적음.

### 3.2 일본어 특이 기준

- **Filtered F1 ≥ 0.85** (5종 원본) — 주입 과정에서 기존 엔티티 보존력 기준선. 현재 측정에서 gemma-4-31B-it BF16 0.8772, gemma-4-31B-it-AWQ-8bit 0.8767로 1차·2차 후보.
- **10종 평면 F1** (PII 혼합) — 주입·검증 역할에서 실제로 필요한 능력. 5종 능력이 같으면 PII 패턴 추출력으로 차별화.
- **외부 API의 disclaimer 응답성** — `system prompt`에 "합성 데이터·정보 추출 전용" 명시 후에도 거부하는 모델은 운영 불가.

### 3.3 베트남어 특이 기준

- **Cross-model κ ≥ 0.55** — 두 후보 모델 사이 합의율. 0.5 미만이면 silver 학습 데이터의 잡음이 너무 커 production 부적합.
- **Wikidata 앵커 일치율** — 외부 silver(P31)와의 정합. 90% 이상이면 LLM 재라벨이 도메인 정답에 강하게 정렬돼 있다고 본다.
- **PROD·EVT 합의율** — 신규 타입은 두 모델 모두 합의해야 안전. 합의율 낮으면 `recall_strict` 정책으로 PROD·EVT는 `high`만 통과시켜 보수화.
- **3-gold(WikiANN PER/LOC/ORG) 대비 silver F1** — 이슈 #30에서 측정 중. 3-gold 절대 F1이 낮으면 재라벨러 자체를 교체.

### 3.4 운영 매트릭스

| 역할 분담 | JA 후보 (2026-04 기준) | VI 재라벨 (2026-04) | VI PII inject (#8) |
|---|---|---|---|
| Primary (recall) | gemma-4-31B-it-AWQ-8bit | gemma-4-31B-it-AWQ-8bit | n/a (suffix 규칙 주입) |
| Secondary (precision) | Qwen3.6-35B-A3B-AWQ-4bit | Qwen3.6-35B-A3B-AWQ-4bit | Qwen3.6-35B-A3B-AWQ-4bit (verify) |
| 비고 | inject=Primary, verify=Secondary (drop_span) | A/B 독립 재라벨 후 `recall_strict` 정책 병합 | LLM inject 미사용 → 운영 비용 ↓ |

상세 권장 설정·전체 모델 비교는 각 리포트의 최종 섹션("권장 설정")을 참조한다.
