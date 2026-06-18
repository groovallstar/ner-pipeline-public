# 합성 PII 주입 — 방식 · LLM 사용 기준 · 검증 정책

> 프로젝트가 합성 PII를 **어떤 방식으로 주입**하고, **LLM을 언제
> 쓰는지**, **검증·병합 정책**은 무엇인지 정리한다.
> 모듈: `src/ner/augmenters/pii/` (파일별 API는 `src/ner/augmenters/AGENTS.md`).
> 라벨 정의의 단일 출처: `docs/manual/data/canonical-entity-schema.md`.

---

## 0. 왜 합성 PII 주입인가

이 파이프라인의 목표 중 하나는 **개인정보(PII) 검출·마스킹**이다. 모델이
PII를 검출하도록 학습하려면 PII가 라벨된 데이터가 필요한데, 공개 NER
데이터셋에는 PII가 없거나(일본어 Stockmark) 라벨 공간이 좁다(베트남어
WikiANN 3종). 따라서 PII를 **데이터 증강으로 합성해 주입**한다.

목표 라벨 공간은 **10종 평면 canonical** = NER 5종(`PER/LOC/ORG/PROD/EVT`)
+ PII/날짜 5종(`EMAIL/PHONE/DAT/ID_NUM/CREDIT_CARD`)이다.

인간 재어노테이션 비용을 피하되 LLM 단일 출력의 환각·shortcut learning
위험을 줄이기 위해, **주입 방식**과 **검증 정책**을 분리해 설계한다.

---

## 1. 주입 방식 — LLM 자연 주입

모든 언어(일본어·베트남어·한국어)에서 **LLM 자연 주입**을 표준으로
사용한다. LLM에 "원문에 PII를 자연스러운 문맥으로 삽입하라"고 요청하고,
생성 텍스트에서 string match로 PII span 오프셋을 추출한다(`LLMInjector`,
`--mode llm`). 주입 후 `harden_pii_format_collisions`로 무라벨 CC·ID_NUM
포맷 열을 일관 relabel 한다. 프롬프트는 언어별
(`_INJECTION_PROMPT_JA`·`_INJECTION_PROMPT_VI` …)이며 `_INJECTION_PROMPTS`
dict에서 `lang` 키로 선택된다.

> **왜 단순 규칙 삽입을 쓰지 않나**: 문장 앞·뒤·랜덤 위치에 PII를 끼워
> 넣으면 "문두·문말 근처의 숫자열·@ = PII"라는 위치 기반 shortcut을
> 학습해, 문장 중간에 자연스럽게 등장하는 PII에 일반화하지 못한다. 또
> 자연 문장의 trigger phrase(「担当の○○」「연락처는 ○○」)가 사라져
> 실제 문서에서 감지력이 떨어진다. 그래서 모든 언어에서 자연 주입을
> 택한다.

규칙 기반 suffix 모드(`PIIInjector`, `--mode suffix`: 문장 끝에 단서어 +
합성 값 접미, LLM 없이 결정론적)도 CLI에 남아 있으나 현재 언어별
파이프라인에서는 사용하지 않는다.

### 언어별 적용

- **일본어** — LLM 자연 주입
- **베트남어** — LLM 자연 주입
- **한국어** — LLM 자연 주입

---

## 2. 2단계 구조 — 주입 + 교차 검증

LLM 주입 출력을 그대로 학습에 쓰지 않고, **독립한 두 번째 LLM**으로
재라벨해 교차 검증한다.

| 단계 | 모듈 | 역할 |
|---|---|---|
| 1. Inject | `LLMInjector` / `PIIInjector` | PII 삽입 + span 오프셋 산출 |
| 2. Verify | `PIIVerifier` | 주입 결과를 독립 재라벨해 span이 PII·원본 엔티티 양쪽에서 재검출되는지 확인 |

검증기는 `label_spans(split=False)`로 호출해 문맥을 보존하며, 각 주입
span을 **confirmed / missed / conflict**로 분류한다. CLI는
`python -m ner.augmenters.pii`이며 `--inject-url/--inject-model`과
`--verify-url/--verify-model`을 분리 지정해 서로 다른 모델을 주입·검증
역할에 배치할 수 있다.

> **검증을 언제 생략하나**: 주입기의 span 추출이 오프셋 정확성을 보장하는
> 경우(한국어 — 자연 주입의 `extract_spans`가 오프셋을 정확히 산출)에는
> verify 단계를 건너뛰어 사람이 만든 원본 gold를 그대로 보존한다.

---

## 3. 정책

### 3.1 검증 정책 (`--verify-policy`)

| 정책 | 동작 | 쓰는 경우 |
|---|---|---|
| `drop_span` (기본) | `missed`(검증기 미검출) + `conflict`(타입 불일치) span을 제거하고 **레코드 자체는 보존** | 학습 품질 우선 — 불확실 span만 솎아냄 |
| `drop_record` | `missed`·`conflict` 중 하나라도 있으면 레코드를 통째 제거 | 보수적 정제가 필요할 때 |
| `keep_all` | 분류만 하고 모두 유지 | 검증 진단·분석용 |

### 3.2 라벨 병합

canonical 정렬을 위해 주입 라벨을 무조건 병합한다: `NAME → PER`,
`ADDRESS → LOC`(`config.py::DEFAULT_MERGE_RULES`). 주입 밀도·대상 PII
종류·seed는 `InjectionConfig(lang, density, pii_labels, seed)`로 제어한다.

---

## 4. 데이터 생성 결과

구체 산출 경로·레코드 수는 데이터 재정리로 의미가 없어 생략하고, **생성
방식**만 정리한다.

- **일본어** — Stockmark 5종 NER 원본에 LLM PII 5종을 자연 주입해 10종
  평면 학습셋을 생성한다(주입 후 검증 적용).
- **베트남어** — WikiANN 재라벨 silver(5종 NER) 위에 PII 5종을 LLM 자연
  주입해 10종 평면을 생성한다. (재라벨 silver 생성·검증은 별개 상류 단계
  — `docs/manual/data/vietnamese-ner.md §7`.)
- **한국어** — KLUE 유래 NER gold에 PII 4종(`DAT` 제외 — 이미 존재)을
  LLM 자연 주입한다. 검증 없이 원본 gold를 보존한다.

각 레코드는 원본 NER span과 주입된 PII span을 함께 보관하며, 주입·검증
통계를 부산물로 남긴다.

---

## 부록: 모델 선정·운영 이력

주입·검증에 투입할 LLM은 ① 베이스 NER span F1(원본 엔티티 보존력),
② 샘플당 처리 시간, ③ VRAM·GPU 수(주입·검증 듀얼 부팅 가능 여부),
④ 합성 PII에 대한 안전 필터 거부율을 기준으로 고른다.

시점별 구체 모델 선택과 벤치 수치는 본문에 박지 않고 리포트에 위임한다:
`docs/reports/japanese-ner-benchmark.md`,
`docs/reports/japanese-ner-pii-benchmark.md`,
`docs/reports/vietnamese-ner-silver-quality.md`.
