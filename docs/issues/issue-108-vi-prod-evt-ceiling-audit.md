# issue-108 — VI PROD/EVT 천장 진단 (JA §3 기준 적용 + 소규모 gold 감사)

- GitHub: #108 (area:classifier, area:augmenters)
- 브랜치: `feat/issue-108-vi-prod-evt-ceiling-audit`
- 선행: 리포트 §천장, issue-103, **issue-84 deferred follow-up**(`:105-107`)

## 목적

VI PROD(strict F1 ~0.71)·EVT 천장이 (a) silver 노이즈 / (b) 모델 한계 /
(c) schema 미규정 중 무엇인지 소규모 감사로 결판. 판정 루브릭 = canonical §3
회색지대 기준(JA #84/#85 확정, 공통).

## 선조사 결론 — JA 기준 적용 가능성

| 축 | 결론 | 근거 |
|---|---|---|
| 의미 규칙(§3·3.1·3.2·3.3) | **직접 이식 가능** | "공통(JA·VI)" 명시 + VI 예시 존재 |
| VI gold 적용 여부 | **미적용** | `wikiann_vi/prompts.py`는 §3 기본만, 3.1/3.2/3.3 누락 |
| 결정적 모순 | **VI PROD에 서비스 포함** | `dịch vụ` ∈ PROD — #84가 JA서 제거(1042→943)한 정의 |
| 어휘 시그널 | **VI 현지화 필요** | `発売`/`本社`→`phát hành`·`ra mắt`/`trụ sở`·`thành lập` |

→ VI 천장은 JA가 #84 전 겪던 "schema 미규정 → gold 비일관" 상태일 공산.

## 측정 기준선 (fold0, phobert)

| | P | R | gold n | FN | FP | 불일치 |
|---|---:|---:|---:|---:|---:|---:|
| PROD | 0.725 | 0.698 | 404 | ~122 | ~107 | ~230 |
| EVT | 0.830 | 0.874 | 95 | ~12 | ~17 | ~29 |

## 계획 (체크박스 = 추적 단위)

- [ ] 불일치 추출 스크립트 — fold0 phobert pred vs silver, PROD+EVT FN/FP 전수
      + 일치 대조군 ~40. (test_predictions.json 재사용 가능)
- [ ] §3 루브릭 판정 — vLLM 1차(§3 전문 주입) → 사용자 flagged 검토
      (schema §4.1 JA 선례). 3분류: silver오류 / 모델오류 / schema-갭
- [ ] IAA — 같은 30건 2회 판정으로 클래스 정의 모호성 측정
- [ ] 적용성 산출 표 — §3 규칙별 VI 직접이식 vs 시그널 현지화 목록
- [ ] 보정 PROD/EVT F1 추정(silver오류 제외) = 진짜 천장
- [ ] 결론 리포트 + refuter PASS

## 범위·전제

- **진단 + 프롬프트 정렬 remediation**(사용자 결정으로 동일 이슈 진행). gold
  re-silver + 재학습은 #103 baseline gold를 변경하고 hours 소요라 **별도 go/no-go**.
- 정직한 전제: JA #84 동종 보정은 F1 레버 아님(비대칭 P-레버, per-fold std
  5.24pp 노이즈 대역 내). 단 VI는 실패 모드가 *누락*(precision)이라 re-silver 시
  JA보다 precision 이동 여지 있음(미검증).
- 한계: 불일치 표본은 모델이 교정한 silver 오류만 포착, 모델도 틀린 건 누락
  → 대조군으로 일부 보완.

## 결정 (사용자 비준)

- **판정 주체 = vLLM 1차 + 사용자 검토**(schema §4.1 JA 선례). vLLM에 §3 전문을
  루브릭으로 주입해 ~300건 1차 분류 → 불확실·flagged 케이스만 사용자 검토.
- 착수 = **계획 검토 후**(본 문서 사용자 확인 뒤 구현).
- 산출 위치: issue-108 md 내장(분량 크면 `docs/reports/` 분리).

## 구현 결과

도구: `src/ner/scripts/audit_vi_prod_evt.py` (extract/adjudicate/analyze).
판정: gemma-4-31B-it-AWQ-8bit @ vLLM, §3 루브릭 주입, temp=0. 대상: fold0
phobert. 산출: `results/classifier/vi/audit/`(gitignore).

**추출**: PROD 불일치 206(FN 122·FP 84) + EVT 26(12·14) + 일치 대조군 40 = 272.

**§3 판정 → 3분류 + 보정 F1**:

| | orig F1 (P·R) | 보정 F1 (P·R) | model_err | silver_err | both | schema_gap | agree |
|---|---|---|---:|---:|---:|---:|---:|
| PROD | 0.711 (.725·.698) | **0.813** (.897·.743) | 116 | 68 | 21 | **0** | 27 |
| EVT | 0.851 (.830·.874) | **0.901** (.910·.892) | 15 | 9 | 3 | **0** | 12 |

(EVT fold0=0.851은 pooled 0.792보다 높음 — fold0 EVT n=95로 작음.)

### 천장 분해 (사용자 원질문 "silver냐 모델이냐"의 답)

- **PROD precision(.725→.897)은 대부분 silver 탓** — 모델 FP 84건 중 **67건이
  silver 누락**(model이 맞음). silver가 창작물(노래·앨범·애니·TV·교향곡)을
  PROD로 안 잡는 체계적 누락이 주범.
- **PROD recall(.698→.743)은 대부분 모델 탓** — 보정해도 거의 안 오름. 모델이
  실제 PROD 121건을 진짜로 놓침(genuine 약점).
- → **천장은 축별로 갈린다**: precision 천장=silver 노이즈, recall 천장=모델
  한계. JA #84의 "비대칭 P-레버"와 정확히 일치(언어 간 재현).
- **보정(진짜) 천장 추정**: PROD ~0.81, EVT ~0.90 (LLM 판정 기준).

### JA §3 기준 적용 가능성 — 판정

- **적용됨, 깨끗이**: `schema_gap=0` — 판정자가 272건 전부 §3로 결정 가능했고
  모호 플래그 0. 즉 **클래스 정의(§3) 자체는 병목 아님** — 내 1차 가설(클래스
  모호성)은 이 판정으론 **미지지**.
- **단 VI gold가 §3을 미반영**이 확증: `'Pháp lệnh...'`(법령)을 silver가 PROD로
  오라벨(§3.1 "법령=비-entity" 위반) + 창작물 대량 누락(§3.2 PROD positive
  미적용). 이식 작업 = (1) VI 프롬프트에 §3.1/3.2/3.3 추가, (2) 창작물 커버리지
  re-silver. 어휘 시그널 현지화는 본 표본에선 미발동(창작물은 명확).

### 정직한 한계

- 보정 F1의 "gold"는 **LLM(gemma)**, 사람 아님 → 추정치. silver_err 77건이
  보정을 견인하므로 사용자 검토(선택한 워크플로)는 이걸 표집 점검해야.
- fold0 단일. 대조군 40 중 1건만 오류(agree 39) → 불일치 표집의 사각(모델도
  같이 틀린 silver 오류)은 작음(~2.5%).
- `schema_gap=0` 보강(IAA): temp0 vs temp0.7 재판정 **라벨 일치 271/271=1.000**,
  양쪽 ambiguous 0, flip 0 → 판정자 자기일관성 완전(케이스가 §3로 명확).
  단 이는 *모델 자기일관성*이지 사람-LLM IAA는 아님 — 그 축은 사용자 검토.
- both_error 24는 다수가 경계(boundary) 불일치(type 합의) — 깊은 모호성 아님.

## remediation — VI 프롬프트 §3 정렬 (사용자 검토 확정 후)

감사가 증명한 VI gold-생성 프롬프트의 §3 divergence를 정렬. 대상 2파일 4사이트:
`augmenters/wikiann_vi/prompts.py`(재라벨 = classifier silver 생성원, single+batch),
`labelers/vi/ner_prompts.py`(LLM 벤치 라벨러, single+system).

- **PROD에서 `dịch vụ`(서비스) 제거** + §3.2 positive 제한(서비스·SaaS·기술
  표준·상 = 비-entity) 명시 — 4사이트 전부.
- **§3.1/§3.3 회색지대 규칙 추가**: 법령/pháp lệnh/nghị định = 비-entity(과라벨
  `Pháp lệnh` 직접 차단), 다년 process(Chiến tranh Lạnh·Minh Trị Duy tân)·
  시대·추상 topic = 비-entity(과라벨 `Minh Trị Duy tân` 직접 차단), 제조 artifact·
  named 프로젝트/전시 = 규칙화.
- 어휘 시그널은 VI로 현지화(법령=`pháp lệnh/nghị định`, 출시=`ra mắt` 등).

**효과 범위(정직)**: 이 변경은 *gold 생성*을 §3에 맞추는 것 — 기존 gold·#103
벤치 수치는 **불변**. 천장을 실제로 옮기려면 re-silver+재학습 필요(별도 go/no-go).
remediation의 직접 검증은 과라벨 2건이 새 프롬프트로 비-entity 처리되는지(규칙
명문화 = 결정적). 누락 75건은 re-silver 시 회복 여부 측정 대상.

## remediation 측정 결과 (격리 re-silver + 5-fold 재학습)

방법(`src/ner/scripts/resilver_vi_isolated.py`): 변수를 *프롬프트 하나*로 격리
— pii_all 각 행의 원문(주입 전)만 새 §3 프롬프트로 **두 모델(Gemma+Qwen)
재라벨 → 동일 `recall_strict` 합의 merge**(2-LLM 합의 기준 *유지*) → 원문
NER만 교체(주입 PII·텍스트·5-fold split 고정) → `pii_all_v2.jsonl`. 검증:
offset 정합 91,543/91,543, PII 주입분 Δ0, 회복 PROD 275종 실제 작품
(`Music of the Sun`·`Vietnam Idol`·`Boeing B-50` 등).

gold 변화(v1→v2): PROD 2,005→2,394(+389, 창작물 회복), EVT 474→436(−38,
과라벨 제거). 과라벨 `Minh Trị Duy tân`·`Pháp lệnh` 제거 확인.

**5-fold pooled strict F1 (v1 #103 → v2 재학습):**

| Entity | phobert v1→v2 | xlm-r-base v1→v2 |
|---|---|---|
| overall | 0.9461 → **0.9529** (+0.7) | 0.9459 → **0.9543** (+0.8) |
| **PROD** | 0.7171 → **0.7920** (**+7.5**) | 0.7097 → **0.7614** (**+5.2**) |
| EVT | 0.7920 → 0.7731 (−1.9) | 0.8031 → 0.7348 (−6.8) |
| PER | 0.9272 → 0.9278 | 0.9317 → 0.9434 |
| LOC | 0.9349 → 0.9563 (+2.1) | 0.9291 → 0.9510 (+2.2) |
| ORG | 0.8814 → 0.8994 (+1.8) | 0.8888 → 0.9037 (+1.5) |

- ✅ **PROD 천장 해결**: +5~7.5pp, recall ~.83 — 모델이 회복된 창작물을 학습해
  찾음. 감사 보정 추정(0.81) 사실상 적중(0.79). 천장은 데이터(gold) 문제였음.
- ✅ overall·LOC·ORG 동반 상승(더 완전한 gold).
- ⚠️ **EVT 회귀**(양 모델 일관, −1.9/−6.8pp): 저support(436) EVT가 새 프롬프트로
  2모델 합의가 더 갈려 sparse·noisy. 일부는 과라벨(쉬운 양성) 제거의 정당한
  결과지만 순회귀는 사실 — **EVT는 미해결, 별도 손질 필요**(EVT few-shot·합의 정책).
- 채점: 옛 phobert 예측을 v2 gold로 재채점하면 0.711→0.655(하락)인데, 이는
  *옛 모델*을 새 gold로 본 불공정 비교 — *재학습*하면 0.792로 오름(데이터 한계 확증).

산출물: `results/classifier/vi/resilver_v2/<model>/fold{0..4}/`, gold
`data/wikiann_vi/pii_all_v2.jsonl`(기존 gold 보존, #103 baseline 불변).

## 검증

- [x] `pytest tests/ner/classifier/ -q` green (52) + labelers/vi·augmenters
  prompt 테스트 80 green
- [x] `ruff check` clean (audit 스크립트 + 2 프롬프트 파일)
- [x] 프롬프트 `.format` 렌더링 무결 + `dịch vụ` 는 "KHÔNG gồm"(제외) 절에만 잔존
- [x] refuter 게이트 PASS (`fafb57f4f1a4` — 진단 축; 수치↔JSON 일치, 보정 F1
  산술 정합. 지적(EVT 0.90 fold0 한정)은 리포트 한정자 추가로 반영)
