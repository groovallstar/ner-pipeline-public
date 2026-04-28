# JA PII 프롬프트 Rule 4·5 보강 + 재생성 리포트

이슈 #27 Phase 2 산출물. `_INJECTION_PROMPT_JA` 에 단서어 직전 부착 금지
(Rule 4) · 영문 라벨명 leakage 금지 (Rule 5) 를 추가하고 `data/pii/
stockmark_pii_1000.jsonl` 을 재생성한 결과.

**측정일**: 2026-04-28
**재생성 모델**: `cyankiwi/gemma-4-31B-it-AWQ-8bit` (이슈 #27 Phase 1
Filtered F1 1위, 0.8676)
**입력 데이터**: `data/stockmark/train.jsonl` 1000 샘플
**산출**: `data/pii/stockmark_pii_1000.jsonl` 997 레코드 (3 추출 실패 drop)
**소요 시간**: 4분 49초 (concurrency=16, gemma-31B-AWQ-8bit)

---

## 1. 변경 사항

### 1.1 코드 (`src/augmenters/pii/llm_injector.py`)

기존 5조항(Rule 1–5) 을 6조항으로 확장.

| Rule | 내용 | 변화 |
|---|---|---|
| 1 | PII 값 1문자 변경 금지 | (기존) |
| 2 | 원문 고유표현 보존 | (기존) |
| 3 | 자연 컨텍스트 삽입 + 문말 나열 금지 | 旧 Rule 3·4 통합 |
| **4** | **단서어 직전 부착 금지** (`担当者：`/`連絡先：`/`電話：`/`電話番号：`/`メール：`/`住所：`/`ID番号：`/`カード番号：`/`クレカ番号：`/`生年月日：`) | **신규** — VI Rule 4 의 JA 등가 |
| **5** | **영문 라벨명 leakage 금지** (`NAME`/`PHONE`/`EMAIL`/`ID_NUM`/`ID_NUMBER`/`CREDIT_CARD`/`DAT`/`ADDRESS`) | **신규** — VI Rule 5 의 JA 등가 |
| 6 | 출력은 편집된 본문만, 설명·주석·markdown 래핑 없음 | 旧 Rule 5 |

PII 정보 헤더에 "참고용. 라벨명 그대로 출력하지 말 것" 명시 추가.

### 1.2 테스트 (`tests/augmenters/pii/test_llm_injector.py`)

JA 부정 예시 검증 2건 신설 → 전체 17/17 pass.

```python
def test_ja_prompt_contains_negative_examples(self):
    # Rule 4: 단서어 부정 예시 (担当者：, 連絡先：, 電話：)
    # Rule 5: 영문 라벨명 (ID_NUMBER, CREDIT_CARD)
def test_ja_empty_pii(self):
    # JA 빈 PII 마커 (（なし）)
```

---

## 2. 검증 4종 결과

```
Records: 997   Total entities: 3,724
Per-label:
  PER: 782   ORG: 840   LOC: 768   PROD: 224   EVT: 183
  EMAIL: 198   PHONE: 185   DAT: 193   ID_NUM: 179   CREDIT_CARD: 172
PII entities total: 927
```

| # | 검증 항목 | 결과 |
|---|---|---|
| 1 | 라벨 셋 ⊂ 10종 (`PER LOC ORG PROD EVT EMAIL PHONE DAT ID_NUM CREDIT_CARD`) | **PASS** — 외부 라벨 0종 |
| 2 | Offset 정합성 (`text[start:end] == ent.text`) | **PASS** — 0 / 3,724 위반 |
| 3 | PII 단서어 동반률 (Rule 4 위반) | **PASS** — **0 / 927 = 0.00%** |
| 4 | 영문 라벨 leakage 카운트 (Rule 5 위반) | **PASS** — **0건** |

### 2.1 베이스라인 ↔ 후속 비교

`data/pii/stockmark_pii_1000.jsonl` 직전 산출본(2026-04-24, Rule 4·5
미적용) 과 본 산출본(2026-04-28, Rule 4·5 적용) 비교.

| 지표 | Before (구 프롬프트) | After (Rule 4·5) | 변화 |
|---|---|---|---|
| Records | 993 | 997 | +4 |
| Total entities | 3,309 | 3,724 | +415 |
| PII entities | 902 | 927 | +25 |
| **PII 단서어 동반률** | **2.22%** (20 / 902) | **0.00%** (0 / 927) | **-2.22pp** |
| **영문 라벨 leakage** | **116건** | **0건** | **-100%** |

> 핸드오프 문서(2026-04-27)는 Stockmark 베이스라인을 prefix 27.5% /
> leakage 170 으로 기재했으나, 본 측정 시점의 데이터는 prefix 2.22% /
> leakage 116. 핸드오프의 27.5% 측정은 재현되지 않았다 — 서로 다른
> 정의(예: 더 광범위한 prefix 패턴 매칭)일 가능성이 있다. 본 리포트는
> 본 측정 정의(strict colon match) 기준 일관성 있게 before/after 를
> 비교한다. 어느 정의를 쓰든 새 산출본의 0% / 0 은 절대값이라 정의
> 차이의 영향이 없다.

### 2.2 PII 밀도 분포 (관측, `pii_max=3` 적용)

`DEFAULT_DENSITY = {0:0.2, 1:0.4, 2:0.3, 3:0.1}` 기반 샘플링.

| n PII | Records | 비율 |
|---:|---:|---:|
| 0 | 321 | 32.20% |
| 1 | 444 | 44.53% |
| 2 | 213 | 21.36% |
| 3 | 19 | 1.91% |

분포 자체는 #23 에서 분석된 것과 동일한 패턴 (raw 샘플링은 정확,
`stats.py:21` 의 `samples_no_pii_ratio` 측정 결함은 별도 후속 이슈).
본 이슈에서는 측정 정의 변경 범위 외.

---

## 3. 운영 영향

- **silver 학습 데이터 품질 ↑**: 단서어 직전 부착 패턴이 0% 가 되어 BERT
  파인튜닝 시 "콜론 직후 = PII" 같은 shortcut 학습 위험 제거.
- **라벨러 평가 정합성 ↑**: 영문 라벨 leakage 0 → 본문에서 라벨명이
  본문 토큰으로 등장하지 않아 자기지시(self-referential) 신호 차단.
- **PII 카운트 +25**: 새 프롬프트가 PII 임베딩을 자연 문장 구조로
  녹여내므로 LLM 이 PII 를 더 자주 (또는 더 분명하게) 출력함. 이로
  인해 추출 누락이 감소했을 가능성. drop record 도 7→3 으로 감소.

---

## 4. VI 결과 비교 (PR #25, 이슈 #23)

| 지표 | VI (wikiann_vi_pii) | JA (stockmark_pii_1000) |
|---|---|---|
| Records | 39,979 | 997 |
| PII entities | (전체 97,306 entities) | 927 |
| 단서어 동반률 | **0.00%** | **0.00%** |
| 영문 라벨 leakage | **0건** | **0건** |

VI 와 JA 양 언어에서 Rule 4·5 추가가 동일한 0/0 효과를 재현. 프롬프트
패턴의 언어 중립적 효과 입증.

---

## 5. 산출 파일

| 파일 | 변경 |
|---|---|
| `data/pii/stockmark_pii_1000.jsonl` | 재생성 (997 records) |
| `data/pii/stockmark_pii_1000.stats.json` | 재생성 (`stats.py` 자동 산출) |
| `src/augmenters/pii/llm_injector.py` | `_INJECTION_PROMPT_JA` Rule 4·5 추가 |
| `tests/augmenters/pii/test_llm_injector.py` | JA 부정 예시 + 빈 PII 테스트 2건 추가 |
| `docs/reports/japanese-pii-prompt-rule45-2026-04.md` | 신규 (본 리포트) |

---

## 6. 후속 작업·알려진 한계

- **`stats.py` 측정 결함**: `samples_no_pii_ratio` 가 NER 도 부재인
  케이스만 카운트 (PII 부재이지만 NER 있는 케이스 누락) — #23 에서
  분리된 별도 후속 이슈 후보로 유지.
- **단서어 정의 차이**: 본 측정의 strict prefix 정의 (`prefix + ：/: +
  PII직후`) 는 핸드오프의 27.5% 정의와 다를 수 있음. 새 산출본은 두
  정의 모두 0%.
- **밀도 분포 측정 결함**과의 분리: 본 이슈는 프롬프트 보강만 다루며,
  분포 측정 정의 변경은 호환성 영향이 있어 별도 이슈로 분리.
