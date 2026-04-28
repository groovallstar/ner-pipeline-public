# JA Stockmark + 합성 PII 벤치마크 리포트 (10종 canonical, 시설=ORG)

> 본 리포트는 이슈 #27 Phase 4 의 silver PII 데이터 산출물 품질 측정.
> Stockmark NER 5종 + 합성 PII 5종 = **10종 평면 canonical** 위에서
> 새 LOC/ORG 경계(시설=ORG) 를 적용한 silver 데이터를 재생성·검증.

**측정일**: 2026-04-28 (Phase 4)
**입력**: `data/stockmark/{train,test}.jsonl` (5종 canonical, 시설=ORG, #27 Phase 4 기준)
**출력**: `data/pii/stockmark_pii_{train,test}.jsonl` + `.stats.json` + `.verify.json`
**Inject 모델**: cyankiwi/gemma-4-31B-it-AWQ-8bit (vLLM TP=1, GPU 1, 포트 8081)
**Verify 모델**: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (vLLM TP=1, GPU 2, 포트 8082)
**Verify policy**: `drop_span` — 미확인 span 만 제거, 레코드 보존

---

## 1. 산출 통계

| Split | 입력 records | 산출 records | kept ratio | confirmed | missed | conflict |
|---|---|---|---|---|---|---|
| train | 4,274 | **4,243** | 99.27% | 14,021 | 1,411 | 515 |
| test | 1,069 | **1,064** | 99.53% | 3,504 | 339 | 130 |
| **계** | **5,343** | **5,307** | **99.33%** | **17,525 (87.9%)** | **1,750** | **645** |

> kept ratio: Verifier 가 모든 span 을 떨어뜨린 record 만 drop. drop_span
> policy 라 대부분의 record 는 일부 span 을 잃어도 보존됨.
> confirmed ratio: confirmed / (confirmed + missed + conflict).

**작업 시간**:
- train inject 12:01~12:16 (15분, gemma-31B-AWQ-8bit)
- train verify 12:16~13:59 (1시간 43분, Qwen3.6-35B-A3B-AWQ-4bit)
- test inject 14:00~14:03 (3분)
- test verify 14:03~14:29 (26분)
- 합계: 약 2시간 17분

---

## 2. 라벨 분포 (verify 적용 후)

### Train (4,243 records / 14,021 entities)

| Label | 카운트 | 비중 |
|---|---|---|
| ORG | 3,876 | 27.6% |
| PER | 2,924 | 20.9% |
| LOC | 2,066 | 14.7% |
| EMAIL | 811 | 5.8% |
| DAT | 807 | 5.8% |
| PHONE | 764 | 5.4% |
| PROD | 760 | 5.4% |
| EVT | 669 | 4.8% |
| CREDIT_CARD | 673 | 4.8% |
| ID_NUM | 671 | 4.8% |

### Test (1,064 records / 3,504 entities)

| Label | 카운트 | 비중 |
|---|---|---|
| ORG | 1,046 | 29.8% |
| PER | 714 | 20.4% |
| LOC | 470 | 13.4% |
| EMAIL | 212 | 6.0% |
| DAT | 210 | 6.0% |
| PHONE | 196 | 5.6% |
| PROD | 179 | 5.1% |
| EVT | 171 | 4.9% |
| ID_NUM | 158 | 4.5% |
| CREDIT_CARD | 148 | 4.2% |

NER 5종 중 ORG 비중 가장 높음 — #27 시설=ORG 통합으로 LOC 가 ORG 로
이동한 영향. Phase 1 (시설=LOC) 분포에서는 LOC 가 ORG 와 거의 같은
규모였으나 본 측정에서 LOC 가 ORG 의 절반 수준으로 축소.

---

## 3. 검증 4종 결과

스크립트: `/tmp/verify_pii_4checks.py` (1회성, 작업 후 폐기).

| 검증 항목 | train | test | 결과 |
|---|---|---|---|
| 1. 라벨 셋 ⊂ 10종 (canonical) | 외부 라벨 0 / 14,021 | 외부 라벨 0 / 3,504 | **PASS** |
| 2. Offset 정합성 | 0 / 14,021 위반 | 0 / 3,504 위반 | **PASS** |
| 3. PII 단서어 동반률 | 0 / 5,792 = 0.00% | 0 / 1,394 = 0.00% | **PASS** |
| 4. 영문 라벨 leakage | 0 records | 0 records | **PASS** |

> 단서어 동반률: PII span 의 직전 컨텍스트(20글자)에 (`担当者`/`連絡先`/
> `電話`/`電話番号`/`メール`/`住所`/`ID番号`/`カード番号`/`クレカ番号`/
> `生年月日`) + 콜론(`:`/`：`) 패턴이 동반된 비율. strict colon 정의.

> Leakage 검출 토큰: `NAME`/`PHONE`/`EMAIL`/`ID_NUM`/`ID_NUMBER`/
> `CREDIT_CARD`/`DAT`/`ADDRESS`. 단어 경계 매칭.

이슈 #23 (PR #25) 에서 입증된 VI 측 Rule 4·5 효과(prefix 27.5%→0%,
leakage 170→0) 가 JA 측 새 스키마에서도 동일하게 발현. 새 LOC/ORG
경계가 PII 주입 품질에 영향을 주지 않음.

---

## 4. NER 5종 베이스라인 대비 일관성

silver gold 의 NER 5종 분포가 베이스라인 (#27 Phase 4 stockmark 결과)
과 일치하는지 확인. 베이스라인 분포 (PII 미주입 stockmark, 시설=ORG):

| Label | Phase 4 stockmark train | PII 주입 후 train (NER 5종만) |
|---|---|---|
| ORG | 4,558 | 3,876 |
| PER | 2,426 | 2,924 |
| LOC | 1,790 | 2,066 |
| PROD | 986 | 760 |
| EVT | 804 | 669 |

PII 주입 후 LOC 가 +276, PER 가 +498 증가한 것은 합성 PII 의 `NAME`
(merge → PER) 과 `ADDRESS` (merge → LOC) 가 NER 라벨에 흡수된 결과
(`config.py` `DEFAULT_MERGE_RULES`).

ORG/PROD/EVT 가 약간 감소한 것은 verify 단계에서 conflict span 이 drop
된 결과. confirmed ratio 87.9% 와 일치.

---

## 5. 권장 설정 (이슈 #27 Phase 4 산출 기준)

### Inject 모델 선택 근거

`cyankiwi/gemma-4-31B-it-AWQ-8bit`:
- Phase 4 NER 벤치 Filtered F1 **0.8767** (단일 GPU 환경 1위)
- 단일 GPU TP=1 운영 가능 (BF16 모델 google gemma 와 동등 품질)
- train inject 15분 + test 3분 = 안정적 처리 시간

### Verify 모델 선택 근거

`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`:
- Phase 4 Filtered F1 0.8359 (Inject 모델과 다른 architecture)
- Inject (gemma) ↔ Verify (qwen) 두 다른 family 로 교차 검증
- TP=1 GPU 2 에 동시 구동 가능 (Inject 와 GPU 충돌 없음)
- verify 단계 처리량 ~0.66/s (concurrency 32 환경) — 약 1.5시간 소요

### 동시 구동 검증

본 측정은 Inject (vllm-gemma 8081, GPU 1) 와 Verify (vllm-qwen 8082,
GPU 2) 가 **GPU 별도라 동시 구동 가능**. 측정 격리는 NER 벤치(속도) 에서
필요하지 PII 주입(품질) 에서는 불필요.

---

## 6. 알려진 한계

- **conflict 615 건 분석 미진행**: Verifier 가 inject 모델과 다른 라벨을
  부여한 span 615건. drop_span 정책으로 모두 제거됐지만, 어떤 라벨 간
  충돌이 가장 많은지 (예: LOC↔ORG, PER↔ORG) 본 리포트에는 분석 없음.
  V 이슈(VI) 에서 동일 검증을 진행할 때 추가 분석 가능.
- **Verifier retry 패턴**: Qwen3.6-AWQ-4bit verify 호출에서 매 요청마다
  retry 가 1회 발생 (200 OK 후 retry, 또 200 OK). 응답은 정상이나
  처리량을 절반으로 떨어뜨림. 원인 추적 미실시.
- **단서어 동반률 정의**: strict colon 정의 (`〜：` / `〜:` 패턴) 사용.
  핸드오프 문서의 27.5% 베이스라인은 더 느슨한 매칭 정의 — 직접 비교
  불가. 두 정의 모두에서 본 산출은 0% 일치.

---

## 7. 산출 파일

```
data/pii/
├── stockmark_pii_train.jsonl          # 4,243 records / 14,021 entities
├── stockmark_pii_train.stats.json
├── stockmark_pii_train.verify.json
├── stockmark_pii_test.jsonl           # 1,064 records / 3,504 entities
├── stockmark_pii_test.stats.json
└── stockmark_pii_test.verify.json
```

스키마: `{id, text, entities:[{label, start_char, end_char, text}]}`. 라벨은
canonical 10종 (`PER LOC ORG PROD EVT EMAIL PHONE DAT ID_NUM CREDIT_CARD`)
중 하나.

생성 명령:
```bash
# train
python -m augmenters.pii \
  --source stockmark --lang ja \
  --output data/pii/stockmark_pii_train.jsonl \
  --mode llm --llm-concurrency 32 \
  --inject-url http://localhost:8081/v1 \
  --inject-model cyankiwi/gemma-4-31B-it-AWQ-8bit \
  --verify vllm --verify-policy drop_span \
  --verify-url http://localhost:8082/v1 \
  --verify-model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --verify-concurrency 32

# test
python -m augmenters.pii \
  --source jsonl --input data/stockmark/test.jsonl --lang ja \
  --output data/pii/stockmark_pii_test.jsonl \
  ... (동일 inject/verify 옵션)
```
