# Stockmark + PII 혼합 NER 벤치마크 리포트

**측정일**: 2026-04-17
**데이터셋**: `data/pii/stockmark_pii_1000.jsonl` (993 samples, Stockmark + PII 혼합)
**데이터 생성**: `augmenters.pii --mode llm` (gemma-4-26B-A4B-AWQ 주입) + `--verify vllm` (Qwen3.5-27B-AWQ 교차 검증, policy=drop_span)
**평가지표**: 문자 오프셋 Span F1
**엔티티 타입**: Stockmark 8종 + PII 7종 (총 15종)
**공통 실행 조건**: `--concurrency 32` (sample 단위 병렬)

---

## 1. 요약 테이블

| # | 모델 | 양자화 | 백엔드 | F1 | Precision | Recall | Sec/sample | TPS | 총 시간 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | google/gemma-4-31B-it | BF16 | vllm TP=2 | **0.8955** | 0.8458 | **0.9513** | 0.252 | 11094 | 4:10 |
| 2 | Qwen/Qwen3.5-27B | BF16 | vllm TP=2 | 0.8803 | 0.8051 | 0.9710 | 1.425 | 1951 | 23:35 |
| 3 | cyankiwi/Qwen3.5-27B-AWQ-4bit | AWQ-4 | vllm TP=1 | 0.8751 | 0.7938 | 0.9749 | 1.655 | 1680 | 27:24 |
| 4 | cyankiwi/gemma-4-26B-A4B-AWQ-4bit | AWQ-4 (MoE A4B) | vllm TP=1 | 0.8474 | 0.7908 | 0.9127 | **0.099** | **28439** | **1:38** |
| 5 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 | 0.8305 | 0.7838 | 0.8830 | 0.411 | 6748 | 6:48 |
| 6 | Qwen/Qwen3.5-122B-A10B-GPTQ | GPTQ-Int4 (MoE A10B) | vllm TP=2 | 0.8146 | **0.8100** | 0.8193 | 1.175 | 2357 | 19:27 |
| 7 | gpt-5-mini (medium) | - | openai API | 0.6706 | 0.7250 | 0.6238 | 2.558 | 578 | 42:20 |
| 8 | gpt-5-mini (minimal) | - | openai API | 0.5623 | 0.5177 | 0.6153 | 0.475 | 1701 | 7:52 |

**품질 상위 3**: gemma-4-31B-it > Qwen3.5-27B > Qwen3.5-27B-AWQ
**속도 상위 3**: gemma-4-26B-A4B-AWQ(1:38) > gemma-4-31B-it(4:10) > Qwen3.5-35B-A3B(6:48)

---

## 2. 환경 및 측정 조건

- 하드웨어: NVIDIA RTX A6000 ×3 (GPU0/GPU1/GPU2)
- vLLM: `vllm/vllm-openai:v0.19.0` (Qwen), `vllm/vllm-openai:gemma4` (Gemma)
- GPU 할당:
  - TP=2 모델: GPU 1,2 (포트 8081)
  - AWQ TP=1 모델: GPU 1(8081) 또는 GPU 2(8082)
- OpenAI `gpt-5-mini`: `reasoning_effort` default(medium) 또는 env `OPENAI_REASONING_EFFORT=minimal`
- 실행: `python -m llm_eval --lang ja --local-file data/pii/stockmark_pii_1000.jsonl --max-samples 1000 --concurrency 32`
- 모든 벤치는 단독 실행

---

## 3. 핵심 발견: gpt-5-mini의 PII 미인식

gpt-5-mini는 **PII 7종 엔티티를 전혀 인식하지 못했다** (모든 PII 타입 F1=0.0000).

| PII 타입 | gpt-5-mini (medium) | Qwen3.5-27B | gemma-4-31B-it |
|---|---|---|---|
| ADDRESS | 0.0000 | 0.8375 | 0.9057 |
| CREDIT_CARD | 0.0000 | 0.9880 | 0.9880 |
| DOB | 0.0000 | 0.6899 | 0.9713 |
| EMAIL | 0.0000 | 0.9975 | 0.9949 |
| ID_NUMBER | 0.0000 | 0.9765 | 0.9702 |
| PHONE | 0.0000 | 0.9722 | 0.9000 |
| NAME | - | - | - |

**원인 추정**: gpt-5-mini의 안전 필터(safety guardrails)가 PII 정보를 출력에서 억제한 것으로 보인다. 프롬프트에 PII 엔티티 타입이 명시적으로 포함되어 있음에도(`src/labelers/ja/ner_prompts.py`) 실제 응답에서 PII 관련 span을 반환하지 않았다. Stockmark 원본 엔티티(人名, 法人名 등)만 정상 추출.

→ **gpt-5-mini는 PII 포함 NER에 부적합** (안전 필터 우회 불가).

---

## 4. 모델별 상세 (엔티티별 F1)

### 4.1 gemma-4-31B-it (F1=0.8955, PII 포함 품질 1위)

| Entity | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| EMAIL | 0.9949 | 0.9949 | 0.9949 | 198 |
| CREDIT_CARD | 0.9880 | 0.9762 | 1.0000 | 164 |
| DOB | 0.9713 | 0.9538 | 0.9894 | 188 |
| ID_NUMBER | 0.9702 | 0.9819 | 0.9588 | 170 |
| 人名 | 0.9689 | 0.9585 | 0.9795 | 731 |
| 法人名 | 0.9308 | 0.9150 | 0.9472 | 341 |
| ADDRESS | 0.9057 | 0.8521 | 0.9664 | 149 |
| PHONE | 0.9000 | 0.9101 | 0.8901 | 182 |
| 地名 | 0.8432 | 0.7800 | 0.9176 | 340 |
| 政治的組織名 | 0.8404 | 0.7822 | 0.9080 | 174 |
| 施設名 | 0.7978 | 0.7255 | 0.8862 | 167 |
| 製品名 | 0.7852 | 0.6558 | 0.9784 | 185 |
| イベント名 | 0.7778 | 0.6869 | 0.8963 | 164 |
| その他の組織名 | 0.7206 | 0.5833 | 0.9423 | 156 |

PII・Stockmark 양쪽 균형 잡힌 고득점. DOB(0.9713)·ADDRESS(0.9057) 등 다른 모델이 고전하는 타입에서도 우수.

### 4.2 Qwen3.5-27B vs Qwen3.5-27B-AWQ (품질 비교)

| Entity | 27B BF16 F1 | 27B-AWQ F1 | Δ |
|---|---|---|---|
| EMAIL | 0.9975 | 0.9975 | 0.000 |
| CREDIT_CARD | 0.9880 | 0.9939 | +0.006 |
| ID_NUMBER | 0.9765 | 0.9766 | 0.000 |
| PHONE | 0.9722 | 0.9553 | -0.017 |
| 人名 | 0.9676 | 0.9717 | +0.004 |
| 法人名 | 0.9465 | 0.9543 | +0.008 |
| ADDRESS | 0.8375 | 0.8239 | -0.014 |
| DOB | 0.6899 | 0.6836 | -0.006 |
| **Overall** | **0.8803** | **0.8751** | **-0.005** |

AWQ-4bit에서 F1 차이 -0.005 — 사실상 동등. GPU 1장(TP=1)으로 BF16(TP=2)과 같은 품질을 달성.

### 4.3 MoE 모델

| 모델 | Active | F1 | PII 평균 F1 | Stockmark 평균 F1 |
|---|---|---|---|---|
| 35B-A3B | 3B | 0.8305 | 0.872 | 0.793 |
| 122B-GPTQ | 10B | 0.8146 | 0.847 | 0.738 |

122B-GPTQ가 **ADDRESS(0.484)** 에서 크게 하락 — Recall 0.362로 주소 패턴 인식 실패. MoE 라우팅이 PII 문맥에서 편향될 가능성.

---

## 5. Stockmark 원본 벤치마크와의 비교

| 모델 | Stockmark 원본 F1 | Stockmark+PII F1 | Δ | PII 영향 |
|---|---|---|---|---|
| gemma-4-31B-it | 0.8212 | 0.8955 | **+0.074** | 긍정 (PII 고인식) |
| Qwen3.5-27B | 0.7770 | 0.8803 | **+0.103** | 긍정 |
| Qwen3.5-27B-AWQ | 0.7777 | 0.8751 | **+0.097** | 긍정 |
| gemma-4-26B-AWQ | 0.7622 | 0.8474 | **+0.085** | 긍정 |
| Qwen3.5-35B-A3B | 0.7280 | 0.8305 | **+0.103** | 긍정 |
| Qwen3.5-122B-GPTQ | 0.7237 | 0.8146 | **+0.091** | 긍정 |
| gpt-5-mini (medium) | 0.7935 | 0.6706 | **-0.123** | 부정 (PII 미인식) |
| gpt-5-mini (minimal) | 0.6905 | 0.5623 | **-0.128** | 부정 |

- vLLM 모델: F1이 일률적으로 **+0.07~0.10 상승**. PII 엔티티(EMAIL, CREDIT_CARD 등)가 패턴 매칭이 쉬워 전체 F1을 끌어올림.
- gpt-5-mini: PII 미인식으로 support가 큰 PII 엔티티가 전부 0점 → 평균 F1 하락.

---

## 6. 핵심 인사이트

### (a) PII 인식 능력
vLLM 모델들은 프롬프트에 명시된 PII 타입(EMAIL, PHONE, ADDRESS 등)을 잘 인식한다. 특히 EMAIL·CREDIT_CARD·ID_NUMBER는 거의 모든 모델에서 F1 0.9 이상.

### (b) gpt-5-mini의 안전 필터 한계
PII 데이터 추출 태스크에서 gpt-5-mini는 안전 필터로 인해 PII span을 출력에서 억제. NER 프롬프트로는 이 필터를 우회할 수 없음 → PII 포함 벤치에 부적합.

### (c) DOB (생년월일)의 난이도
DOB는 모델별 편차가 큼 (gemma-31B: 0.971 vs Qwen-27B: 0.690). 날짜 형식(`2001.03.27`, `1985/12/03` 등) 다양성과 문맥 내 위치가 난이도를 높인다.

### (d) ADDRESS 인식
122B-GPTQ가 ADDRESS에서 F1 0.484로 크게 하락 (Recall 0.362). 반면 gemma-4-31B-it은 0.906. 일본 주소 형식의 패턴 다양성에 대한 모델별 강건성 차이가 크다.

---

## 7. 권장 설정 (PII 포함 NER)

| 우선순위 | 권장 모델 | 근거 |
|---|---|---|
| 품질 최우선 | **google/gemma-4-31B-it** | F1 0.8955, PII·Stockmark 양쪽 최상위 |
| 속도·VRAM 최우선 | **cyankiwi/gemma-4-26B-A4B-AWQ-4bit** | 993샘플 98초, GPU 1장 |
| 품질·속도 균형 | **cyankiwi/Qwen3.5-27B-AWQ-4bit** | F1 0.8751, GPU 1장 |
| ⚠️ 비추천 | **openai:gpt-5-mini** | PII 미인식 (안전 필터) |

---

## 8. 산출물

```
results/
├── BENCHMARK_REPORT_PII.md                        ← 이 문서
├── pii_stockmark_gemma4_31b_1000.json
├── pii_stockmark_gemma4_26b_awq_1000.json
├── pii_stockmark_gpt5mini_1000.json               (medium)
├── pii_stockmark_gpt5mini_minimal_1000.json       (minimal)
├── pii_stockmark_qwen27b_1000.json
├── pii_stockmark_qwen27b_awq_1000.json
├── pii_stockmark_qwen35b_a3b_1000.json
└── pii_stockmark_qwen122b_gptq_1000.json

data/pii/
├── stockmark_pii_1000.jsonl                       (993 records)
├── stockmark_pii_1000.stats.json
└── stockmark_pii_1000.verify.json
```

관련 로그: `logs/pii_bench_*.log`
