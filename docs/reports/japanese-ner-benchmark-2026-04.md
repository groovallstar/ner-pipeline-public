# Stockmark NER 벤치마크 리포트

**측정일**: 2026-04-16
**데이터셋**: `stockmark/ner-wikipedia-dataset` (test split, 1000 samples, PII 주입 없음)
**평가지표**: 문자 오프셋 Span F1 (`src/metrics/span_metrics.py`)
**엔티티 타입**: 人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名

---

## 1. 요약 테이블

| # | 모델 | 양자화 | 백엔드 | F1 | Precision | Recall | Sec/sample | TPS | 총 시간 | 병렬화 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | google/gemma-4-31B-it | BF16 | vllm TP=2 | **0.8212** | 0.8180 | 0.8244 | 2.964 | 747 | 49:24 | 직렬 |
| 2 | gpt-5-mini (medium, 병렬) | - | openai API | 0.7935 | 0.7973 | 0.7898 | 2.060 | 607 | 34:20 | 병렬=32 |
| 3 | gpt-5-mini (medium, 직렬) | - | openai API | 0.7892 | 0.7920 | 0.7865 | 8.756 | 144 | 2:25:56 | 직렬 |
| 4 | cyankiwi/Qwen3.5-27B-AWQ-4bit | AWQ-4 | vllm TP=1 | 0.7777 | 0.7414 | 0.8178 | 1.287 | 1716 | 21:26 | 병렬=32 |
| 5 | Qwen/Qwen3.5-27B | BF16 | vllm TP=2 | 0.7770 | 0.7388 | 0.8195 | 3.476 | 636 | 57:55 | 직렬 |
| 6 | cyankiwi/gemma-4-26B-A4B-AWQ-4bit | AWQ-4 (MoE A4B) | vllm TP=1 | 0.7622 | 0.7525 | 0.7721 | **0.059** | **37419** | **0:59** | 병렬=32 |
| 7 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 | 0.7280 | 0.7381 | 0.7181 | 1.074 | 2052 | 17:53 | 직렬 |
| 8 | Qwen/Qwen3.5-122B-A10B-GPTQ | GPTQ-Int4 (MoE A10B) | vllm TP=2 | 0.7237 | **0.8104** | 0.6538 | 2.013 | 1091 | 33:32 | 직렬 |
| 9 | gpt-5-mini (reasoning=minimal) | - | openai API | 0.6905 | 0.6224 | 0.7754 | 2.346 | 326 | 39:06 | 직렬 |

> **주의**: Sec/sample·총 시간은 sample 단위 병렬화(`concurrency=32`) 적용 전/후가 섞여 있다. 속도 비교 시 "병렬화" 열을 반드시 확인.

**품질 상위 3**: gemma-4-31B-it > gpt-5-mini(parallel, 기본) > gpt-5-mini(직렬, 기본)
**속도 상위 3**: gemma-4-26B-A4B-AWQ(59초) > Qwen3.5-27B-AWQ(21분) > gpt-5-mini(parallel, 34분)

---

## 2. 환경 및 측정 조건

- 하드웨어: NVIDIA RTX A6000 ×3 (GPU0/GPU1/GPU2)
- vLLM: 이미지 `vllm/vllm-openai:v0.19.0` (Qwen/Gemma용) / `vllm/vllm-openai:gemma4` (Gemma-4 계열)
- GPU 할당:
  - 일반 vLLM 모델 (TP=2): GPU 1,2
  - AWQ 모델 (TP=1): Qwen=GPU 2(포트 8082), Gemma=GPU 1(포트 8081)
- OpenAI: `gpt-5-mini`, `reasoning_effort` 기본값(medium) 또는 명시값(minimal)
- 벤치마크 CLI: `python -m llm_eval --lang ja --max-samples 1000 --concurrency 32`
- 모든 벤치는 GPU 경합 없이 단독 실행 (사용자 지시: vLLM 동시 구동 시 속도 측정 오염 방지)

---

## 3. 용어 및 모델 해설

### 3.1 양자화 / 수치 정밀도

| 용어 | 정식 명칭 | 설명 |
|---|---|---|
| **BF16** | Brain Float 16 | 16-bit 부동소수. 추론 기본 정밀도. 품질 손실 없음. |
| **Int4** | 4-bit integer | 가중치를 4비트 정수로 표현. 이론상 메모리 1/4, 추론 속도 ↑. |
| **AWQ** | Activation-aware Weight Quantization | 활성값(activation) 분포를 기반으로 **중요 가중치를 보존**하며 4-bit 양자화. 가중치별 스케일을 조정해 양자화 오차를 최소화 → **정확도 손실이 작고**, vLLM/TRT-LLM 등 런타임이 지원. 동일 GPU에서 BF16 대비 메모리 1/4~1/3, 처리량 상승. |
| **GPTQ** | (Post-Training) Generative Pre-trained Transformer Quantization | 각 레이어를 순차적으로 풀며 **Hessian 기반 오차 보정**으로 4-bit 양자화. AWQ와 유사한 목적이지만 접근법이 다름 (AWQ=activation-aware scale, GPTQ=second-order Hessian 보정). 초대형 모델(122B 등)에서 주로 사용. |

### 3.2 아키텍처 / 구조

| 용어 | 설명 |
|---|---|
| **Dense** | 매 토큰마다 **전체 파라미터**가 활성. 전통적 Transformer. |
| **MoE** (Mixture of Experts) | 여러 전문가(FFN) 중 라우팅 네트워크가 **일부만 선택**해 활성. 총 파라미터 중 소수만 계산에 참여 → 큰 모델을 적은 연산으로 운영 가능. |
| **A{N}B** | "Active N Billion" — MoE에서 **토큰당 실제 활성되는 파라미터 수**. 예: `35B-A3B` = 총 35B, 토큰당 3B 활성. |
| **TP=k** (Tensor Parallel) | 단일 layer의 가중치를 **k개 GPU에 분할**하여 병렬 계산. 큰 모델을 여러 GPU에 나눔. TP=1은 단일 GPU. |

### 3.3 모델 식별자 해석

| 모델 | 해설 |
|---|---|
| `google/gemma-4-31B-it` | 총 31B Dense, instruction-tuned ("it"). BF16. |
| `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | 총 26B MoE, 활성 4B, instruction-tuned, AWQ-4 양자화. |
| `Qwen/Qwen3.5-27B` | 총 27B Dense. BF16. |
| `cyankiwi/Qwen3.5-27B-AWQ-4bit` | 27B Dense의 AWQ-4 양자화본. |
| `Qwen/Qwen3.5-35B-A3B` | 총 35B MoE, 활성 3B. BF16. |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | 총 122B MoE, 활성 10B, GPTQ-Int4 양자화. |
| `openai:gpt-5-mini` | OpenAI API (외부 호출). 내부적으로 reasoning(hidden CoT) 토큰 생성. |

### 3.4 `cyankiwi/*` 접두어
공개 허브의 커뮤니티 양자화본 재배포 계정. 원본 모델 가중치를 AWQ로 변환해 재배포한 체크포인트 (공식 빌드 아님).

---

## 4. 모델별 상세 (엔티티별 F1)

### 4.1 gemma-4-31B-it (BF16, F1=0.8212, 품질 1위)

| Entity | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| 人名 | 0.9414 | 0.9526 | 0.9305 | 518 |
| 地名 | 0.8382 | 0.8657 | 0.8123 | 373 |
| 法人名 | 0.8125 | 0.9602 | 0.7042 | 480 |
| 政治的組織名 | 0.7956 | 0.7605 | 0.8341 | 217 |
| 施設名 | 0.7939 | 0.7787 | 0.8097 | 226 |
| イベント名 | 0.7781 | 0.7563 | 0.8011 | 186 |
| 製品名 | 0.7589 | 0.6830 | 0.8538 | 212 |
| その他の組織名 | 0.7135 | 0.6120 | 0.8551 | 214 |

### 4.2 Qwen3.5-27B vs Qwen3.5-27B-AWQ-4bit (품질 동등, 속도·VRAM 우위는 AWQ)

| Entity | 27B BF16 F1 | 27B-AWQ F1 | Δ |
|---|---|---|---|
| 人名 | 0.9416 | 0.9364 | -0.005 |
| 法人名 | 0.8015 | 0.8034 | +0.002 |
| 地名 | 0.7989 | 0.7735 | -0.025 |
| 政治的組織名 | 0.7887 | 0.7837 | -0.005 |
| 施設名 | 0.7905 | 0.8009 | +0.010 |
| イベント名 | 0.7601 | 0.6839 | **-0.076** |
| 製品名 | 0.7576 | 0.7522 | -0.005 |
| その他の組織名 | 0.6920 | 0.7016 | +0.010 |
| **Overall** | **0.7770** | **0.7777** | **+0.001** |

AWQ-4bit에서 평균 F1 거의 동일(0.7770 ↔ 0.7777). イベント名에서만 -0.076의 상대적 하락. TP=1로 GPU 절반 사용하면서 동등 품질 — 운영 관점에서 AWQ가 강한 선택지.

### 4.3 gpt-5-mini: reasoning_effort 영향 + 병렬화 영향

| 설정 | F1 | Sec/sample | 총 시간 | 비고 |
|---|---|---|---|---|
| medium(기본) + 직렬 | 0.7892 | 8.756 | 2:25:56 | 최초 기준선 |
| **medium + 병렬=32** | **0.7935** | **2.060** | **0:34:20** | 4.25× 가속, 품질 유지 |
| minimal + 직렬 | 0.6905 | 2.346 | 0:39:06 | 9.9%p 하락 |

- Reasoning=minimal 은 "事件名", "学校法人", "法令", "法律・条文" 등 **지시에 없는 타입을 환각** — precision 크게 하락(0.792 → 0.622).
- 병렬화만 적용(minimal 없이) 하면 품질 손실 없이 4.25배 가속 가능 — **기본 운영 설정으로 권장**.

### 4.4 MoE 모델 (Qwen3.5-35B-A3B, Qwen3.5-122B-GPTQ)

| 모델 | Active Params | F1 | 특이점 |
|---|---|---|---|
| 35B-A3B | 3B | 0.7280 | 속도 최상위(TPS 2052), 품질 중하위 |
| 122B-A10B-GPTQ | 10B | 0.7237 | Precision 최상(0.8104), Recall 약함(0.6538) |

두 MoE 모두 dense 27B(F1 0.7770)보다 전반적 F1은 낮음. 특히 122B-GPTQ는 **recall 0.6538** 로 낮아 누락이 많다. Precision 지향 용도에는 적합.

---

## 5. 핵심 인사이트

### (a) Sample 단위 병렬화의 필요성
벤치마크 러너가 sample 단위 직렬 호출을 하던 구조에서, `asyncio.Semaphore(32)` 기반 sample 단위 병렬로 전환 후 gpt-5-mini가 **2:25:56 → 0:34:20 (4.25배)** 가속. 품질 차이는 노이즈 수준(F1 0.7892 → 0.7935).
- 변경 범위: `src/llm_eval/benchmark_runner.py` 의 `_run_offset_span` async 재구현, `BaseOpenAILabeler`·`BaseVllmLabeler` 에 `alabel_spans` async API 추가.

### (b) AWQ-4bit 양자화 효과 (Qwen3.5-27B)
BF16(TP=2) 대비 품질 동등(F1 +0.0007)하면서 GPU를 절반 사용. 운영 비용·VRAM 측면에서 우수한 선택.

### (c) Reasoning 토큰의 실질 기여
gpt-5-mini 속도 저하 원인의 대부분은 **hidden reasoning 토큰**(sample당 ~500개). minimal 로 낮추면 reasoning 토큰이 급감(562 → 68)하지만 레이블 준수율·경계 정확도가 저하되어 품질 저하. NER 에서는 medium 유지가 안정적.

### (d) Precision vs Recall 트레이드오프
- Recall 상위: Qwen3.5-27B(0.8195), Qwen-AWQ(0.8178), gemma-4-31B(0.8244)
- Precision 상위: 122B-GPTQ(0.8104), gemma-4-31B(0.8180), gpt-5-mini parallel(0.7973)
- F1·균형형: **gemma-4-31B-it**

---

## 6. 권장 설정

| 우선순위 | 권장 모델 | 근거 |
|---|---|---|
| 품질 최우선 | **google/gemma-4-31B-it** | F1 0.8212, 인/법/지명 균형 |
| 속도·VRAM 최우선 | **cyankiwi/gemma-4-26B-A4B-AWQ-4bit** | 1000샘플 59초, GPU 1장 |
| 품질/속도/비용 균형 | **cyankiwi/Qwen3.5-27B-AWQ-4bit** | F1 0.7777, GPU 1장, 21분 |
| 외부 API 허용 시 | **openai:gpt-5-mini** (기본 medium + 병렬=32) | F1 0.7935, 34분, 운영 단순 |

---

## 7. 산출물

```
results/
├── BENCHMARK_REPORT.md                       ← 이 문서
├── stockmark_gemma4_31b_1000.json
├── stockmark_gemma4_26b_awq_1000.json
├── stockmark_gpt5mini_1000.json              (medium, 직렬)
├── stockmark_gpt5mini_minimal_1000.json      (minimal, 직렬)
├── stockmark_gpt5mini_parallel_1000.json     (medium, 병렬=32)
├── stockmark_qwen27b_1000.json               (BF16, 직렬)
├── stockmark_qwen27b_awq_1000.json           (AWQ-4, 병렬=32)
├── stockmark_qwen35b_a3b_1000.json
└── stockmark_qwen122b_gptq_1000.json
```

관련 로그: `logs/bench_*.log`
