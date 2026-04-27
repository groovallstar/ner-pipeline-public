# Stockmark NER 벤치마크 리포트 (5종 canonical)

> 본 리포트는 이슈 #21 에서 8종 → 5종(`PER · LOC · ORG · PROD · EVT`)으로
> 축소된 canonical 스키마 위에서 측정한 **PII 주입 전 원본 NER 라벨러
> 정당성** 측정이다. 이슈 #27 Phase 1 산출물.
>
> 직전 8종 baseline(2026-04-16) 결과는 본 리포트의 "8종↔5종 비교" 컬럼에
> 보존됐다. 별도 8종 리포트 파일은 삭제됐다.

**측정일**: 2026-04-27
**데이터셋**: `data/stockmark/test.jsonl` (1069 samples, 5종 canonical, PII 주입 없음)
**총 엔티티**: 2,621 개 (PER 554 / LOC 637 / ORG 996 / PROD 229 / EVT 205)
**평가지표**: 문자 오프셋 Span F1 (`metrics/span_metrics.compute_offset_span_f1`)
**엔티티 타입**: PER · LOC · ORG · PROD · EVT
**프롬프트**: 10종 평면 (`PER LOC ORG PROD EVT EMAIL PHONE DAT ID_NUM CREDIT_CARD`) — `src/labelers/ja/ner_prompts.py`
**측정 격리**: vLLM 컨테이너 한 번에 1개씩 단독 부팅·측정·정지 (GPU 경합 0)

---

## 0. 측정 방법론 — Raw F1 vs Filtered F1

JA 라벨러는 5종 NER + 5종 PII 의 **10종 통합 프롬프트** 를 사용한다 (PII
주입본 평가와 코드 공유). 본 측정의 gold 는 5종 NER 만 포함하므로,
모델이 출력한 PII 5종(`DAT EMAIL PHONE ID_NUM CREDIT_CARD`)은 **모두
False Positive** 로 잡혀 precision 이 인위적으로 하락한다.

이 효과를 분리하기 위해 본 리포트는 두 가지 F1 을 모두 보고한다.

| F1 종류 | 정의 | 용도 |
|---|---|---|
| **Raw F1** | 10종 프롬프트 출력 그대로 5종 gold 에 매칭. PII 출력은 FP. | 운영 시점 라벨러 동작의 실측치. silver 데이터 생산 시 후처리 필요량의 직관 |
| **Filtered F1** | 예측에서 PII 5종을 제외한 뒤 5종에 대해서만 매칭. | **8종 baseline 과 직접 비교 가능한 정량값**. 5종 NER 능력의 순수 측정 |

8종 baseline (2026-04-16) 은 8종 단일 스키마 프롬프트로 측정됐기 때문에
PII FP 가 없다. 5종 비교는 **Filtered F1 ↔ 8종 F1** 로 한다.

---

## 1. 요약 테이블 (Filtered F1 정렬)

| # | 모델 | 양자화 | 백엔드 | **Filtered F1** | Raw F1 | 8종 F1 (Δ) | Sec/sample | 총 시간 |
|---|---|---|---|---|---|---|---|---|
| 1 | cyankiwi/gemma-4-31B-it-AWQ-8bit | AWQ-8 (Dense) | vllm TP=1 | **0.8676** | 0.7750 | 0.8253 (+0.042) | 0.246 | 4:23 |
| 2 | google/gemma-4-31B-it | BF16 (Dense) | vllm TP=2 | 0.8670 | 0.7748 | 0.8212 (+0.046) | 0.186 | 3:19 |
| 3 | cyankiwi/gemma-4-31B-it-AWQ-4bit | AWQ-4 (Dense) | vllm TP=1 | 0.8670 | 0.7775 | 0.8190 (+0.048) | 0.197 | 3:30 |
| 4 | Qwen/Qwen3.5-27B | BF16 (Dense) | vllm TP=2 | 0.8638 | 0.7704 | 0.7770 (+0.087) | 1.068 | 19:02 |
| 5 | cyankiwi/Qwen3.5-27B-AWQ-4bit | AWQ-4 (Dense) | vllm TP=1 | 0.8576 | 0.7665 | 0.7777 (+0.080) | 1.229 | 21:54 |
| 6 | cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit | AWQ-8 (MoE A4B) | vllm TP=1 | 0.8413 | 0.7540 | 0.7734 (+0.068) | 0.073 | 1:19 |
| 7 | cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit | AWQ-4 (MoE A4B) | vllm TP=1 | 0.8357 | 0.7470 | 0.7622 (+0.074) | 0.062 | 1:06 |
| 8 | openai:gpt-5-mini | - | openai API | 0.8049 | 0.7232 | 0.7935 (+0.011) | 1.527 | 27:13 |
| 9 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 | 0.8010 | 0.7244 | 0.7280 (+0.073) | 0.302 | 5:23 |
| 10 | Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 | GPTQ-Int4 (MoE A10B) | vllm TP=2 | 0.7491 | 0.6762 | 0.7237 (+0.025) | 0.835 | 14:52 |

> Δ 컬럼은 `Filtered F1 - 8종 F1` 차이. 모든 모델이 **+** 방향 — 5종
> 축소가 라벨러 정합성을 일관되게 개선했다.

**품질 상위 3** (Filtered): gemma-4-31B-AWQ-8bit (0.8676) ≈ gemma-4-31B-BF16 (0.8670) ≈ gemma-4-31B-AWQ-4bit (0.8670)
**속도 상위 3**: gemma-4-26B-AWQ-4bit (1:06) > gemma-4-26B-AWQ-8bit (1:19) > gemma-4-31B-BF16 (3:19)

---

## 2. 환경 및 측정 조건

- 하드웨어: NVIDIA RTX A6000 ×2
- vLLM 이미지: `vllm/vllm-openai:gemma4` (Gemma-4 계열) / `vllm/vllm-openai:v0.19.0` (Qwen 계열)
- GPU 할당:
  - TP=2 모델 (BF16/GPTQ): GPU 1·2
  - TP=1 모델 (AWQ): Gemma=GPU 1, Qwen=GPU 2
- OpenAI: `gpt-5-mini`
- **모든 측정 동일 조건**: `--concurrency 32` (sample-level 병렬), `--no-bertscore`, vLLM 컨테이너 단독 구동 (8종 baseline 의 BF16 직렬 측정과 달리 본 측정은 BF16 도 동일 병렬도)
- 측정 스크립트: `/tmp/run-ja-5type-bench.sh` (이슈 #27 Phase 1 일회성 오케스트레이션)

---

## 3. PII 출력 분석 (Raw vs Filtered 차이)

5종 gold 에는 PII 가 없는데도 모델은 평균 600+ 개의 PII 출력을 생성한다.
**라벨러를 silver 데이터 생산자로 운영할 때 후처리 필터링이 필수**.

| 모델 | Raw P | Filtered P | PII FP (derived) | PII FP / total pred |
|---|---|---|---|---|
| gemma-4-31B-AWQ-8bit | 0.7009 | 0.8688 | 626 | 19.3% |
| gemma-4-31B-BF16 | 0.7012 | 0.8684 | 623 | 19.2% |
| gemma-4-31B-AWQ-4bit | 0.7081 | 0.8674 | 604 | 18.4% |
| Qwen3.5-27B-BF16 | 0.6930 | 0.8603 | 638 | 19.5% |
| Qwen3.5-27B-AWQ | 0.6894 | 0.8523 | 627 | 19.1% |
| gemma-4-26B-AWQ-8bit | 0.6753 | 0.8530 | 600 | 19.6% |
| gemma-4-26B-AWQ-4bit | 0.6679 | 0.8479 | 619 | 20.4% |
| gpt-5-mini | 0.6539 | 0.8009 | 595 | 18.4% |
| Qwen3.5-35B-A3B | 0.6742 | 0.8204 | 542 | 17.9% |
| Qwen3.5-122B-GPTQ | 0.6858 | 0.8543 | 503 | 17.0% |

\* PII FP = `(Raw_TP / Raw_P) - (Raw_TP / Filtered_P)` (raw vs filtered 의
TP 는 동일, pred 차이만 PII type FP).

PII 출력률은 모든 모델에서 17~21% 수준. 모델 크기·구조와 무관하게 거의
일정 → 프롬프트 자체의 효과지 모델 능력의 함수가 아니다. silver 생산
파이프라인은 `pred_type ∈ {PER, LOC, ORG, PROD, EVT}` 필터를 항상
적용해야 한다.

---

## 4. 모델별 상세 (Filtered F1, 5종 per-entity)

### 4.1 gemma-4-31B-it 3-model 비교 (1·2·3 위, ±0.001 동일성)

| Entity | AWQ-8bit F1 | BF16 F1 | AWQ-4bit F1 | Support |
|---|---|---|---|---|
| PER | 0.9545 | 0.9536 | 0.9462 | 554 |
| ORG | 0.8743 | 0.8760 | 0.8759 | 996 |
| LOC | 0.8417 | 0.8384 | 0.8447 | 637 |
| EVT | 0.8068 | 0.8010 | 0.8048 | 205 |
| PROD | 0.7617 | 0.7633 | 0.7636 | 229 |
| **Overall (Filtered)** | **0.8676** | **0.8670** | **0.8670** | 2,621 |

| 지표 | AWQ-8bit | BF16 | AWQ-4bit |
|---|---|---|---|
| Filtered F1 | 0.8676 | 0.8670 | 0.8670 |
| Filtered Precision | 0.8688 | 0.8684 | 0.8674 |
| Filtered Recall | 0.8665 | 0.8657 | 0.8666 |
| 총 시간 | 4:23 | 3:19 | 3:30 |
| Sec/sample | 0.246 | 0.186 | 0.197 |
| GPU | 1장 | 2장 (TP=2) | 1장 |

- 세 양자화 변종이 **0.001 단위 차이**로 본질적 동등. 양자화로 인한
  품질 손실 사실상 없음.
- BF16 (TP=2) 가 sec/sample 에서 가장 빠르지만 GPU 2장 사용. AWQ-8bit
  단일 GPU 로 동등 품질 확보 가능 → **운영 효율 1위는 AWQ-8bit**.

### 4.2 Qwen3.5-27B BF16 vs AWQ-4bit (4·5위)

| Entity | 27B BF16 | 27B AWQ | Δ |
|---|---|---|---|
| PER | 0.9349 | 0.9370 | +0.002 |
| ORG | 0.8746 | 0.8698 | -0.005 |
| LOC | 0.8472 | 0.8396 | -0.008 |
| EVT | 0.7991 | 0.7936 | -0.006 |
| PROD | 0.7633 | 0.7390 | -0.024 |
| **Overall (Filtered)** | **0.8638** | **0.8576** | **-0.006** |

8종 baseline 에서 두 모델은 0.7770 ↔ 0.7777 (Δ +0.001) 거의 동일. 5종
Filtered 에서도 0.8638 ↔ 0.8576 (Δ -0.006) 으로 BF16 소폭 우위. 양자화
영향 미미.

### 4.3 gemma-4-26B-A4B (MoE) AWQ 8bit vs 4bit (6·7위)

| Entity | 4bit F1 | 8bit F1 | Δ | Support |
|---|---|---|---|---|
| PER | 0.8973 | 0.9024 | +0.005 | 554 |
| ORG | 0.8543 | 0.8525 | -0.002 | 996 |
| LOC | 0.8170 | 0.8195 | +0.003 | 637 |
| EVT | 0.7610 | 0.7800 | +0.019 | 205 |
| PROD | 0.7237 | 0.7349 | +0.011 | 229 |
| **Overall (Filtered)** | **0.8357** | **0.8413** | **+0.006** | 2,621 |

- 5종 환경에서도 **8bit 가 4bit 대비 +0.006 우위** (8종 측정 +0.011 과
  유사 추세).
- 속도 차이는 1:06 ↔ 1:19 (+13초) 로 사용 가능 범위 내.

### 4.4 MoE / 외부 API (8·9·10위)

| 모델 | Filtered F1 | 특이점 |
|---|---|---|
| gpt-5-mini | 0.8049 | 8종 baseline 4위(0.7935) → 5종 8위. **8종↔5종 개선폭 +0.011 로 가장 작음** — 외부 API 라 기존부터 8종 일부 라벨에 강했고 5종 축소 이득이 작음 |
| Qwen3.5-35B-A3B (MoE) | 0.8010 | active 3B 라 31B/27B Dense 보다 capacity 낮음. 그러나 8종 0.7280 → 5종 0.8010 으로 +0.073 큰 개선 |
| Qwen3.5-122B-A10B-GPTQ | 0.7491 | recall 0.6669 (전 모델 중 최저) — 누락 패턴이 많은 precision-지향 모델 |

---

## 5. 8종↔5종 비교 종합

5종 축소 효과(Δ = Filtered F1 - 8종 F1) 는 모델별로 +0.011 ~ +0.087
범위. 평균 +0.058. 가장 큰 개선:

| Δ 상위 | 모델 | 8종 | 5종(Filtered) |
|---|---|---|---|
| +0.087 | Qwen/Qwen3.5-27B | 0.7770 | 0.8638 |
| +0.080 | Qwen3.5-27B-AWQ-4bit | 0.7777 | 0.8576 |
| +0.074 | gemma-4-26B-A4B-AWQ-4bit | 0.7622 | 0.8357 |

| Δ 하위 | 모델 | 8종 | 5종(Filtered) |
|---|---|---|---|
| +0.011 | gpt-5-mini | 0.7935 | 0.8049 |
| +0.025 | Qwen3.5-122B-GPTQ | 0.7237 | 0.7491 |

**해석**: 5종 reduction 의 효과는 "8종 시절 헷갈렸던 ORG 변종(政治的組織名/その他の組織名/法人名) 통합"·"地名/施設名 → LOC 통합" 등이 컸다.
이미 8종에서 잘 분리하던 모델(gpt-5-mini, 122B-GPTQ)은 추가 이득이 적었다.

**라벨러 정당성 결론**: 모든 모델의 5종 Filtered F1 이 8종 baseline 보다
높음. 5종 축소가 라벨링 품질을 일관되게 개선했고, **silver 데이터
생산자로 사용할 정당성이 정량적으로 확보됨**.

---

## 6. 권장 설정

| 우선순위 | 권장 모델 | 근거 |
|---|---|---|
| **silver 생산 품질·효율 1위** | **cyankiwi/gemma-4-31B-it-AWQ-8bit** | Filtered F1 0.8676, TP=1 단일 GPU, 4:23 |
| 품질 동등 + GPU 2장 가용 | **google/gemma-4-31B-it (BF16)** | Filtered F1 0.8670, sec/sample 최저 (0.186) |
| VRAM 17GB 한계 환경 | **cyankiwi/gemma-4-31B-it-AWQ-4bit** | Filtered F1 0.8670, BF16 동등 품질 |
| 속도·VRAM 최우선 | **cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit** | 1:06, F1 0.8357 (-0.032 vs 1위) |
| 속도 + 품질 소폭 ↑ | **cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit** | 1:19, F1 0.8413 |
| Qwen 계열 단일 GPU | **cyankiwi/Qwen3.5-27B-AWQ-4bit** | Filtered F1 0.8576, 21:54 (느림) |
| 외부 API | **openai:gpt-5-mini** | Filtered F1 0.8049, 운영 단순 |

silver PII 데이터 재생성(이슈 #27 Phase 2) 의 기본 후보는 **gemma-4-31B-AWQ-8bit**.

---

## 7. 운영 주의사항

- **PII 후처리 필터 필수**: 모든 모델이 17~21% 수준의 PII 타입 출력을
  생성. silver 생산 시 `pred_type ∈ {PER, LOC, ORG, PROD, EVT}` 필터
  적용해야 정확도 유지.
- **샘플 카운트 정정**: 직전 8종 리포트는 "1000 samples" 라고 기재했으나
  실제 `data/stockmark/test.jsonl` 은 **1069 samples** / 2621 entities.
  본 측정부터 정확값을 사용한다.
- **속도 비교 한계**: 직전 8종 리포트의 BF16 측정은 직렬(`concurrency=1`)
  이라 시간이 부풀려져 있다(예: BF16 31B 49:24). 본 측정은 모든 모델
  동일 `concurrency=32` → BF16 31B 가 3:19 로 측정. 8종↔5종 직접 속도
  비교는 무의미.
- **Raw F1 사용처**: 운영 환경에서 후처리 필터를 적용하지 않을 때의 F1
  추정값. 일반적인 학습/평가 목적에는 Filtered F1 만 사용.

---

## 8. 산출 파일

```
results/ja-5type-bench-2026-04/
├── 01-gemma-4-31B-it-AWQ-8bit.json       # cyankiwi/gemma-4-31B-it-AWQ-8bit
├── 02-gemma-4-31B-it-AWQ-4bit.json
├── 03-gemma-4-26B-A4B-it-AWQ-8bit.json
├── 04-gemma-4-26B-A4B-it-AWQ-4bit.json
├── 05-google-gemma-4-31B-it.json         # BF16
├── 06-Qwen3.5-27B-AWQ-4bit.json
├── 07-Qwen3.5-27B.json                   # BF16
├── 08-Qwen3.5-35B-A3B.json
├── 09-Qwen3.5-122B-A10B-GPTQ-Int4.json
├── 10-gpt-5-mini.json
└── run.log                               # 측정 진행 로그
```

각 JSON: `metrics.span_f1.{overall, per_entity}` + `latency.{total_seconds, samples_per_second, avg_per_sample, prompt_tokens, completion_tokens, total_tokens, tokens_per_second}`.
