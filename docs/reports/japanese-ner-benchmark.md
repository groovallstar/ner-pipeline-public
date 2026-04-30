# Stockmark NER 벤치마크 리포트 (5종 canonical, 시설=ORG)

> 본 리포트는 이슈 #21 에서 8종 → 5종(`PER · LOC · ORG · PROD · EVT`)으로
> 축소되고, 이슈 #27 Phase 4 에서 `LOC/ORG` 경계가 재정의된 canonical
> 스키마 위에서 측정한 **PII 주입 전 원본 NER 라벨러 정당성** 측정이다.
>
> #27 LOC/ORG 재정의 요지: `LOC` = 지명·주소만 (국가·행정구역·자연
> 지명·주소). 인공 시설(역·공항·병원·초중고·대학(본체·캠퍼스·시설
> 모두)·점포·박물관·도서관·종교시설 등) 은 모두 **ORG** 로 통합.

**측정일**: 2026-04-28 (Phase 4)
**데이터셋**: `data/stockmark/test.jsonl` (1,069 samples, 5종 canonical, PII 미주입)
**총 엔티티**: 2,621 개 (PER 554 / LOC 400 / ORG 1,233 / PROD 229 / EVT 205)
**평가지표**: 문자 오프셋 Span F1 (`metrics/span_metrics.compute_offset_span_f1`)
**엔티티 타입**: PER · LOC · ORG · PROD · EVT
**프롬프트**: 10종 평면 (`PER LOC ORG PROD EVT EMAIL PHONE DAT ID_NUM CREDIT_CARD`) — `src/ner/labelers/ja/ner_prompts.py`
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
| **Filtered F1** | 예측에서 PII 5종을 제외한 뒤 5종에 대해서만 매칭. | 5종 NER 능력의 순수 측정값 — silver 라벨러 정당성 평가 기준 |

silver 라벨러 정당성 결론은 **Filtered F1** 기준으로 내린다.

---

## 1. 요약 테이블 (Filtered F1 정렬)

| # | 모델 | 양자화 | 백엔드 | **Filtered F1** | Raw F1 | sec/sample | Total TPS | Out TPS | 총 시간 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | google/gemma-4-31B-it | BF16 (Dense) | vllm TP=2 | **0.8772** | 0.7837 | 0.175 | 12,848 | 323.7 | 3:13 |
| 2 | cyankiwi/gemma-4-31B-it-AWQ-8bit | AWQ-8 (Dense) | vllm TP=1 | 0.8767 | 0.7833 | 0.237 | 9,508 | 240.3 | 4:19 |
| 3 | Qwen/Qwen3.5-27B | BF16 (Dense) | vllm TP=2 | 0.8588 | 0.7665 | 1.125 | 1,992 | 44.9 | 20:09 |
| 4 | cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit | AWQ-8 (MoE A4B) | vllm TP=1 | 0.8549 | 0.7657 | 0.075 | 29,946 | 699.2 | 1:25 |
| 5 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | AWQ-4 (MoE A3B) | vllm TP=1 | 0.8359 | 0.7556 | 0.276 | 8,122 | 178.0 | 5:00 |
| 6 | openai:gpt-5.4-mini | - | openai API | 0.8234 | 0.7377 | 0.277 | 4,182 | 181.7 | 5:01 |
| 7 | openai:gpt-5-mini | - | openai API | 0.8160 | 0.7332 | 2.166 | 750 | 238.8 | 38:41 |
| 8 | Qwen/Qwen3.5-35B-A3B | BF16 (MoE A3B) | vllm TP=2 | 0.7900 | 0.7138 | 0.317 | 7,050 | 146.6 | 5:45 |

**품질 상위 3** (Filtered): google/gemma-4-31B-it (0.8772) ≈ gemma-4-31B-AWQ-8bit (0.8767) > Qwen3.5-27B (0.8588)
**속도 상위 3** (Output TPS): gemma-4-26B-A4B-AWQ-8bit (699 tps) > google/gemma-4-31B-it (324 tps) > gemma-4-31B-AWQ-8bit (240 tps)

> Total TPS = (prompt+completion)/sec, Output TPS = completion/sec.
> 출력 TPS 는 디코딩 처리량으로 라벨러 운영 비용의 직접 지표.

---

## 2. Phase 1 (시설=LOC, 이전 스키마) 대비 변화

| # | 모델 | Phase 1 Filtered F1 | Phase 4 Filtered F1 | Δ |
|---|---|---|---|---|
| 1 | google/gemma-4-31B-it | 0.8670 | 0.8772 | **+0.0102** |
| 2 | cyankiwi/gemma-4-31B-it-AWQ-8bit | 0.8676 | 0.8767 | **+0.0091** |
| 3 | Qwen/Qwen3.5-27B (BF16) | 0.8638 | 0.8588 | -0.0050 |
| 4 | cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit | 0.8413 | 0.8549 | **+0.0136** |
| 5 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | 0.8316 | 0.8359 | +0.0043 |
| 6 | openai:gpt-5-mini | 0.8049 | 0.8160 | **+0.0111** |
| 7 | Qwen/Qwen3.5-35B-A3B | 0.8010 | 0.7900 | -0.0110 |
| 8 | openai:gpt-5.4-mini | — | 0.8234 | (신규 측정) |

평균 Δ +0.005. LOC/ORG 라벨 분포가 바뀌었음에도 라벨러 NER 능력은
**거의 일관되게 유지**되거나 소폭 개선. 시설=ORG 통합이 라벨러에 추가적인
모호성을 주지 않았다는 정량 근거.

> Phase 1 결과는 동일 데이터셋(stockmark test 1069)·동일 프롬프트·동일
> 실행 조건(concurrency 32, 단독 컨테이너) 으로 측정되었다. Phase 1 시점
> 의 LOC 정의는 시설을 포함했고, Phase 4 는 시설을 ORG 로 재배치한 것이
> 유일한 차이.

---

## 3. 환경 및 측정 조건

- 하드웨어: NVIDIA RTX A6000 ×2
- vLLM 이미지: `vllm/vllm-openai:gemma4` (Gemma-4 계열) / `vllm/vllm-openai:v0.19.0` (Qwen 계열)
- GPU 할당:
  - TP=2 모델 (BF16): GPU 1·2
  - TP=1 모델 (AWQ): Gemma=GPU 1, Qwen=GPU 2
- OpenAI: `gpt-5-mini`, `gpt-5.4-mini`
- **모든 측정 동일 조건**: `--concurrency 32` (sample-level 병렬), `--no-bertscore`, vLLM 컨테이너 단독 구동
- 측정 스크립트: `/tmp/run-ja-phase4-bench.sh` (이슈 #27 Phase 4 일회성 오케스트레이션)

---

## 4. PII 출력 분석 (Raw vs Filtered 차이)

5종 gold 에는 PII 가 없는데도 모델은 평균 600+ 개의 PII 출력을 생성한다.
**라벨러를 silver 데이터 생산자로 운영할 때 후처리 필터링이 필수**.

| 모델 | Raw P | Filtered P | PII pred (derived) | PII FP / total pred |
|---|---|---|---|---|
| google/gemma-4-31B-it | 0.7105 | 0.8808 | 623 | 19.3% |
| gemma-4-31B-AWQ-8bit | 0.7099 | 0.8798 | 623 | 19.3% |
| Qwen3.5-27B BF16 | 0.6899 | 0.8554 | 634 | 19.4% |
| gemma-4-26B-A4B-AWQ-8bit | 0.7007 | 0.8661 | 603 | 19.1% |
| Qwen3.6-35B-A3B-AWQ-4bit | 0.6907 | 0.8379 | 556 | 17.6% |
| gpt-5.4-mini | 0.6517 | 0.7983 | 628 | 18.4% |
| gpt-5-mini | 0.6617 | 0.8098 | 596 | 18.3% |
| Qwen3.5-35B-A3B | 0.6723 | 0.8216 | 539 | 18.2% |

\* PII pred = `(Raw_TP / Raw_P) - (Filtered_TP / Filtered_P)` (PII 는 gold
에 없어 raw_TP = filtered_TP).

PII 출력률은 모든 모델에서 17~20% 수준. Phase 1 측정값 (17~21%) 과 일관.
모델 크기·구조·스키마와 무관 → 프롬프트 자체의 효과지 모델 능력의
함수가 아니다. silver 생산 파이프라인은 `pred_type ∈ {PER, LOC, ORG,
PROD, EVT}` 필터를 항상 적용해야 한다.

---

## 5. silver 라벨러 정당성 결론

8모델 Filtered F1 분포:

- 0.85 이상 (4 모델): gemma-31B 2종 + Qwen3.5-27B BF16 + gemma-26B-A4B-AWQ-8bit
- 0.80~0.85 (3 모델): Qwen3.6-35B-A3B + gpt-5.4-mini + gpt-5-mini
- 0.80 미만 (1 모델): Qwen3.5-35B-A3B BF16

**production silver 권장선**: Filtered F1 ≥ 0.85 (상위 4 모델). 1위
google/gemma-4-31B-it BF16 (0.8772) 과 2위 gemma-4-31B-AWQ-8bit (0.8767) 가
공동 후보. 양자화 영향 미미(-0.0005), 실측 noise 수준.

이슈 #27 Phase 4 PII 재생성에는 **gemma-4-31B-AWQ-8bit** (TP=1, 단일 GPU)
를 inject 모델로, **Qwen3.6-35B-A3B-AWQ-4bit** 를 verify 모델로 배치
(TP=1 단일 GPU 동시 구동).

---

## 6. 권장 설정

| 우선순위 | 권장 모델 | 근거 |
|---|---|---|
| **silver 생산 품질·효율 1위** | **google/gemma-4-31B-it (BF16)** | Filtered F1 0.8772, sec/sample 0.175, GPU 2장 |
| 단일 GPU 환경 1위 | **cyankiwi/gemma-4-31B-it-AWQ-8bit** | Filtered F1 0.8767, TP=1, 4:19 |
| 속도·VRAM 최우선 | **cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit** | Out TPS 699 (1위), F1 0.8549, 1:25 |
| Qwen 계열 단일 GPU | **cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit** | F1 0.8359, 5:00, GPU 1장, MoE |
| 외부 API (속도·비용) | **openai:gpt-5.4-mini** | F1 0.8234, 5:01 (gpt-5-mini 38:41 대비 7배 빠름) |
| 외부 API (품질) | **openai:gpt-5-mini** | F1 0.8160, 운영 단순. 단 처리 시간 38분(rate limit 영향 추정) |

PII 재생성 워크플로우 (이슈 #27 Phase 4):
- **Inject**: cyankiwi/gemma-4-31B-it-AWQ-8bit (8081, GPU 1)
- **Verify**: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (8082, GPU 2)
- 동시 구동, train+test 합쳐 ~2시간. 검증 4종 모두 PASS.

---

## 7. 운영 주의사항

- **PII 후처리 필터 필수**: 모든 모델이 17~20% 수준의 PII 타입 출력을
  생성. silver 생산 시 `pred_type ∈ {PER, LOC, ORG, PROD, EVT}` 필터
  적용해야 정확도 유지.
- **샘플 카운트**: `data/stockmark/test.jsonl` 은 1,069 samples / 2,621
  entities. 표·계산 모두 이 정확값을 사용.
- **Raw F1 사용처**: 운영 환경에서 후처리 필터를 적용하지 않을 때의 F1
  추정값. 일반적인 학습/평가 목적에는 Filtered F1 만 사용.
- **속도 측정 격리**: 모든 vLLM 모델이 동일 `concurrency=32` + 단독 컨테이너
  구동. GPU 경합 0 으로 sec/sample·TPS 비교가 공정.
- **OpenAI 측정 변동성**: gpt-5-mini 측정 시간이 Phase 1 (27분) 대비
  Phase 4 (38분) 로 늘었음. API rate limit · 외부 인프라 변동의 영향
  으로 추정. 품질 지표(F1)는 일관 — TPS/시간만 영향.

---

## 8. 산출 파일

```
results/ja-phase4-bench-2026-04/
├── 01-gemma-4-31B-it-AWQ-8bit.json       # cyankiwi/gemma-4-31B-it-AWQ-8bit
├── 02-gemma-4-26B-A4B-it-AWQ-8bit.json
├── 03-Qwen3.6-35B-A3B-AWQ-4bit.json      # cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
├── 04-google-gemma-4-31B-it.json         # BF16
├── 05-Qwen3.5-35B-A3B.json                # BF16 (MoE A3B)
├── 06-Qwen3.5-27B.json                    # BF16
├── 07-gpt-5-mini.json
├── 08-gpt-5.4-mini.json
└── run.log                                 # 측정 진행 로그
```

각 JSON: `metrics.span_f1.{overall, per_entity}` + `latency.{total_seconds, samples_per_second, avg_per_sample, prompt_tokens, completion_tokens, total_tokens, tokens_per_second, output_tokens_per_second}`.

이전 Phase 1 측정값 (시설=LOC) 의 모델별 F1 은 §2 비교 표에 보존.
원본 JSON 산출은 별도 보관 안 함 (재측정 시 위 명령 재실행).
