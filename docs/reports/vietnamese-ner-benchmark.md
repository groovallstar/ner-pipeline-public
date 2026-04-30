# WikiANN-vi NER 벤치마크 리포트 (5종 canonical)

> 본 리포트는 이슈 #8 의 **PII 주입 전 원본 NER 라벨러 정당성** 측정이다.
> WikiANN-vi 원본은 PER/LOC/ORG 3종이지만 본 프로젝트는 #30 에서
> 정의한 canonical 5종(`PER · LOC · ORG · PROD · EVT`) 으로 silver 를
> 재라벨하고, 그 silver 를 gold 로 삼아 라벨러 F1 을 측정한다.

**측정일**: 2026-04-30
**데이터셋**: `data/wikiann_vi/test.jsonl` 의 첫 1000 샘플 — canonical
5-gold (recall_strict policy, #30 silver), `data/wikiann_vi/test_canonical_5gold_1000.jsonl`
**총 엔티티**: 1,054 개 (PER 402 / LOC 351 / ORG 228 / PROD 54 / EVT 19)
**평가지표**: 문자 오프셋 Span F1 (`metrics/span_metrics.compute_offset_span_f1`)
**엔티티 타입**: PER · LOC · ORG · PROD · EVT
**프롬프트**: 10종 평면 (`PER LOC ORG PROD EVT EMAIL PHONE DAT ID_NUM CREDIT_CARD`) — `src/labelers/vi/ner_prompts.py`
**측정 격리**: vLLM 컨테이너 단독 부팅, concurrency=32, 모델별 1000 샘플

---

## 0. 측정 방법론 — Raw F1 vs Filtered F1

VI 라벨러는 5종 NER + 5종 PII 의 **10종 통합 프롬프트** 를 사용한다 —
PII 주입본 평가와 코드를 공유하기 위함. 본 측정의 gold 는 5종 NER 만
포함하므로, 모델이 출력한 PII 5종(`DAT EMAIL PHONE ID_NUM CREDIT_CARD`)
은 **모두 False Positive** 로 잡혀 precision 이 인위적으로 하락한다.

JA 벤치와 동일 방법론으로 본 리포트는 두 가지 F1 을 보고한다.

| F1 종류 | 정의 | 용도 |
|---|---|---|
| **Raw F1** | 10종 프롬프트 출력 그대로 5종 gold 에 매칭. PII 출력은 FP. | 운영 시점 라벨러 동작의 실측치 |
| **Filtered F1** | 예측에서 PII 5종을 제외한 뒤 5종에 대해서만 매칭. | 5종 NER 능력의 순수 측정값 |

silver 라벨러 정당성 결론은 **Filtered F1** 기준.

> 본 측정에서는 PII 출력 빈도가 매우 낮아 Raw≈Filtered. WikiANN-vi
> 본문에 PII 단서어가 거의 없어 모델이 PII 를 거의 예측하지 않는다.

---

## 1. 요약 테이블

| # | 모델 | 양자화 | 백엔드 | F1 | Precision | Recall | sec/sample | Total TPS | 총 시간 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | cyankiwi/gemma-4-31B-it-AWQ-8bit | AWQ-8 (Dense) | vllm TP=1 | **0.8764** | 0.8121 | 0.9516 | 0.113 | 17,217 | 1:53 |
| 2 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | AWQ-4 (MoE A3B) | vllm TP=1 | 0.8121 | 0.8481 | 0.7789 | 0.189 | 9,742 | 3:09 |
| 3 | openai:gpt-5.4-mini | - | openai API | 0.7907 | 0.7132 | 0.8871 | 0.231 | 2,814 | 3:51 |

**품질 1위**: gemma-4-31B-AWQ-8bit (F1 0.8764) — recall 0.95 가 견인.
**precision 1위**: Qwen3.6-35B-A3B-AWQ-4bit (P 0.8481) — 보수적 예측.
**recall 1위**: gemma-4-31B-AWQ-8bit (R 0.9516).

> 본 라인업은 Phase 1 측정분 (현재 가용 vLLM 컨테이너 + OpenAI). JA
> 벤치 미러링 라인업의 잔여 3종 (`google/gemma-4-31B-it` BF16 TP=2,
> `Qwen/Qwen3.5-27B` BF16 TP=2, `cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit`)
> 은 vLLM 컨테이너 swap 후 측정 예정.

---

## 2. Per-Entity Breakdown

### 2.1 cyankiwi/gemma-4-31B-it-AWQ-8bit

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.937 | 0.913 | 0.963 | 402 |
| LOC | 0.938 | 0.925 | 0.952 | 351 |
| ORG | 0.927 | 0.908 | 0.947 | 228 |
| PROD | 0.612 | 0.448 | 0.963 | 54 |
| EVT | 0.778 | 0.824 | 0.737 | 19 |

### 2.2 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.842 | 0.885 | 0.804 | 402 |
| LOC | 0.832 | 0.930 | 0.752 | 351 |
| ORG | 0.859 | 0.962 | 0.776 | 228 |
| PROD | 0.708 | 0.678 | 0.741 | 54 |
| EVT | 0.850 | 0.810 | 0.895 | 19 |

### 2.3 openai:gpt-5.4-mini

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.875 | 0.847 | 0.906 | 402 |
| LOC | 0.836 | 0.817 | 0.855 | 351 |
| ORG | 0.880 | 0.849 | 0.912 | 228 |
| PROD | 0.393 | 0.257 | 0.833 | 54 |
| EVT | 0.720 | 0.581 | 0.947 | 19 |

---

## 3. 자체 평가 편향 주의

gemma-4-31B-AWQ-8bit · Qwen3.6-35B-A3B-AWQ-4bit 는 본 silver(`recall_strict`
policy 의 5-type canonical) 를 직접 생산한 모델이다. 따라서 두 모델의
F1 은 자기 silver 에 대한 자기일치(self-bench) 효과를 일부 포함하며,
절대 성능보다는 다음 두 값을 비교에 활용한다:

1. **gpt-5.4-mini F1** — 외부 모델로서 silver 합의에 얼마나 근접하는지의
   독립 측정. 0.7907 은 두 silver 생성 모델보다 6~9 pp 낮다.
2. **per-entity 분포** — PER/LOC/ORG (gold 가 되기 좋은 명확한 카테고리)
   에서는 모든 모델이 0.83 이상. PROD/EVT 는 모델별 변동 폭이 크며
   특히 gpt-5.4-mini PROD F1 0.393 — 작품·곡 등 generative entity 인식
   능력 차.

---

## 4. 속도·자원 비교

| 모델 | sec/sample | Out TPS | GPU | TP |
|---|---:|---:|---|---:|
| gemma-4-31B-AWQ-8bit | 0.113 | 17,217 | 1 GPU 49 GB | 1 |
| Qwen3.6-35B-A3B-AWQ-4bit | 0.189 | 9,742 | 1 GPU 47 GB | 1 |
| gpt-5.4-mini | 0.231 | 2,814 | OpenAI API | - |

> 본 측정 시점 GPU 0 가 idle 상태였으므로 vLLM 컨테이너의 KV cache 가
> 충분히 확보됨. 생산 환경 (다른 워크로드 동시 구동) 에서는 sec/sample
> 이 1.5~2x 증가할 수 있다.

---

## 5. 데이터셋 구조

```
data/wikiann_vi/
├── test_canonical_5gold_1000.jsonl   # 1000 샘플 gold (#30 recall_strict)
└── ... (다른 split)
```

각 record: `{id, text, entities: [{text, label, start_char, end_char}]}`.
WikiANN HF 원본 (PER/LOC/ORG 3종) 과 1000 ID 가 정합되도록 `id` 기준
정렬된 부분집합.

---

## 6. 재현 명령

```bash
# 데이터 (이미 생성됨, 재생성 불필요)
# data/wikiann_vi/test.jsonl 에서 첫 1000 ID 를 알파벳 순으로 추출

# 벤치
source ./.env
uv run python -m llm_eval --lang vi \
  --local-file data/wikiann_vi/test_canonical_5gold_1000.jsonl \
  --models vllm:cyankiwi/gemma-4-31B-it-AWQ-8bit \
  --vllm-url http://localhost:8081/v1 \
  --max-samples 1000 --no-bertscore \
  --output results/issue-8/canonical_5gold/01-gemma-4-31B-AWQ-8bit.json
```

---

## 7. 한계

- silver gold 자체가 LLM 산출이며 인간 검수 미경유. 절대 정확도가 아닌
  silver 합의 일치도 측정.
- Phase 1 라인업 (3 모델). JA 미러링 풀 라인업 (6 모델) 측정 진행 시
  본 리포트 §1 표를 확장한다.
- WikiANN-vi 본문의 짧은 문장 특성상 문장 내 entity 개수가 평균 1.05
  로 낮음. 긴 문서에서의 라벨러 동작은 별도 평가 필요.
