# 베트남어 WikiANN-vi → Stockmark 8종 스키마 확장 리포트 (2026-04)

> 대상 이슈: #10 `feat/issue-10-vi-ner-8type-relabel`
> 스펙: `docs/specs/entities/vietnamese-ner-8types.md`
> 데이터 범위: WikiANN-vi test split 앞 1000 샘플
> 실행일: 2026-04-21

## 1. 요약

- 두 LLM(`gemma-4-31B-it-AWQ-8bit`, `Qwen3.5-27B-AWQ-4bit`)으로 WikiANN-vi
  1000샘플을 Stockmark 8종 스키마로 재라벨했다. 재라벨 에러 **0건**, 프롬프트
  기반 few-shot만으로 8종 모두를 실제 샘플에서 탐지.
- Cross-model Cohen's kappa = **0.6620** (Landis-Koch "substantial").
- 그러나 **타입 일치율(둘 다 라벨한 경우) 97.06%** vs **coverage 일치율**
  (동일 offset에 둘 다 라벨한 비율) 75.55% — kappa를 끌어내리는 주원인은
  타입 혼동이 아니라 **"엔티티로 볼지 여부"에 대한 coverage 차이**.
- WikiANN 원본에 없던 製品名·イベント名도 실제 탐지됨(gemma 129개, qwen 83개)
  → 합성 보충(PII 구조 재사용) 경로는 당장 필수가 아님. 빈도 보강은
  Wikipedia 인터링크 앵커 검증 후 재판단.
- **권장**: Gemma를 주(primary) 재라벨 모델로 지정하고, Qwen은 해당 span의
  보수적 검증(validator)으로 활용. 두 모델이 모두 라벨한 span을 "고신뢰
  silver", gemma만 라벨한 span을 "확장 silver"로 분리 관리.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| 데이터셋 | `unimelb-nlp/wikiann` config=`vi`, split=`test` |
| 샘플 수 | 1000 (앞 1000건, max_samples=1000) |
| 모델 A | `cyankiwi/gemma-4-31B-it-AWQ-8bit` @ http://localhost:8081/v1 |
| 모델 B | `cyankiwi/Qwen3.5-27B-AWQ-4bit` @ http://localhost:8082/v1 |
| 컨테이너 | vllm-gemma · vllm-qwen (별도 GPU) |
| 프롬프트 | `src/augmenters/wikiann_vi/prompts.py` SINGLE (~3.8K chars) |
| 요청 제어 | concurrency=16, temperature=0, max_tokens=2048, timeout=120s |
| max_model_len | 8192 (양 vLLM 동일) |

재라벨 CLI: `python -m augmenters.wikiann_vi`.

산출물:
- `data/wikiann_vi_relabel/gemma_8type.jsonl` (318 KB, 1000 records)
- `data/wikiann_vi_relabel/qwen_8type.jsonl` (303 KB, 1000 records)
- `data/wikiann_vi_relabel/kappa_gemma_vs_qwen.json` (상세 kappa·혼동행렬)

## 3. 타입 분포 비교

레코드별 `gold_spans_8type` 내 엔티티 수를 타입별로 집계.

| 타입 | Gemma | Qwen | Δ (G−Q) |
|---|---:|---:|---:|
| 人名 | 415 | 384 | +31 |
| 地名 | 368 | 314 | +54 |
| 製品名 | 108 | 67 | +41 |
| その他の組織名 | 87 | 62 | +25 |
| 政治的組織名 | 68 | 70 | −2 |
| 施設名 | 49 | 38 | +11 |
| 法人名 | 36 | 28 | +8 |
| イベント名 | 21 | 16 | +5 |
| **총 스팬** | **1152** | **979** | **+173** |
| 엔티티 레코드 | 853 | 695 | +158 |

- 두 모델의 **타입 순위 동일**: 人名 > 地名 > 製品名 > その他の組織名 >
  政治的組織名 > 施設名 > 法人名 > イベント名
- 政治的組織名만 역전(+2 Qwen)이며, 나머지 7종은 Gemma가 더 많이 탐지 →
  Gemma가 전반적으로 더 관대
- WikiANN이 제공하지 않는 製品名·イベント名도 양쪽 모두 **실제 샘플에서
  발견** (Gemma 129, Qwen 83) → 경로 B(재어노테이션)만으로도 두 타입 확보
  가능성이 뒷받침됨

## 4. Cross-model Cohen's kappa

offset별 타입을 pair로 수집하고(한쪽만 라벨한 경우 없는 쪽 `O`), 전체 8종 +
`O` 9개 클래스 기준 kappa를 계산.

| 지표 | 값 |
|---|---:|
| Paired spans (A∪B offsets) | 1214 |
| Observed agreement `po` | 0.7331 |
| Chance agreement `pe` | 0.2104 |
| **Cohen's κ** | **0.6620** |

Landis-Koch 해석 기준(0.61–0.80)으로 **"substantial agreement"**.

### 타입별 일치율 (Gemma 관점, Gemma 라벨 중 Qwen도 같은 타입인 비율)

| 타입 | agreed / total_A | ratio |
|---|---|---:|
| 政治的組織名 | 61 / 68 | 0.8971 |
| 人名 | 349 / 415 | 0.8410 |
| 地名 | 293 / 368 | 0.7962 |
| 施設名 | 33 / 49 | 0.6735 |
| 法人名 | 24 / 36 | 0.6667 |
| その他の組織名 | 54 / 87 | 0.6207 |
| イベント名 | 13 / 21 | 0.6190 |
| 製品名 | 63 / 108 | 0.5833 |

## 5. 핵심 관찰: coverage vs type 이중 구조

kappa 0.66은 "전반적 불일치"로 읽히기 쉽지만, span 레벨에서 분해하면 결이
다르다.

| 지표 | 값 |
|---|---:|
| Gemma span 수 | 1152 |
| Qwen span 수 | 979 |
| 동일 offset에 둘 다 라벨 (교집합) | 917 |
| Gemma만 라벨 | 235 |
| Qwen만 라벨 | 62 |
| **동일 offset 내 타입 일치율** | **890/917 = 97.06%** |

핵심 분리:
- **타입 분류의 일관성은 매우 높다(97%)**. 둘 다 엔티티로 본 span에 대해
  Gemma/Qwen은 거의 항상 같은 8종을 고른다.
- **엔티티로 볼지에 대한 coverage는 비대칭**. Gemma-only 235 vs Qwen-only
  62 → Gemma가 엔티티를 더 넓게 추출, Qwen은 더 보수적.

혼동 행렬 상위 off-diagonal 10건 중 8건이 `X → O` 또는 `O → X` 형태
("한쪽만 라벨"). 실제 타입 간 cross-confusion은 `地名 → 政治的組織名 5건`,
`その他の組織名 → イベント名 2건` 수준으로 극소.

```
상위 disagreement 유형(빈도순)
  Gemma 人名 / Qwen 없음           : 66
  Gemma 地名 / Qwen 없음           : 65
  Gemma 製品名 / Qwen 없음          : 41
  Gemma その他の組織名 / Qwen 없음   : 30
  Qwen 人名 / Gemma 없음           : 30
  Qwen 地名 / Gemma 없음           : 20
  Gemma 施設名 / Qwen 없음          : 16
  Gemma イベント名 / Qwen 없음      : 7
  Gemma 法人名 / Qwen 없음          : 6
  Gemma 地名 / Qwen 政治的組織名     : 5
```

이 패턴의 함의:
1. 8종 매핑 스펙 자체(`vietnamese-ner-8types.md`)에 **구조적 모호성이 적다**
   — LLM 두 개가 독립으로 같은 선택지에 수렴.
2. "엔티티로 볼지" 결정(≈ recall)은 프롬프트 민감도가 크다. 이 축을
   개선하려면 어느 모델을 신뢰할지가 아니라, 어느 span을 신뢰할지를
   골라야 한다.

## 6. 운영 제안

### 6.1 주·보조 라벨러 분리 운영

- **Gemma = primary 재라벨러** (재현·추출 능력 우수, 5× 빠름)
- **Qwen = 검증자(validator)**. 둘 다 라벨한 span은 "고신뢰 silver"로 격상
- Gemma-only 235 span은 "확장 silver"로 분리 보관. 학습 데이터로 쓸 경우
  가중치 조정 또는 분할 실험(포함 vs 제외)으로 영향 측정

### 6.2 합성 보충(이슈 #10 5단계) 필요성

- 製品名 108 / イベント名 21 (Gemma)은 학습 데이터로서는 여전히 희소.
- 다만 본 실험은 앞 1000샘플만 사용. WikiANN-vi test 전체(약 1만) 기준으로
  환산하면 製品名 ~1080 / イベント名 ~210 수준 예상.
- **권장**: Wikipedia 인터링크 앵커 검증(이슈 #10 4단계)을 먼저 수행해 두
  타입의 gold 신호를 확보. 그 후 **빈도가 낮은 타입에 한해서만** 합성
  보충을 재검토.

### 6.3 WikiANN silver 오라벨 관찰

3샘플 스모크에서 이미 확인된 바 있다.
- `Ốc móng tay` (굴·조개류): WikiANN LOC → Gemma/Qwen 모두 skip
- `Chính cha Lãm`: WikiANN은 "Lãm"을 non-entity → Gemma가 人名으로 탐지

이는 WikiANN silver 노이즈의 전형. 재라벨이 오히려 원본 오라벨을 교정하는
유익한 부작용이 있으나, **gold 평가 없이는 정량화 불가** — 이슈 #10 4단계의
Wikipedia 앵커로 부분 측정 가능.

## 7. 한계

- **이중 silver**: WikiANN(자동) + LLM 재라벨(자동)이 누적된 실버 데이터.
  Stockmark(gold)와의 F1 절대값 비교는 수행하지 않는다.
- **1000샘플 표본**: 전체 test(약 1만) 분포와 유의한 차이가 없을 확률이
  높지만, 저빈도 타입(イベント名 21건) 통계는 표본 오차가 크다.
- **모델 편향**: Gemma/Qwen 2종만 사용. 3종 이상으로 확장하면 kappa와
  coverage 판정이 달라질 수 있다.
- **프롬프트 민감도**: SINGLE 프롬프트 기반. BATCH·system 프롬프트 사용 시
  타입 결정 경계가 달라질 가능성.

## 8. 다음 단계 (이슈 #10 진행 매트릭스)

- [x] 1. VI loader·라벨러 이전 (`labelers/vi/`)
- [x] 2. 8종 매핑 스펙 확정 (`docs/specs/entities/vietnamese-ner-8types.md`)
- [x] 3. 경로 B 재라벨 + 2모델 kappa 측정 (본 리포트)
- [ ] 4. Wikipedia 인터링크 → Wikidata P31/P279 앵커 서브셋 생성 + 본 재라벨
      결과와 일치율 측정
- [ ] 5. 製品名·イベント名 빈도 재집계(전체 test 기준) + 합성 보충 필요성
      사용자 승인
- [ ] 6. 최종 리포트 갱신: 3중 검증(kappa + Wikipedia 앵커 + 3종 공정 비교)
      통합 해석

## 9. 재현

```bash
# 재라벨 (각 모델별)
python -m augmenters.wikiann_vi \
    --max-samples 1000 --concurrency 16 \
    --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi_relabel/gemma_8type.jsonl

python -m augmenters.wikiann_vi \
    --max-samples 1000 --concurrency 16 \
    --base-url http://localhost:8082/v1 \
    --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
    --output data/wikiann_vi_relabel/qwen_8type.jsonl

# Cross-model kappa
python -m augmenters.wikiann_vi.kappa \
    --a data/wikiann_vi_relabel/gemma_8type.jsonl \
    --b data/wikiann_vi_relabel/qwen_8type.jsonl \
    --json-out data/wikiann_vi_relabel/kappa_gemma_vs_qwen.json
```

## 10. 관련 커밋·문서

- 스펙: `docs/specs/entities/vietnamese-ner-8types.md`
- 이슈 플랜: `docs/issues/issue-10-vi-ner-8type-relabel.md`
- 재라벨 CLI: `src/augmenters/wikiann_vi/__main__.py`
- Kappa CLI: `src/augmenters/wikiann_vi/kappa.py`
- 테스트: `tests/augmenters/wikiann_vi/`
