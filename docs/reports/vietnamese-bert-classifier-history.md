# 베트남어 BERT NER 분류기 — 실험 히스토리 상세

- 현 production 요약·영구 인용 표: `docs/reports/vietnamese-bert-classifier-benchmark.md`
- 측정 프로토콜: pooled micro-average char-offset span F1 (`seed=42`).
  suffix 코퍼스 38,371 행(stratified 5-fold), 자연 코퍼스(Phase 3) 37,706 행
  (`--group-key orig` 누출-free group-kfold).
- 본 문서는 phase별 `직전상태 → 시도 → 결과 → 잔여한계` 연대기. phase별 전체
  P/R/F1·검증·refuter는 각 `docs/issues/issue-N` 에, 현 수치는 benchmark.md 에.
- **gold 계보**: v1 `pii_all.jsonl`(#103) → v2 `pii_all_v2.jsonl`(#108 re-silver)
  → v3 `pii_all_v3.jsonl`(#112) → **Phase 3 자연 주입**으로 단일
  `pii_all.jsonl` 재생성(#116, suffix 코퍼스·버전 접미사 폐기). benchmark.md
  표 baseline 은 **v1 불변**.

| phase | 이슈 | gold | overall leaked (phobert/xlm-r) | overall grouped (phobert/xlm-r) | 핵심 |
|---|---|---|---|---|---|
| 0 | #103·#106 | v1 suffix | 0.9461 / 0.9459 | ≈0.939 / ≈0.937† | 누출 제거 + offset-trim + PhoBERT |
| 1 | #108 | v2 suffix | 0.9529 / 0.9543 | ≈0.946 / ≈0.946† | PROD 천장 §3 re-silver(+5~7.5pp), EVT 회귀 |
| 2 | #112 | v3 suffix | 0.9513 / 0.9525 | ≈0.944 / ≈0.944† | EVT 회귀 해결(recall_strict_evt), PROD give-back |
| 3 | #116 | natural | 0.9506 / 0.9517 | **0.9437 / 0.9430** | suffix→자연 주입; PII 0.99=포맷학습(가설 반증) |

> **grouped 열 — 측정 vs 추정**: Phase 3(자연 코퍼스)은 **측정값**(원문
> group-kfold 직접 평가). Phase 0–2(suffix)는 **추정(≈†)** = leaked − Phase 3
> 측정 인플레(overall phobert −0.69pp / xlm-r −0.87pp). suffix v1/v2/v3 코퍼스
> 파일은 단일 `pii_all.jsonl` 정책으로 폐기돼 직접 재측정 불가 → Phase 3(누출률
> 27.9%로 유사) 측정의 대표 추정으로 환산. Phase 3 측정 anchor(leaked→grouped,
> phobert/xlm-r): overall 0.9506/0.9517 → 0.9437/0.9430 · NER 5종 micro
> 0.9198/0.9226 → 0.9086/0.9083 · PII 5종 micro 0.9985/0.9972 → 0.9983/0.9974.
>
> **⚠️ leaked 열은 원문 cross-fold 누출 인플레 포함**: NER 5종 절대값
> +1.1~1.4pp(EVT/ORG/PROD +3~5pp) 부풀려져 있다. PII 5종·**phase 간 델타는
> 유효**(같은 누출이 동일 적재). 정량·정정: benchmark.md §원문 누출 정정 (#124).

## Phase 0 — 캐노니컬 5-fold 베이스라인 확립 (#103, 추론비용 #106)

### 직전 상태
- 구 production 표(0.8985, sweep, v1~v4 ablation)에 두 혼입: fast-tokenizer
  offset 정렬 버그 + WikiANN-vi 내재 train/test 누출.

### 본 단계에서 시도
- **누출 제거(불완전 — #124 정정)**: 40,000→**38,371**(−1,629) "원문 키 중복
  제거". 그러나 원본 WikiANN-vi(40,000행)의 유니크 원문은 **29,343개뿐**(#124
  측정)인데 1,629행만 제거 → 실제로는
  *주입 후* 전체텍스트 중복만 제거됐고 'cross-split 중복 0' 재검증도 주입 텍스트
  (행마다 유니크) 기준이라 **원문 cross-fold 누출이 남았다**(test 행 ~30%). 본
  phase 와 #108/#112 의 **NER 5종 절대값은 +1.1~1.4pp(EVT/ORG/PROD +3~5pp)
  인플레** — 상세·정정은 benchmark.md §원문 누출 정정 (#124).
- **offset-trim 정렬 수정**(`data_utils._encode_vi`): SentencePiece `▁` 선행
  공백·숫자 entity 후행 부호 흡착 교정 → **mmBERT 0.53→0.94 구제**, XLM-R·
  CafeBERT 정렬 0.966→1.0.
- **PhoBERT 통합**(`_encode_phobert`+`pyvi`): 단어분절 char-span 정렬(왕복 0.996).
- 인코더 5종 stratified 5-fold sweep. (#106: 추론 비용 비교 — phobert 메모리
  41%↓·단문 레이턴시 20%↓로 정확도 동률을 비용축에서 결판.)

### 결과 (v1 gold, pooled strict)
- 상위 무승부: **phobert 0.9461 ≈ xlm-r-base 0.9459**. CafeBERT 0.9367(VI
  continued-pretrain 무이득), mmBERT 0.9395, xlm-r-large 0.8364(fold0 all-O 붕괴).
- 천장: **PROD 0.717/0.710, EVT 0.792/0.803**(저support·silver 노이즈). PII 5종
  0.98~1.00 포화, PER/LOC/ORG 0.88~0.93.

### 잔여 한계 → 다음 가설
- PROD/EVT 천장이 (a) silver 노이즈 / (b) 모델 한계 / (c) schema 미규정 중
  무엇인지 미규명 → #108 소규모 감사로 결판.

## Phase 1 — PROD/EVT 천장 §3 감사 + 프롬프트 정렬 re-silver (#108)

### 직전 가설
- 천장 원인 미상. JA #84/#85의 canonical §3 회색지대 기준을 VI gold 생성에
  이식 가능한지 미검증(`wikiann_vi/prompts.py`는 §3 기본만, 3.1/3.2/3.3 누락).

### 본 단계에서 시도
- **§3 기준 gold 감사**(fold0, gemma-4-31B 판정, 272건): 불일치를 silver오류/
  모델오류/schema-갭 3분류 + IAA(temp0 vs temp0.7).
- **VI 프롬프트 §3 정렬**: 서비스(`dịch vụ`) 제거·법령/다년 process=비-entity·
  창작물 positive few-shot — 재라벨·LLM 라벨러 4사이트.
- **격리 re-silver**(`scripts/resilver_vi_isolated.py`): 원문 NER만 2모델
  (Gemma+Qwen) `recall_strict` 합의로 재라벨, 주입 PII·텍스트·split 고정(변수=
  프롬프트 하나) → `pii_all_v2.jsonl`. 5-fold×2 재학습.

### 결과 (v1→v2)
- **천장 분해**(사용자 원질문 "silver냐 모델이냐"): PROD precision(.725→보정
  .897)=silver 창작물 누락, recall(.698→.743)=모델 실측 약점. **schema_gap 0·
  IAA 1.0**(클래스 정의는 병목 아님 — VI gold가 §3.1/3.2 미반영이 원인).
- **PROD pooled 0.7171→0.7920(phobert)·0.7097→0.7614(xlm-r)** (+5~7.5pp, 보정
  추정 0.81 적중). overall 0.9461→0.9529·0.9459→0.9543, LOC·ORG 동반 상승.
- gold PROD 2,005→2,394(+389 창작물 회복), EVT 474→436(−38 과라벨 제거).

### 잔여 한계 → 다음 가설
- **EVT 회귀**: 0.7920→0.7731(phobert)·0.8031→0.7348(xlm-r), 양 모델 일관.
  저support(436) EVT가 새 프롬프트로 2모델 합의가 더 갈려 sparse·noisy → #112.

## Phase 2 — EVT 합의 회귀 해결 (recall_strict_evt) (#112)

### 직전 가설
- EVT 회귀가 (a) 합의 불일치 / (b) 과라벨 제거 / (c) 진짜 손실 중 무엇인지
  분해 필요. 합의 기준 유지하며 few-shot 보강 vs 합의 메커니즘 조정 중 택일.

### 본 단계에서 시도 (기존 re-silver 아티팩트 재사용, 진단은 GPU 없이)
- **진단**: EVT 합의 Jaccard **0.548**. Gemma-only EVT 178건 중 **151(85%)이
  Qwen 무-span**(type 분쟁 아닌 순수 recall 미스). 원인 = VI EVT는 보통명사-핵
  서술구(Cúp·Trận·Công ước·Bão…)가 다수인데 Qwen(MoE active 3B·4bit)의 엄격한
  고유명사 편향이 EVT에만 집중 타격 → strict 교집합이 EVT 를 *EVT에 가장 약한
  모델*의 recall 에 천장. v1→v2 EVT diff: removed 131 중 정당 제거(다년process·
  시대) 19, **legit 손실(연도대회·조약·전쟁·재해) ~98**.
- **메커니즘 조정**(few-shot 아님 — 예시 이미 존재해 헤드룸 불확실): `merge_
  confidence.recall_strict_evt` 정책 — single-model EVT(gemma_only/qwen_only
  대칭) 중 §3 명시 legit 카테고리(연도대회·조약/공의회·전쟁·재해·선거) regex
  매칭분만 구제. **§3 결정론 규칙의 독립 재적용**이라 LLM 판정자 없이 self-
  confirm/유도 회피. 미매칭 long-tail 보수 drop.
- 기존 Gemma/Qwen relabel 재-merge → `pii_all_v3.jsonl`(EVT +183 구제·0 제거·
  구제 오탐 0, 비-EVT는 v2와 바이트 동일). 5-fold×2 재학습.

### 결과 (v2→v3)
- **EVT pooled 0.7731→0.8224(phobert)·0.7348→0.8370(xlm-r)** — 양 모델 **v1
  초과**(+3.0/+3.4pp). EVT precision 회복(xlm v2 0.658→v3 0.806).
- **per-fold std 붕괴**: EVT std phobert 0.089→**0.011**, xlm 0.075→0.043 —
  회귀의 본질이 *평균*이 아니라 *합의 불안정(분산)*이었음을 재학습이 직접 확증.
- **PROD give-back −2.2~2.4pp**(phobert 0.792→0.769, xlm 0.762→0.740 — EVT↔PROD
  경쟁; PROD gold 불변이라 모델 효과). 단 v1 대비 +3~5pp로 #108 이득 보존.
  overall flat(0.9513/0.9525), PII 5종·비-EVT 불변.

### 잔여 한계 → 다음 가설
- EVT long-tail(투어·금융위기·영화제 등 ~65건) 패턴 미커버 — 의도적 보수 drop.
  PROD recall 천장(모델 약점)은 미해결.
- PII 주입이 suffix 스캐폴딩(81.7%)이라 PII F1 0.99가 과대평가일 수 있음 →
  #116(자연 주입) 측정 후 결정 → **Phase 3 에서 반증**(자연 주입에서도 PII
  ~0.99 유지, 오히려 +0.5~0.6pp).

## Phase 3 — suffix 스캐폴딩 제거, PII 자연 주입 (#116)

### 직전 가설
- suffix 모드 PII(전체 81.7%가 `Liên hệ:`·`CCCD:`·`Email:` 등 경직 접두어)가
  "접두어 뒤 = PII" 표면 단서를 만들어 PII F1 0.99 가 과대평가일 수 있다
  (Phase 2 잔여). 자연 문맥 PII 일반화가 미검증.

### 본 단계에서 시도 (재학습 신규 없음 — #124 누출감사와 코퍼스 공유)
- **자연 주입 재생성**(`augmenters/pii --mode llm`, Qwen3.6-35B): 원문 NER(v3)
  은 string-match 재배치만, PII 는 문중 자연 삽입. suffix gold → 원문 추출
  (`build_vi_natural_source`) → 주입 → 단일 `pii_all.jsonl`(37,706 행, 버전
  접미사 폐기). LLM verify 제거 — 라벨이 구성상 정답이라 무가치 + 기본 drop 이
  "모델이 못 알아본 어려운 자연 PII"를 제거해 무결성 훼손(suffix 0.99 낙관을
  다른 경로로 재현).
- **결정론 품질 게이트**(배포 코퍼스 직접 재검증, `validate_vi_natural`):
  suffix 마커 **0**, span offset round-trip **100%**, 무라벨 email **0**/
  phone **7**, PII 무변형. NER 54,895·PII 35,511 span 이 #124 감사 support 와
  정확 일치 → 배포=감사 코퍼스 provenance-lock.
- **5-fold×2 평가 = #124 누출감사 재사용**: 자연 코퍼스 leaked vs grouped
  전수 1회. suffix v3(resilver_v3, leaked)와 같은 leaked 분할 *방식*으로
  비교해 주입 스타일 효과를 **근사** 분리(코퍼스 크기·내용 차 confound 는
  잔여 한계).

### 결과 (suffix v3 → 자연, phobert / xlm-r)
- **PII 가설 반증 — 방향까지**: 같은 leaked 분할 PII 5종 micro
  **0.9921→0.9985(+0.64) · 0.9917→0.9972(+0.55)** — 자연 주입 PII 가 오히려
  *약간 높다*(DAT/CC). 0.99 는 suffix 표면 단서가 아니라 **PII 내부 포맷
  (이메일·전화·ID·카드 형태) 학습**의 산물 → 스캐폴딩 인플레 가설 기각.
- **NER 5종 micro −0.5pp**(0.9252→0.9198 · 0.9274→0.9226): 문중 삽입이 주변
  문맥을 약간 더 어렵게 하나 붕괴 없음(v3 라벨 재사용이라 체계적 회귀 가능성
  낮음, 보존율 99.4%).
- **overall flat**(0.9513→0.9506 · 0.9525→0.9517, −0.07/−0.08).
- **배포(누출-free) 수치 = grouped**: overall 0.9437/0.9430, NER5 0.9086/
  0.9083, PII5 0.9983/0.9974 — 자연 코퍼스의 정직한 일반화 성능.

### phobert 10-fold 상세 (VI baseline 확정)

phobert 를 VI 분류기 baseline 으로 확정 → 누출-free 10-fold(group-kfold,
`cross_fold_orig_dups=0`)로 overall + per-entity P/R/F1 재측정. 5-fold 대비
train 90%(vs 80%)라 overall +0.20pp(0.9437→0.9457). xlm-r 은 baseline 비채택
으로 10-fold 미측정(5-fold 값은 위 표·phase 0–2 유지).

| entity | F1 | P | R | support |
|---|---|---|---|---|
| **overall** | **0.9457** | 0.9390 | 0.9524 | 90,406 |
| PER | 0.9177 | 0.9117 | 0.9238 | 21,848 |
| LOC | 0.9472 | 0.9408 | 0.9536 | 22,141 |
| ORG | 0.8738 | 0.8717 | 0.8758 | 8,033 |
| PROD | 0.7014 | 0.6321 | 0.7878 | 2,314 |
| EVT | 0.8081 | 0.7690 | 0.8515 | 559 |
| DAT | 0.9983 | 0.9975 | 0.9992 | 7,104 |
| EMAIL | 0.9998 | 0.9997 | 0.9999 | 7,182 |
| PHONE | 0.9983 | 0.9973 | 0.9993 | 7,059 |
| ID_NUM | 0.9968 | 0.9962 | 0.9975 | 7,101 |
| CREDIT_CARD | 0.9989 | 0.9989 | 0.9990 | 7,065 |

n_sentences=37,706 · 10 folds · 전수 1회 pooled micro. PII 5종 0.997~1.000
유지(누출-free에서도 — §Phase 3 가설 반증 재확인). PROD/EVT recall 천장은
5-fold 와 동일 경향(저support·모델 약점). 출처:
`results/classifier/vi/kfold10/grouped/phobert-base-v2/pooled_metrics.json`.

### 잔여 한계
- **보존율 정확 재측정 불가**: v3 source NER 이 단일 `pii_all.jsonl` 정책으로
  삭제 → 주입시점 게이트(99.41% overall, EVT per-label 92.9%)가 유일 측정.
  #124 NER5 F1 무붕괴가 무결성 교차검증.
- NER5 자연 −0.5pp 가 (a)문중 문맥 난이도 (b)코퍼스 크기차(38.4k→37.7k) 중
  무엇인지 미분리 — suffix 코퍼스 폐기로 동일분할 재측정 불가.
- 측정 출처: `results/classifier/vi/audit/{natural_gate_deployed,
  natural_vs_suffix_comparison}.json`, `leak_audit/leak_inflation_summary.json`.
