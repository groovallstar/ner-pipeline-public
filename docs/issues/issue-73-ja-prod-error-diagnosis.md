# issue-73: JA classifier PROD F1/P/R 진단 — 오류 구조 분석·천장 판정 + 미측정 증강 폐기

- Issue: https://github.com/groovallstar/ner_pipeline/issues/73
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-73-ja-prod-error-diagnosis`
- 승인일: <!-- 승인 후 채움 -->

## 목적

청정 10-fold pooled 기준선에서 **F1 최하위**인 PROD(support 994 = 5.4%,
F1 0.7943 / P 0.7493 / R 0.8451)의 FP/FN 경로를 pooled 예측(5,270문장)
으로 진단한다. ORG(#71)와 **동일한 프로토콜**로 오류를 HALLUCINATION /
BOUNDARY / TYPE_MISMATCH / ambiguous 로 분해하고, 노이즈 바닥(±1.28pp)
위로 올릴 **구체적 처방 경로가 존재하는지** 또는 ORG처럼 **gold 천장**
인지 방어 가능한 판정을 내린다.

## 왜 하드 메트릭 게이트를 두지 않는가 (핵심 전제)

- 직전 #71에서 ORG를 model-side·data-side 세 각도로 처방했으나 전부
  노이즈 바닥을 못 넘고 "gold 천장" 결론. PROD는 ORG보다 **F1이 더 낮고
  (0.79) support가 1/5(994)** 이며 도메인이 long-tail(법안·서적·식품·
  교통카드·음악)이라 천장 위험이 구조적으로 더 크다.
- 따라서 "PROD F1 +Npp"를 성공 기준으로 박지 않는다. **진단이 먼저
  오류 구조를 드러내고**, 그 결과에 따라 처방 1개를 설계하거나 천장을
  문서화한다. (성공 = 둘 중 하나의 정직한 결론)

## 정직한 한계 (게이트와의 관계)

PROD가 완벽(F1→1.0)해져도 overall 기여는 support 5.4% 가중 → ~+1.1pp
수준. 본 이슈는 게이트(0.95) 도달 이슈가 아니라 **잔여 최하위 엔티티의
오류 구조를 닫는 진단**이다.

## 선행 제약 — fold 예측 산출물이 디스크에 없음

#71 ORG 진단이 쓴 `results/classifier/ja_sweep/kfold10_phonediv_noextra/
fold0..9` 의 `test_predictions.json` 은 현재 **존재하지 않는다**
(`results/` 는 gitignore + 머지 후 정리됨; issue-71 doc 의 "타 세션
results/ 정리 충돌 주의 #69 전례" 경고대로). 따라서 PROD pooled 진단
전에 **청정 10-fold 를 재생성**해야 한다. fold 1회 ≈ 140초, 10-fold
≈ GPU **~50분**.

진단 도구(`error_analysis.py --from-predictions --fold-dirs`)는 #71
에서 이미 구현됨 → 본 이슈는 **도구 재사용만** 하고 새로 만들지 않는다.

## 단계 (다섯)

### 0. 청정 10-fold 예측 재생성 (~50분 GPU, 코드 변경 없음)

```bash
for fold in 0 1 2 3 4 5 6 7 8 9; do
  uv run python -m ner.classifier --lang ja \
    --data data/stockmark/pii_all_phonediv.jsonl \
    --kfold 10 --fold-index ${fold} --seed 42 --metric-mode strict \
    --output-dir results/classifier/ja_sweep/kfold10_phonediv_noextra/fold${fold}
done
uv run python -m ner.classifier.kfold_pool \
  --fold-dirs results/classifier/ja_sweep/kfold10_phonediv_noextra/fold{0..9}
```

- pooled overall strict F1 이 기준선 **0.9195 ± 노이즈** 안에 드는지 확인
  (재현성 sanity). 벗어나면 데이터/시드 변동 의심 → 중단·재확인.

### 1. PROD pooled 진단 실측 (재학습 없음)

```bash
uv run python -m ner.classifier.error_analysis --from-predictions \
  --lang ja \
  --fold-dirs results/classifier/ja_sweep/kfold10_phonediv_noextra/fold{0..9} \
  --output-dir results/classifier/ja_sweep/kfold10_phonediv_noextra/diag_prod \
  --with-diagnosis --diagnosis-fp-types PROD --diagnosis-fn-types PROD \
  --diagnosis-top-n 80
```

- PROD FP 를 HALLUCINATION / BOUNDARY / TYPE_MISMATCH / ambiguous 비율로
  분해 (P 0.7493 = 과예측 leak 의 정체).
- PROD FN 을 같은 축으로 분해 (R 0.8451 = miss 의 정체; long-tail
  도메인 miss 인지 BOUNDARY miss 인지).
- TYPE_MISMATCH 는 어느 타입과 혼동되는지(ORG↔PROD, PROD↔EVT 등)
  confusion matrix 로 확인.

### 2. 처방 = gold gap census (3-way 독립 판정)

진단 결과 PROD FP 의 leak 이 HALLUCINATION 69% 지배 → 상당수가 **gold
누락**(모델이 PROD 로 맞췄으나 gold 미라벨) 가설. 이를 **순환 함정 없이**
검증·교정하기 위해, 평가 대상 BERT 와 독립인 3개 판정자의 합의만 gold
gap 으로 확정한다 (gold 는 Stockmark 인간 주석이라 LLM·BERT 양쪽과 독립).

- **2a. dual-vLLM 투표** — gemma-4-31B(8081) + Qwen3.6-35B(8082) 에 동일
  문장을 JA NER 라벨링(기존 `VllmNERLabeler` 재사용, split=False) → 각
  HALL FP span 을 PROD 로 독립 라벨하는지 투표.
  (`/tmp/census_vllm_adjudicate.py`, 산출 `census_vllm_votes.jsonl`)
- **2b. Claude 판정 + 합의** — 둘 다 PROD 인 후보를 canonical schema
  기준으로 재검수. 두 vLLM 합의도 schema 상 오류(회계용어·소행성·게임
  모드명 등)면 reject. accept = strong ∩ schema-clean.
- **2c. gold 코퍼스 보정** — accept span 을 `pii_all_phonediv.jsonl` 에
  text 정확 일치 + offset 검증으로 추가 (백업 `.bak`, 겹침 0 검증).
- **2d. 재학습·held-out 측정** — 보정 gold 로 10-fold 재학습
  (`kfold10_phonediv_census`) → pooled PROD delta 가 노이즈 바닥
  (±1.28pp)을 넘는지 측정.

> **정직성 단서**: accept 26→25 건은 *모델이 이미 PROD 로 예측한* 위치라
> gold 추가 시 FP→TP 직접 전환. 즉 이 처방은 모델 능력 향상이 아니라
> **gold 오류 보정**(측정 정정)이며, 측정 delta 는 gold-fix 효과의
> *상한*(모델에 유리한 위치만 골라 고쳤으므로)이다. 그래도 노이즈
> 바닥을 못 넘으면 천장이 a fortiori 로 확정된다.

### 3. 미측정 증강 도구 폐기 (선제적 정리)

`prod_seed`/`negative` 는 PROD 천장 회복용으로 #61 에 도입됐으나 청정
K-fold 에서 검증된 적 없음(S8 production 수치는 train/test overlap 로
~1.7pp 부풀려짐). 채택 안 된 미측정 실험 → 코드·테스트·문서 참조 정리.

폐기 대상 (삭제 전 live import 부재 확인):

- `src/ner/augmenters/ja/prod_seed/` (전체)
- `src/ner/augmenters/ja/negative/` (전체)
- `src/ner/augmenters/ja/__init__.py` (→ `augmenters/ja/` 빈 디렉토리 제거)
- `tests/ner/augmenters/ja/prod_seed/`, `tests/ner/augmenters/ja/negative/`
- 문서 참조: `src/ner/augmenters/AGENTS.md`, 루트 `CLAUDE.md`
  (`python -m ner.augmenters.ja.{negative,prod_seed}` CLI 예시 +
  디렉토리 구조 주석), `docs/manual/` 내 해당 모듈 맵 항목

> 삭제 범위는 구현 시 `grep` 으로 잔여 참조 0 을 확인한 뒤 확정한다
> (모호하면 사용자에게 범위 재확인).

### 4. 마무리

- 진단 결과·판정을 `docs/issues/issue-73-…md` 결과 섹션 + 필요 시
  `docs/reports/` 히스토리에 기록 (축약형: 과정은 인라인 요약, 영구
  인용 표만 유지).
- `pytest` green (폐기로 사라진 테스트 외 회귀 없음), `docs/` 갱신 확인.
- 이슈 md 최초 커밋 → PR(`closes #73`) → 머지 전 `git status` 확인.

## 커밋 입도 (예정)

- 진단은 코드 변경 없음(도구 재사용) → 산출물은 커밋 대상 아님(gitignore).
- 처방 채택 시: 코드+테스트+docs 한 커밋 (`refs #73`).
- 증강 폐기: 코드·테스트·문서 참조를 **한 커밋**으로(동일 관심사=폐기).
- 이슈 md(계획+결과)는 마무리 단계 최초 커밋.

## 리스크

- **재측정 GPU 10회**: 타 세션 `results/` 정리와 충돌 주의(#69·#71 전례).
- **천장 가능성 높음**: PROD가 ORG보다 불리 → 처방 없이 천장 문서화로
  끝날 수 있고, 그것이 정직한 성공 결과다(negative result 아님).
- **폐기 범위**: `negative` 는 구 S7 sweep 에 쓰였으므로 과거 산출물
  재현 불가가 됨 — 채택 안 됐으니 수용. 삭제 전 import 0 확인 필수.

## 구현 결과·검증

### S0 청정 baseline (원본 gold 994 PROD)

overall strict **0.9199** / PROD F1 **0.7895** (P 0.7656 / R 0.8149) —
#69 기준 0.9195 와 노이즈 내 일치. PROD 는 10종 중 F1 최하위.

### S1 PROD pooled 진단

- FP 248 = HALLUCINATION 171 / BOUNDARY 61 / TYPE 16
- FN 184 = MISS 88 / BOUNDARY 61 / TYPE 35
- 모델은 PROD 과예측(P<R), leak 은 정밀도 쪽. TYPE 혼동은 PROD↔ORG 지배.

### S2 census — 두 방식의 대조가 핵심 결과

독립 판정자 3종(평가 BERT 와 무관): gemma-4-31B(8081) + Qwen3.6-35B
(8082) + Claude(canonical schema). gold 는 Stockmark 인간 주석이라
LLM·BERT 양쪽과 독립 → 순환 함정 없음.

| census | 탐색 범위 | 확정 gap | PROD F1 | Δ vs 0.7895 |
|---|---|---|---|---|
| model-FP (편향) | 모델 HALL FP 171 | 25 | 0.8049 | +1.54pp |
| **model-neutral (전수)** | 5,270 전수 gemma∩qwen | **49** | **0.7933** | **+0.38pp** |

model-FP 의 +1.54pp 중 ~1.2pp 가 **모델-편향**(모델이 이미 맞춘 자리만
gold 화). 전수 독립으로 편향 제거 시 **+0.38pp = 노이즈 바닥(±1.28pp)
미달** → **PROD = gold 천장 model-neutral 확정**(ORG #71 census +0.5pp
선례와 일치). 서명: HALL FP 171→151(↓) 이지만 MISS 88→96(↑) — 독립
탐색 gap 은 모델이 못 잡아 새 FN 이 되어 F1 이 안 오른다.

### 전수 census 판정 분포 (GAP 149)

ACCEPT **61** / REJECT 41 / **BORDERLINE 47**. BORDERLINE = 군함·전차·
함포·법령·프로젝트·전시·랭킹 등 **canonical schema 미규정 범주** —
ACCEPT 와 맞먹는 규모이며 **gold 비일관·PROD 천장의 구조적 원인**.
OVER 20(gold=PROD, LLM 둘 다 부정)은 대부분 LLM span 분할 누락, 진짜
over-label 은 법령 3건(`律令法`·`思想犯保護観察法`·`連邦制定法`)뿐 →
gold 의 PROD over-labeling 은 사실상 없고 누락(gap) 한 방향.

### gold 상태 (사용자 결정)

ACCEPT 61 중 **49 건 적용**(994→1043; 12 스킵 = 기존 gold 와 경계겹침·
surface 불일치). `pii_all_phonediv.jsonl` 은 gitignore 라 gold 보정은
**로컬 한정·미커밋**.

### 현재 gold 권위 측정 (ORG·PROD 수정 누적 반영)

이후 ORG gold 에도 수정이 누적(현재 ORG 5300 = #69 5133 + #71 census 89
+ 추가 78)되어, 현재 gold(ORG 5300 / PROD 1043)로 전체 재측정:

| 기준 | overall | ORG | PROD |
|---|---|---|---|
| #69 (재현 가능) | 0.9195 | — | 0.7895 |
| **현재 gold** | **0.9167** | **0.8932** | **0.7826** |

gold 수정(ORG +167 / PROD +49)에도 per-entity 변동은 전부 노이즈 바닥
내 → **천장 재확인** (구 목표 0.95 대비 −3.3pp; 게이트는 `33b2bdc`
에서 제거). 현재 gold 는 gitignore 로컬
이라, 재현 가능한 영구 기준은 #69 의 **0.9195** 를 유지한다.

### 증강 도구 폐기 (측정으로 정당화)

`prod_seed`/`negative` 삭제 — 코드·테스트(잔여 163 pass)·문서 참조 정리.
전수 census 가 "PROD 신호 추가 → HALL 재생성"(census2 에서 새 환각 발생)
을 직접 측정 → `prod_seed`(recall oversample)는 지배적 precision leak 을
악화, `negative` 가 노릴 HALL 은 대부분 gold 누락(모델 정답)이라 무효임이
확정. 미측정 추론이 아닌 **측정 근거**로 폐기.

### 산출물 (results/ gitignore, 로컬)

- 진단: `kfold10_phonediv_noextra/diag_prod/`
  (confusion·fp/fn_top·`census_fullcorpus_{candidates,verdicts}.json`)
- 전수 라벨: `diag_prod/fullcorpus_prod_labels.jsonl`
- 재학습: `kfold10_phonediv_census`(model-FP) / `…_census2`(model-neutral)
