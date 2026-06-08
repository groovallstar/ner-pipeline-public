# issue-80: JA classifier EVT F1/P/R 진단 — 3-갈래 오류 분해 + 레버 판정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/80
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-80-ja-evt-error-diagnosis`
- 승인일: <!-- 승인 후 채움 -->

## 목적

청정 결합 gold 10-fold pooled 에서 PROD 다음으로 낮은 EVT
(F1 0.8454 / P 0.8056 / R 0.8893, support 876) 의 오류 구조를 pooled
예측(재추론 0)으로 분해해, 노이즈 바닥(±1.21pp) 위로 올릴 **레버가
있는지** 또는 PROD/ORG 처럼 **gold 천장**인지 방어 가능한 판정을 내린다.
하드 메트릭 게이트는 두지 않는다(#73 전례) — 진단이 오류 구조를 닫고,
레버 1개 설계 또는 천장 문서화로 끝낸다(둘 중 하나가 정직한 성공).

## 정직한 한계

EVT 가 완벽(F1→1.0)해도 overall 기여는 support 4.7% 가중 → **~+0.7pp
상한**(LOC support 2832 의 1/3 레버리지). 게이트 도달이 아니라 잔여
차하위 엔티티의 오류 구조 진단이다.

## 단계 (진단-우선, #73 프로토콜 재사용)

1. error_analysis.py `--from-predictions` 로 EVT FP/FN 전수 분해
   (ccfix fold 예측 재사용, 재추론 0). 3-갈래 정량화.
2. 경계 레버 판정: gold extent 자기일관성 검사(`オリンピック` bare vs
   modified) → 학습 모순인지 모델 한계인지.
3. 환각 census: HALLUCINATION FP 가 gold 누락인지 dual-vLLM∩Claude
   독립 판정(#73 census 프로토콜 재사용) — model-FP(상한) 먼저.
4. accept gold 보정 → 10-fold 재학습 → pooled EVT delta vs 노이즈 바닥.
5. 결과를 docs/issues + per-entity-diagnosis §EVT 에 축약 기록. 레버 채택 시
   처방·재측정, 아니면 천장 문서화.

## 구현 결과·검증

### 진단 — EVT pooled 3-갈래 (ccfix 예측, 재추론 0)

strict TP 779 / FP 188 / FN 97 (P 0.8056 / R 0.8893 / F1 0.8454 —
per-entity-diagnosis 권위값과 정확히 일치, 분해 검증됨).

| 경로 | FP | FN | 정체 |
|---|---:|---:|---|
| BOUNDARY | 46 | 46 | 모델 under-extent 85%(39/46), gold 가 항상 더 긺 |
| HALLUC / MISS | 117 | 34 | ∅→EVT 과예측 / 완전 누락 |
| TYPE (EVT↔ORG) | 25 | 17 | 양방향 ORG 24건 (委員会総会·武道会 회색지대) |

confusion: gold EVT→pred ORG 12 / pred EVT←gold ORG 12 (EVT↔ORG 지배).

### 경계 레버 가설 **반증** (핵심 결과)

이슈 선두 가설("gold extent 비일관 = 학습 모순")을 직접 측정으로 반증.
gold 는 고도로 일관적 — `オリンピック` bare=0/modified=35, `選手権`
0/82, `大会` 0/67, `選挙` 0/47, `戦争` 0/37, `会議` 0/22. bare 혼재는
`ワールドカップ`(2/15)·`ダービー`(1/4) 단 3건. **gold-보정 boundary
레버는 존재하지 않는다.** 경계의 실제 정체:

- 연도 prefix 경합: EVT-내부 연도 표면 16/24 가 standalone **DAT 으로도
  라벨**(DAT 264회 : EVT-prefix 38회). 모델의 DAT prior 가 `1996年` 을
  EVT 에서 분리 = intrinsic schema 경합, gold 오류 아님.
- suffix 이질 24종(祭·作戦·本大会·選手権·コンサート… 각 1건): 단일
  fixable 패턴 없음 = 모델 compositional 한계.

→ BOUNDARY = 모델/schema 천장. strict↔relaxed +2.5pp 격차는 학습
모순이 아니라 진짜 compositional 난이도.

### HALLUCINATION census — 두 방식의 대조가 핵심 (#73 프로토콜)

독립 판정자: gemma-4-31B@8081 ∩ Qwen3.6-35B@8082 (평가 BERT·gold 와
무관). 두 census 를 대조:

| census | 탐색 범위 | 적용 gap | EVT F1 | Δ vs 0.8454 | EVT P | EVT R |
|---|---|---:|---:|---:|---:|---:|
| model-FP (편향) | 모델 HALL FP 117 → ACCEPT 17 | 17 | 0.8559 | **+1.05pp** | 0.8175 | 0.8981 |
| **model-neutral (전수)** | 5,270 전수 gemma∩qwen → ACCEPT 92 | **92** | **0.8495** | **+0.41pp** | 0.8208 | 0.8802 |

- **model-FP**: 모델이 EVT 로 예측한 117 자리만 판정 → 둘 다 EVT 21 /
  모호 24 / 둘 다 아님 72(62% = 진짜 precision leak). 21 schema 판정 →
  ACCEPT 17. accept 가 *모델이 이미 맞춘 자리*라 gold 추가 시 FP→TP 직접
  전환 = **상한**(모델 편향).
- **model-neutral**: 5,270 문장 전체를 두 LLM 이 처음부터 라벨 → gold
  無 GAP 124 → schema 판정 ACCEPT 92(reject 7 = 질병·법령·그룹·정치구상)
  → gold 876→968. 모델 예측과 무관한 대칭 탐색이라 **무편향**.

### 결론 — EVT = gold 천장 (model-neutral 확정)

model-FP +1.05pp 중 ~0.64pp 가 **모델 편향**. 전수 독립으로 편향 제거
시 **+0.41pp = 노이즈 바닥(±1.21pp) 미달**(EVT per-seed 분산 ±3.69pp 는
한참 위) → **EVT = gold 천장 model-neutral 확정**. 결정적 서명: 무편향
보정에서 **R 0.8893→0.8802(↓)** — 독립 탐색 gap 92 의 대부분이 모델이
못 잡는 자리라 새 FN 이 되어 F1 이 안 오른다(P 만 +1.52, F1 순증 +0.41).
정확히 #73 PROD(편향 +1.54 → 무편향 +0.38) 와 동일 패턴.

세 갈래 종합:

- **BOUNDARY**: gold 일관(bare=0) 으로 레버 가설 반증; 연도=DAT 경합 +
  suffix 이질 = 모델/schema 천장.
- **HALLUC**: model-FP 62% 가 진짜 leak; 무편향 gold-fix(+92) 도
  +0.41pp < 바닥 → 천장.
- **TYPE (EVT↔ORG)**: 委員会総会·武道会 canonical 미규정 회색지대.

→ **EVT 는 PROD(#73)·ORG(#71) 와 같은 gold 천장.** 무편향 census 의
92 보정은 인간 주석 누락 교정(gold 품질 향상)이자 천장의 직접 증명 —
로컬 채택(#73/#78 와 동일, gitignore 미커밋). 현재 권위: overall
0.9268 / EVT 0.8495(support 968).

### 산출물 (results/ gitignore, 로컬)

- 진단: `kfold10_phonediv_ccfix/diag_evt/`
  (`error_analysis_pooled.json`·`confusion_matrix_pooled.md`·
  `fp_top_EVT`/`fn_top_EVT`)
- model-FP census: `census_candidates_evt.jsonl`·
  `census_vllm_votes_evt.jsonl`·`census_verdicts_evt.json`
- model-neutral census: `census_fullcorpus_labels_evt.jsonl`·
  `census_fullcorpus_candidates_evt.jsonl`·
  `census_fullcorpus_verdicts_evt.json`
- 보정 gold: `pii_all_phonediv.jsonl`(EVT 968, 무편향; 백업
  `.preevtcensus`=876) — gitignore 로컬·미커밋
- 재학습: `kfold10_phonediv_evtcensus`(model-FP)·
  `kfold10_phonediv_evtcensus_neutral`(model-neutral, 권위)

## 커밋 입도

- 진단·census 는 코드 변경 없음(도구 재사용) → 산출물 커밋 대상 아님.
- 이슈 md(계획+결과) + per-entity-diagnosis §EVT 갱신은 마무리 단계 한 커밋.

## 리스크

- 천장 가능성 높음(PROD #73 전례, EVT 레버리지 더 낮음). 처방 없이
  천장 문서화로 끝날 수 있고 그것이 정직한 결과.
- gold 보정은 로컬·미커밋(gitignore) — 코퍼스 재생성 시 재유입.
