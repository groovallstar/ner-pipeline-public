# issue-82: JA classifier LOC 최대레버 진단 — 3-갈래 오류 분해 + 레버/천장 판정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/82
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-82-ja-loc-error-diagnosis`
- 승인일: <!-- 승인 후 채움 -->

## 목적

청정 결합 gold 10-fold pooled 에서 ORG·PROD·EVT 천장 확정(#71·#73·#80)
후 유일하게 남은 미진단 NER 엔티티 LOC(F1 0.8987 / P 0.8757 / R 0.9230,
support 2832)의 오류 구조를 pooled 예측(재추론 0)으로 분해해, 노이즈
바닥(±1.21pp) 위로 올릴 레버가 있는지 또는 gold 천장인지 판정한다.
하드 게이트 없음(#73·#80 전례) — 진단이 오류 구조를 닫고 레버 1개 설계
또는 천장 문서화로 끝낸다.

## 정직한 레버리지

LOC 완벽 시 overall 기여 support 15.2% 가중 → ~+2pp 상한 = 잔여 엔티티
중 최대 레버(EVT 의 3배, ORG 동급). 레버가 있다면 overall 이득이 가장 큰
후보.

## 단계 (진단-우선, #80 이중 census 프로토콜)

1. error_analysis.py `--from-predictions` 로 LOC FP/FN 전수 분해 (ccfix
   예측 재사용, 신규 학습 0). 3-갈래 정량화. **[완료]**
2. 경계·attributive 레버 판정: gold extent·국가명 자기일관성 검사. **[완료]**
3. HALLUCINATION 이중 census (model-FP 상한 + model-neutral 권위) → gold
   보정 → 10-fold 재학습 → delta vs 노이즈 바닥. **[완료]**
4. LOC↔ORG 회색지대: canonical metonymy 규칙 확인. **[완료]**
5. docs/issues + per-entity-diagnosis §LOC 축약 기록 + canonical §2.3
   attributive 규칙. **[완료]**

## 구현 결과·검증

### 진단 — LOC pooled 3-갈래 (ccfix 예측, 재추론 0)

strict TP 2614 / FP 371 / FN 218 (P 0.8757 / R 0.9230 / F1 0.8987 —
per-entity-diagnosis 권위값과 정확히 일치, 분해 검증됨).

| 경로 | FP | FN | 정체 |
|---|---:|---:|---|
| HALLUC / MISS | 247 | 77 | ∅→LOC 과예측(P leak 지배) / 완전 누락 |
| BOUNDARY | 94 | 94 | 행정·자연 extent (州·市·県·区·山, 주소 blob) |
| TYPE (LOC↔ORG) | 30 | 47 | 양방향 66 (gold LOC→ORG 40·ORG→LOC 26) |

confusion: gold LOC→pred ORG 40 / pred LOC←gold ORG 26 (LOC↔ORG 지배,
LOC↔EVT 7·LOC↔PER 3 미미).

### HALLUCINATION 247 분해 (precision leak 의 정체)

| 구성 | 건수 | 비중 | 성격 |
|---|---:|---:|---|
| attributive 국가·대지명 | 82 | 33% | 日本/中国/アメリカ/ロシア… gold 모순 클래스 |
| 〒·주소 blob | 23 | 9% | 합성 PII 주입 아티팩트 (이슈 범위 제외) |
| 일회성 고유 지명 | 142 | 58% | 전부 빈도 1, 반복 패턴 0 = EVT식 산발 |

### 경계·attributive 자기일관성 — #80 과 정반대 결과 (핵심)

#80 EVT 는 gold extent 가 고도 일관(bare=0)이라 boundary 레버 가설을
반증했다. LOC 는 **attributive 국가명에서 gold 자기모순을 직접 확인**:

| 日本+문맥 | gold-tagged | gold-NULL(model찍음) | 판정 |
|---|---:|---:|---|
| 日本では (referential) | 18 | 0 | 일관 |
| 日本において | 6 | 0 | 일관 |
| 日本国内 | 3 | 3 | **모순** |
| 日本の実業家 | 8 | 3 | **모순** |
| 日本の女性… | 4 | 2 | **모순** |
| 日本公開 | 2 | 1 | **모순** |

→ **referential 日本(장소 지칭)은 일관 태깅, attributive 日本(국적·수식)
은 gold 가 흔들린다.** 同一 표면 日本国内 이 gold有 3·gold無 3 으로 정확히
반반 = 주석 모순. 모델도 192 EXACT vs 8 MISS 로 흔들림 → attributive
nationality 의 본질적 모호성. **#80 과 달리 LOC 엔 정의 가능한 gold
모순 클래스가 존재** = 레버 후보.

경계(BOUNDARY FN 94): 모델 under-extent 68(72%)·over-extent 26 —
EVT 와 동일 방향(gold 가 더 긴 행정·주소 복합체, 州·市·県·区 suffix +
〒/丁目 주소 blob 14건을 모델이 분할). 행정 지명 복합체 single-LOC 규칙
(東京都千代田区1丁目… = 한 덩어리)을 모델이 못 지키는 compositional 한계.

### schema 결정 — attributive nationality GAP 정정 (제안)

canonical 은 `日本` 을 LOC 예시로 들지만(line 127) *국적·수식 용법* 규정이
없다(`[지명]+代表→ORG` line 122 만 존재). 50/50 모순의 근본 원인. 기존
원칙("LOC=지리적 위치 **자체**" + `代表→ORG` 환유 선례)에 맞춘 정정:

| 규칙 (§2.3 JA 추가 제안) | 판정 | 예시 |
|---|---|---|
| 국적·언어·계통 복합어 (지명+人/語/系/製) | **비-엔티티** | `日本人`, `ドイツ語`, `日系`, `アメリカ製` |
| `[지명]+代表` (기존) | `ORG` | `日本代表` |
| 지명 단독·속격·영역 지칭 (referential) | `LOC` | `日本では`, `日本の市場`, `日本国内` |

이 규칙은 F1 레버가 아니라 **gold 품질·일관성 정정**(미래 50/50 모순
예방, 모델 attributive 과예측 억제 기준). 채택 시 §2.3 표에 3행 추가 +
§7 변경 로그 1줄. **canonical 단일 출처 수정이라 사용자 승인 후 반영.**

### LOC↔ORG 회색지대 — canonical metonymy 규칙 (schema 확인)

`canonical-entity-schema.md`: `東京都` 가 행정구역 지칭=LOC, 지방정부
조직=ORG — **표면형 의미로 결정**(line 91·263). TYPE 양방향 66 의 정체:
バチカン(국가=기관)·北朝鮮政府·エジプト軍·イラン当局·ソビエト連邦が打ち上げ
= 지명의 정부/군/조직 환유. schema 가 명시적으로 표면-의미 판정을 위임한
**규정된 회색지대** = BORDERLINE 천장 귀속. attributive 국가명은 schema
에 LOC 예시(日本)는 있으나 **국적·수식 용법 규정 부재 = schema GAP**.

### attributive 82 해석적 상한 — 방어 가능 fix 는 sub-floor

attributive 82 를 직후 문맥으로 하위분류:

| 하위패턴 | 건수 | LOC 여부 (canonical 원칙) |
|---|---:|---|
| bare/other (日本初·日本馬·米英·中国上海…) | 40 | 혼재 (수식·등위·경계분할) |
| genitive (日本の+X) | 20 | 방어 가능 LOC (장소 of) |
| spatial-compound (国内·全国·本土…) | 10 | 방어 가능 LOC (영역 지칭) |
| compound +人 (アメリカ人…) | 8 | **비-LOC** (국적=사람) |
| compound +代表 | 3 | **ORG** (기존 규칙 line 122) |
| compound +語 (ドイツ語) | 1 | **비-LOC** (언어) |

해석적 F1 상한 (HALLUC→gold 보정 시):

| 시나리오 | ΔF1 | 바닥(±1.21pp) |
|---|---:|---|
| all-82 (미규정·人/語/代表 포함, 방어 불가) | +1.53pp | 겨우 위 |
| **genitive+spatial 30 (방어 가능)** | **+0.57pp** | **미달** |
| genitive 20만 | +0.38pp | 미달 |

→ **방어 가능한 어떤 schema 결정으로도 attributive 보정은 sub-floor.**
#80 EVT 와 동형(편향 상한 +1.05 → 무편향 +0.41 미달). 단 비-LOC(人/語/
代表) 12 건은 *모델 과예측*이라 gold-fix 가 아니라 모델 억제 대상.
**attributive 82 = 천장(+schema GAP 정정으로 미래 모순 예방).**

### model-FP 타깃 census — long-tail 실측 (gemma∩qwen, 247 HALL_FP)

247 HALL_FP 를 평가 BERT·gold 와 독립인 gemma@8081 ∩ qwen@8082 에
canonical 스키마로 재라벨시켜 각 span 을 LOC 로 보는지 투표:

| 판정 | 건수 | 비중 |
|---|---:|---:|
| 둘 다 LOC (강한 gold-gap) | 64 | 26% |
| 하나만 LOC (모호) | 82 | 33% |
| 둘 다 LOC 아님 (model 오류) | 101 | 41% |

schema 필터(compound +人/+代表 5건 제거) → **ACCEPT 59 gold-gap**.
구성:

| 구성 | 건수 | 성격 |
|---|---:|---|
| genuine 신규 지명 | 25 | `東京`·`京都`·`パリ`·`仙台`·`カナダ`·`38度線`… **gold 누락 = 주석 결함** |
| attributive 국가 | 32 | `日本`·`ロシア`·`アメリカ`… = schema-GAP 클래스 |
| 경계(행성) | 2 | `火星`·`天王星` (canonical 미규정 천체) |

**핵심: EVT(boundary 완전 일관, gold 무결)와 달리 LOC 는 진짜 gold
누락이 존재** — 東京·京都·パリ 를 gold 가 빠뜨린 명백한 주석 결함을
독립 dual-LLM 이 확인.

### 노이즈 바닥 정정 — LOC 는 ±0.6pp (이슈의 ±1.21pp 는 EVT 차용)

측정 프로토콜은 **이항 SE**(per-entity-diagnosis §측정): EVT support 876 → pooled
±1.21pp. LOC support 2832(3.2×) → **±1.21×√(876/2832) ≈ ±0.67pp**
(직접 이항 SE 계산 ±0.57pp 와 일치). 즉 LOC 바닥 ≈ **±0.6pp**, 이슈가
빌린 ±1.21pp 보다 빡빡.

### 판정 — borderline (attributive schema 결정이 lever/ceiling 을 가름)

ACCEPT 를 gold-fix 시 FP→TP 산술 투영 (바닥 ±0.6pp):

| 시나리오 | +N | ΔF1 | 바닥 |
|---|---:|---:|---|
| all-59 (편향 상한, attributive 포함) | 59 | **+1.11pp** | 위 |
| novel+행성 (referential-only 방어) | 27 | +0.51pp | 경계 |
| novel만 | 25 | +0.47pp | 미달 |

→ **lever 냐 ceiling 이냐가 attributive 국가명 schema 결정에 정확히
달림** ("schema 결정 먼저"가 옳았던 이유). referential-only 규칙이면
방어 delta +0.5pp ≈ 바닥(천장), tag-all 이면 +1.11pp(레버). 단 **모든
수치는 편향된 model-FP 투영** — 실제 held-out 은 재학습해야 확정(#80
에서 투영 vs 재학습 괴리 전례). 4 NER 엔티티 중 **가장 빠듯한 경계**.

| 경로 | 판정 | 근거 |
|---|---|---|
| TYPE LOC↔ORG | 천장(borderline) | schema 위임 환유, #80 EVT↔ORG 동형 |
| BOUNDARY | 천장(compositional) | under-extent 72%, 행정·주소 복합체 분할 |
| HALLUC attributive 82 | schema-GAP | 방어 fix sub-floor, 규칙 정정 대상 |
| HALLUC 주소 23 | 범위 제외 | injector 노이즈 |
| **HALLUC novel 25** | **gold 누락 확정** | dual-LLM ∩, 東京·京都·パ리 결함 |

→ census 가 25 신규 gold 누락(품질 결함)을 발견 — EVT 와 질적으로
다름. 그러나 방어 가능 F1 이득은 ±바닥 경계라, lever 확정엔 gold-fix
재학습 실측이 필요.

### 무편향 census → gold-fix → 10-fold 재학습 (권위 실측)

전수 census: GAP(gemma∩qwen LOC, gold無) **180** + OVER(gold LOC, 양
LLM 부정) 85. GAP 중 `47都道府県`(수량구) 제외, 문장 id 별 적용(기존
엔티티 겹침 보존) → **+129 LOC gold (2832→2961)**. seed 42 동일 분할
10-fold 재학습(`kfold10_phonediv_loccensus_neutral`):

| | overall F1 | LOC F1 | LOC P | LOC R | TP/FP/FN |
|---|---:|---:|---:|---:|---|
| baseline (evtcensus_neutral) | 0.9268 | 0.8964 | 0.8810 | 0.9124 | 2584/349/248 |
| **loccensus_neutral** | **0.9282** | **0.9043** | 0.8829 | 0.9267 | 2744/364/217 |
| 델타 | +0.14pp | **+0.79pp** | +0.19 | **+1.43** | — |

maximal +129 은 +0.79pp(바닥 위)였으나, **referential-only robustness
재학습이 이를 schema 인공물로 규명**:

| gold-fix | LOC F1 | P | R | ΔF1 | 바닥 |
|---|---:|---:|---:|---:|---|
| baseline (2832) | 0.8964 | 0.8810 | 0.9124 | — | — |
| **refonly +67 (구체 지명)** | **0.8995** | 0.8743 | 0.9262 | **+0.31pp** | **미달** |
| maximal +129 (attributive 포함) | 0.9043 | 0.8829 | 0.9267 | +0.79pp | 위 |

→ **방어 가능한 referential-only 보정(東京·京都·パリ 등 구체 지명 67)은
+0.31pp < 바닥 = 천장.** 바닥 위(+0.79)는 attributive 국가명(日本の/
アメリカの)을 LOC 로 인정하는 tag-all 규칙에만 의존(maximal 초과분
+0.48pp 가 전부 attributive=모델이 이미 찍은 FP→TP). refonly 는 P 가
0.8810→0.8743(↓) — 추가 67 이 대부분 모델 미예측 자리(주소·행정구역)라
새 FN, R 만 +1.38pp. overall 도 refonly 0.9268→0.9256(노이즈 내).

### 최종 판정 — LOC = gold 천장 (4연속, 단 정정 가능 결함 존재)

**LOC = ORG·PROD·EVT 와 같은 gold 천장.** 방어 가능 보정 +0.31pp < 바닥
±0.6pp. "레버" 외관은 attributive=LOC 라는 약한 schema 선택의 인공물.
"schema 결정 먼저"·"referential-only 변종"이 정확히 이 함정을 드러냄.

단 **EVT(완전 무결)와 질적 차이**: LOC 엔 진짜 gold 결함이 존재 —
GAP 67(구체 지명 누락 東京·京都·パリ) + OVER 85(attributive 과태깅
日本·ドイツ + 합성 주소 경계) = 대칭 gold 노이즈. 천장이되 **gold 품질
정정 여지가 있는 천장**. 실질 가치는 F1 이 아니라 ① 67 구체 누락 교정
② attributive schema GAP 규칙화(50/50 모순 예방).

### 산출물 (results/ gitignore, 로컬)

- 진단: `kfold10_phonediv_ccfix/diag_loc/`
  (`error_analysis_pooled.json`·`confusion_matrix_pooled.md`·
  `fp_top_LOC_pooled.md`·`fn_top_LOC_pooled.md`)

## 커밋 입도

- 진단·census 는 코드 변경 없음(도구 재사용) → 산출물 커밋 대상 아님.
- 이슈 md(계획+결과) + per-entity-diagnosis §LOC 갱신은 마무리 단계 한 커밋.

## 리스크

- 천장 가능성(ORG·PROD·EVT 3연속 전례). 단 LOC attributive gold 모순은
  #80 에 없던 신호라 census 결과에 따라 4연속 천장이 깨질 수 있음.
- 재측정 GPU 시 타 세션 results/ 정리 충돌 주의(#69·#71·#73·#80 전례).
- gold 보정은 로컬·미커밋(gitignore).
