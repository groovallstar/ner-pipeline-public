# issue-84: JA PROD 회색지대 schema 규정 — 4범주 일관화 + relabel·재측정

- Issue: https://github.com/groovallstar/ner_pipeline/issues/84
- PR: <!-- 머지 직전 채움 -->
- 브랜치: <!-- #82(미머지) 의존 — base 정리 후 확정 -->
- 승인일: 2026-06-08

## 동기 — #73 천장 결론 재감사에서 파생

"PROD P/R 향상" 요청 → #73 이 PROD = gold 천장으로 이미 닫은 클래스라
먼저 **재감사**했다. 디스크 산출물(noextra/census/census2)로 재현:

| run | PROD F1 | P | R | overall |
|---|---:|---:|---:|---:|
| baseline | 0.7895 | 0.7656 | 0.8149 | 0.9199 |
| model-FP (편향) | 0.8049 | 0.7726 | 0.8400 | 0.9188 |
| **model-neutral** | 0.7933 | **0.7744** | 0.8130 | 0.9180 |

- **F1 천장 재현**: model-neutral ΔF1 +0.38pp ≪ 바닥 ±1.28pp → #73 견고.
- **#73 미보고 사실**: gold-fix 는 대칭 천장이 아니라 **비대칭 P-레버**
  (ΔP +0.88 / ΔR −0.19) — "환각 FP" 다수가 gold 누락(모델이 맞음)이라
  보정 시 P 만 오른다. "P·R 동시 향상"은 gold 로는 불가(맞교환).
- **진짜 벽 = 측정 노이즈**: per-fold PROD F1 std **5.24pp**(폭 0.697–
  0.850). ±1.28pp 는 이항 SE 만; 학습 확률성 분산은 미측정(single-run).

→ "데이터 더 수급"도 #45 가 28+8회로 부정(oversample 0/neg, KWDLC
type drift −5~−10pp, LLM aug 는 FP 증가형 P 회귀). PROD 는 data-starved
가 아니라 **schema 미규정으로 gold 비일관**. 그래서 #82 LOC §2.3 선례대로
**schema 규정**으로 전환.

## 규정 (사용자 비준, canonical §3.1)

#73 BORDERLINE 47(canonical 미규정 회색지대)을 4범주로 규정:

| 규칙 | 판정 | 근거 |
|---|---|---|
| A 제조 명명 탈것·무기·함정·항공기·우주선·기관차 | `PROD` | 형식·급·모델명 제조품 = 제품 (`Galaxy S24` 동형) |
| B 법령·법안·규정·칙유 | 비-entity | 법적 문서·제도, LAW 타입 없음 (조약=EVT 와 구분) |
| C 전시·투어·프리미어·1회성 방송이벤트 | `EVT` | 1회성 행사 |
| D named 프로젝트·계획·전략·프로그램 | `EVT` | 조직 활동; generic 단독은 제외 |

보조: generic vs named, 番組(PROD)/방송이벤트(EVT), 法 표면≠법령
(`魔法にかけられて`·`法政大学校歌`·`ホール・エルー法` 유지), 조약·랭킹 기존
규정 유지. 상세는 canonical §3.1.

## relabel (로컬·gitignore, 백업 `.preprodschema`)

전-occurrence 일관 적용, offset·overlap 무결(기존 overlap 7 은 직교,
신규 0):

- A 군용 제조 **+15** PROD (`スペースシャトルエンデバー` 는 gold 부분
  `エンデバー` 존재로 제외)
- B 법령 **−14** PROD (allowlist; 영화·교가·공정 오탐 3 제외)
- C+D **+7** EVT 추가 + **2** PROD→EVT 이동
  (`Google Summer of Code Project in 2005`·`京都電気自動車プロジェクト`)
- 순 support: **PROD 1043→1042 / EVT 968→977**

## 측정 (10-fold, baseline = loccensus_refonly)

baseline(현재 gold 권위): overall 0.9256 / PROD 0.8004·0.7725·0.8303 /
EVT 0.8593·0.8366·0.8833. 보정 gold 재학습 → `kfold10_phonediv_prodschema`.

## 구현 결과·검증

### 1) 4범주 규정 (prodschema)

보정 gold 재학습 `kfold10_phonediv_prodschema` (10-fold pooled, strict):

| 지표 | baseline | prodschema |
|---|---:|---:|
| PROD F1·P·R | 0.8004·0.7725·0.8303 | 0.8090·0.7898·0.8292 |
| overall F1 | 0.9256 | 0.9222 |

예측대로 **비대칭 P-레버**(ΔP +1.73 / ΔR −0.11), F1 변화는 per-fold
std 5.24pp 노이즈 대역 안. 성능 레버 아님 확인.

### 2) 추가 정정 — 서비스 제외 (ADR 0001)

4범주 규정 후 재감사에서 **남은 비일관의 주범이 "서비스"**임이 드러나,
PROD 정의를 positive 재규정(서비스·온라인 운영물·기술 표준 제외).
상세·근거: `docs/decisions/0001-prod-service-exclusion.md`, schema §3.2.

- PROD 1042 → **943** (이탈 99 = 재라벨 20[ORG·EVT] + 제거 79)
- gold 승격: `pii_all_phonediv.jsonl` ← prodclean, 백업 `.preprodclean`
- 프롬프트(JA `ner_prompts.py`) positive 정의 6 사이트 통일

재학습 `kfold10_phonediv_prodclean` (10-fold pooled, strict):

| 지표 | prodschema | **prodclean** | baseline |
|---|---:|---:|---:|
| PROD F1 | 0.8090 | **0.8168** | 0.8004 |
| PROD P·R | 0.7898·0.8292 | 0.7913·0.8441 | 0.7725·0.8303 |
| ORG F1 | 0.8913 | 0.8965 | 0.8982 |
| EVT F1 | 0.8366 | 0.8353 | 0.8593 |
| overall F1 | 0.9222 | 0.9251 | 0.9256 |

- overall이 baseline 수준 회복(0.9251), overall P 최고(0.9111), ORG 회복.
- **PROD F1 상승은 비교 불가** — 가장 비일관·난해했던 79 스팬 제거의
  기계적 효과(support 943≠1043). 성능 개선으로 보고 금지.
- **EVT −2.4pp(vs baseline)** 는 single-run이라 실재/노이즈 미판정 →
  **별도 이슈에서 multi-seed(s42·43·44)로 재확인 예정.**

### 미반영(후속)

- VI 라벨러·`wikiann_vi` 재라벨: VI gold 미분석이라 프롬프트만 바꾸면
  desync → VI는 gold+프롬프트 동반 정정으로 별도 처리.
- 프롬프트 EVT/ORG 정의의 §3.1 C/D 반영(named 계획·전시→EVT)도 후속.

## 정직한 전제

F1 은 바닥 아래일 공산이 큼 — 본 이슈는 성능 레버가 아니라 **gold
일관성·방어가능성** 보정(정의상 올바른 쪽으로 규정, 메트릭 미타겟).
