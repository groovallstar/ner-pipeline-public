# issue-153(재정의): 한국어 LOC/ORG/시설 외연 재정의 — narrow-ORG 전수 재라벨·재학습

- Issue: https://github.com/groovallstar/ner_pipeline/issues/153
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-153-ko-loc-org-facility-schema`
- 선행: `issue-153-ko-org-diagnosis.md`(1단계 ORG 일관성 레버 +0.028) →
  2단계 공정 census 천장 확정(ORG ΔF1 +0.006, 2σ 내) → **3단계 본 재정의**

## 목적

ORG F1 추격이 천장(+0.006)임이 확정된 뒤, 목표를 **F1 상승이 아니라
LOC/ORG/시설 경계의 일관·well-posedness** 로 전환한다. 인간 감사로 드러난
사용자 실제 ORG 외연(정부·행정·공공·정치 기관만)에 맞춰 gold 를 전수
재라벨하고, 불변 타입 무회귀를 확인하며 재학습한다.

## 확정 스키마 (narrow-ORG, locked)

- **LOC** = 행정구역·정착지·국가·대륙 + 자연지명(산·강·바다·섬·호수·화산)
  + 주소. 지명이 구단·팀·환유로 쓰여도 표면형이 지명이면 LOC.
- **ORG** = **정부·행정·공공·정치 기관만** — 부·청·위원회·정당·노조·공공기관·
  공기업·국제기구·정치조직 + 환유 정부건물명(청와대·백악관·국회의사당·정부청사).
- **DROP(비-entity)** = 나머지 전부 — 민간 회사·은행·언론/방송·스포츠구단
  (지명 아닌 팀명)·군·군부대·법원·검찰·종교 조직/시설·모든 인공 시설·venue.
- 복합 span 은 분할: "미국 백악관"→「미국」LOC+「백악관」ORG,
  "서울 명동성당"→「서울명동」LOC(성당 버림).
- EVT/PER/PROD/DAT/PII 정의 불변(범위 밖). 단일 출처:
  `docs/manual/data/korean-entity-labeling-rules.md`.

## 결과 요약

narrow-ORG 는 **인간 감사로 검증**됐고(ORG precision 0.50→1.00), 전수 재라벨
후 **불변 타입 무회귀**(최대 하락 DAT −0.0041, fold σ 내)로 재학습됐다.
ORG span 은 9,638→2,537(−74%)로 급감했으나 신-스키마 ORG F1 0.830 을
유지 — narrow-ORG 가 더 학습 가능·well-posed 함을 시사한다. LOC/ORG 는
스키마가 바뀌어 old baseline 과 직접 비교 불가(신 성능 기록만).

## 결과: pooled strict P/R/F1 (baseline → retrain)

10-fold pooled(전 코퍼스 각 문장 1회 test), strict span match. ¹=변경 타입
(narrow-ORG 신-스키마 — old 와 정의·support 달라 직접 비교 불가). overall
support 는 하드 LOC/ORG 제거로 84,908→76,860 이라 overall 은 비교 부적절 —
무회귀 판정은 **불변 타입**(동일 라벨·동일 fold·동일 config) 기준.

| 타입 | base P | R | F1 | new P | R | F1 | sup(b→n) |
|---|---|---|---|---|---|---|---|
| **overall** | 0.904 | 0.921 | 0.912 | 0.909 | 0.931 | 0.920 | 84908→76860 |
| PER | 0.919 | 0.921 | 0.920 | 0.919 | 0.921 | 0.920 | 18131 |
| DAT | 0.831 | 0.865 | 0.847 | 0.821 | 0.867 | 0.843 | 9934 |
| PROD | 0.678 | 0.750 | 0.712 | 0.694 | 0.725 | 0.709 | 3554 |
| EVT | 0.561 | 0.681 | 0.615 | 0.581 | 0.705 | 0.637 | 1388 |
| EMAIL | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 8322 |
| PHONE | 0.997 | 0.999 | 0.998 | 0.997 | 0.999 | 0.998 | 8448 |
| ID_NUM | 0.997 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 | 8467 |
| CREDIT_CARD | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 8580 |
| LOC ¹ | 0.868 | 0.864 | 0.866 | 0.838 | 0.889 | 0.862 | 8446→7499 |
| ORG ¹ | 0.815 | 0.853 | 0.833 | 0.777 | 0.891 | 0.830 | 9638→2537 |

불변 타입은 P/R/F1 모두 baseline 근접(최대 F1 하락 DAT −0.0041, fold σ 내).
변경 타입 ORG 는 sup 74% 감소·narrow 재정의로 recall 0.853→0.891 상승(정의가
균질해짐)·precision 0.815→0.777, F1 0.833→0.830 유지.

## 수락 기준 결과 (검증 가능 항목)

- [x] **narrow-ORG 인간 감사 게이트**: blind 150건(비복합 124) 채점 —
  broad→narrow 로 accuracy 0.645→0.831, **ORG precision 0.50→1.00**
  (허위 ORG=회사·방송·군·법원 소멸), ORG recall 0.667, LOC 0.891/0.891,
  DROP 0.889/0.842. (`results/.../census_relabel/audit_score_narrow.json`)
- [x] **전수 재판정(dual-LLM 합의)**: 18,084 LOC/ORG 를 gemma-4-31B +
  Qwen3.6-35B 이중 독립판정. 합의 93.4%·HOLD 6.6%. flip ORG→DROP 5,903·
  LOC→DROP 755·ORG→LOC 238·LOC→ORG 19. (`relabel_metrics.json`)
- [x] **복합 분할**: 공백 포함 LOC/ORG 2,838건 dual-LLM 분할, 합의 1,623
  (실제 분할 1,008→1,102 sub-span·drop 합의 615)·HOLD 1,215. HOLD 는
  결정론적 제거(합의 실패 span 미보존). (`split_narrow.jsonl`)
- [x] **행 보존 재라벨**: 25,989행·id 순서·text 불변, LOC/ORG 만 변경.
  keep 8,726·flip 208·drop 5,454·split_emit 1,102·split_hold 1,215·hold
  858. 엔티티 LOC 8,446→7,499·ORG 9,638→2,537, **불변 타입 PER/PROD/EVT/
  DAT/PII 완전 동일**. (`pii_all.final.jsonl`)
- [x] **fold 멤버십 동일**: split_kfold_stratified 층화가 PROD/EVT 에만
  의존 → relabel 전후 fold0 test id 완전 동일(baseline 대조 확인). 통제
  per-fold 비교 성립.
- [x] **불변 타입 무회귀**: 10-fold pooled strict F1, baseline(census_
  baseline)과 동일 config(fp16·group_key None·seed 42). overall
  0.9124→0.9198, **최대 하락 DAT −0.0041(fold σ 0.0136 내)**, PROD
  −0.0028·EVT +0.0215·PII 안정. 회귀 없음. leak-free(dups=0).
  (`kfold_pool/pooled.json`)
- [x] **신-스키마 LOC/ORG**: LOC 0.8624(sup 7,499)·ORG 0.8300(sup 2,537).
  fold collapse 없음.
- [ ] 결과-시점 refuter PASS · PR(closes #153).

## 구현 (분해)

1. **census harness**(`src/ner/labelers/ko/ko_loc_org_census.py`, 1회성
   비커밋 도구): judge/split/apply/sample/score 모드. dual-LLM 합의분만
   적용, HOLD 제거. per-instance provenance JSONL + summary metrics 로
   재inference 없이 recount 가능(결과-시점 refuter 대비).
2. **narrow-ORG 재판정**: 1차 census(broad-ORG)+blind 감사에서 ORG prec
   0.48 게이트 FAIL → 사용자 실제 ORG=정부·정치만 발견 → `_SCHEMA`
   narrow-ORG 로 수정, 전수 재-judge(`judged_narrow.jsonl`).
3. **split 버그 수정**: split 프롬프트가 judge용 `_SYSTEM`("한 단어만
   출력")+split JSON 지시로 **명령 충돌** → gemma 가 단어 1개만 뱉어
   파싱 실패 → HOLD 79%. `_SCHEMA`(규칙)를 judge tail 과 분리해 split 은
   tail 없이 사용 → HOLD 42.8%·도구실패 0. `_SYSTEM`(judge) 불변으로
   judged_narrow 재사용 유효.
4. **재라벨 gold**: apply 로 judged_narrow + split_narrow 병합 →
   `pii_all.final.jsonl`(행·id·text 보존 assert).
5. **재학습**: 10-fold, baseline 동일 config, GPU 격리(vLLM 무중단).
   pooling → `pooled.json`.

## 방법론 노트

- **무회귀 통제성**: baseline 이 fp16·group_key None·seed 42 라 재학습도
  동일하게 맞춰 relabel 이외 변수를 제거(핸드오프 초안의 bf16 은 baseline
  과 어긋나 정정 — koelectra 는 ELECTRA 라 bf16 권장 대상도 아님).
  stratification 이 PROD/EVT(불변)에만 의존해 fold 멤버십이 relabel 전후
  동일 → per-fold 직접 비교. 불변 타입 라벨까지 동일하므로 ΔF1 은 순수
  LOC/ORG 학습분포 변화의 간접 효과.
- **ORG recall 정제 미채택**: HOLD/누락 ORG(지원단·연수원·정부센터 등)
  정제를 검토했으나, 실제 누락은 각 1~5건(대부분 HOLD 로 이미 배제)인
  반면 센터/연수원 확장은 민간 시설을 ORG 로 흘려 precision 재-파손 +
  well-posedness 역행. 순-손해로 판정해 narrow-ORG 확정.
- **HOLD 제거 설계**: dual-LLM 불일치(HOLD)는 stale 유지 금지 원칙에 따라
  train·eval 양쪽에서 제거(틀린 라벨 주입 대신 불확실분 배제). 복합
  HOLD 도 동일 — 임의 분절보다 제거가 well-posed.

## 미결 / 후속 (사용자 결정)

- **프로덕션 프롬프트 롤아웃**: `ner_prompts.py` 와 KO 규칙문서가 아직
  broad-ORG(삼성전자·KBS·대법원=ORG)로, narrow-ORG 학습 gold 와 불일치.
  프로덕션 라벨러를 narrow-ORG 로 바꾸면 회사·방송·법원을 더 이상 추출하지
  않는 급격한 변경이라 별도 확인 필요.
- **이슈 구조**: 본 재정의는 #153(1단계 진단) 아래 진행됐으나 별도 스코프.
  #153 재범위 vs 신규 이슈 분리는 PR 시 결정.
- **재학습 σ**: 단일 seed(42). 다-seed 분산 미측정 — fold σ 로 무회귀
  마진만 근사.

## 결론

ORG 천장 확정 후 목표를 well-posedness 로 전환, 인간 감사로 사용자 실제
ORG 외연(정부·정치)을 확정하고 전수 재라벨했다. 불변 타입은 무회귀,
narrow-ORG 는 sup 74% 감소에도 F1 유지 — 좁고 일관된 정의가 학습에
유리함을 보인다. 프로덕션 롤아웃은 라벨 의미의 급변이라 사용자 확인 후 별도.
