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
+ 정부기관 공통명사 gold 완성(lever ②) 후 재학습됐다. **불변 타입 무회귀**
(최대 하락 DAT −0.0058, fold σ 내), **ORG F1 0.830→0.849**(공통명사 완성으로
+0.019·8/10 fold 일관). ORG span 9,638→2,797. 좁고 일관된 정의 + gold
완성이 학습에 유리함을 보인다.

## 결과: pooled strict P/R/F1 (최종)

10-fold pooled(전 코퍼스 각 문장 1회 test), strict span match. narrow-ORG
재정의 + 정부기관 공통명사 gold 완성(lever ②) 반영 후 최종 상태.

| 타입 | P | R | F1 | support |
|---|---|---|---|---|
| **overall** | 0.907 | 0.931 | 0.918 | 77120 |
| PER | 0.916 | 0.919 | 0.917 | 18131 |
| LOC | 0.834 | 0.892 | 0.862 | 7499 |
| ORG | 0.811 | 0.891 | 0.849 | 2797 |
| DAT | 0.819 | 0.866 | 0.842 | 9934 |
| PROD | 0.691 | 0.732 | 0.711 | 3554 |
| EVT | 0.566 | 0.714 | 0.631 | 1388 |
| EMAIL | 0.993 | 0.995 | 0.994 | 8322 |
| PHONE | 0.997 | 0.999 | 0.998 | 8448 |
| ID_NUM | 0.998 | 0.997 | 0.997 | 8467 |
| CREDIT_CARD | 0.999 | 0.998 | 0.999 | 8580 |

불변 타입(PER/PROD/EVT/DAT/PII)은 원 baseline 대비 무회귀(최대 하락 DAT
−0.0058·fold σ 내; EMAIL −0.005 는 fold9 단일 이상치=노이즈). ORG 는
narrow-ORG 0.830 → 공통명사 완성 0.849(+0.019·8/10 fold 일관, precision
0.777→0.811·recall 0.891 유지). LOC/ORG 는 스키마가 바뀌어 old broad
baseline 과 직접 비교 불가(신 성능 기록).

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
- [x] **신-스키마 LOC/ORG**(narrow 재학습): LOC 0.862·ORG 0.830. fold
  collapse 없음.
- [x] **정부기관 공통명사 gold 완성(lever ②)**: error_analysis 로 ORG 오류
  지배 채널=정부·국회·국정원 등 공통명사 gold 비일관 규명(FP 90% 가 gold
  누락). 무표시 등장 292건 dual-LLM 판정 → ORG 260 추가 → 재학습 **ORG F1
  0.830→0.849**(+0.019·8/10 fold). (`kfold_pool_gov/pooled.json`)
- [x] 결과-시점 refuter PASS(최종 diff 96f77c9, 순환/게이밍 반증) · PR #160.

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
6. **정부기관 공통명사 완성(lever ②)**: error_analysis 로 ORG 오류 지배
   채널이 정부·국회·국정원 등 공통명사 gold 비일관임을 규명 → 전 문장
   standalone 등장을 dual-LLM 판정 → ORG 260 추가
   (`pii_all.govcomplete.jsonl`) → 재학습(`kfold_pool_gov`). production 승격.

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

## 후속 / 한계

- **프로덕션 반영 완료**: `ner_prompts.py`·KO 규칙문서 narrow-ORG 반영,
  재라벨 gold(정부기관 공통명사 완성본) production 승격
  (`data/klue/pii_all.jsonl`). #153 재범위(제목 갱신).
- **재학습 σ**: 단일 seed(42). 다-seed 분산 미측정 — fold σ 로 무회귀 마진·
  ORG 이득(+0.019, fold-σ 0.013)만 근사. EMAIL fold9 이상치가 보이듯 단일
  seed fold 불안정 존재 — 정밀화 시 ≥3 seed 권장.

## 결론

ORG 천장 확정 후 목표를 well-posedness 로 전환, 인간 감사로 사용자 실제
ORG 외연(정부·정치)을 확정하고 전수 재라벨했다. 이어 error_analysis 로 ORG
오류 지배 채널(정부기관 공통명사 gold 비일관)을 규명, 공통명사 완성으로 ORG
F1 0.830→0.849(+0.019)를 실현했다(불변 타입 무회귀). 좁고 일관된 정의 +
gold 완성이 학습에 유리함을 보인다.
