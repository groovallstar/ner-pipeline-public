# issue-133 — VI EVT 누출-free 천장 재감사 + 헤드룸 게이트 처방

- 브랜치: `feat/issue-133-vi-evt-headroom-gated-lift` (develop 기반)
- 선행: #108(PROD/EVT §3 감사), #112(EVT 합의 회귀 해결·long-tail drop),
  #124(원문 cross-fold 누출 정량·group-kfold 영구 수정), #131(PROD 헤드룸
  게이트·prodrecover 승격·EVT per-fold std 측정)

## 목적

현 배포 gold(prodrecover, `pii_all.jsonl`) 기준 leak-free **EVT F1
0.8020**(P 0.752 / R 0.859, support 559=10종 최저)를 #131과 동형으로
재감사한다. #108 천장(~0.90)은 leaked fold0(n=95)이라 무효 — 현 gold 기준
천장을 먼저 재추정하고, 회복가능 헤드룸이 임계 이상일 때만 처방한다(go/no-go).
임계는 **std-상대**: support 559의 EVT는 per-fold 분산이 10종 최고라, 노이즈
대역 안의 lift 는 검증 불가하기 때문이다.

## 진단 (AC1·AC2) — 누출-free 전수 10-fold

도구: `audit_vi_prod_evt.py` extract→adjudicate→analyze, prodrecover 10-fold
`test_predictions.json` 전수(37,706행), 현 배포 gold. 추출 1,426 케이스(EVT
FN 87·FP 103 포함 + PROD + 대조군 40), **unparsed 0**. gemma-4-31B §3 판정.

EVT 보정 천장(영구 인용):

| 축 | base (현 배포) | 보정 천장 | 분해 |
|---|---|---|---|
| **F1** | **0.8020** | **0.8574** | 헤드룸 **5.54pp** |
| P | 0.7524 | 0.8433 | FP 158→100: silver-갭 58(37%) / 모델-FP 100(63%) |
| R | 0.8587 | 0.8720 | FN 79→79: **fn_drop=0 → recall 100% 모델** |

- 분류: silver_error 58, model_error 74, both_error 50, schema_gap **0**,
  agree 8, unparsed 0.
- **EVT precision 약축의 정체**: silver-갭(gold 누락 legit event)과 모델-FP가
  37:63. recall 약축은 **전부 모델**(FN 79 중 silver-갭 0 — gold 허위 없음).
- silver-갭 59건의 gold 오라벨 분포: **O 40**(누락) / ORG 14(국회·올림피아드)
  / LOC 4(태국 총선 2007 등) / EVT 1(경계). **PROD 출처 0** →
  EVT↔PROD gold 경쟁 없음.
- **per-fold std = 0.0543**(EVT per-fold F1 0.697~0.887, support 559 최저 →
  10종 최고 분산). PROD std 0.0280, overall std 0.0031 대조.

## 게이트 판정 (AC3) — std-상대 → 실측 검증

- **헤드룸 5.54pp ≈ 1.02 std**(임계 1 std = 5.43pp) → 기계적으론 go 턱걸이.
- 분석: 헤드룸이 ≈1 std라 천장을 100% 회복해도 lift 는 1.02 std로 retrain
  노이즈에 묻힐 것으로 예측. 헤드룸 **100%가 precision silver-갭**(recall
  헤드룸 0, 레버 없음). #131 PROD(헤드룸 14pp = 2.6 std)와 정반대.
- **판정 = 검증가능한 모델 lift 없음.** 단 (1) gemma §3 가 독립 확인한 gold
  누락은 실재하고, (2) 이 예측을 *재학습으로 실측 검증*하기로 결정 — gold-fix
  후 phobert 10-fold 재학습.

## 처방 — gold-fix + 재학습

**① gold-fix**: `audit_vi_prod_evt apply-silver-gap --type EVT`(신규 서브커맨드).
선택 = direction=FP·model=EVT·gemma correct=EVT·not ambiguous·**silver=O**
(additive 안전). 중복 제거 후 gold id 매칭 행에 surface 일치·비-겹침 재검증하여
EVT span 삽입, 원본 필드(orig) 보존. 40건 → dedup 39건 전부 삽입(skip 0).
회복 long-tail 예: "Liên hoan phim Cannes 2012", "Euro 2008", "Destiny's
Child World Tour" — **#112가 drop 한 투어·영화제를 gemma 가 회복**. 배포 gold
승격 `pii_all.jsonl` **EVT 559→598**, 구판 `pii_all_pre_evtfix.jsonl`. gold
계보: …→natural(#116)→prodrecover(#131)→**evtfix(#133)**.

**② 재학습**: 보정 gold(598)로 phobert-base-v2 10-fold(group-key orig) 재학습
→ `results/classifier/vi/evtfix/`. cross-type 19건(ORG/LOC→EVT)은 기존 라벨
변경이라 제외(위험).

## 결과 — strict span F1, leak-free (evtfix 승격)

영구 인용; relaxed(0.8604)은 게이트 지표 아님이라 비교에서 제외.

| entity | evtfix (새 gold·재학습) | prodrecover (구) | Δpp |
|---|---|---|---|
| **EVT** | **0.8249** (P 0.7972 / R 0.8545, sup 598) | 0.8020 (sup 559) | **+2.29** |
| PROD | 0.7992 (P 0.7974 / R 0.8010) | 0.7913 | +0.79 |
| **overall** | **0.9474** | 0.9459 | +0.15 |
| ORG / PER / LOC | 0.8730 / 0.9197 / 0.9467 | 0.8734 / 0.9162 / 0.9468 | −0.05 / +0.35 / −0.02 |

- **EVT +2.29pp = 0.38 std**(새 per-fold std 0.0599). **흔들림 안 — 검증가능한
  lift 아님.** 헤드룸≈1 std 예측을 재학습으로 실측 확인(이론 → 경험). 게다가
  이 +2.29pp 는 gold 변경(support 559→598)과 재학습이 섞여 순수 모델 개선도
  아니다.
- evtfix 는 더 깨끗한 gold 로 학습됐고 어디서도 회귀 없음 → **새 canonical 모델**.

## 비-회귀 (AC4)

- **PROD 0.7913→0.7992**(+0.79pp, std 0.0198 내) ≥ 0.763 floor ✓.
- **overall 0.9459→0.9474**(+0.15pp), ORG/PER/LOC 평탄(±0.35pp 내) — 비-회귀.
- **EVT↔PROD net span**: gold-fix silver-갭 39건 PROD 출처 0 → gold측 PROD
  무손실. 진단상 경쟁 model=EVT/gold=PROD 8 + 역 6 으로 미세·균형.

## 정직한 한계

- **lift 는 노이즈 안**: +2.29pp = 0.38 std. support 598 의 노이즈 바닥
  (std 0.060)이 천장 미만 lift 를 통계적으로 비가시화 — "EVT 가 더 좋아졌다"고
  주장 불가. 본 이슈 산출물은 **(a) 정직한 천장 재추정 0.8574, (b) gold 품질
  +39 event, (c) 회귀-free 재학습 모델**이지 검증된 점수 향상이 아니다.
- **혼재 델타**: 0.8020→0.8249 는 gold 변경 + 재학습 합산이라 분해 불가. 단
  순환(현 모델 예측을 그 gold 에 넣고 재채점)은 회피 — evtfix 는 *새 모델이 새
  gold 를 평가*한 비-순환 측정.
- **gold=LLM 추정**: gemma 단일 §3 판정(사람 전수 아님, 제약대로).
- **모델 측 헤드룸**: 모델-FP 100·모델-FN 79(경계오류 45)는 가용 레버 없음.

## 검증 (AC5·AC6)

- [x] leak-free: `cross_fold_orig_dups=0`(evtfix·prodrecover 모두), n=37,706
- [x] gold 무결성: 37,706행·id/orig/text 불변, **EVT +39·비-EVT 9종 0 변동**
- [x] `ruff check` clean / `pytest tests/ner -q` green(363, 신규 8건 포함)
- [x] 측정 정합(머지 시점): #133 승격런 본문 수치 ↔ 당시
  `evtfix/pooled_metrics.json`(EVT 0.8249·PROD 0.7992·overall 0.9474) ↔
  `prod_evt_analysis_prodrecover10.json`(천장 0.8574). **이후 de5acf2 출하
  재실행이 JSON 을 덮어씀 → 현 디스크값 EVT 0.8159·PROD 0.8070·overall
  0.9478(GPU 비결정성 노이즈 내). #133 승격런 값은 기록 보존, 현 canonical
  수치는 `docs/reports/vietnamese-bert-classifier-spec.md`.**
- [x] refuter 게이트 PASS (Opus 격리, 머지 시점 diff 기준 — std 0.0599
  재계산·+2.29pp=0.38std 노이즈 내 정직 프레이밍·3-문서 정합·과대주장 0·
  테스트 8건 무결 독립 확인)

산출물(gitignore): `data/wikiann_vi/{pii_all,pii_all_pre_evtfix}.jsonl`,
`results/classifier/vi/evtfix/phobert-base-v2/{fold0..9,pooled_metrics}`,
`results/classifier/vi/audit/{prod_evt_*_prodrecover10,added_evt}.*`.
