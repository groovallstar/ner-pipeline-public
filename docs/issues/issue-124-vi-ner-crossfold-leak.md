# issue-124 — VI 분류기 gold NER 원문 cross-fold 누출 정량 + 영구 수정

- GitHub: #124 (area:augmenters, area:classifier)
- 브랜치: `feat/issue-124-vi-ner-crossfold-leak` (develop 기반)
- 선행: #116(VI PII 자연 주입)에서 분리된 스핀오프. 영향받는 과거 측정 =
  #103/#108/#112(VI 분류기 canonical 5-fold).

## 배경 / 근본 원인

#103 은 "원문 키 중복 제거 40,000→38,371, cross-split 중복 0 재검증"으로
WikiANN-vi 내재 train/test 누출을 제거했다고 보고했다. 그러나 이는 **거짓
안심**이었다:

- 원본 WikiANN-vi(`all.jsonl`, 40,000행)의 **유니크 원문은 29,343개뿐**
  (#124 측정). #103 dedup 은 1,629행만 제거 → 38,371행에 원문 중복이 ~9,000행
  그대로 남았다.
- 즉 실제로 일어난 건 *주입 후* 전체텍스트 기준 중복 제거였고(주입 PII 가
  행마다 달라 원문 중복을 가림), 'cross-split 중복 0' 재검증도 행마다 유니크한
  *주입 텍스트* 기준이라 **원문 단위 누출이 통과됐다**.
- 근본은 `split_kfold_stratified` 가 **행 단위** 라운드로빈이라 같은 원문에서
  파생된 여러 행이 fold 로 흩어진 점. 어떤 reshape(re-silver·PII 주입)에도
  잠복 누출이 새 형태로 재발했다.

**실측(자연 코퍼스 `pii_all.jsonl` 37,706행, 실제 split 재현)**: 행 단위
분할에서 test 행의 **30.2%(11,392)**, NER 엔티티의 **27.7%(15,229/54,895)**
가 원문이 train/valid 에도 존재 → 모델이 일반화가 아니라 암기로 맞춤. PII 는
행별 랜덤값이라 누출 0.

## 결정 — 측정·수정 방식 = group-kfold (dedup 아님)

원안 goal 1("원문 dedup 29,343 재학습 vs 누출 baseline 37,706")은 변수 2개
(누출 제거 + 학습데이터 22% 손실)를 동시에 바꿔 인플레 폭을 누출 단독에
귀속할 수 없다(교란). 대신:

- **코퍼스를 37,706행으로 고정**, 분할만 leaked(행 단위) vs grouped(원문
  group-kfold)로 바꿔 **누출 효과만 분리**.
- group-kfold 는 **데이터 손실 0** + split 단계에 박히는 영구 방지책을 겸한다.
- 자연 코퍼스는 행마다 주입 전 원문을 담은 `orig` 필드를 이미 보유 → 별도
  재구성 없이 그룹 키로 사용.

## 구현

- `data_utils.split_kfold_stratified(group_key=)`: 같은 `row[group_key]` 를
  공유하는 행을 한 unit 으로 묶어 통째로 한 fold 에 배정. `group_key=None` 은
  기존 행 단위 분할과 **bit-for-bit 동일**(백워드 호환). 층화는 unit 내
  strat_labels 합집합 기준.
- 분류기 CLI `--group-key`(예: `orig`), `test_predictions.json` 에 `orig` 기록.
- `kfold_pool` 가드를 **원문(orig) 단위 cross-fold 검사**로 격상: 같은 fold 안
  원문 중복은 group 정상이라 허용, fold 가 갈리면 누출. `require_no_leak` /
  CLI `--allow-cross-fold-leak` 로 누출 baseline 측정 시에만 카운트 모드
  (`cross_fold_orig_dups`).
- 테스트: group 분할 cross-fold 원문 누출 0·그룹 불가분·전수 1회 test·
  group_key=None 동일성(test_data_utils 5개), pool 가드 cross-fold raise/
  카운트(test_kfold_pool 3개).

## 측정 결과 (자연 코퍼스 37,706행 고정, 분할만 변경)

leaked = 행 단위 stratified 5-fold, grouped = 원문 group-kfold. 둘 다 전수
1회 test (pooled micro char-offset span F1). `n_sentences=37,706` 양쪽 동일.

| 모델 | 지표 | leaked | grouped | 인플레 |
|---|---|---:|---:|---:|
| phobert-base-v2 | overall | 0.9506 | 0.9437 | **+0.69** |
| phobert-base-v2 | NER 5종 micro | 0.9198 | 0.9086 | **+1.12** |
| phobert-base-v2 | PII 5종 micro | 0.9985 | 0.9983 | +0.02 |
| xlm-roberta-base | overall | 0.9517 | 0.9430 | **+0.87** |
| xlm-roberta-base | NER 5종 micro | 0.9226 | 0.9083 | **+1.43** |
| xlm-roberta-base | PII 5종 micro | 0.9972 | 0.9974 | −0.02 |

### per-entity strict F1 (leaked → grouped, 인플레 pp)

| ent | phobert leaked→grouped (Δpp) | xlm-r leaked→grouped (Δpp) | sup |
|---|---|---|---:|
| EVT | 0.8372 → 0.7973 (**+3.98**) | 0.8242 → 0.7713 (**+5.29**) | 559 |
| ORG | 0.8882 → 0.8647 (**+2.35**) | 0.8901 → 0.8557 (**+3.44**) | 8033 |
| PROD | 0.7362 → 0.7076 (**+2.86**) | 0.7116 → 0.6811 (**+3.05**) | 2314 |
| PER | 0.9235 → 0.9145 (+0.90) | 0.9382 → 0.9254 (+1.28) | 21848 |
| LOC | 0.9506 → 0.9443 (+0.63) | 0.9457 → 0.9394 (+0.63) | 22141 |
| PII 5종 | 0.998~1.000 (±0.05) | 0.997~1.000 (±0.06) | ~35.5k |

### 해석

- ✅ **NER 5종 절대값 인플레 확정**: micro +1.1~1.4pp, 저support·고난도
  엔티티에 집중 — **EVT +4.0/+5.3, ORG +2.4/+3.4, PROD +2.9/+3.1pp**. 암기가
  희소·어려운 엔티티에서 가장 크게 작동.
- ✅ **PII 5종 불변(±0.05pp)** — 누출 0 주장 검증(행별 랜덤값이라 train·test
  값 비중복).
- ✅ **grouped cross-fold 원문중복 0**(`kfold_pool` 가드, leaked 는 6,973).
- ✅ **상대 비교·델타 유효**: phobert≈xlm-r, #108/#112 re-silver 이득(PROD·EVT
  델타)은 같은 누출이 양쪽에 동일 적재돼 보존.

## 영구 방지 + 과거 정정

- 영구 방지: `--group-key orig` 를 VI 분류기 평가 프로토콜로. split 단계 +
  `kfold_pool` 원문 가드 + pytest(cross-fold 원문중복 0 강제)로 어떤 reshape
  에도 재발 불가.
- 과거 정정: `docs/reports/vietnamese-bert-classifier-{benchmark,history}.md`
  에 §원문 누출 정정 (#124) 추가 — #103/#108/#112 NER 5종 절대값을 인플레로
  표기. suffix v1/v3 코퍼스 파일은 폐기돼 정확 재측정 불가 → 자연 코퍼스 측정의
  **대표 추정**(suffix v3 누출률 27.9% 로 유사 규모). PII·델타·상대비교는 유효.

산출물: `results/classifier/vi/leak_audit/{leaked,grouped}/<model>/fold{0..4}`
+ `pooled_metrics.json` + `leak_inflation_summary.json`(results 는 gitignore —
리포트 표가 영구 인용 출처).

## 검증

- [x] `pytest tests/ -q` green (**352**) — group 분할/pool 가드 신규 8개 포함,
  기존 테스트 삭제·skip·약화 없음
- [x] `ruff check` clean (변경 .py 5개)
- [x] 실 코퍼스 grouped 분할 cross-fold 원문 누출 0.0%(행 단위 30.2%) 재확인,
  group_key=None bit-for-bit 백워드 호환
- [x] refuter 게이트 PASS (`2f92ed3e6cb5` — 코드 정합성·측정 무결성·테스트
  무결성 3축; 리포트 18값 ↔ pooled_metrics.json 일치, NER5/PII5 micro 복원
  검산, n_sentences·cross_fold_orig_dups 대조)
