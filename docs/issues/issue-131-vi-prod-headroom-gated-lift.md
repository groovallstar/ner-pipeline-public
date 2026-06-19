# issue-131 — VI PROD 누출-free 천장 재감사 + 헤드룸 게이트 처방

- GitHub: #131 (area:classifier, area:augmenters)
- 브랜치: `feat/issue-131-vi-prod-headroom-gated-lift` (develop 기반)
- 선행: #108(PROD/EVT §3 감사·re-silver), #112(EVT 회귀 해결), #124(원문
  cross-fold 누출 정량·group-kfold 영구 수정), #116(자연 주입 단일 gold)

## 목적

현 leak-free baseline(phobert 10-fold grouped) **PROD F1 0.7014**(P 0.632 /
R 0.788, 10종 중 최저)를 개선. 단 #108 천장 추정(0.81)은 **leaked v1 fold0**
값이라 두 가지로 무효 — (a) cross-fold 누출 인플레(+2.9~3.1pp, #124),
(b) gold가 v2→v3→자연으로 바뀜. 따라서 현재 배포 gold 기준으로 천장을 먼저
재추정하고, 회복가능 헤드룸이 임계 이상일 때만 처방한다(go/no-go).

## 진단 (AC1·AC2) — 누출-free 전수 10-fold

도구: `scripts/audit_vi_prod_evt.py`(다중-fold 확장) extract→adjudicate→analyze.
판정 = gemma-4-31B-it-AWQ-8bit @ vLLM, §3 루브릭, temp 0. 대상: phobert grouped
10-fold `test_predictions.json` 전수(37,706행), 현재 배포 gold. 추출 1,549
케이스(PROD FN 496·FP 836 + EVT 170 + 대조군 60), unparsed 0. 무결성: analyze
`orig_f1` PROD(0.7014/0.632/0.788) = `pooled_metrics.json` 정확 일치.

| | P | R | F1 |
|---|---:|---:|---:|
| base (현 배포 gold) | 0.632 | 0.788 | 0.7014 |
| 보정 (gemma §3 천장) | 0.841 | 0.839 | **0.84** |

- **헤드룸 ≈ 14pp**. schema_gap **0**(§3로 전부 결정 — 클래스 정의 병목 아님,
  #108 재확인).
- **precision이 압도적 레버**: 모델 FP 1,061 중 **603이 실제 정답**(silver
  창작물 미라벨) → FP 1,061→458, P 0.632→0.841. recall은 거의 천장(허위 gold
  25건, FN 491→466).
- 불일치 분류: silver_error 628 / model_error 479 / both_error 220 / agree 46.
- **go/no-go**: 헤드룸 14pp ≫ 임계 3pp → **GO**. (계획 단계 Contrarian 가설
  "PROD는 천장 ~2pp 이내"는 반증됨.)

## 결정 (사용자 비준)

- **성공 기준**: 천장 재추정 후 상대목표 — leak-free PROD F1 **≥0.77**(헤드룸
  ≥50%), EVT·overall 비-회귀(per-fold std 내).
- **천장 재추정** = gemma 자동 판정만(사람 검토 생략; 한계: gold=LLM 추정,
  사람-LLM IAA 없음).
- **허용 레버** = gold re-silver(precision) + 모델측. 제외 = 3번째 LLM·외부
  코퍼스.
- **메커니즘 전환(진단 후 확정)**: 추가 분석 결과 silver-갭 606의 **97%가
  창작물 제목(노래·영화·앨범·TV·만화)** — EVT와 달리 §3 표면 패턴이 없어
  `recall_strict_evt`식 결정론 regex 구제 **이식 불가**(모델코드 패턴은 2%만
  매칭). 따라서 레버 = **PROD high→high+gemma_only 완화(단일모델 신뢰) +
  사람 spot-audit 게이트**.

## 구현

1. **`recall_strict_prod` 병합 정책**(`augmenters/wikiann_vi/merge_confidence.py`):
   `recall_strict_evt`와 동일(EVT legit 구제 유지 = 비-회귀)하되 PROD 를
   high+medium_recall(gemma_only)로 완화. conflict·qwen_only PROD 는 drop.
   단위 테스트 추가(EVT 동작 동일성 포함).
2. **자연-코퍼스 re-silver**(`scripts/resilver_vi_natural.py`, 신규): suffix
   전용 `resilver_vi_isolated.py`(경계 로직)는 자연 주입(#116)에 못 써, 배포
   `text`를 직접 Gemma+Qwen 재라벨 → `recall_strict_prod` PROD를 **기존
   엔티티에 겹치지 않게 additive 추가**(기존 PII·PER·LOC·ORG·EVT 불변).
   Gemma 전수(37,706), Qwen은 후보 텍스트(1,412)만 타겟 재라벨(MoE-4bit +
   prefix-cache 0%로 전수 시 ~5h라 conflict 필터 역할만 타겟; 25배 단축).
3. **Wikidata 종-FP 필터**: spot-audit 1차가 추가 후보의 ~22%가 생물 종(種)
   FP임을 적발(Gemma·Qwen 공유 편향이라 consensus 무력). 기존 `wikidata_anchor`
   P31로 1,025 고유 표면형 검증 → taxon(Q16521 등) 145 + 다른 타입 35 drop,
   나머지 keep(precision 우선·미해석 보존). 재감사서 종 소멸 확인.

후보 흐름: Gemma PROD 3,793 → gold에 없는 후보 1,447 → Qwen conflict 필터
1,410 → Wikidata 필터 **1,194 추가**. gold PROD **2,314 → 3,508**. 무결성:
orig 보존 37,706/37,706, 비-PROD 엔티티 바이트 불변, PROD 오프셋 0 mismatch.

## 측정 결과 (phobert 10-fold grouped, leak-free)

`cross_fold_orig_dups=0`, n_sentences=37,706.

| entity | baseline | prod-recover | Δpp |
|---|---:|---:|---:|
| **PROD** | 0.7014 | **0.7913** | **+8.99** |
| EVT | 0.8081 | 0.8020 | −0.61 |
| PER | 0.9177 | 0.9162 | −0.15 |
| LOC | 0.9472 | 0.9468 | −0.04 |
| ORG | 0.8738 | 0.8734 | −0.03 |
| DAT | 0.9983 | 0.9973 | −0.10 |
| EMAIL | 0.9998 | 0.9996 | −0.02 |
| PHONE | 0.9983 | 0.9980 | −0.03 |
| ID_NUM | 0.9968 | 0.9970 | +0.02 |
| CREDIT_CARD | 0.9989 | 0.9983 | −0.06 |
| **overall** | 0.9457 | **0.9459** | +0.03 |

- ✅ **목표 초과**: PROD 0.7014→0.7913 (헤드룸 14pp의 ~65% 회복). P
  0.632→0.775(+14.3pp, 창작물 silver-갭 해소), R 0.788→0.808. PROD support
  2,314→3,508.
- ✅ **유의성**(per-fold): PROD 0.7025(std 0.038)→0.7905(std 0.028) — 이득
  >2 std, 분산도 안정. EVT 0.8110(0.071)→0.8017(0.054) — 변화 <1 std =
  **노이즈 대역(비-회귀)**. EVT gold는 바이트 불변이라 순수 모델 효과
  (PROD↔EVT 경쟁, #112 동일 기제).
- ✅ overall·PER·LOC·ORG·PII 전부 평탄(±0.15pp).

배포 gold 승격: `pii_all.jsonl` ← prodrecover(계보 v1→v2→v3→natural→
**prodrecover** 신 phase). baseline 리포트 표는 영구 인용으로 불변. 구 gold
백업 `pii_all_pre_prodrecover.jsonl`.

## 정직한 한계

- **gold=LLM+Wikidata 추정**: 추가 PROD의 정답성은 Gemma+Qwen 재라벨 +
  Wikidata P31에 의존(사람 전수 검증 아님). spot-audit 표본으로만 점검.
- **약한 순환**: Gemma가 재라벨 labeler이자 천장 판정자 → Wikidata(독립
  근거)가 최종 게이트로 순환 일부 차단. recall gain(+2pp)이 일반화 증거.
- **종-FP 잔여**: Wikidata 미해석(no-qid) keep 378에 종이 일부 잔존 가능
  (재감사 추정 ~5-8% 노이즈, silver 수준 이하).
- **헤드룸 미회복분(~35%)**: 보정 천장 0.84 대비 0.7913 — 잔여는 모델
  recall 약점(하드 플로어 fp458≈fn466 대칭) + 미해석 창작물 보수 drop.

## 검증

- [x] `ruff check` clean (변경 .py 4개)
- [x] `pytest tests/ner/augmenters tests/ner/classifier -q` green (212) —
  `recall_strict_prod` 단위테스트(PROD gemma_only 구제·EVT 비-회귀 동일성) 포함
- [x] gold 무결성: orig 보존·비-PROD 바이트 불변·PROD 오프셋 0 mismatch
- [x] leak-free: `cross_fold_orig_dups=0`
- [x] refuter 게이트 PASS (Opus 격리 컨텍스트 — 측정 무결성 3축: 리포트
  66값 ↔ pooled JSON 정합, 순환성(gold 확장 FP→TP)을 독립 출처(Gemma+Qwen
  +Wikidata)·recall +2pp 일반화·EVT/overall 비-회귀로 반증 통과, 코드·테스트
  무결)

산출물(gitignore): `results/classifier/vi/audit/{prod_evt_*_grouped10,
natural_relabel_{gemma,qwen},added_prod_filtered,wikidata_keep_surfaces}.*`,
`results/classifier/vi/prodrecover/phobert-base-v2/{fold0..9,pooled_metrics}`.
