# KO EVT F1 천장 조사 & 성능 향상 핸드오프

> 자유형식 실험 리포트(이슈 무관). #163(EVT 타입 정합)의 **후속 — EVT F1
> 향상** 조사. 다음 세션이 이 문서만 읽고 이어갈 수 있게 작성.

## TL;DR

- **현 gold = 1,442**(#163 재라벨 완료·최대-span). EVT F1 = **0.664**(strict,
  10-fold pooled, kf-deberta-base, `cross_fold_orig_dups: 0`). relaxed 0.736.
- 아래 **천장 조사(0.640·오류 구조·레버 반증)는 완전성 회수 이전 1,346 gold**
  에서 측정한 기록이다. 완전성 회수(무라벨 EVT 137건 add + generic drop +
  span 최대화)로 gold 가 1,442 로 늘며 0.640→0.664(+0.024)가 됐으나, gold
  정의가 달라진 값이라 "레버로 올린 F1"이 아니다.
- **값싼 레버 4개 전부 반증**(아래 표, 1,346 기준). 0.80은 튜닝이 아니라
  불확실한 연구 투자. 남은 경로 = **② gazetteer *feature* 통합(재학습)**.
  낙관적 천장 ~0.70–0.75, 0.80은 aspirational.

## 현 상태 (fact)

- **#163 재라벨 완료**: `data/klue/pii_all.jsonl` EVT **1,442**, 불변 타입
  byte-identical. 3축(1회성 포함·복합 행사명사만 bare·최대-span)으로 재판정 —
  generic drop 10 · 타입 재판정(drop 8·add 114) · span 최대화 32. 타입
  일관성(무라벨 0인 표면형) 92%→99%. **미커밋** — 커밋 전 결과-시점 refuter.
  상세: `docs/issues/issue-163-ko-evt-gold-schema-consistency.md`.
  - 산출물·백업·provenance: `results/classifier/ko/evt_maxspan/`.
  - 착수점(§3.5 폐기·되돌림) gold: `results/classifier/ko/evt_census/pii_all.evt.jsonl`(1,346).
- **EVT 라벨 기준은 커밋된 canonical §3·§3.3 그대로**(JA·VI와 동일). 세 축:
  1회성 named 사건 포함(`세월호 참사`·`하이옌`) / 제도화된 복합 행사명사만
  bare EVT(`총선`·`여론조사`, 단 `조사`·`선거` 단독은 아님, §3.3 보조원칙) /
  **최대-span**(`2014 브라질 월드컵` 통째, 수식어 트림 금지).
- **"KO EVT 협소 재정의"(§3.5) 시도는 폐기됨.** 재발 행사만 EVT로 좁히고
  수식어를 트림해 gold 를 984 로 줄였으나, 고유명(`세월호 참사`)을 버리고
  일반명사(`조사`·`회담`)를 남겨 NER 정의가 뒤집혔다. 트림 경계는 두 LLM 이
  16%(131/840) 불일치할 만큼 비결정적이었고, 결과 gold 가 오히려 더 비일관했다
  (무라벨 0인 표면형 80% vs 최대-span 92%). 무엇보다 hapax 52%→32%,
  span 평균 7.15자→4.63자로 **과제를 쉽게 만들어 F1 을 올리는** 변경이었다.
  아래 0.640 은 최대-span 1,346 gold 기준이므로 그대로 유효하다.

## 오류 구조 (측정 — `results/classifier/ko/evt_census/probe_scripts/evt_error_analysis.py`)

10-fold pooled EVT: P **0.60** / R **0.69**. TP 923 · FP 617 · FN 423.

| precision 손실 (FP 617) | | recall 손실 (FN 423) | |
|---|---|---|---|
| gold공백(누락+spurious) | 47% | 경계오류 | 49% |
| 경계오류(EVT↔EVT) | 39% | 완전누락(anchorless) | 39% |
| 타입혼동(DAT/LOC/ORG) | 14% | 타입혼동 | 12% |

**경계 오류의 정체**(`evt_boundary_probe.py`): 240쌍 중 **77%가 모델 과축소**
(gold보다 짧게) — `한국시리즈`→`한국`, `도쿄 올림픽`→`올림픽`, `후쿠시마 원전
사고`→`사고`. gold는 대부분 **정확**(descriptive 꼬리 2%뿐). 즉 모델이 hapax
긴 span을 못 배워 익숙한 조각으로 후퇴.

## 반증된 cheap 레버 (재현 명령 포함)

| 레버 | 결과 | 방법 |
|---|---|---|
| **커버리지 축소**(#125 T3, anchorless 19% 강등) | 0.640→**0.650** (+0.01, −19% 커버리지) | `build_restricted.py` → restricted gold 10-fold |
| **gold 경계 재주석**(#2) | **refuted** — gold 이미 정확, 재주석 무의미 | `evt_boundary_probe.py`(77% 모델 과축소) |
| **경계 loss 가중**(`--boundary-b/i-weight`) | i3=0.623↓ / b2i2=0.658 flat | fold0 sweep, `--boundary-i-weight 3.0` 등 |
| **gazetteer 후처리**(①) | 0.640→0.60(fix)/0.57(fix+recall), 클린해도 net-음 | `gen_gazetteer.py`+`merge_measure.py` |

각 축이 knob/후처리로 안 풀리는 이유가 아래 근본 원인.

## 근본 원인 (얽힌 3구조)

1. **세계지식 부재** — `한국시리즈`≠`한국`을 모델·정적사전 둘 다 모름.
2. **문맥의존 경계** — 같은 이벤트가 문장별 bare(`월드컵`)/수식(`브라질 월드컵`)
   → 정적 규칙·사전 불일치(gazetteer 후처리가 맞던 것도 깬 이유).
3. **hapax** — 표면형 83% 1회, 일반화 단서 없음(19%는 anchorless=recall≈0).

## 다음 경로 & 가이드 (EVT 성능 향상 계속)

**② gazetteer feature 통합** (유일 정공법, 후처리와 다름):
- 후처리(①)는 문맥 없이 하드 매칭이라 과발화. **feature 통합은 모델이
  문맥 조건부 사용을 *학습*** → "여기선 사전 믿고 저기선 무시".
- 구현: (a) 사전 개선 — Wikidata SPARQL(KO 이벤트 라벨) + `gen_gazetteer.py`
  확장 + 도메인목록, §3.5 경계로 정규화. (b) `data_utils.py` 토크나이즈 시
  토큰별 사전 B/I/O 매치 feature 계산. (c) `train_eval.py`에서 그 feature
  임베딩을 BERT 토큰표현에 concat 후 분류 head. (d) 10-fold 재학습 vs 0.640.
- **누출 주의**: 사전은 외부 세계지식만(gold 표면형 금지). fold-blind.
- 기대: 과발화가 학습으로 완화되나 문맥경계 문제 잔존 → **~0.70–0.75 추정,
  0.80 불확실**. 다주 프로젝트.

**③ 완전성 회수 — 완료**(#163 재라벨로 gold 1,346→1,442):
- 전수 조사(경계 가드): 무라벨 0인 표면형 92%→**99%**, 무라벨 278→**25건**
  (잔여는 전부 동형이의·다른 개체 이름 일부). dual-LLM 문맥 판정 + 사람 검수로
  타입 재판정(drop 8·add 114) + 단일 generic drop 10 + span 최대화 32.
- **주의(교훈)**: 라벨률이 낮다고 자동 drop 금지. `올림픽`은 라벨률 0.34지만
  일반명사가 아니라 gold 가 놓친 것이다. 지표는 "일반명사(제거)"와 "과소라벨
  (추가)"을 구분 못 한다 — 문맥 판정만이 구분한다. 표면형 하드코딩 열거로
  처리하면 실패한다(폐기된 `regen_narrow.py` 전례).
- 결과: EVT F1 0.640(1,346)→**0.664**(1,442). 단 이는 완전성·정합 개선의
  부작용이지 F1 레버가 아니며, gold 정의가 달라 두 값의 직접 비교는 무의미하다.
- 상세: `docs/issues/issue-163-ko-evt-gold-schema-consistency.md`,
  provenance `results/classifier/ko/evt_maxspan/`.

**안 되는 것(재시도 금지)**: gold 경계 재주석(#2), 순 loss 가중, 순 후처리
gazetteer(①), 커버리지 축소, **gold 협소화로 F1 올리기**(§3.5). 위에서 다 반증됨.

## 산출물 위치 (전부 `results/classifier/ko/evt_census/`, gitignore-but-on-disk)

- `pii_all.evt.jsonl`(#163 gold 1346) · `pii_all.pre163.jsonl`(백업 1388)
- `{baseline,new,restricted}_pooled.json`(10-fold 결과)
- `judged.jsonl`·`audit_{key,human}.jsonl`·`trim_confirm.jsonl`·`evt_metrics.json`
- `kfold_new_preds/fold{0..9}.json`(error analysis 재현용)
- `gazetteer{,_clean}.txt`(861/827 entries) · `probe_scripts/*.py`(6 도구)

**재학습 명령**(참고): `CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python
-m ner.classifier --lang ko --data <gold> --kfold 10 --fold-index <i>
--no-stratify --seed 42 --precision fp16 --output-dir <dir>` × 10 →
`python -m ner.classifier.kfold_pool --fold-dirs <dirs> --allow-cross-fold-leak`.
GPU: 0 여유(1·2는 vLLM gemma/qwen). fold당 ~13분.

## 미결정 (open)

1. **#163 잔여 작업** — 단일 generic 위반 11건 정리 + 완전성 회수 137건.
   수락 기준 확정 → 정의-시점 refuter → 구현 → 결과-시점 refuter → 커밋.
   `docs/issues/issue-163-*.md` 미작성.
2. **② feature 통합 착수 여부** — 불확실 payoff(~0.70–0.75) 다주 투자. spine
   변경이면 새 이슈+수락기준+정의-시점 refuter 필요. 사전은 외부 지식만
   (gold 표면형 금지·fold-blind).
3. 0.80이 사업상 필수인가 — 아니면 0.64가 정직한 종료점(#163 정합성이 실 성과).
   **gold 를 좁혀 0.80을 만드는 길은 폐기됐다**(§3.5, 위 "현 상태" 참조).
