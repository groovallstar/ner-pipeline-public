# issue-163: 한국어 EVT gold 라벨 기준 정립·스키마 일관성 재검증

## 목적

한국어 EVT gold 라벨을 canonical EVT 스키마(§3·§3.1·§3.3)에 대해 재판정해
**라벨 기준을 정립하고 일관성을 확립**한다. 성능 향상이 목표가 아니다 —
스키마 불일치 span 은 교정하며, 그 결과 EVT F1 이 변해도 수용한다. 게이트는
F1 이 아니라 **라벨 일관성**이다.

## 범위

- KO EVT 라벨 기준 3축 확정(1회성 사건 포함 / 복합 행사명사만 bare / 최대-span).
- 현 gold 의 축 위반·비일관 교정: 단일 generic 명사 제거 · 타입 재판정 ·
  완전성 회수 · span 최대화.
- 불변 타입(PER/LOC/ORG/PROD/DAT/PII) 라벨 무영향.

## 성공 기준

- EVT 스키마 정합 규칙 명문화(canonical 무엇이 EVT/비-EVT/최대-span 인가).
- dual-LLM 합의 재판정 provenance JSONL + 재판정 분포.
- 불변 타입 byte-identical.
- 라벨 일관성 개선(무라벨 출현 감소) 측정.
- EVT F1 은 참고치(게이트 아님).

## 현 상태 (fact)

- **최종 gold**: `data/klue/pii_all.jsonl`, EVT **1,442건 / 표면형 842종**,
  25,989 문장. 불변 타입 9종 byte-identical(재라벨 착수점 1,346 대비).
- **타입 일관성**: 무라벨 0인 표면형 92% → **99%**, 무라벨 총 278 → **25건**
  (잔여 25건은 전부 동형이의·다른 개체 이름 일부 — 의도적 제외).
- 재판정 산출물·provenance JSONL·중간 gold 백업·10-fold 체크포인트는 검증
  (결과-시점 refuter PASS) 후 제거됐다(`results/` gitignore, 일회성). 최종
  gold 만 `data/klue/pii_all.jsonl` 에 남는다. 아래 표·검증 섹션의 파일명은
  작업 시점 기록이며, 재현이 필요하면 gold + 재판정 코드로 재생성한다.

## KO EVT 라벨 기준 (3축 확정)

KO EVT 는 **커밋된 canonical §3·§3.3 그대로**이며 JA·VI 와 동일하다. §3.5
"협소 재정의" 시도(아래 결정 로그)는 폐기됐다.

- **축 1 — 1회성 named 사건 포함.** 역사상 한 번 일어난 개별 사건도 EVT.
  전쟁·참사·재해·조약·항쟁·named 위기·named 사건(`임진왜란`·`세월호 참사`·
  `외환위기`·`3·1운동`·`부림사건`)과 명명 자연재해(`하이옌`).
- **축 2 — 제도화된 복합 행사명사만 bare EVT.** `총선`·`대선`·`수능`·
  `여론조사`·`국정조사`·`정상회담` 은 고유명 수식 없어도 EVT. 단일 generic
  사건명사(`조사`·`선거`·`투표`·`대회`·`경기`·`사고` 단독)는 named·수식이
  붙을 때만 EVT (canonical §3.3 "주기적 복합 행사명사 예외").
- **축 3 — 최대-span.** span 은 그 행사를 지시하는 고유명 전체
  (`2014 브라질 월드컵` 통째, `MBC 연예대상`, `F1 그랑프리`). 개별 회차
  수식어를 벗겨 `월드컵`으로 트림하지 않는다. 선행 지시어(`이번`·`지난`)와
  조사만 제외. **겹침 정책**: 이미 라벨된 인접 엔티티(DAT/LOC/PER/ORG/PII)는
  삼키지 않는다 — 평면 10종 스키마라 겹침을 표현할 수 없으므로 EVT span 은
  그 경계에서 멈춘다(`보스턴 테러`의 `보스턴`=LOC 보존).

### 재판정 세부 규칙 (R1~R3)

타입·span 재판정 중 dual-LLM HOLD 를 사람이 판정하며 확립한 규칙:

- **R1** 수식 위치라도 그 행사를 지시하면 EVT(`올림픽 신기록`·`올림픽 금메달`),
  다른 개체의 이름 일부이면 제외(`올림픽 공원`=LOC 일부·`올림픽 대표팀`=팀·
  `월드컵 북로`=도로명·`F1 조직위`=ORG).
- **R2** 복합 행사명사는 제도화된 절차·회의(`청문회`·`국정조사`·`정기국회`) ·
  임의 개최 회견·회의(`기자회견`·`최고위원회의`) · 의식·시상식(`개막식`·
  `취임식`·`갈라쇼`) 모두 bare EVT.
- **R3** 순수 경기 단계명(`결승전`·`준결승`·`포스트시즌`·`플레이오프`) 단독은
  비-entity.

## 결정 로그 (append-only)

- **§3.5 "KO EVT 협소 재정의" 폐기.** 재발·정기 행사만 EVT 로 좁히고 수식어를
  트림해 gold 를 1,346→984 로 줄인 미커밋 시도를 전면 폐기(gold·docs·프롬프트
  diff + `results/classifier/ko/evt_r1/` 삭제). 폐기 근거:
  - **NER 정의 역전**: 고유명(`세월호 참사`)을 버리고 일반명사(`조사`·`회담`)를
    남겼다. 상위 빈도 47%가 일반명사가 되고, generic 가드 예외 목록으로 구멍을
    손으로 막아야 했다(`조사`·`선택`·`유로` 잔재).
  - **비결정성**: 트림 경계에서 두 LLM 이 16%(131/840) 불일치. 규칙이 결정을
    내려주지 못한다는 신호.
  - **비일관 악화**: 트림 gold(984)가 최대-span gold(1,346)보다 비일관 —
    무라벨 0인 표면형 80% vs 92%.
  - **은닉 F1 레버**: 제거된 761건의 64%가 hapax, 평균 9.09자. hapax 52%→32%,
    span 평균 7.15→4.63자로 **과제를 쉽게 만들어 F1 을 올리는** 변경인데
    "F1 레버 아님"으로 기록돼 있었다. 구 gold 예측을 새 gold 로 재채점하면
    F1 0.640→0.354(트림만 0.489) — 트림은 모델이 맞히던 923건 정확매치를 깬다.
- **정의-시점 refuter(설계 반증)**: 초기 완전성-회수 설계에 치명 4건 지적
  (rubric 미검증·AC 순환성·Track B recall 미측정·gemma anchoring 비대칭).
  이후 설계를 최대-span 기준 재판정으로 전환하며 반영.
- **rubric 캘리브레이션 게이트**: 재판정 프롬프트를 canonical 예시 20건에
  dry-run → gemma 20/20·qwen 19/20(유일 오답은 규칙 명시된 `선거`, 모델 한계).
  통과 후 본판정 착수.
- **KO default 백본을 koelectra 로 통일.** `koelectra-base-v3` 는 #122 벤치가
  선정한 baseline 이자 #125·#128·#140·#153 전 진단이 쓴 production 모델인데,
  CLI default(`__main__.py`)만 `kf-deberta-base` 로 어긋나 있었다(`--lang ko`
  도입 커밋부터 미정합·근거 미문서화). #163 이 그 default 로 EVT 를 재며
  불일치가 드러났다. 재측정상 두 백본은 overall 동률·EVT 백본불변이고
  koelectra 가 전 KO 계보와 정합하므로 default·문서를 koelectra 로 정렬한다
  (`--precision fp16` 기본; koelectra 는 fold ~0.3~3% 학습붕괴 → F1≈0 fold 만
  `--train-seed` 재실행). 벤치 리포트의 kf-deberta 비교행은 히스토리라 불변.

## 구현 결과

착수점 gold: EVT 1,346 (§3.5 폐기·되돌림 후. #163 1단계 dual-LLM 재판정
1,388→1,346 반영분. 산출물은 검증 후 제거).

| 단계 | EVT | 내용 | provenance |
|---|---|---|---|
| 착수점 | 1,346 | 최대-span·1회성 포함 | (되돌림) |
| generic 제거 | 1,336 | 단일 generic 명사 drop 10 (`조사`1·`선거`7·`투표`2) | `generic_drop.jsonl` |
| 타입 재판정 | 1,442 | dual-LLM 444건 판정 → drop 8·add 114 | `judged66.jsonl`·`apply_provenance.jsonl` |
| span 최대화 | 1,442 | 경계 확장 32건(건수 불변) | `span_pass.jsonl`·`span_provenance.jsonl` |

- **타입 재판정(444건)**: 무라벨 출현이 있던 66종 × 라벨 292 + 무라벨 152.
  자동 확정 412건(dual-LLM 합의 336 + R2/R3 규칙 76), 사람 검수 32건.
  결과 keep 284·drop 8·add 114·skip 36. drop 8 = R3 경기 단계명 5
  (`준결승`·`포스트시즌`2·`플레이오프`·`결승전`) + 고유명 없는 사건명사 3
  (`총기사고`2·`토네이도`).
- **span 최대화(32건)**: 합의 확장 8 + HOLD 사람 승인 24. 겹침 정책으로
  DAT 삼킴 146건은 span_pass 단계에서 자동 기각. 예: `F1`→`F1 그랑프리`,
  `연예대상`→`MBC 연예대상`, `하이옌`→`제 30호 태풍 하이옌`.
- **모델 편향 관측**: 타입 판정에서 qwen 이 EVT 과호출(HOLD 88%가
  qwen=EVT/gemma≠EVT — gemma 는 원 라벨러라 자기 라벨 방어 anchoring),
  span 판정에서는 정반대로 gemma 가 확장적·qwen 이 보수적. 두 편향이 방향별로
  달라 사람 검수를 각 방향에 배치.

## 변경 요약

- `data/klue/pii_all.jsonl` — EVT 1,346→1,442 재라벨(불변 타입 byte-identical).
- `docs/manual/data/canonical-entity-schema.md` — §5.3 KO 결정표에 3축·R1~R3
  8행 추가.
- `docs/reports/ko-evt-f1-ceiling-investigation.md` — §3.5 폐기·현 상태 정합화.

## 검증

- 불변 타입 9종 byte-identical(독립 재검증, apply assert 외 별도).
- span↔원문 불일치 0 · 엔티티 겹침 0 · 행수·id·text 보존.
- 타입 일관성: 무라벨 0인 표면형 92%→99%, 무라벨 278→25건.
- **10종 전체·entity별 메트릭** (1,442 gold, 10-fold pooled, strict/exact-span,
  `--no-stratify` seed 42, kf-deberta-base, `cross_fold_orig_dups: 0` = 원문
  단위 cross-fold 누출 없음). 학습 산출물(pooled.json·fold 체크포인트)은
  검증 후 제거 — 아래 표 수치는 그 pooled.json 에서 전사·기계 대조한 값이다.

  | entity | P | R | F1 | support |
  |---|---|---|---|---|
  | **overall** | 0.9195 | 0.9248 | **0.9221** | 77,174 |
  | PER | 0.9268 | 0.9100 | 0.9183 | 18,131 |
  | LOC | 0.8418 | 0.8736 | 0.8574 | 7,499 |
  | ORG | 0.8033 | 0.8645 | 0.8328 | 2,797 |
  | PROD | 0.7507 | 0.7226 | 0.7363 | 3,554 |
  | EVT | 0.6324 | 0.6990 | 0.6640 | 1,442 |
  | DAT | 0.8412 | 0.8615 | 0.8512 | 9,934 |
  | EMAIL | 0.9999 | 0.9999 | 0.9999 | 8,322 |
  | PHONE | 0.9976 | 0.9989 | 0.9983 | 8,448 |
  | ID_NUM | 0.9979 | 0.9966 | 0.9972 | 8,467 |
  | CREDIT_CARD | 0.9970 | 0.9986 | 0.9978 | 8,580 |

  relaxed(partial): overall F1 0.9390 · EVT F1 0.7362 (P 0.7011 / R 0.7750).
  EVT 는 참고치이며 게이트 아님 — 구 gold(1,346·§3.5-984) 숫자와 **직접 비교
  금지**(gold 정의가 다름). 불변 타입 9종은 gold 무변경이므로 이 표의 변동은
  fold 분할·학습 seed noise 범위.
- **백본 각주.** 위 표는 당시 CLI default 였던 `kf-deberta-base` 로 측정됐다.
  프로젝트의 다른 KO 엔티티 진단(#122·#125·#128·#140·#153)은 전부 production
  baseline `koelectra-base-v3` 로 측정돼 이 표는 백본이 달라 그들과 직접
  비교되지 않았다. 이후 KO default 백본을 koelectra 로 통일(결정 로그 참조)
  하며 같은 1,442 gold 를 koelectra 로 재측정(동일 recipe·백본만 상이)한 대조:

  | entity | kf-deberta | koelectra | Δ(strict F1) |
  |---|---|---|---|
  | overall | 0.9221 | 0.9187 | −0.003 |
  | EVT | 0.6640 | 0.6654 | +0.001 |
  | ORG | 0.8328 | 0.8514 | +0.019 |
  | PROD | 0.7363 | 0.7043 | −0.032 |
  | LOC | 0.8574 | 0.8591 | +0.002 |
  | PER | 0.9183 | 0.9167 | −0.002 |
  | DAT | 0.8512 | 0.8421 | −0.009 |

  EVT 는 백본-불변(0.664≈0.665, relaxed 0.7362→0.7327) — support-limited
  진단(#125) 재확인이자 백본 교체가 EVT 레버가 아님을 실측한 것. ORG(koelectra
  0.851)는 #153 narrow-ORG 최종(koelectra·sup 2,797)의 0.849 와 정합해 #163 표의
  낮은 ORG(0.833)가 백본 차이임을 확인해준다. PROD 는 kf-deberta 가 우위, overall
  은 동률(±fold noise) — koelectra 는 recall 형·kf-deberta 는 precision 형.
- 결과-시점 refuter: **PASS** (`.omc/state/refuter/082736db0e14.json`, round 2).

## 관련 커밋

- `docs: 한국어 EVT gold 최대-span 재라벨 기준 정립·§3.5 폐기` (feat/issue-163)

## 후속 작업

- gazetteer feature 통합(외부 지식만·fold-blind) — 별도 이슈. gold 표면형
  금지, 본 재라벨 gold 의 F1 을 gazetteer 투자 판단 근거로 쓰지 않는다.
  상세 설계·근거는 아래 부록 참조.

## 부록: EVT F1 천장 조사 (1,346 gold 기준)

> 원래 자유형식 리포트(`docs/reports/ko-evt-f1-ceiling-investigation.md`)였으나
> #163 에 통합. 아래 수치는 **완전성 회수 이전 1,346 gold** 의 10-fold 측정
> 기록이며, 재현 산출물(probe 스크립트·pooled·fold 예측)은 검증 후 제거됐다.
> 최종 1,442 gold 의 F1·메트릭은 위 "검증" 섹션 참조. 두 gold 는 정의가 달라
> 직접 비교하지 않는다.

**요지**: 이 setup(KLUE silver + base 모델)의 EVT F1 은 1,346 기준 **0.640**
이 견고한 구조적 천장이었다. 값싼 레버 4개가 전부 반증됐고, 남은 정공법은
gazetteer *feature* 통합(재학습)뿐이다(낙관적 ~0.70–0.75, 0.80 은 aspirational).

### 오류 구조 (1,346 gold, 10-fold pooled)

EVT: P **0.60** / R **0.69**. TP 923 · FP 617 · FN 423.

| precision 손실 (FP 617) | | recall 손실 (FN 423) | |
|---|---|---|---|
| gold공백(누락+spurious) | 47% | 경계오류 | 49% |
| 경계오류(EVT↔EVT) | 39% | 완전누락(anchorless) | 39% |
| 타입혼동(DAT/LOC/ORG) | 14% | 타입혼동 | 12% |

**경계 오류의 정체**: 240쌍 중 **77% 가 모델 과축소**(gold 보다 짧게) —
`한국시리즈`→`한국`, `도쿄 올림픽`→`올림픽`. gold 는 대부분 정확(꼬리 2%뿐).
모델이 hapax 긴 span 을 못 배워 익숙한 조각으로 후퇴한다. (이 진단이 축3
최대-span 결정의 근거 중 하나 — 트림은 모델 과축소를 gold 로 고착시킨다.)

### 반증된 cheap 레버 4개

| 레버 | 결과 |
|---|---|
| 커버리지 축소(anchorless 19% 강등) | 0.640→0.650 (+0.01, −19% 커버리지) |
| gold 경계 재주석 | refuted — gold 이미 정확, 재주석 무의미 |
| 경계 loss 가중(`--boundary-b/i-weight`) | i3=0.623↓ / b2i2=0.658 flat |
| gazetteer 후처리 | 0.640→0.60(fix)/0.57(fix+recall), 클린해도 net-음 |

**재시도 금지**: 위 4개 + **gold 협소화로 F1 올리기**(§3.5, 위 결정 로그).

### 근본 원인 (얽힌 3구조)

1. **세계지식 부재** — `한국시리즈`≠`한국`을 모델·정적사전 둘 다 모름.
2. **문맥의존 경계** — 같은 이벤트가 문장별 bare(`월드컵`)/수식(`브라질 월드컵`).
3. **hapax** — 표면형 83% 1회, 일반화 단서 없음(19%는 anchorless=recall≈0).

### 남은 경로 — ② gazetteer feature 통합 (별도 이슈)

후처리(문맥 없는 하드 매칭)와 달리 **모델이 문맥 조건부 사용을 *학습*** 한다.
- (a) 사전 개선 — Wikidata SPARQL(KO 이벤트 라벨) + 도메인 목록, 최대-span
  경계로 정규화. (b) 토크나이즈 시 토큰별 사전 B/I/O 매치 feature 계산.
  (c) 그 feature 임베딩을 BERT 토큰표현에 concat 후 분류 head.
  (d) 10-fold 재학습.
- **누출 주의**: 사전은 외부 세계지식만(gold 표면형 금지), fold-blind.
- 기대: ~0.70–0.75 추정, 0.80 불확실. 다주 프로젝트.

**재학습 명령**(참고): `CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python
-m ner.classifier --lang ko --data <gold> --kfold 10 --fold-index <i>
--no-stratify --seed 42 --precision fp16 --output-dir <dir>` × 10 →
`python -m ner.classifier.kfold_pool --fold-dirs <dirs> --allow-cross-fold-leak`.
fold당 ~13분.

### 미결정 (open)

- **② feature 통합 착수 여부** — 불확실 payoff(~0.70–0.75) 다주 투자. spine
  변경이면 새 이슈 + 수락 기준 + 정의-시점 refuter 필요.
- **0.80 이 사업상 필수인가** — 아니면 0.664 가 정직한 종료점(#163 정합성이
  실 성과). gold 를 좁혀 0.80 을 만드는 길은 폐기됐다(§3.5).
