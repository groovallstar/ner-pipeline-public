# issue-140: 한국어 PROD/EVT gold §3.4 룰로 코퍼스 재생성

- Issue: https://github.com/groovallstar/ner_pipeline/issues/140
- PR: https://github.com/groovallstar/ner_pipeline/pull/142
- 브랜치: `feat/issue-140-ko-prod-evt-rule-regen`
- 승인일: 2026-06-23

## 목적                                                            [필수]

canonical §3.4·§5.3(KO PROD 경계 결정표, 이미 명문화 `9f86879`)을 **코퍼스
재생성으로 KO gold 에 실제 적용**한다. 4종 seed 부터 §3.4 프롬프트로 PROD/EVT 를
처음부터 재라벨해 #111 구 rubric 산출을 완전 대체한다. KLUE 4종(PER/LOC/ORG/
DAT)은 결정적 유지(단 eponymy 감사로 PER 일부 교정).

## 범위                                                            [권장]

- 포함: §3.4 프롬프트 이식(+R8 신규) → seed 재생 → relabel → merge → eponymy
  감사 → 재학습 무회귀 → production `pii_all.jsonl` 재빌드.
- 제외: PER/LOC/ORG/DAT 통째 재라벨, PII 주입 로직 변경.

## 성공 기준                                                       [필수]

- [x] A. §3.4 R1–R8 relabel 프롬프트 이식 + **R8 원자로 호기=비-entity** canonical
  보강. pilot 10/10 정확, ruff clean.
- [x] B. KO **row-level split**(`group_key=None`) — 행 원자적이라 누출-free
  (`orig` 는 VI/JA 전용, KO 해당 없음).
- [x] C. **고정 split**: `--no-stratify`(label-불변) → relabel 전후 fold 동일
  (단위 테스트 2종으로 불변성·대조 검증).
- [x] D. eponymy **전수 감사**: PER↔PROD 직접충돌 333 occurrence 를 사전등록
  치환 테스트로 독립 에이전트 1차 판정 → 2차 독립 에이전트 blind 재판정
  **κ=0.888 · 94.4% 일치** → 312 flip + 차량 3 flip 적용(id+offset, fold-blind).
- [x] E. PROD·EVT·변경 PER 은 **보고만**. 무회귀 게이트(gold 불변
  LOC/ORG/DAT + 미접촉 PER)는 ±0.5pp 이내 = train-seed 노이즈 범위.
- [x] F. merge partial-overlap drop 로깅(skipped=525, `merge_decisions` report).
- [ ] G. `pytest tests/ner` green · ruff clean · 결과-시점 refuter PASS.

## 결정 로그 (append-only)                                         [불변식]

- 2026-06-23: §3.4 는 *이미 명문화*(`9f86879`) — 본 이슈는 작성이 아니라 *적용*.
  초기 stale develop 체크아웃에서 §3.4 누락 오판 → 워크트리 기준으로 정정.
- 2026-06-23: eponymy = **B-전수**(독립 에이전트+refuter), ㉠-strict·blanket merge
  기각. 감사 범위는 추정 155–250 → 실측 **333**(직접충돌)로 확대.
- 2026-06-23: split 정정 — KO 는 `orig` 없음 → row-level. 비교는 `--no-stratify`
  고정 split.
- 2026-06-23: 무회귀 비교는 **clean NER-only**(PII 없이). llm-PII 가 원문을
  재작성해 PII-텍스트 graft 시 1.7% span 누락(비대칭 편향) → PII 제외가 직교·
  무편향. 프로토콜은 `docs/manual/data/korean-ner.md §7` 에 명문화.

## 현 상태 (fact)                                                  [권장]

신규 NER gold `origin.boundary.eponymy.jsonl` 분포: PER 18,238 / PROD 3,593 /
EVT 1,390 / LOC 7,944 / ORG 10,434 / DAT 10,080. relabel PROD 3,726·EVT 1,467 →
merge(961 replace·525 skip) → eponymy 315 PER→PROD. production `pii_all.jsonl`
은 확정 gold 에 §3.4-동일 llm 주입으로 재빌드(`docs/issues/issue-115` 방식).

## 검증                                                            [필수]

- 테스트: `python -m pytest tests/ner` / ruff: 변경 .py clean.
- **무회귀 (koelectra-base-v3, clean NER-only, 10-fold pooled, `--no-stratify`,
  seed 42)** — baseline `origin.jsonl`(#111) vs 신규 `origin.boundary.eponymy`:

  | type | baseline | boundary | Δ(pp) | gate |
  |---|---|---|---|---|
  | LOC | 0.8292 | 0.8319 | +0.28 | ✅ |
  | ORG | 0.8116 | 0.8094 | −0.22 | ✅ |
  | DAT | 0.8563 | 0.8572 | +0.09 | ✅ |
  | PER | 0.9239 | 0.9218 | −0.20 | 보고 |
  | PROD | 0.6608 | 0.6917 | +3.09 | 보고 |
  | EVT | 0.5878 | 0.6078 | +2.00 | 보고 |
  | micro | 0.8478 | 0.8475 | −0.03 | |
  | macro | 0.7783 | 0.7866 | +0.84 | |

  게이트(LOC/ORG/DAT) 모두 ±0.3pp = 노이즈 범위, PER·micro 불변 → **무회귀 PASS**.
  PROD/EVT 향상은 gold 정의 변경 포함이라 *보고만*. koelectra(ELECTRA) fold 1개
  학습 붕괴는 동일 seed 재학습으로 수렴(cuDNN 비결정성). (Δ 는 4-dec 반올림 전
  full-precision 기준.)
- **Caveat**: production 모델 koelectra 로 측정 — #128 과 같은 모델이나 eval
  (10-fold pooled·clean NER-only) 이 달라 절대값(#128 single-split 0.86대)과
  직접 비교 불가. 게이트 유효성은 baseline↔boundary **델타**. dup leak 12행 무시.
- 산출: `results/classifier/ko/cmp_{baseline,boundary}_koel/pooled.json`.

## 관련 커밋                                                       [옵션]

- `5ba397b`: feat(labelers) §3.4 코퍼스 재생성 + eponymy 감사 + 무회귀 검증
