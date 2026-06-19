# 한국어 BERT NER 분류기 — 엔티티별 성능 진단·보정

한국어 canonical 분류기(`python -m ner.classifier --lang ko`, gold
`data/klue/`)의 엔티티별 잔여 성능을 진단·기록한다. **Part 0**은 gold 계보
(KLUE 원본 → canonical 10종 구축 단계), **Part 1**은 엔티티별 진단이다.

- 대상: ko gold `data/klue/origin.jsonl`(NER) + `data/klue/pii_all.jsonl`(10종)
- 단일 스키마 출처: `docs/manual/data/canonical-entity-schema.md`
- baseline 수치: `korean-bert-classifier-benchmark.md` (#122, koelectra-base-v3)

## Part 0 — gold 계보

### 단계 요약

| 단계 | 이슈 | 한 일 | gold |
|---|---|---|---|
| 1 | #109 | KLUE→canonical 정렬(`PS/LC/OG/DT→PER/LOC/ORG/DAT`, `TI/QT` 드롭) | 4종 seed |
| 2 | #111 | KLUE 문장 LLM 재라벨로 `PROD/EVT` 증분(§3.1~3.3 회색지대) | NER 5종 + DAT |
| 3 | #115 | PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`) llm 자연삽입 | **canonical 10종** |

### 단계 3 (#115) — PII 4종 → 10종 완성 (2026-06-16)

ja/vi의 suffix PII 주입 파이프라인을 한국어로 지역화하며, *분류기 학습용*
gold 품질을 위해 다음 설계 결정을 거쳤다.

**설계 결정 흐름** (이번 세션의 핵심):

- **주입 라벨 = PII 4종만** — KLUE 유래 `PER/LOC/DAT`는 합성분 없이 순수 유지
  (`--pii-labels` CLI 신규).
- **주입 방식 = llm 자연삽입(≠ suffix)** — suffix는 접두(`연락처:`) + 문말
  나열이라 BERT가 접두 패턴으로 쉽게 풀어 학습 편향(`augmenters/AGENTS.md`
  "suffix BERT 적합성 낮음"). llm은 PII를 문중 자연삽입(경직 접두 0%).
- **verify 제거** — `--verify` drop_span은 레코드 전체를 LLM과 대조해 못
  맞힌 항목을 삭제하는데, ko 원본은 *사람 KLUE gold*라 LLM(F1~0.9)이 재현
  못 한 ~13%가 삭제됨(스모크 측정). 게다가 llm `extract_spans`가 string-match
  로 offset을 이미 정확 보장 → verify 무용. (drop_span은 silver·두-LLM 검증
  용이라 ja/vi엔 정당했으나 ko 사람 gold엔 부적합.)
- **ko 라벨러 10종 확장 revert** — verify를 먹이려 도입했던 것이라 verify
  제거와 함께 되돌림. ko LLM NER 10종 벤치는 별도 이슈로.
- **`extract_spans` 공백 허용 매칭** — LLM이 카드번호 이중공백을 단일로
  정규화 → exact 실패로 레코드 2% 드롭. 공백 run을 `\s+`로 재탐색(매칭
  표면형을 엔티티로 저장) → 드롭 2%→0.07%.

**결과 (full run, gemma-4-31B)**: gold `data/klue/pii_all.jsonl` 25989행
(26008 입력 → 19 스킵 0.073%).

| 검증 | 결과 |
|---|---|
| 10종 분포 | 전부 존재, PII 4종 각 ~8.3k |
| offset 무결 | 0 mismatch (33819 PII 전수) |
| 자연삽입 | 경직 접두 직후 0/33819 (0.00%) |
| 주민등록번호 체크섬 | 8469/8469 무효 (실유효 번호 비생성) |
| 원문 NER 보존 | 99.30% (스킵 제외 99.40%) |

- 원문 손실 0.7%: LLM 재작성이 상대날짜(`7년전`)·복합 작품명을 패러프레이즈해
  string-match 미발견 → drop된 것(허용 범위).
- 주입 PII는 문법은 자연스러우나 KLUE 뉴스 주제와 의미상 어색 — 형식(format)으로
  인식되는 타입(13자리=ID_NUM 등)이라 분류 학습엔 무해하고, 접두 의존을 차단해
  "너무 쉽지 않게"라는 의도와 부합.
- 검증: `pytest tests/ner` 323 green, ruff clean, 격리 반박자(refuter) PASS
  (측정 숫자 독립 재계산 일치, 테스트 무결성 확인).

산출: 커밋 `dcfb2e0`, PR #118. 재현:

```bash
python -m ner.augmenters.pii --source jsonl --input data/klue/origin.jsonl \
    --lang ko --pii-labels EMAIL PHONE ID_NUM CREDIT_CARD --mode llm \
    --inject-url http://localhost:8081/v1 \
    --inject-model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/klue/pii_all.jsonl
```

## Part 1 — 엔티티별 진단

### EVT — support-limited (학습가능성 절벽) (#125)

koelectra-base-v3 baseline에서 EVT가 NER-5 최저(F1 0.581). 정체를 진단한
결과 **gold 양 부족(support-limited)** 으로 확정 — 경계·커버리지 결함이 아니다.

**진단 경로:**

- **오류 감사** (test 오류 distinct 86): gold-side 45%(경계 비일관·gold 누락)
  / model-side 34%(recall 실패·조각 예측) / EVT↔ORG·DAT 정의모호 21%. EVT
  gold는 KLUE 문장 LLM 재라벨 silver(#111)이고 표면형 83%가 hapax.
- **커버리지 축소 가설 → 반증**: 앵커(행사유형 head) 없는 idiosyncratic
  1회성 고유명을 비-entity로 강등(EVT 1187→904)하면 EVT가 **개선이 아니라
  악화**(F1 0.581→0.447). 강등 span이 예측에서 FP로 전환되고 recall이
  과붕괴 → flat-span NER에서 "비학습 꼬리 제거"는 역효과(커버리지 축소·
  관련 스키마 절 시도는 폐기).
- **support-limited 가설 → 확정**: 클래스별 Pearson(log-support, F1)=0.986,
  EVT가 최소 support=최저 F1(ORG만 ORG↔LOC confusion으로 outlier). EVT train
  서브샘플 learning curve(고정 full test)에서 **viability cliff** — ~600–890
  span 미만이면 모델이 EVT를 전부 abstain(F1=0), 그 위에서 켜짐. 현재 1187은
  절벽을 막 넘은 가파른 상승 구간(100%에서도 Δ+0.061, 포화 아님).

**EVT P/R/F1 측정** (strict, koelectra-base-v3, 고정 full test):

| 조건 | EVT gold(train) | P | R | F1 | test sup |
|---|---|---|---|---|---|
| baseline (#122) | 1187 | 0.530 | 0.642 | **0.581** | 123 |
| 커버리지 축소 (반증) | 904 | 0.462 | 0.433 | 0.447 | 97 |
| learning curve 25% | 297 | 0.000 | 0.000 | **0.000** | 123 |
| learning curve 50% | 594 | 0.000 | 0.000 | **0.000** | 123 |
| learning curve 75% | 891 | 0.574 | 0.537 | **0.555** | 123 |
| learning curve 100% | 1187 | 0.579 | 0.659 | **0.616** | 123 |

> 컨트롤(EVT만 변경): 같은 런들에서 ORG F1 range 0.011·PROD range 0.070 —
> EVT 신호(0→0.616)는 저support 단일런 noise를 압도. 100%(0.616)와 #122
> baseline(0.581) 차이는 fresh 재학습 분산 범위.

**진단 결론**: EVT = support-limited(비학습 아님 — gold가 PROD급 ~3k이면
~0.70, LOC급 ~8k이면 ~0.85 기대). 천장 레버는 경계·커버리지·스키마가 아니라
**EVT gold 증강**(곡선이 근거, 단 1187 너머는 외삽). cliff 근처라 high-variance
동반. 재현용 EVT 서브샘플러(`subsample_evt.py`)는 1회 측정 후 제거 —
로직(EVT span 안정 해시 rate% nested 서브샘플)은 커밋 이력 보존.

## 다음 (후속 이슈 후보)

- EVT gold 증강 — KLUE 미검출 이벤트 문장 추가 relabel로 절벽에서 끌어올리기.
- ko LLM NER 10종 라벨러 확장·벤치마크 — 측정과 함께 별도.
