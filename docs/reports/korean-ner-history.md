# 한국어 canonical NER — 트랙 히스토리

한국어 canonical 파이프라인(`src/ner/labelers/ko/`, `data/klue/`)이 KLUE
원본에서 canonical **10종 평면** gold까지 도달한 과정을 단계별로 누적
정리한다. 각 단계의 "무엇을·왜·결과"를 축약해 누적한다.

- 대상: ko gold `data/klue/origin.jsonl`(NER) + `data/klue/pii_all.jsonl`(10종)
- 단일 스키마 출처: `docs/manual/data/canonical-entity-schema.md`

## 단계 요약

| 단계 | 이슈 | 한 일 | gold |
|---|---|---|---|
| 1 | #109 | KLUE→canonical 정렬(`PS/LC/OG/DT→PER/LOC/ORG/DAT`, `TI/QT` 드롭) | 4종 seed |
| 2 | #111 | KLUE 문장 LLM 재라벨로 `PROD/EVT` 증분(§3.1~3.3 회색지대) | NER 5종 + DAT |
| 3 | #115 | PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`) llm 자연삽입 | **canonical 10종** |

## 단계 3 (#115) — PII 4종 → 10종 완성 (2026-06-16)

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

## 다음 (후속 이슈 후보)

- ko 분류기 학습·벤치마크 — canonical 10종 gold(`pii_all.jsonl`) 소비.
- ko LLM NER 10종 라벨러 확장·벤치마크 — 측정과 함께 별도.
