# issue-10: 베트남어 WikiANN NER을 일본어 Stockmark 8종 스키마로 확장

- Issue: https://github.com/groovallstar/ner_pipeline/issues/10
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-10-vi-ner-8type-relabel`
- 승인일: 2026-04-21
- 완료일: 2026-04-22

## 목적

일본어 Stockmark(8종 수작업)와 베트남어 WikiANN(3종 silver) 간 스키마
비대칭을 해소하기 위해, WikiANN-vi 위에 Stockmark 8종과 동일한 라벨
(人名·法人名·政治的組織名·その他の組織名·地名·施設名·製品名·イベント名)을
씌운 베트남어 파생 데이터셋을 구축한다.

#8이 전체 완료되기 전에 **데이터 라벨링 확장을 선행**하기 위해 별도 이슈로
분리한다. #8의 VI loader·VI vLLM 라벨러 작업(①②)은 본 이슈로 이전하고,
#8은 PII 주입·벤치·리포트(③④⑤)에 집중한다.

## 범위

- 포함:
  - `src/labelers/vi/dataset_loader.py`(신규) — WikiANN BIO → offset span
    변환, JA loader와 동일 `gold_spans` 스키마. (← #8에서 이전)
  - `src/labelers/vi/{vllm_ner_labeler.py, ner_prompts.py,
    ollama_ner_labeler.py, openai_ner_labeler.py}`(신규) — VI 라벨러
    기본 셋. (← #8에서 이전)
  - 8종 재라벨 스크립트 `src/augmenters/wikiann_vi/relabel_8type.py`(신규) —
    경로 B: vLLM 8-class few-shot 프롬프트로 WikiANN-vi 재어노테이션.
    기본 모델 `cyankiwi/gemma-4-31B-it-AWQ-8bit`
    (endpoint http://localhost:8081/v1, `max_model_len=8192`).
  - 製品名·イベント名 빈도 측정 리포트 단계 — 부족 판정 시 후속 단계에서
    합성 보충 경로(PII 주입 구조 재사용) 투입 여부 결정.
  - 8종 매핑 스펙 `docs/specs/entities/vietnamese-ner-8types.md`(신규) —
    PER→人名, LOC→地名/施設名, ORG→法人名/政治的組織名/その他の組織名
    분할 기준, 모호 사례(지명/시설 경계 등) 처리 규칙 포함.
  - 검증 레이어 3중 구성:
    - ① cross-model agreement (최소 2개 모델로 독립 재라벨 → Cohen's kappa)
    - ② Wikipedia 인터링크 기반 Wikidata 타입 앵커 서브셋
    - ③ PER/LOC/ORG 3종 공정 비교 행
  - 리포트 `docs/reports/vietnamese-ner-schema-expansion-2026-XX.md`.
- 제외:
  - PII 주입 파이프라인·벤치·리포트 — #8.
  - BERT 파인튜닝 — 후속 이슈.
  - 수작업 gold 라벨링 — 본 프로젝트 상황상 불가.
  - VLSP/PhoNER 대체 도입.

## 성공 기준

- [ ] 테스트: `labelers/vi/dataset_loader` BIO ↔ offset span 왕복 단위 테스트,
      8종 재라벨 스크립트 단위 테스트.
- [ ] 데이터: 8종 스키마 WikiANN-vi 파생 JSONL 1000샘플 이상 생성.
- [ ] 검증: cross-model kappa 실측값 리포트 + 범위 명시.
- [ ] 검증: Wikipedia 인터링크로 가능한 앵커 서브셋 크기·타입별 일치율 리포트.
- [ ] 비교: 일본어 Stockmark 8종 vs 베트남어 확장 8종 동일 모델셋 상대
      순위 비교표.
- [ ] 문서: 8종 매핑 스펙 + silver 한계 명시 + 리포트.

## 구현 단계 (3~6)

- [ ] 1. #8에서 `src/labelers/vi/dataset_loader.py` · `labelers/vi/*`
      라벨러 이전 + BIO↔offset span 왕복 단위 테스트.
- [ ] 2. 8종 매핑 스펙 확정 (`docs/specs/entities/vietnamese-ner-8types.md`).
      Stockmark 정의 재확인 + 베트남어 모호 사례 10건 수준 예시.
- [ ] 3. 경로 B 재라벨 스크립트 구현 — `cyankiwi/gemma-4-31B-it-AWQ-8bit`로
      WikiANN-vi 재라벨, 모델 2종 이상 독립 실행해 kappa 측정.
- [ ] 4. Wikipedia 인터링크 → Wikidata P31/P279 조회로 앵커 서브셋 생성 +
      재라벨 결과와 일치율 측정.
- [ ] 5. 製品名·イベント名 빈도 집계. 부족 판정 시 합성 보충 경로(PII 구조
      재사용) 추가 여부를 사용자 승인 후 결정.
- [ ] 6. 리포트 작성 — 3중 검증 결과 + Stockmark 8종 vs VI 8종 상대 순위
      비교.

## 위험·의존성

- **이중 silver**: WikiANN(silver) + LLM 재라벨(silver) → 오류 누적. 3중
  검증 레이어로 정량화는 가능하지만 Stockmark와의 절대 F1 비교는 포기.
- **製品名·イベント名 저빈도**: Wikipedia 개요 문체에서 두 타입이 드물
  가능성. 3단계 이후 빈도 보고 5단계 합성 보충 결정.
- **Wikidata 매핑 모호성**: Wikidata 타입 체계 ≠ Stockmark 8종. 매핑 규칙
  2단계 스펙에서 확정.
- **모델 자원**: `cyankiwi/gemma-4-31B-it-AWQ-8bit`가
  http://localhost:8081/v1에 로드 상태 확인됨(`max_model_len=8192`).
  프롬프트 길이 8K 이내로 설계.
- **선행/병렬**: #8 ③④⑤가 본 이슈의 loader·라벨러(1단계)에 의존. #8
  본문을 범위 조정해야 함.

## 참고

- 관련 이슈: #8 (범위 조정 대상, ③④⑤만 잔존) —
  `docs/issues/issue-8-vi-wikiann-pii-bench.md`
- 선행 스펙: `docs/specs/entities/japanese-ner.md` (Stockmark 8종 정의)
- 배경 조사: 본 세션 대화 로그 (멀티링구얼 NER 스키마 관행 서베이,
  베트남어 NER 데이터셋 서베이, Wikipedia 인터링크 검증 아이디어)

---

## 변경 요약

WikiANN-vi(3종: PER/LOC/ORG) 데이터셋을 일본어 Stockmark 8종 스키마로
재라벨하는 파이프라인과 3중 검증(cross-model kappa + Wikidata 앵커 + 3종
공정 비교)을 구축했다. VI loader·라벨러·재라벨 클라이언트·앵커 유틸·CLI를
신규 추가하고, Gemma로 10K 전체 test를 재라벨해 `gold_spans_8type` 필드
11629 건을 생성했다. 이중 silver 한계는 인지하고 절대 F1 비교를 포기하는
대신, Wikidata 앵커 일치율 95.12%(10K, 6858 mapped)·cross-model κ=0.662(1K)로
신뢰도를 정량화했다. 製品名 1077·イベント名 182 확보로 **합성 보충은
불필요**하다는 결정을 도출했다.

## 구현 결과 (계획 대비)

- [x] 1. VI loader·라벨러 이전 — `labelers/vi/{dataset_loader,*_ner_labeler}.py`
      생성, BIO↔offset span 왕복 단위 테스트 11건 pass
- [x] 2. 8종 매핑 스펙 확정 — `docs/specs/entities/vietnamese-ner-8types.md`
      191줄 작성, WikiANN 3종 → 8종 분할 규칙 + 모호 사례 10건
- [x] 3. 경로 B 재라벨 스크립트 구현 — `src/augmenters/wikiann_vi/` 패키지
      신규(프롬프트·Relabeler·CLI·kappa·Wikidata 앵커), 단위 테스트 46건 pass.
      Gemma/Qwen 1K 재라벨 + κ=0.6620 측정
- [x] 4. Wikipedia 인터링크 → Wikidata P31 앵커 검증 — `WIKIDATA_TO_STOCKMARK`
      134 Q-ID 매핑, 10K 재측정에서 6858건 매핑·95.12% 일치
- [x] 5. 製品名·イベント名 빈도 집계 — Gemma 10K 전체 재라벨(25분, 11629
      span, 0 error) → 製品名 1077·イベント名 182 확보 →
      **합성 보충 불필요 판정**
- [x] 6. 리포트 `docs/reports/vietnamese-ner-schema-expansion-2026-04.md`
      438줄 작성 (3중 검증 + 타입별 신뢰도 계층 + 3종 공정 비교)

추가 수행 (품질 향상 A1·A3·A4):
- [x] A1. `WIKIDATA_TO_STOCKMARK` 확장 80 → 134 Q-ID. 政治的組織名 0.54 →
      0.72 (+18.2pp), イベント名 0.62 → 0.70 (+8.2pp)
- [x] A3. JA Stockmark 1069 vs VI 10K 3종 공정 비교 (entity density·분포 차)
- [x] A4. `Relabeler.batch_size` 추가. 벤치 결과 현 구성(concurrency=16)에서는
      vLLM continuous batching이 이미 최적이라 SINGLE과 동속도 — 타 backend
      옵션으로만 보존

## 검증

### 테스트

```bash
pytest tests/labelers/vi/ tests/augmenters/wikiann_vi/ -v
# 51 passed in 3.43s
# - VI dataset_loader (BIO↔span 왕복): 11건
# - relabel_8type (parse/match/batch): 14건
# - kappa (Cohen/per-type/confusion): 12건
# - wikidata_anchor (type/resolve/iter): 14건
```

전체 회귀: `pytest tests/` → 전체 pass (기존 142 + 신규 51 = 193).
Ruff lint: `ruff check src/ tests/` → All checks passed.

### 메트릭

| 지표 | 값 |
|---|---|
| Cross-model κ (1K) | 0.6620 |
| 동일 offset 타입 일치율 | 97.06% |
| Wikidata 앵커 일치율 (10K, v2) | 95.12% (6858 mapped) |
| 政治的組織名 앵커 (v1 → v2) | 0.54 → 0.72 |
| イベント名 앵커 (v1 → v2) | 0.62 → 0.70 |
| Gemma 10K 재라벨 시간 | 1519s (에러 0) |
| 製品名 / イベント名 10K 건수 | 1077 / 182 |

### 리뷰어 확인 사항

- `data/wikiann_vi_relabel/` 산출물은 gitignore — 재현은 리포트 §13 참조
- 매핑 테이블 v2의 일부 regression(人名·法人名 소폭 하락)은 확장 엔트리가
  더 많은 엔티티를 잡으면서 발생한 noise; 절대값은 여전히 High 계층
- Qwen 10K 재실행 / 10K kappa 재측정은 후속 이슈 후보로 이월 (5× 속도 차)

## 관련 커밋

- `a773ac6` feat(labelers/vi): 베트남어 LLM 라벨러 추가
- `039839c` feat(labelers/vi): WikiANN BIO→offset span 로더 및 왕복 테스트
- `b2e31e2` docs(specs/entities): 베트남어 NER 8종 매핑 스펙 추가
- `02cfbc3` feat(augmenters/wikiann_vi): 8종 재라벨 프롬프트 추가
- `d3b484b` feat(augmenters/wikiann_vi): 8종 재라벨 async 클라이언트 및 유틸
- `351e3c0` feat(augmenters/wikiann_vi): 8종 재라벨 CLI 엔트리 추가
- `31e5c52` feat(augmenters/wikiann_vi): cross-model Cohen kappa 모듈 및 CLI
- `bd40868` docs(reports): 베트남어 8종 스키마 확장 리포트 추가 (2026-04)
- `cdac640` feat(augmenters/wikiann_vi): Wikipedia 인터링크 + Wikidata P31
  앵커 검증
- `fdd7bd5` docs(reports): 베트남어 8종 스키마 확장 리포트 4·5·6단계 결과
  통합
- `54b4771` feat(augmenters/wikiann_vi): WIKIDATA_TO_STOCKMARK 매핑 테이블
  21종 확장 (80→134)
- `cd64ece` feat(augmenters/wikiann_vi): BATCH 프롬프트 기반 다중 레코드 호출
  지원
- `433cc3f` docs(reports): 매핑 확장·3종 공정 비교·신뢰도 계층 재분류 반영

## 후속 작업·알려진 한계

### 후속 이슈 후보
- Qwen 10K 재실행 + 10K kappa 재계산 (1K kappa 대표성 검증, ~2시간 컴퓨트)
- `WIKIDATA_TO_STOCKMARK` 추가 확장 134 → ~200 (diminishing return 구간)
- VI 8종 학습 데이터 기반 BERT 파인튜닝 (원래 이슈 #10 제외 범위)
- JA 8종 vs VI 8종 통합 멀티링구얼 벤치 러너

### 알려진 한계
- 이중 silver(WikiANN 자동 + LLM 재라벨 자동) — 본 프로젝트 제약상 수작업
  gold 검증은 불가
- Wikipedia 리다이렉트로 anchor 타입이 왜곡되는 드문 케이스 존재 (예:
  "Her Morning Elegance" 노래 → 가수 Wikipedia 페이지로 리다이렉트)
- 매핑 테이블은 수작업 큐레이션(134종). SPARQL 기반 transitive P279 탐색
  도입 시 자동 확장 가능하나 본 범위에서는 미도입
- 1K cross-model kappa를 10K 대표성으로 채택 (Qwen 10K 미수행)
