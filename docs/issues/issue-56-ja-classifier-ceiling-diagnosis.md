# issue-56: JA classifier 천장 진단 — confusion matrix + FP/FN dump

- Issue: https://github.com/groovallstar/ner_pipeline/issues/56
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-56-ja-classifier-ceiling-diagnosis`
- 승인일: 2026-05-18

## 목적

v3+boundary production 모델의 천장 미달 4클래스 (LOC/ORG/EVT/PROD) FP/FN
패턴을 confusion matrix + surface dump 로 정밀 진단. 후속 시나리오
(S1~S5) 우선순위 결정 근거 확보.

## 범위

- 포함: `error_analysis.py` 확장 / 진단 산출물 / 진단 리포트
- 제외: 학습·데이터 변경, backbone/loss 수정 (S1~S5 의 몫)

## 성공 기준

- [ ] 테스트: `tests/ner/classifier/test_error_analysis.py` 신규 4종 pass
- [ ] 문서 갱신: 진단 리포트 + 핸드오프 update + 본 이슈 md
- [ ] 메트릭: confusion matrix 가 PROD/EVT/ORG FP leak 경로를 클래스당
  1~2개로 좁힘, LOC FN OOV vs ambiguous 비율 정량화

## 위험·의존성

- v3_boundary best/ inference (~수십 초, GPU 사용 가능 시)
- tokenizer `tohoku-nlp/bert-base-japanese-v3` 별도 주입 필요
- test 셋 inference 만 — 학습·모델 선택 노출 없음

## 현 상태 (fact)

- 기준선: strict 0.9368 / relaxed 0.9505 (v3+boundary)
- 천장 미달: PROD-P 0.8515 / EVT-P 0.8681 / ORG-P 0.8946 / LOC-R 0.9068
- 패턴: 3종 over-predict + 1종 under-predict
- 기존 `error_analysis.py` 는 4-카테고리 분류·per-type 집계 지원,
  confusion matrix·surface dump 미지원

## 결정 로그 (append-only)

- 2026-05-18: 핸드오프 (post-#51) E8-JA DAPT / Tier 2.B-JA self-training
  대신 진단 기반 S0~S5 채택. 진단 결과로 S1~S5 우선순위 재조정 예정.
- 2026-05-18: 진단 결과 type 혼동 ≤ 12% 확인 → S3 (contrastive) 우선순위
  강등, S7 (HALL negative oversampling) / S6 (BOUNDARY 정책 명문화) 신규
  도출. 다음 작업은 S1 + S7 병렬.

## 미해결 질문

- 해소: PROD/EVT/ORG FP 주 leak 경로 → **HALL 60 (47%) + BOUNDARY 55
  (43%), type 혼동 11%**. ORG HALL 28 이 단일 최대 (전체 HALL 의 47%).
- 해소: LOC FN OOV vs ambiguous boundary 비율 → **MISS 12 (41%, 짧은
  ambiguous 지명) + BOUNDARY 12 (41%, 우편번호·복합 지명) + TYPE_MISMATCH
  5 (17%, 외국 지명 → ORG)**.

## 구현 단계 (가변)

- [x] `error_analysis.py` 확장 (`build_confusion_matrix`,
      `top_errors_by_type`, `render_diagnosis_md`, `--with-diagnosis`)
- [x] 단위 테스트 추가 (confusion 3종 + top_errors 1종)
- [x] 진단 inference + 산출물 생성 (`results/.../v3_boundary/`)
- [x] 진단 리포트 작성
  (`docs/reports/japanese-bert-classifier-ceiling-diagnosis.md`)
- [x] 핸드오프 '다음 작업' 섹션을 S1+S7 병렬 + 후속 후보 재정렬로 교체

---

## 변경 요약

- `error_analysis.py` 에 confusion matrix + top-N surface dump 분석 추가
  (신규 4 함수 + CLI flag `--with-diagnosis`, BC 유지)
- 신규 단위 테스트 4종 (confusion 3 + top_errors 1) 추가
- v3+boundary inference (527 sentences, 5초) → 5개 진단 산출물 생성
  (`results/.../v3_boundary/`, `.gitignore` 처리)
- 진단 결과를 본 이슈 md 에 통합 + 핸드오프 갱신 (S0 완료 + S1/S7 다음
  작업)

## 검증

- 테스트: `uv run python -m pytest tests/ner/classifier/test_error_analysis.py -q`
  → 14 passed (신규 4 포함)
- Lint: `uv run ruff check src/ner/classifier/error_analysis.py
  tests/ner/classifier/test_error_analysis.py` → All checks passed
- 진단 CLI: stdout 에 FN/FP 분포 + 5 files saved 확인
- 핵심 발견은 §진단 결과 참조

## 진단 결과

### 천장 미달 4클래스

| 종류 | 정밀도 | 재현율 | 어느 쪽이 부족한가 |
|---|---:|---:|---|
| PROD | **0.8515** | 0.9053 | 정밀도 (너무 많이 표시) |
| EVT | **0.8681** | 0.8977 | 정밀도 |
| ORG | **0.8946** | 0.9152 | 정밀도 |
| LOC | 0.9369 | **0.9068** | 재현율 (충분히 못 표시) |

### 오답 5종 분류 — 종류 혼동은 본질이 아님

- 정답 일치 (EXACT) — 위치·종류 모두 맞음
- 경계 어긋남 (BOUNDARY) — 종류는 맞고 시작·끝만 다름
- 종류 혼동 (TYPE_MISMATCH) — 위치 겹치지만 종류 다름
- 누락 (MISS) — 정답 있는데 모델 미반응
- 환각 (HALLUCINATION) — 정답 없는데 모델이 entity 라벨 부여

집계 (test 527문장):
- 정답 1922 / 예측 1936 / 정답 일치 1807
- 놓침 (FN) 115: 경계 어긋남 55 (48%) / 누락 46 (40%) / 종류 혼동 14
  (12%)
- 잘못 짚음 (FP) 129: 환각 60 (47%) / 경계 어긋남 55 (43%) / 종류 혼동 14
  (11%)

**종류 혼동이 양쪽 12% 이하 — "ORG·PROD 헷갈림"이 본질이 아님**. 진짜
문제는 환각 + 경계 어긋남.

### 혼동 행렬 (NER 5종 only)

행 = 정답, 열 = 예측, ∅ = 정답 또는 예측 없음.

| 정답＼예측 | PER | LOC | ORG | PROD | EVT | ∅ (누락) |
|---|---:|---:|---:|---:|---:|---:|
| PER | 373 | - | 2 | - | - | 2 |
| LOC | 1 | 294 | 4 | - | - | 12 |
| ORG | 1 | - | 545 | 2 | 1 | 17 |
| PROD | - | - | - | 92 | - | 3 |
| EVT | - | - | - | 1 | 83 | 4 |
| **∅ (환각)** | 3 | 7 | **28** | 6 | 7 | - |

→ 환각 60건 중 **ORG 28건 (47%) = 단일 최대 leak**.

### 클래스별 잘못 짚는 패턴

- **PROD 환각 6**: 짧은 영어 약어·일반 명사 (`SFET`, `Mac`, `旗幟`, `黒`,
  `納豆`, `PWR`)
- **EVT 환각 7**: 영어 약어·숫자·행사명 일부 (`FI`, `選手`, `第`, `1999`,
  `ヨーロッパリーグ`)
- **ORG 환각 28** (최대):
  - 단일 한자: `米`, `局`, `連`, `第`, `鍋`
  - 영어 약어: `PC`, `CG`, `GM`, `KD`(KDE 일부)
  - 일반 명사: `厚生`, `工場`, `劇団`, `海軍`, `徳川`
  - 역명·시설명: `鎌取駅`, `東京駅`, `小作駅`, `東洋大学`
- **LOC 놓침 29**:
  - 누락 12 — 짧은 지명 (`日本` 2회, `朝鮮`, `米国`, `ユーゴ`, `米`)
  - 경계 어긋남 12 — 우편번호·복합 지명 (`〒652-...`, `中華人民共和国`,
    `スペイン北部`)
  - 종류 혼동 5 — **4건이 외국 지명을 ORG로 잘못 분류** (`モーデン`,
    `コロネルズ・ロウ`, `コニー・アイランド`, `ニトラ`)

경계 어긋남 패턴: 영문 모델명 (`富士フイルム FinePix F200EXR`), 인용부호
(`「黒」納豆`), 우편번호 prefix, 복합 조직·지명 splitting.

### 진단으로 알게 된 것

1. 종류 혼동은 본질이 아니다 (12% 이하) — ORG↔PROD 분리 학습 ROI 낮음
2. 환각 + 경계 어긋남 = 잘못 짚음의 88%
   - 환각: 짧은 영어 약어·단일 한자·일반 명사를 entity 로 잘못 부름.
     학습 데이터에서 "이건 entity 아님" 신호 부족
   - 경계 어긋남: 인용부호·우편번호·영문 모델명·복합 명사 자르는 정책
     불일치
3. LOC 놓침의 본질은 외국 지명 사전 부재 + 우편번호 포함 정책 부재
4. ORG 환각 28건이 전체 환각의 47% — 단일 최대 표적

### 후속 시나리오 우선순위 (효과·비용 순)

| 순위 | 시나리오 | 효과 | 비용 | 한 줄 |
|---|---|---|---|---|
| 1 | **S1** 클래스별 기준값 + 비대칭 손실 | ★★★ | ★ | 신뢰도 기준값 올림 (학습 없이 즉시 확인). 미달 시 잘못 짚을 때 큰 페널티 주는 손실함수로 1회 학습 |
| 2 | **S7** 환각 부정 예시 추가 학습 (신규) | ★★★ | ★★ | 짧은 영어 약어·일반 명사를 "entity 아님" (O) 으로 명시 학습. ORG 환각 28건 직격 |
| 3 | **S6** 경계 규칙 명문화 + 데이터 보정 (신규) | ★★ | ★★ | 인용부호·우편번호·모델명 자르는 규칙 schema 명문화. 경계 어긋남 55건 직격 |
| 4 | **S2** Wikidata 지명 사전 (LOC 한정) | ★★ | ★★ | 외국 지명 사전 토큰 단서로 주입. LOC 누락 + 외국 지명 종류 혼동 직격 |
| 5 | **S4** 사람 검수 추가 | ★★ | ★★ | 신뢰도 0.4~0.7 구간 400~500건. 라운드 2 에서 40건 실시 — 추가 한계 |
| 6 | **S3** 종류 분리 학습 | ★ | ★★ | 종류 혼동 12% 뿐 — ROI 낮음. S7 흡수 |
| 7 | **S5** 기반 모델 교체 | ★ | ★★★ | 최후 옵션 |

### 권장 진행

가장 효과·비용 비율 좋은 두 시나리오 병렬 분기:

1. **S1** — 학습 없이 신뢰도 기준값만 올려도 정밀도 회복 가능한지 즉시
   측정
2. **S7** — 환각 28건의 ORG 가 가장 큰 표적. 부정 예시 풀 구축 + 1회
   학습

두 결과 모두 미달 시 **S6** (경계) / **S2** (LOC 사전) 로 분기.

### 한계

- test 527문장 단일 inference. seed/split 변경 시 패턴 재검증 가능하지만
  라운드 2 에서 80/10/10 seed=42 동결
- 환각 surface 패턴은 *경향성*. 일부 (`徳川`, `海軍`, `工場`) 는 진짜
  entity 모호 케이스 — 라벨 정합성 별도 검토 필요
- 경계 규칙 명문화 (S6) 는 라운드 2 canonical schema 4영역의 연속선,
  라운드 3 schema update 로 통합 가능

## 관련 커밋

- `<hash>`: <!-- commit 직후 채움 -->

## 후속 작업

- **S1** per-class threshold + asymmetric focal loss
- **S7** HALL negative oversampling (신규)
- 두 결과 미달 시: S6 (BOUNDARY 정책 schema 라운드 3) / S2 (LOC gazetteer)
