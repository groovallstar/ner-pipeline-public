# issue-153: 한국어 ORG 엔티티 진단 — gold 일관성 레버 규명·적용

- Issue: https://github.com/groovallstar/ner_pipeline/issues/153
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-153-ko-org-diagnosis`
- 승인일: 2026-06-29

## 목적

ORG F1 0.8167(support 대비 저성능 outlier)의 잔여를 진단하고 천장 레버를
규명한다. EVT(#125)·PROD(#128) 와 동형의 per-entity 진단.

## 결과 요약

천장이 아니라 **gold 일관성 레버**였다. 국가명 ORG↔LOC 양분(gold 비일관)이
model 의 국가→ORG 과예측을 학습시켜 ORG·LOC F1 을 동시에 눌렀다. 일관화+재학습
으로 ORG **+0.028**(clean 동일-test)·LOC **+0.041**(보고만), ORG↔LOC confusion
**−55%**.
JA ORG #71 의 "gold-천장(F1 무중재)" 예상을 반증한다.

## 수락 기준 결과

- [x] 10-fold pooled ORG baseline 확립: P0.817/R0.816/F1 **0.8167**, fold std
  0.012(#128 single-split 0.816 정합). leak-free 확인(`cross_fold_orig_dups=0`,
  KO 행 원자적).
- [x] FP/FN 4버킷 MECE(pooled distinct **3,207**): ORG↔LOC 851·HALLUCINATION
  773·MISS 710·BOUNDARY 609·ORG↔기타 264.
- [x] ORG↔LOC confusion 정량화: gold ORG→pred LOC 449 + gold LOC→pred ORG 402
  = 851(국가명이 ~40%, 양방향 동일 표면형).
- [x] gold-누락 falsifiable + zero-retrain rescore: B 기관추가 +0.009·C 조각제거
  +0.007(gold-side 확정·모델 맞음), **A 국가→LOC −0.012**(model 과예측 →
  천장 가설 반증).
- [x] 지배 레버 판정(사전 임계): model-side(ORG↔LOC) 지배 ∧ clean ΔF1 +0.028
  > 2σ(0.018) → **레버 있음**(천장 아님).
- [x] 순환성 가드: gold-변경 포함 F1 은 보고만, **구·신 모델을 동일 교정-test
  로 채점**(0.8093→0.8371)해 비순환 ΔF1 +0.028 확정.
- [x] joint gate 무회귀: PER 0.918 불변, micro +0.007, paired-seed 분산 안.
- [ ] `pytest tests/ner` green · ruff clean · 결과-시점 refuter PASS.

## 구현 (분해)

1. **진단**: baseline 10-fold(`koel-org-diag`) → `error_analysis --from-
   predictions` pooled → 4버킷·confusion·표면형 집중도. 원인 = gold-side
   (국가 비일관 / 기관 누락 / 단일글자 조각).
2. **rescore**(zero-retrain): 교정 규칙을 test gold 에 적용·모델 고정 재채점
   → A 국가→LOC 가 음(−0.012). 비일관 gold 로 학습한 국가→ORG 과예측이 원인
   임을 규명(천장 반증).
3. **gold 일관성 확보**(dual-LLM gemma-4-31B + Qwen3.6-35B 이중 독립판정):
   - ① 룰(canonical 확정 케이스): 국가→LOC 580 · 기관(국회·청와대·국정원)
     →ORG 20 · 단일조각 제거 247.
   - ② dual-LLM 기타 104 판정(합의 79%, **합의분만** 적용) + HOLD triage
     (지리명→LOC·명확기관→ORG·진짜 모호 유지).
   - ③ 행정복합(서울시류)·polysemy 유지. cross-type minority 162 도
     per-instance dual-LLM 으로 clear 오류 **33 만** 정리(120 은 적법
     polysemy 보존).
   - split 표면형 **166→17**(잔여 전부 적법 문맥의존). PER/PROD/EVT 불변.
4. **레버 측정**: 교정 gold 10-fold pooled → ORG 0.8371·LOC 0.8694·micro
   0.9133. clean 동일-test ORG 0.8093→0.8371(+0.028). confusion 851→385.
5. **gold 통합**: 교정본을 production 승격(`data/klue/pii_all.jsonl` ·
   `origin.boundary.eponymy.jsonl`), 구버전·중간 삭제 → **2파일**.
6. **문서**: 진단 리포트 §ORG · canonical 변경이력 #153 · 본 아카이브.

## 방법론 노트

- **1회성 일관성 툴은 비커밋**(프로젝트 norm, `c113ed3` 동형). 방법·provenance
  (룰·합의율·flip 카운트)는 본 문서·리포트에 흡수. 규칙(국가=LOC·일반명사 기관
  =ORG)은 기존 canonical(`korean-ner` 149·150)이라 신규 룰이 아닌 **enforcement**.
- **dual-LLM 이중 독립판정**으로 polysemy(배트맨 캐릭터/작품·서울시 행정/지방
  정부·바르셀로나 도시/클럽)를 보존 — 합의분만 적용해 over-flatten 방지(#140
  독립 에이전트 κ 검증과 동형).
- baseline 모델·예측은 `results/classifier/ko/koel-org-diag/`, 교정 후는
  `koel-org-consist/`(gitignore — 표가 영구 인용 단일 출처).

## 결론

ORG 천장은 gold-천장이 아니라 gold 일관성 레버였다. 국가명 ORG↔LOC 양분이
유일 최대 오류 채널이자 model 오류의 원천이었고, 일관화+재학습으로 ORG·LOC 가
동반 상승했다. 잔여 split 17 은 canonical 이 문맥의존이라 규정한 적법 다의성
(축소 불가). 후속(외부 ORG 데이터)은 천장 도달 시 별도.
