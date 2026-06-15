# issue-98: 베트남어 NER 전체 체인 정합성 감사

- Issue: https://github.com/groovallstar/ner_pipeline/issues/98
- PR: <!-- 승인·구현 후 채움 -->
- 브랜치: `feat/issue-98-vi-ner-fullchain-audit` (워크트리 `/tmp/ner_vi_audit`)
- 베이스: c251f40 (JA 마무리 #96 tip) — develop은 6커밋 뒤(CRF·핸드오프 잔존)라 부적합
- 승인일: <!-- 승인 후 채움 -->

## 목적

베트남어 NER 체인(canonical 10종 평면 = NER 5 + PII 5)의 end-to-end 정합성을
감사한다. JA #95(umbrella)의 VI 대응. VI는 미출하 상태이므로 **출하 아티팩트
채택(#95 자식 A)은 제외**하고, 정합성 확인·stale 참조 정리·현재 metric 천장
확인에 집중한다.

**게이트 비중립**: 본 감사는 게이트 수치를 확정하지 않는다. per-entity
P/R/F1 천장을 results 실측으로 확인·정리만 하고, 게이트는 그 지표를 보고
유동적으로 별도 설정한다(범위 밖). 성능 천장 *개선*(F2~F6)도 범위 밖.

## 체인 지형 (정찰 결과)

```
labelers/vi (vllm·openai·dataset_loader)   ── LLM NER 라벨러, canonical 10종 참조
augmenters/wikiann_vi  ── WikiANN-vi → NER 5종(PER/LOC/ORG/PROD/EVT) 재라벨
augmenters/pii         ── PII 5종(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) 주입
        └→ data/wikiann_vi/pii_all.jsonl (합본, gitignore)
classifier (data_utils·train_eval)  ── pii_all.jsonl 소비, BIO↔char-span, span F1
llm_eval (benchmark_runner·report)  ── VI offset-span 러너·메트릭
results/classifier/vi/*.json  ── metrics / sweep(4) / v2·v3·v4 (로컬 존재 확인)
docs/reports/vietnamese-*.md (4종)  ── 영구 인용 표 (results 백킹)
```

## 계획 (체크박스 — 동일 항목 Issue 본문에도 기록)

- [x] **B1. 라벨·오프셋 계약 정합**: `labelers/vi` ↔ `augmenters/wikiann_vi` +
  `augmenters/pii` ↔ `classifier/data_utils` 의 canonical 10종 라벨 집합 일치,
  BIO ↔ char-span 변환, `[start,end)` 반열림 계약 일관성. **PII 5종 주입 경로가
  NER 5종과 한 계약(`pii_all.jsonl`)으로 합류**하는지 확인.
- [x] **B2. 리포트 수치 ↔ results 무결성 + 천장 확인**: VI 리포트 4종
  (`vietnamese-bert-classifier-benchmark` / `-ner-benchmark` / `-ner-pii-benchmark`
  / `-ner-silver-quality`)의 인용 표 ↔ `results/classifier/vi/*.json` 실측 대조.
  per-entity P/R/F1 천장을 results 실측으로 단일 출처에 정리(**게이트 미설정**).
- [x] **C. llm_eval VI 재점검**: offset-span 러너·report 메트릭 정의가 classifier
  span F1과 비교 가능한지, `labelers/vi`(vllm/openai) 프롬프트·파싱이 canonical
  10종 평면과 일치하는지.
- [x] **D. stale 참조 정리**: 삭제된 `handoff-post-issue-40-classifier-followups.md`
  dangling reference 정리 — `vietnamese-bert-classifier-benchmark.md`(3곳),
  `issue-45-ja-classifier-external-corpus.md`(2곳).
- [x] **V. 검증**: 워크트리 `uv sync` → `pytest tests/ner/` 통과 + 변경 `.py`
  `ruff check` clean → `refuter` 게이트로 diff 반증(PASS).

## 성공 기준

- VI 체인 라벨·오프셋 계약 일관(코드 점검 통과).
- VI 리포트 수치 = `results/*.json` 실측 일치(불일치는 수정 또는 명시).
- per-entity P/R/F1 천장이 results 실측으로 단일 출처에 정리됨(게이트 미설정).
- llm_eval VI 메트릭 정의가 classifier span F1과 비교 가능함을 재확인.
- dangling reference 0건.
- `pytest tests/ner/` 통과, `ruff check` clean.

## 결정 로그 (append-only)

- 2026-06-11: 범위 = VI 전체 체인 정합성 감사. #95(JA)의 VI 대응, 단일 이슈
  (출하 아티팩트 채택 자식 A 미적용).
- 2026-06-11: 게이트 중립 확정 — 0.95를 고정 목표로 다루지 않고, P/R/F1 천장만
  확인. 게이트 수치는 확인 후 유동적 별도 설정.
- 2026-06-11: 베이스 = c251f40(in-flight HEAD). develop 머지 후 rebase, PR은 그
  시점 develop 대상.

## 해결된 질문 (resolved)

- B2 천장 단일 출처 = **기존 `vietnamese-bert-classifier-benchmark.md` 갱신**.
  VI 미출하라 spec.md 신설은 과잉 — 기존 영구 인용 표를 results 실측에 맞추고
  게이트 표기 제거. (2026-06-11 결정)
- D 정리: `issue-45`(JA 문서)의 VI 핸드오프 dangling ref 2곳도 **본 #98에서 함께
  정리**(benchmark 3곳 + issue-45 2곳 = dangling ref 0건 일괄 달성). (2026-06-11 결정)

## 구현 결과

### 정합 확인 (변경 불필요 — 클린)

- **핵심 학습 계약 일관**: classifier `data/wikiann_vi/pii_all.jsonl`(실측) =
  `{text, entities:[{label,start_char,end_char,text}], id}`, canonical 10종,
  half-open `[start,end)`. wikiann_vi(NER 5종 재라벨) → augmenters/pii(PII 5종
  주입, `--source jsonl`/`load_jsonl`) → `pii_all.jsonl` → classifier
  `data_utils.load_jsonl`(미지 라벨 reject) 까지 한 계약으로 합류 확인.
- **B2 측정 무결성 클린**: `vietnamese-bert-classifier-benchmark.md` 의 모든 표
  (baseline overall + per-entity 10종 / sweep 5모델 / large·mmBERT per-entity /
  v2·v3·v4 + v4 per-entity)가 `results/classifier/vi/*.json` 실측과 **정확 일치**.
  천장: production baseline 0.8985, 최선 변종 v4 0.9147(80/20).
- **C1 메트릭 비교가능성**: llm_eval `benchmark_runner.py` ↔ classifier
  `train_eval.py` 가 동일 `ner.metrics.span_metrics.compute_offset_span_f1` 사용 →
  VI LLM span F1 ↔ BERT 분류기 span F1 직접 비교 가능.

### 근본 원인 (수렴)

발견된 불일치는 모두 한 뿌리 — **VI eval 이 JA식 `offset_span`+`entities`
(`load_local`) 경로로 마이그레이션됐는데, 구 BIO/recall-merge 경로와 그 문서가
dead 로 잔존**. 코드 *버그*는 없음.

### 수정 (FINDING별)

- **FINDING-A (code 수리)** — `labelers/vi/dataset_loader.py`:
  `DEFAULT_PATH` 를 실재 파일 `{train,valid,test}.jsonl` 로, `load(split=)` 의
  기본 경로를 entities 스키마(`load_local`)로 분기. 명시 `path`/`span_key` 로 읽는
  recall-merge 덤프 reader(`_read_records`)는 보존. `--local-file` 없이도 VI eval
  기본 동작 복구. 회귀 테스트 `test_default_split_reads_entities` 추가.
- **FINDING-B (doc)** — `vietnamese-ner.md §1`: silver 흐름도에 merge_confidence
  산출(`gold_spans_8type_merged`)·entities 변환·PII 주입 단계 반영.
- **FINDING-C (doc+comment)** — VI eval 은 `offset_span`(BIO 아님): `vietnamese-
  ner.md §6.2/§6.4/§1`, `benchmark_runner.py:44·66` 주석 정정.
- **D (dangling)** — 삭제된 handoff 참조 5곳 정리(benchmark 3 + issue-45 2).
  benchmark 리포트는 동시에 **게이트 중립화**(0.95 컬럼·gate 행·프레이밍 제거,
  P/R/F1 전부 보존, 천장만 기술; F2~F6 ★ 로드맵은 개선 레버 불릿으로 축약).
- **NIT** — `labelers/vi DEFAULT_ENTITY_TYPES` 순서 비-canonical(무영향)은 미수정.

### 의도적 비변경

- 역사 문서(`issue-23/30`, `data-regen-canonical10-2026-04.md`,
  `vietnamese-ner-silver-quality.md`)의 `vi_wikiann_recall_*` 언급은 과거 빌드
  스냅샷이라 보존.
- LLM 벤치 3종(`vietnamese-ner-benchmark`/`-pii-benchmark`/`-silver-quality`)은
  `results/issue-8·issue-30` 가 로컬 부재(gitignore) → 수치 재검증 불가, 표를
  영구 단일 출처로 유지(변경 없음).

## 검증

- `pytest tests/ner/` 전체 **309 passed**(워크트리 src). 변경 `.py` `ruff check`
  clean. 삭제 handoff dangling ref **0건**.
- 워크트리 env 주의: 셸 `PYTHONPATH=/work/git/ner_pipeline/src/` 가 워크트리
  editable install 을 가리므로, 검증은 `env -u PYTHONPATH -u VIRTUAL_ENV uv run`
  으로 워크트리 src 를 강제해 수행.
