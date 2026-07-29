# src/ner/ — 소스 코드 가이드

## 모듈 구조

패키지 루트: `src/ner/`. import는 `from ner.<모듈>.xxx import Xxx` 형태.

### labelers/

언어별 LLM NER 라벨러 + 공용 유틸. 상세: `src/ner/labelers/CLAUDE.md`.

| 파일/서브모듈 | 역할 |
|---------|------|
| `labeler_base.py` | 라벨러 공통 베이스 클래스, `parse_json_response()` |
| `dataset_loader.py` | HuggingFace datasets 로딩 (`NERRecord` 반환) |
| `tag_aligner.py` | BIO 태그 정렬·정규화·span 추출 유틸리티 |
| `hf_ner_labeler.py` | HuggingFace BERT 기반 NER 라벨러 (벤치마크 베이스라인) |
| `ko/` | 한국어 NER 라벨러 (vllm, openai) — canonical NER 5종 (PER/LOC/ORG/PROD/EVT) + DAT, KLUE 유래·TI/QT 드롭 (PROD/EVT는 LLM 재라벨 증분) |
| `ja/` | 일본어 NER 라벨러 (vllm, openai) — canonical 10종 평면 |
| `vi/` | 베트남어 NER 라벨러 (vllm, openai) — canonical 10종 평면 |

### llm_eval/

벤치마크 오케스트레이션 + 메트릭 + 리포트. 상세: `src/ner/llm_eval/CLAUDE.md`.

| 파일 | 역할 |
|------|------|
| `__main__.py` | CLI: `python -m ner.llm_eval --lang {ko,ja,vi}` |
| `benchmark_runner.py` | KO BIO / JA·VI offset-span 공용 러너 |
| `report.py` | span-match·seqeval·per-entity 리포트 |
| `error_analysis.py` | 문장별 오류 유형 분류 CLI |
| `span_evaluator.py` / `span_evaluator_cli.py` | span 단위 평가 유틸 |
| `vi_silver_quality.py` | silver vs gold 비교 |
| `wikiann_vi_gold.py` | WikiANN-vi gold 기준 평가 |

### augmenters/

학습 데이터 증강 모듈. 상세: `src/ner/augmenters/CLAUDE.md`.

| 서브모듈 | 역할 |
|---------|------|
| `pii/` | 합성 PII 주입기 (suffix/llm 모드, vLLM 교차 검증). CLI: `python -m ner.augmenters.pii` |
| `wikiann_vi/` | WikiANN-vi(3종 BIO) → canonical **NER 5종** 재라벨 + Wikidata 검증 (PII 5종은 `pii/` 가 별도 주입해 10종 평면이 된다). CLI: `python -m ner.augmenters.wikiann_vi` |

### classifier/

JA·VI·KO canonical 10종 평면 BERT 토큰 분류 파인튜닝. 상세: `src/ner/classifier/CLAUDE.md`.

| 파일 | 역할 |
|------|------|
| `__main__.py` | CLI: `python -m ner.classifier --lang {ja,vi,ko}` (`--group-key orig` 로 누출-free group K-fold) |
| `data_utils.py` | JSONL 로딩 (augmenters contract 소비) / fast(offset-trim)·PhoBERT(pyvi)·JA(slow) tokenizer 분기 정렬 / BIO ↔ char-span 변환 / 층화·group K-fold 분할 |
| `train_eval.py` | HF Trainer 래퍼 + char-offset span F1 (`src/ner/metrics` 공용) |
| `confidence_threshold.py` | per-class 신뢰도 임계값 fit·apply |
| `error_analysis.py` | 오류 유형 분류 CLI |
| `kfold_pool.py` | fold별 test 예측 pooled span F1 + 원문(orig) 단위 cross-fold 누출 검증 |

### metrics/

span/BIO 메트릭 공용 구현 (classifier·llm_eval 공유).

| 파일 | 역할 |
|------|------|
| `bio_metrics.py` | `MetricsCalculator` — static 메서드 3종의 채점 경로. `compute_seqeval`(seqeval 기반 BIO 레벨) · `compute_span_f1`(BIO → char-offset span 추출 후 exact match, KLUE 방식) · `compute_span_match`(BIO 없이 텍스트 span 직접 비교, exact + 포함관계 relaxed). 뒤 둘은 seqeval 을 쓰지 않는 별도 알고리즘이다 |
| `span_metrics.py` | `compute_offset_span_f1`(strict — `(start, end, type)` 정확 매칭) · `compute_offset_span_f1_relaxed`(SemEval'13 Partial 매칭) |

### validity/

K-fold 실험 비교 유효성 게이트 (재학습 0회). `fold*/metrics.json`·`pooled_metrics.json` 만 읽어 두 실험을 비교해도 되는지·개선이 노이즈인지 실측인지 판정한다. `metrics/`(한 run 의 F1 계산)를 한 줄도 import 하지 않는 다른 고도의 관심사라 최상위 패키지로 분리했다. 상세: `src/ner/validity/CLAUDE.md`. CLI: `python -m ner.validity {std,repro,compare}`.

| 파일 | 역할 |
|------|------|
| `comparability.py` | 같은 자(`RULER=lang·data_fingerprint·kfold·group_key·seed·stratify`)로 쟀나 — 내용 지문이 경로를 대체해 gold 를 고치면 비교 거부 |
| `leakage.py` | pooled cross-fold 누출 카운터 + 근거(`leak_check_basis`) — 신뢰 근거(group·orig)로 센 0 만 통과, 관측된 누출은 근거 약해도 FAIL |
| `variance.py` | σ_fold·σ_repro·paired Δ·방향 일관성(조이기 전용·false PASS 불가) — 노이즈 밴드 재료 |
| `gate.py` | `compare` — 세 축 통합 verdict(`PASS/FAIL/INVALID/INCONCLUSIVE`) |
| `_common.py` | fold/pooled IO 공유 |
| `__main__.py` | CLI: `std`·`repro`·`compare` 세 서브커맨드를 각 판정 함수로 dispatch 후 결과 JSON 을 stdout(·`--out` 파일)으로 |
| `__init__.py` | 공개 API re-export (`from ner.validity import compare`) |

세 검증은 각자 독립 함수로 호출 가능하다. 어떤 검증 부분집합을 통과해야 '실험 완료'인지의 조합·우선순위·선언은 채점규칙(자)을 바꾸는 별도 하네스의 몫이다 — 기준 파일을 건드리므로 커밋이 잠기고, 반박자가 누출·조작을 점검한다 (루트 `CLAUDE.md` §하네스).

### scripts/

배포 추론 스크립트 — JA·VI 각 한 벌(`eval_ja_ner_test.py`·`eval_vi_ner_test.py` + 동명 `.sh` uv 래퍼). 학습 없이 고정 test 와 저장된 임계값으로 추론·채점만 한다. 상세: `src/ner/scripts/CLAUDE.md`.

## 라벨 스키마

- **KO·JA·VI 공통**: canonical 10종 평면 = NER 5종(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`)
- **KO**: NER 5종 + `DAT` 은 KLUE 유래(`PROD/EVT` 는 LLM 재라벨 증분, TI/QT 드롭), PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`)은 합성 주입
- 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## 코딩 컨벤션

- `PYTHONPATH` **설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 자동 등록
- import: `from ner.labelers.xxx import Xxx` (src 접두어 없이)
- 패키지 관리자: **UV** (`uv sync` 또는 `uv pip install -e .`)
- 타입 힌트, `logging` 표준 라이브러리 사용
- **주석/문서 언어**: print/log/예외·argparse help는 영문, docstring·인라인 주석은 한국어
- 상세 규칙: `docs/specs/coding-conventions.md`
