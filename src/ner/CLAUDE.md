# src/ner/ — 소스 코드 가이드

## 모듈 구조

패키지 루트: `src/ner/`. import는 `from ner.<모듈>.xxx import Xxx` 형태.

### labelers/

언어별 LLM NER 라벨러 + 공용 유틸. 상세: `src/ner/labelers/AGENTS.md`.

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

벤치마크 오케스트레이션 + 메트릭 + 리포트. 상세: `src/ner/llm_eval/AGENTS.md`.

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

학습 데이터 증강 모듈. 상세: `src/ner/augmenters/AGENTS.md`.

| 서브모듈 | 역할 |
|---------|------|
| `pii/` | 합성 PII 주입기 (suffix/llm 모드, vLLM 교차 검증). CLI: `python -m ner.augmenters.pii` |
| `wikiann_vi/` | WikiANN-vi → canonical 10종 평면 재라벨 + Wikidata 검증. CLI: `python -m ner.augmenters.wikiann_vi` |

### classifier/

JA·VI·KO canonical 10종 평면 BERT 토큰 분류 파인튜닝. 상세: `src/ner/classifier/AGENTS.md`.

| 파일 | 역할 |
|------|------|
| `__main__.py` | CLI: `python -m ner.classifier --lang {ja,vi,ko}` (`--group-key orig` 로 누출-free group K-fold) |
| `data_utils.py` | JSONL 로딩 (augmenters contract 소비) / fast(offset-trim)·PhoBERT(pyvi)·JA(slow) tokenizer 분기 정렬 / BIO ↔ char-span 변환 / 층화·group K-fold 분할 |
| `train_eval.py` | HF Trainer 래퍼 + char-offset span F1 (`src/ner/metrics` 공용) |
| `abstention.py` | per-class 신뢰도 임계값 fit·apply |
| `error_analysis.py` | 오류 유형 분류 CLI |
| `kfold_pool.py` | fold별 test 예측 pooled span F1 + 원문(orig) 단위 cross-fold 누출 검증 |

### metrics/

span/BIO 메트릭 공용 구현 (classifier·llm_eval 공유).

| 파일 | 역할 |
|------|------|
| `bio_metrics.py` | seqeval 기반 BIO 레벨 메트릭 |
| `span_metrics.py` | `compute_offset_span_f1` 등 span 레벨 메트릭 |

### scripts/

보조 스크립트 (`eval_ja_ner_test.py` 등).

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
