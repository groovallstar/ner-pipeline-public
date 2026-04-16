# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트를 이해하기 위한 가이드이다.

## 프로젝트 개요

**다국어 NER(Named Entity Recognition) 파이프라인** — LangChain 기반 멀티 백엔드 LLM 지원 + BERT 토큰 분류 파인튜닝 + PII 증강 + 크롤링 기반 학습 데이터 생성.

- 한국어: KLUE NER (6 엔티티: PS, LC, OG, DT, TI, QT)
- 일본어: Stockmark NER Wikipedia (8 엔티티) + 합성 PII 6종 통합 학습
- 베트남어: 3 엔티티 (PER, LOC, ORG)

## 개발 환경

- **호스트에서 직접 개발** (Docker는 Ollama·vLLM 등 외부 서비스 전용 — 개발 컨테이너에 진입하지 않는다)
- 패키지 관리자: **UV** (`uv sync` 또는 `uv pip install -e .` 으로 editable install)
- `PYTHONPATH` **설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 `sys.path`에 자동 등록한다
- import 형태: `from labelers.xxx import Xxx`, `from llm_eval.xxx import Xxx`, `from classifier.xxx import Xxx` 등 (src 접두어 없이)
- 상세 원리·설정: `docs/wiki/concepts/src-layout-packaging.md`

## 핵심 디렉토리 구조

```
src/
├── labelers/          # NER 라벨링 모듈 (상세: src/CLAUDE.md)
│   ├── ko/            # 한국어 (ollama, vllm, openai)
│   ├── ja/            # 일본어 (ollama, vllm, openai)
│   ├── vi/            # 베트남어 (ollama, vllm, openai)
│   ├── dataset_loader.py
│   ├── labeler_base.py
│   ├── tag_aligner.py     # BIO 태그 정렬/정규화/span 추출
│   └── hf_ner_labeler.py  # HuggingFace BERT NER 라벨러
├── llm_eval/          # 벤치마크 오케스트레이션 + 리포트 (상세: src/llm_eval/AGENTS.md)
│   ├── __main__.py          # CLI (python -m llm_eval --lang ja|ko ...)
│   ├── benchmark_runner.py  # BenchmarkRunner (한국어 BIO / 일본어 offset-span 공용 러너)
│   ├── report.py            # ReportGenerator (span-match, seqeval, per-entity 테이블)
│   ├── error_analysis.py    # 문장별 오류 유형 분류 CLI
│   └── span_evaluator.py / span_evaluator_cli.py
├── metrics/           # span/BIO 메트릭 공용 구현 (classifier·llm_eval 공유)
│   ├── bio_metrics.py       # seqeval 기반 BIO 레벨 메트릭
│   └── span_metrics.py      # compute_offset_span_f1 등 span 레벨 메트릭
├── augmenters/        # 학습 데이터 증강 (상세: src/augmenters/AGENTS.md)
│   ├── pii/           # 합성 PII 주입 (suffix/llm 모드, vLLM 교차 검증)
│   └── crawlers/ko/   # 한국어 Yonhap RSS 크롤러 + NER 태깅
├── classifier/        # BERT 토큰 분류 파인튜닝 (상세: src/classifier/AGENTS.md)
│   ├── data_utils.py       # Stockmark 로딩, wordpiece/sentencepiece 정렬
│   ├── train_eval.py       # HF Trainer, offset-span F1
│   └── pii_benchmark.py    # 7 PII 엔티티 BERT 파인튜닝
└── scripts/           # 보조 셸 스크립트 (eval_spans.sh 등)
docker/
├── dev/               # 개발 컨테이너 (상세: docker/CLAUDE.md)
├── ollama/            # Ollama 서비스
└── vllm/              # vLLM 서비스
results/               # 벤치마크 결과 JSON + 리포트
tests/                 # 테스트 (상세: tests/CLAUDE.md)
docs/                  # 문서 (상세: docs/wiki/schema.md)
│   ├── wiki/          # 프로젝트 독립적 도메인 지식
│   ├── entities/      # 언어·엔티티·데이터셋 스펙
│   ├── guides/        # 실행 가이드 + 코드 구현 맵
│   └── issues/        # 이슈별 계획/보고서 히스토리
```

## 주요 CLI 엔트리포인트

```bash
# LLM NER 벤치마크
python -m llm_eval --lang ja --models "vllm:Qwen/Qwen3.5-27B" --max-samples 200
python -m llm_eval --lang ko --models "vllm:Qwen/Qwen3.5-27B" --max-samples 500

# BERT 파인튜닝 & 평가 (classifier)
python -m classifier                        # Stockmark NER 기본 벤치마크
python -m classifier.pii_benchmark          # PII 합성 데이터 기반 BERT 학습

# PII 합성 주입 (augmenters.pii)
python -m augmenters.pii --source stockmark --lang ja \
    --output /data/ner/ja_stockmark_pii.jsonl --n-samples 1000 \
    --mode llm --vllm-url http://localhost:8081/v1 \
    --vllm-model Qwen/Qwen3.5-27B --verify vllm

# 뉴스 크롤러 + NER 태깅 (augmenters.crawlers)
python -m augmenters.crawlers.ko --source yna --max-sentences 100 \
    --output-dir data/ner/raw \
    --vllm-base-url http://localhost:8081/v1 --model Qwen/Qwen3.5-27B
```

## 개발 3원칙

- **원자 단위**: 한 번의 요청은 검증 가능한 최소 기능 단위로 처리한다.
- **검증 의무**: 테스트 통과 + `docs/` 반영 전까지 '완료'로 간주하지 않는다.
- **분해는 사람 주도**: 단일 단계로 검증이 어려우면 작업자에게 먼저 분해 방식을 묻는다.

## 코딩 컨벤션 (주석/문서 언어)

- `print()` / `logger.*` / 예외 메시지 / argparse `help`: **영문**
- 함수·클래스·모듈 docstring, 인라인 `#` 주석: **한국어**
- 문자열 리터럴: **홑따옴표(`'`) 기본**, escape 필요 시 `"` 허용
- 한 줄 **79자 이내**, 함수 사이는 **한 줄만** 비운다, trailing whitespace 금지
- 상세 규칙 및 예시: `docs/guides/coding-conventions.md`

## 커밋 컨벤션

- 제목·본문은 **한국어**로 작성
- `type` 접두사만 영문 허용 (`feat`, `fix`, `chore`, `docs`, `build`, `refactor`, `test`)

## Wiki 운영

`docs/wiki/` 운영 규칙·트리거 키워드·배치 기준은 `docwiki` 스킬과 `docs/wiki/schema.md`에 위임한다. 코드 변경 후에는 `docs/guides/` 하위의 관련 구현 맵(`base_labelers.md`, `bio-*`, `span-evaluator.md` 등)도 최신 상태인지 확인한다.

## 이슈 관리

- **GitHub Issues**(`groovallstar/ner_pipeline`)로 작업 단위를 관리한다. 자동 채번으로 중복을 방지한다.
- 마일스톤은 사용하지 않는다. 영역은 라벨로 구분한다 (`area:labelers`, `area:classifier`, `area:augmenters`, `area:llm-eval`, `area:infra`, `docs` 등).
- 이슈 등록: `gh issue create --title "제목" --body "설명" --label <area>` (`.github/ISSUE_TEMPLATE/` 템플릿 사용 권장)
- 브랜치명: `feat/issue-{번호}-{짧은-슬러그}` 예) `feat/issue-12-vi-crawler`
- 커밋 메시지: 기존 컨벤션(한국어 제목 + `type(스코프):` 접두사) 유지, 본문 끝에 `refs #12` 참조. 최종 PR 또는 마지막 커밋에는 `closes #12`로 이슈 종결.
- **계획·보고 문서는 GitHub Issue와 `docs/issues/` 양쪽에 모두 남긴다.** Issue는 실시간 협의·승인 기록, `docs/issues/`는 기능 설계·구현 히스토리의 영구 보관 용도.

### 이슈 문서 구조

이슈별로 `docs/issues/issue-{번호}-{슬러그}/` 디렉토리를 만든다. 상세 규칙과 템플릿은 `docs/issues/README.md` 참조.

```
docs/issues/issue-12-vi-crawler/
├── plan.md     # 구현 계획 (하위 작업 3~6개 체크박스)
└── report.md   # 최종 결과 보고서 (완료 후 작성)
```

- `plan.md`는 Issue 본문/댓글과 동일 내용을 복사·정리해 저장 (단일 출처는 Issue, `docs/issues/`는 스냅샷)
- `report.md`는 머지 직전 작성 — 변경 요약, 검증 결과, 관련 커밋/PR 링크 포함

### 이슈 진행 절차 (경량 5단계, 승인 1회)

1. **등록**: GitHub Issue 작성 — 목적, 성공 기준(테스트/메트릭), 범위 정리
2. **브랜치**: `develop`에서 `feat/issue-{N}-slug` 분기
3. **구현 계획 수립 → 승인 요청**: 하위 작업 3~6개를 Issue 본문의 체크박스로 분해하고, 동일 내용을 `docs/issues/issue-{N}-{slug}/plan.md`에 저장
4. **구현 & 원자 커밋**: 하위 작업 단위로 커밋, 각 커밋 본문에 `refs #N`
5. **마무리**: 테스트 통과 + `docs/` 갱신 확인 → `docs/issues/.../report.md` 작성 → PR 생성(제목 또는 본문에 `closes #N`) → 머지 전 `git status`로 미커밋 파일 확인
