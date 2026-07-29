# NER Pipeline

다국어 Named Entity Recognition(NER) 파이프라인. 멀티 백엔드 LLM 라벨링 + BERT 토큰 분류 + PII 증강.

## 지원 언어

| 언어 | 데이터셋 | 라벨 공간 |
|------|---------|----------|
| 한국어 | KLUE NER (PROD/EVT LLM 재라벨 + PII 합성 주입) | canonical 10종 평면 |
| 일본어 | Stockmark NER Wikipedia (PII 합성 주입) | canonical 10종 평면 |
| 베트남어 | WikiANN-vi (silver 재라벨 + PII 합성 주입) | canonical 10종 평면 |

세 언어 공통 **canonical 10종 평면** = NER 5종(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`). 한국어는 `DAT`가 KLUE 유래, `PROD/EVT`는 KLUE 문장 LLM 재라벨 증분이며 KLUE `TI/QT`는 드롭, PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`)은 합성 주입이다.

라벨 정의·매핑·경계 규칙: [`docs/manual/data/canonical-entity-schema.md`](docs/manual/data/canonical-entity-schema.md)

## 파이프라인

원천 데이터가 네 단계를 거쳐 학습된 BERT 분류기가 된다.

```mermaid
flowchart LR
    L["1 · 라벨링<br/>LLM으로 엔티티를 뽑아<br/>정답과 비교·채점"] --> A["2 · 증강<br/>PII를 문맥에 자연 주입<br/>(VI는 3종→5종 재라벨도)"]
    A --> V["3 · 검증<br/>silver 품질·원문 누출·<br/>PII 교차검증"]
    V --> C["4 · 분류<br/>BERT 파인튜닝 +<br/>span F1 채점"]
```

단계별 상세: [`docs/manual/pipeline/`](docs/manual/pipeline/)

## 백엔드

vLLM (로컬 GPU), OpenAI, HuggingFace BERT baseline.

## 프로젝트 구조

```
src/ner/
├── labelers/{ko,ja,vi}/   # 언어별 LLM 라벨러
├── llm_eval/              # 벤치마크 오케스트레이션·리포트
├── augmenters/{pii,wikiann_vi}/  # 학습 데이터 증강
├── classifier/            # BERT 토큰 분류 파인튜닝
├── metrics/               # span/BIO 메트릭 공용 구현
├── validity/              # K-fold 분산·비교타당성 게이트 (재학습 0회)
└── scripts/               # 보조 스크립트
src/server/                # ja·vi NER REST API 서비스 (FastAPI)
docker/{server,vllm}/      # REST API 배포 + vLLM
results/                   # 벤치마크 산출물 scratch (gitignore·휘발)
certified/                 # 커밋된 결과 원장 — 인용 근거 metric JSON (숫자 검사 기준)
tests/{ner,server}/        # pytest 테스트
docs/                      # manual·reports·issues·wiki·specs
```

상세 가이드: [`CLAUDE.md`](CLAUDE.md), 모듈 레퍼런스: [`docs/manual/`](docs/manual/)

## 설치

```bash
uv sync                # 또는: uv pip install -e .
```

## 사용법

```bash
# LLM NER 벤치마크
python -m ner.llm_eval --lang {ko,ja,vi} \
    --models "vllm:<model>" --max-samples 200 \
    --vllm-url "http://localhost:8081/v1"

# BERT 파인튜닝·평가 (canonical 10종 평면) — --group-key 는 필수
python -m ner.classifier --lang ja --group-key id     # 일본어
python -m ner.classifier --lang vi --group-key orig   # 베트남어
python -m ner.classifier --lang ko --group-key id     # 한국어
python -m ner.classifier --lang vi --kfold 10 --fold-index 0 --group-key orig  # 원문 단위 누출-free K-fold

# 실험 비교 유효성 게이트 (재학습 0회 — 기존 fold 산출물만 읽는다)
python -m ner.validity std --run <run_dir>
python -m ner.validity compare --baseline <base_dir> --candidate <cand_dir> --target ORG

# 데이터 증강
python -m ner.augmenters.pii          # 합성 PII 주입
python -m ner.augmenters.wikiann_vi   # WikiANN-vi NER 5종 재라벨
```

각 CLI 의 전체 옵션은 `--help` 참조.

## REST API 서비스

ja·vi NER 추론을 FastAPI 로 서빙한다 (`POST /v1/ner` 단일·배치,
`GET /health`, `GET /` 웹 데모 UI). 언어 자동감지(가나→ja, vi 전용 결합부호→vi,
그 외 unsupported), 동시성 세마포어(과부하 429), fp32 배치 추론을 지원한다 —
fp32 고정이라 배치화가 단건 결과를 바꾸지 않는다. 모든 에러는 구조화 봉투
(`{error:{status,message}}`) 하나로 통일돼 있어 소비자는 파싱 경로를 하나만
둔다. 웹 데모에는 한국어 번역 글로스가 있다(기본 비활성 additive 기능 —
PII 는 마스킹-복원으로 원문 보존, 고유명사는 한글 음차).

```bash
python -m server   # uvicorn 기동
```

컨테이너 배포는 `docker/server/`, 상세는
[`src/server/CLAUDE.md`](src/server/CLAUDE.md) 참조.

## 테스트

```bash
uv run pytest tests/ -v
```

## 벤치마크 리포트

[`docs/reports/`](docs/reports/) — 언어·실험별 최신 측정치.

## License

Private repository.
