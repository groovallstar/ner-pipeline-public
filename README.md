# NER Pipeline

한국어·일본어·베트남어·영어의 개체명을 인식하는 다국어 NER 파이프라인입니다.
vLLM 기반 LLM 라벨링·평가, 합성 개인정보(PII) 증강, BERT 계열 토큰 분류기
학습·평가와 FastAPI 기반 REST API 서빙을 제공합니다.

## 지원 언어와 라벨

네 언어는 다음 **canonical 10종 라벨**을 공유합니다.

| 구분 | 라벨 |
|---|---|
| 개체명 | `PER`, `LOC`, `ORG`, `PROD`, `EVT` |
| 날짜 | `DAT` |
| 개인정보 | `EMAIL`, `PHONE`, `ID_NUM`, `CREDIT_CARD` |

라벨의 적용 범위와 경계에는 언어별 차이가 있습니다. 예를 들어 한국어의
`ORG`는 정부·행정·공공·정치 기관을 중심으로 제한됩니다. 정의·매핑·경계 규칙의
기준은 [canonical 엔티티 스키마](docs/manual/data/canonical-entity-schema.md)입니다.

| 언어 | 원천 데이터셋 | 학습 코퍼스 구성 방식 |
|---|---|---|
| 한국어 (`ko`) | KLUE NER | `PROD/EVT` 재라벨과 언어별 기준 정제, PII 4종 합성 주입 |
| 일본어 (`ja`) | Stockmark NER Wikipedia | 원천 NER에 `DAT`을 포함한 PII 5종 합성 주입 |
| 베트남어 (`vi`) | WikiANN-vi | NER 3종을 5종으로 silver 재라벨한 뒤 PII 5종 합성 주입 |
| 영어 (`en`) | OntoNotes5 (tner 판) | 원천 라벨을 canonical로 매핑한 뒤 PII 4종 합성 주입 |

한국어와 영어는 원천의 날짜 라벨을 사용합니다. 일본어와 베트남어의 학습
코퍼스는 합성 주입한 날짜를 사용하므로 자연 날짜 표현의 주석 범위가 다릅니다.

## 설치

Python 3.13 이상과 `uv`가 필요합니다. 저장소 루트에서 실행합니다.

```bash
uv sync
```

런타임과 개발 의존성(`pytest`, `ruff` 등)을 함께 설치합니다. PyTorch는
`pyproject.toml`에 지정된 CUDA 13.0용 인덱스를 사용합니다.
개발은 호스트에서 진행하며, Docker는 외부 추론 서비스와 서버 배포에 사용합니다.

## REST API 빠른 시작

NER 서버에는 학습된 모델과 토크나이저가 별도로 필요합니다. `uv sync`는 배포
모델을 설치하지 않습니다. 기본 모델 루트는 `/data/ner`이며, 언어별 배치는
다음과 같습니다.

```text
/data/ner/
└── {ja,ko,vi,en}/
    ├── model/             # 학습된 HuggingFace 모델과 토크나이저
    └── thresholds.json    # 선택 사항: 라벨별 신뢰도 임계값
```

```bash
uv run python -m server
```

기본 주소는 `0.0.0.0:8008`입니다. 모델 경로나 바인딩 주소는 다음처럼 바꿀 수 있습니다.

```bash
uv run python -m server --model-root /path/to/ner --host 127.0.0.1 --port 8008
```

서버가 시작되면 모델 로드 상태를 확인한 뒤 요청을 보냅니다.

```bash
curl -s http://localhost:8008/health

curl -s http://localhost:8008/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"홍길동은 서울에 살고 있다.","lang":"ko"}'

curl -s http://localhost:8008/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"texts":["東京は晴れです。","Hà Nội là thủ đô.","Barack Obama visited Hawaii."]}'
```

일부 모델을 로드하지 못해도 서버는 시작되지만, 해당 언어가 필요한 요청은
HTTP 503을 반환합니다. `NER_SERVER_API_KEY`를 설정했다면 추론 요청에
`x-api-key` 헤더를 추가해야 합니다.

| 엔드포인트 | 기능 |
|---|---|
| `POST /v1/ner` | 단일 `text` 또는 배치 `texts`를 추론합니다. |
| `GET /health` | 언어별 모델 로드 상태를 확인합니다. |
| `GET /` | 웹 데모 UI를 제공합니다. |
| `POST /v1/translate` · `GET /v1/translate/status` | 웹 데모의 한국어 번역과 상태 조회를 제공합니다. 번역은 기본 비활성입니다. |

`lang`을 생략하면 각 텍스트에서 가나 → 한글 → 베트남어 고유 부호 → 라틴 문자
순서로 신호를 검사합니다. 앞의 조건에 해당하지 않는 라틴 문자 입력은 `en`으로
처리하므로 실제 언어와 다를 수 있습니다. 언어를 알고 있다면 `lang`을 지정합니다.
감지 신호가 없으면 HTTP 200과 `{"lang":"unsupported","entities":[]}`를 반환합니다.
자세한 규칙과 한계는 [언어 자동 감지](docs/manual/language-detection.md)를 참고합니다.

응답 엔티티는 `label`, `start_char`, `end_char`, `text`를 포함합니다.
문자 오프셋은 원문 기준이며 `end_char`는 포함하지 않습니다.
OpenAPI에는 `/v1/ner`만 노출하며, `/docs`와 `/openapi.json`에서 확인할 수 있습니다.

요청·응답 계약은 [REST API 명세](docs/manual/rest-api-spec.md), 연동 예시는
[연동 가이드](docs/manual/rest-api-integration-guide.md), 환경변수와 운영 설정은
[서버 구현 문서](docs/manual/server-implementation.md)를 참고합니다.

## 학습·평가 파이프라인

전체 과정은 라벨링·평가 → 데이터 증강 → 검증 → 분류기 학습·평가로 구성됩니다.
언어별 원천 데이터에 따라 필요한 단계가 다릅니다.

| 단계 | 현재 제공하는 기능 | 상세 문서 |
|---|---|---|
| 라벨링·평가 | vLLM의 OpenAI 호환 API를 호출하는 언어별 라벨러와 벤치마크 | [라벨링](docs/manual/pipeline/1-labeling.md) |
| 증강 | 언어별 합성 PII 생성과 문맥 주입 | [증강](docs/manual/pipeline/2-augmentation.md) |
| 검증 | 주입 PII 교차검증과 분류 평가의 원문 누출 검사 | [검증](docs/manual/pipeline/3-verification.md) |
| 분류 | BERT 계열 토큰 분류기 학습과 span F1 평가 | [분류](docs/manual/pipeline/4-classification.md) |

영어 코퍼스는 원천에 `PRODUCT`, `WORK_OF_ART`, `EVENT`가 있어 NER 재라벨링을
거치지 않았습니다. 영어 LLM 라벨러는 PII 주입 결과의 교차검증에 사용합니다.
베트남어 재라벨·silver 검증 도구와 영어 원천 변환·그룹 복원 도구는 코퍼스 구축 후
삭제되었습니다. 위의 코퍼스 구성 방식은 데이터 생성 이력이며, 모든 준비 단계를
현재 저장소에서 다시 실행할 수 있다는 뜻은 아닙니다.

LLM 라벨링 백엔드는 vLLM이며, 비교용 HuggingFace BERT 베이스라인은 `hf:`
접두사로 선택할 수 있습니다. NER API는 학습된 분류기를 직접 로드하므로
추론에 vLLM을 요구하지 않습니다. 웹 데모 번역은 별도의 OpenAI 호환 엔드포인트를
사용합니다.

단계 간 데이터는 문자 오프셋 span을 사용합니다. 단계별 코드와 언어별 차이는
[파이프라인 구현 맵](docs/manual/pipeline/README.md)에 정리되어 있습니다.

## 프로젝트 구조

```text
src/ner/
├── labelers/{ko,ja,vi,en}/ # 언어별 LLM 라벨러
├── llm_eval/              # 벤치마크 실행과 리포트
├── augmenters/            # 합성 PII 생성·주입·검증
├── classifier/            # 토큰 분류기 학습·평가
├── metrics/               # span/BIO 메트릭
├── validity/              # K-fold 분산·비교타당성 게이트
└── scripts/               # 보조 스크립트
src/server/                # REST API와 웹 데모
docker/                    # 서버 배포와 외부 추론 서비스
```

- [`docker/server/`](docker/server/): REST API 컨테이너 배포 파일입니다.
- [`docker/vllm/`](docker/vllm/): 외부 LLM 추론 서비스 실행 파일입니다.
- [`tests/`](tests/): NER 모듈과 서버의 회귀 검사입니다.
- `results/`: Git에서 제외되는 임시 벤치마크 산출물 경로입니다.
- [`docs/`](docs/): 사용·운영 매뉴얼, 설계, 평가 보고서와 작업 기록입니다.

## 검증

```bash
uv run pytest tests/ -v -rs
uv run ruff check
```

`integration` 마커는 HuggingFace 데이터셋 캐시를, `live` 마커는 실제 서버와
배포 모델을 사용하는 테스트를 구분합니다. 이 밖에도 모델·데이터 유무에 따라
skip되는 테스트가 있으며, `-rs`로 사유를 확인할 수 있습니다.
Skip된 경로는 검증에 성공한 것으로 간주하지 않습니다.

## 관련 문서

- [개발·작업 지침](AGENTS.md)
- [Canonical 엔티티 스키마](docs/manual/data/canonical-entity-schema.md)
- [한국어 데이터·학습 경로](docs/manual/data/korean-ner.md)
- [웹 데모 번역](docs/manual/web-demo-translation.md)
- [평가 보고서와 수치 인용 기준](docs/reports/README.md)
