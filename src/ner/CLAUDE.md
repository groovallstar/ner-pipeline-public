# src/ — 소스 코드 가이드

## 모듈 구조

현재 주요 모듈 경로는 `src/ner/labelers/`이며, 새 모듈은 `src/` 하위에 추가한다.

### augmenters/

학습 데이터 증강 모듈. 상세: `src/ner/augmenters/AGENTS.md`.

| 서브모듈 | 역할 |
|---------|------|
| `pii/` | 합성 PII 주입기 — 기존 NER 데이터셋에 PII 엔티티를 삽입하고 span을 재계산하여 통합 학습 데이터셋 생성 (`python -m ner.augmenters.pii`) |

### labelers/

| 파일 | 역할 |
|------|------|
| `__init__.py` | 패키지 공개 API (DatasetLoader) |
| `dataset_loader.py` | HuggingFace datasets 로딩 (KLUE, KMounLP NER), `NERRecord` 반환 |
| `labeler_base.py` | 라벨러 공통 베이스 클래스 |
| `tag_aligner.py` | BIO 태그 정렬, 태그 정규화 (PER→PS 등), span 추출 유틸리티 |
| `hf_ner_labeler.py` | HuggingFace BERT 기반 NER 라벨러 (벤치마크 베이스라인) |
| `ko/` | 한국어 NER 라벨러 (vllm, openai) |
| `ja/` | 일본어 NER 라벨러 (vllm, openai) |

## NER 엔티티 타입

| 태그 | 의미 |
|------|------|
| PS | 인물 (Person) |
| LC | 장소 (Location) |
| OG | 기관 (Organization) |
| DT | 날짜 (Date) |
| TI | 시간 (Time) |
| QT | 수량 (Quantity) |

## 코딩 컨벤션

- `PYTHONPATH=/work/git/ner_pipeline/src/` — import는 `from ner.labelers.xxx import Xxx` 형태
- 패키지 관리자: **UV** (`uv pip install`)
- 타입 힌트 사용 (typing 모듈)
- 로깅: `logging` 표준 라이브러리
- **주석/문서 언어**: print/log/예외 메시지·argparse help는 영문, docstring·인라인 주석은 한국어 (상세: `docs/specs/coding-conventions.md`)
- 문자열 리터럴은 **홑따옴표(`'`)** 를 기본으로 쓴다 (escape 필요 시 `"` 허용)
- 한 줄은 **79자 이내**로 유지 (초과 시 줄바꿈으로 가독성 확보)
- 메서드·함수 사이는 **한 줄만 비운다** (연속 빈 줄 금지)
- Trailing whitespace 제거
