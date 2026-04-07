# src/ — 소스 코드 가이드

## 모듈 구조

현재 주요 모듈 경로는 `src/labelers/`이며, 새 모듈은 `src/` 하위에 추가한다.

### labelers/

| 파일 | 역할 |
|------|------|
| `__init__.py` | 패키지 공개 API (DatasetLoader, OllamaNERLabeler) |
| `dataset_loader.py` | HuggingFace datasets 로딩 (KLUE, KMounLP NER), `NERRecord` 반환 |
| `labeler_base.py` | 라벨러 공통 베이스 클래스 |
| `ko/` | 한국어 NER 라벨러 (ollama, openai, vllm) |
| `ja/` | 일본어 NER 라벨러 (ollama, openai, vllm) |

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

- `PYTHONPATH=/work/git/ner_pipeline/src/` — import는 `from labelers.xxx import Xxx` 형태
- 패키지 관리자: **UV** (`uv pip install`)
- 타입 힌트 사용 (typing 모듈)
- 로깅: `logging` 표준 라이브러리
