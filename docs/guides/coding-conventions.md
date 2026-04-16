# 코딩 컨벤션

이 프로젝트의 공통 코딩 규칙. 모든 신규 코드 및 수정에 적용.

## 주석 / 문서 언어 정책

| 위치 | 언어 |
|------|------|
| `print()` / `logger.*` 출력 메시지 | **영문** |
| Exception 메시지 | **영문** |
| 함수·메서드 docstring | **한국어** |
| 클래스 docstring | **한국어** |
| 모듈 상단 docstring (스크립트 헤더) | **한국어** |
| 인라인 코드 주석 (`#`) | **한국어** |
| CLI `--help` (argparse `description`, `help`) | **영문** |

### 이유

- 로그/예외/CLI는 외부 도구(grep, monitoring, 외부 라이브러리 traceback)와 섞이므로 ASCII로 유지.
- 주석/docstring은 팀 내부 문서이므로 한국어로 의도와 맥락을 명확히.

### 예시

```python
"""한국어 NER 라벨링 결과를 평가한다.

벤치마크 입력은 KLUE 데이터셋이며, span F1과 seqeval 메트릭을 함께 계산한다.
"""

def compute_f1(gold: List[str], pred: List[str]) -> float:
    """gold/pred 태그 시퀀스의 F1 점수를 계산한다."""
    if not gold:
        raise ValueError("gold tags are empty")  # 예외는 영문
    logger.info("Computing F1 over %d samples", len(gold))  # 로그는 영문
    # 빈 태그는 'O'로 정규화한다  ← 인라인 주석은 한국어
    ...
```

## 기타 규칙 (src/CLAUDE.md 참조)

- 문자열 리터럴: 홑따옴표(`'`) 우선, escape 필요 시 `"` 허용
- 한 줄 79자 이내
- 함수·메서드 사이는 한 줄만 비움
- Trailing whitespace 제거
- 타입 힌트 사용 (`typing` 모듈)
