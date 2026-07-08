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

## 코드·주석에 이슈/PR 번호·날짜 금지

코드 본문, 주석, docstring, 로그·예외 메시지, 변수·함수·파일 이름 어디에도
GitHub Issue 번호(`#42`, `issue-42`), PR 번호, 날짜(`2026-05-04`), 사람 이름,
"방금 고친 버그" 류 일시적 맥락을 남기지 않는다.

### 이유

- 코드는 머지 후에도 수년간 살아남지만 이슈·PR·날짜는 회수되거나 의미가 바뀐다 — 주석이 거짓말을 시작한다.
- 영구 보관이 필요한 맥락은 아래 채널이 이미 존재한다:
  - **왜 이렇게 됐는가**: 커밋 메시지 본문 + `refs #N`
  - **언제·누가**: `git blame` / `git log`
  - **설계·검증 히스토리**: `docs/issues/issue-{N}-{slug}.md`
  - **자유 형식 실험 기록**: `docs/reports/`
- 코드에 박힌 이슈 번호는 grep 결과를 오염시키고, 리네이밍·아카이브 후 dead link가 된다.

### 금지 예시

```python
# ❌ 이슈 번호 박제
def load_dataset():  # issue #40 에서 추가
    ...

# ❌ 날짜 박제
# 2026-05-04: PII 라벨 영문화 후 재학습
LABELS = [...]

# ❌ 일시적 맥락
# TODO(승현): 다음 스프린트에 제거
legacy_path = ...
```

### 허용 예시

```python
def load_dataset():
    """canonical 10종 평면 BIO 시퀀스를 JSONL에서 읽는다."""
    ...

# WikiANN의 LOC는 GPE를 포함하므로 LOC로 정규화한다  ← WHY만 남김
LABELS = [...]
```

장기 마커가 필요한 미해결 작업은 GitHub Issue로 분리하고 코드에는 이슈 번호 없이 의미만 남긴다.

## docstring·주석의 참조 무결성

docstring·인라인 주석·모듈 헤더는 **실재하는 대상만** 가리킨다. 존재하지
않거나 이름이 바뀐 파일·경로·데이터셋·CLI 플래그, 이미 폐기된 계획·측정
방식을 참조로 남기지 않는다. 코드가 참조 대상을 바꾸면(파일 삭제·rename,
기능 구현·폐기, 플래그 추가·제거) 그것을 가리키는 docstring·주석을 **같은
커밋에서** 동기화한다.

### 이유

- 죽은 참조는 grep·문서 탐색을 오염시키고, 폐기된 계획이 docstring에 남으면
  나중에 그대로 되살릴 유혹(attractive nuisance)이 된다.
- 검증이 여러 모듈에서 반복 적발한 재발 결함이다 — 파일 rename·기능 구현
  후 그것을 가리키던 docstring만 stale 로 남는 사례.

### 예시

```python
# ❌ 존재하지 않는/이름 바뀐 대상 참조
def build_eval():
    """pii_all.jsonl 을 읽어 평가셋을 만든다."""  # 파일은 rename 됐음
    load(CLEANED_PATH)

# ❌ 폐기된 계획·없는 플래그를 docstring 에 존속
"""측정은 A->B->C 3지점으로 완성한다(`--mode seq`)."""  # 계획 폐기·플래그 없음

# ✅ 참조 대상과 동기화
def build_eval():
    """정제된 gold(CLEANED_PATH)를 읽어 평가셋을 만든다."""
    load(CLEANED_PATH)
```

## 기타 규칙

- 한 줄 79자 이내
- 함수·메서드 사이는 한 줄만 비움
- Trailing whitespace 제거
- 타입 힌트 사용 (`typing` 모듈)
