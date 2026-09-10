# docs/reports/ — 자유 형식 벤치마크·실험 리포트

GitHub Issue에 매이지 않는 단발성 측정·비교 실험 결과를 **시점 스냅샷**으로 누적 보관한다.

이슈 단위 작업(plan/report)은 `docs/issues/`에, 장기 스펙·규칙은 `docs/specs/`에, 기술 개념·구조·실행 가이드는 `docs/manual/`에 둔다. 이 폴더는 **"언제 어떤 조건에서 어떤 숫자가 나왔는가"** 기록만 담는다.

## 지금 있는 리포트

| 무엇을 재는가 | 파일 |
|---|---|
| LLM 라벨링 품질 (언어별) | `japanese-ner-benchmark.md` · `japanese-ner-pii-benchmark.md` · `vietnamese-ner-benchmark.md` · `vietnamese-ner-pii-benchmark.md` |
| BERT 분류기 (언어별 요약) | `japanese-bert-classifier-benchmark.md` · `korean-bert-classifier-benchmark.md` · `vietnamese-bert-classifier-benchmark.md` · `english-bert-classifier-benchmark.md` |
| BERT 분류기 (엔티티별 진단) | `japanese-bert-classifier-per-entity-diagnosis.md` · `korean-bert-classifier-per-entity-diagnosis.md` |
| BERT 분류기 (동결 히스토리) | `japanese-bert-classifier-history.md` · `vietnamese-bert-classifier-history.md` |
| BERT 분류기 (출하 스펙) | `japanese-bert-classifier-spec.md` · `vietnamese-bert-classifier-spec.md` |
| 데이터 품질 | `vietnamese-ner-silver-quality.md` · `data-regen-canonical10-2026-04.md` |
| 서버 | `language-detection-benchmark.md` |

**언어마다 벌 수가 다르다.** JA 는 넷(요약·히스토리·진단·스펙), VI 는 셋(스펙
없는 진단 대신 silver 품질), KO 는 둘, EN 은 하나다. 나뉜 이유는 측정을 몇 번
반복했는지이지 정책이 아니다 — 여러 번 잰 언어일수록 동결분이 갈라져 나왔다.

## 파일 규칙

리포트 파일명에 `YYYY-MM`을 붙이는지 여부는 **파일이 앞으로 수정되는가, 아니면 동결되는가**로 결정한다. 같은 주제를 새 시점에 다시 측정해야 한다면 *덮어쓰지 않고 새 파일을 만들어 비교 가능하게 남기는 것*이 핵심이고, 그렇지 않으면 접미사는 노이즈가 된다.

- **상시 갱신형 리포트**: `<주제>.md` (시점 접미사 없음). 같은 주제·같은 평가 프레임에서 측정값을 갱신해 가는 경우. 본문 상단의 측정일·환경을 갱신과 함께 업데이트한다.

  > 지금 이 형태로 남은 LLM 라벨링 리포트 넷은 실제로는 갱신이 멈춰 있다 —
  > 그 뒤 작업이 분류기로 옮겨 갔기 때문이다. 다시 재는 일이 생기면 그때
  > 갱신하거나 `-YYYY-MM` 을 붙여 동결로 넘긴다.
- **시점 스냅샷 리포트**: `<주제>-<YYYY-MM>.md`. 모델·데이터·하이퍼파라미터를 바꿔 가며 같은 주제를 반복 측정할 가능성이 있는 경우, 또는 이슈 종료 시점의 일회성 비교·실험이라도 결과를 *동결*해 두고 싶은 경우. 예: `data-regen-canonical10-2026-04.md`. 한 번 작성한 뒤로는 **수정하지 않는다**. 새 측정이 필요하면 `YYYY-MM`이 다른 새 파일을 만든다.
- 두 형태 모두 측정 단위가 커서 여러 파일로 갈라지면 주제별 접미사를 붙인다 (`-pii`, `-baseline` 등).
- 어느 쪽인지 모호하면 **상시 갱신형으로 시작**하고, 두 번째 측정이 실제로 발생하는 시점에 기존 파일을 `<주제>-<YYYY-MM>.md`로 옮기면서 스냅샷 체계로 전환한다.

## 권장 상단 구조

```markdown
# <리포트 제목>

- 측정일: YYYY-MM-DD
- 대상: <모델/데이터셋/조건 요약>
- 실행 명령: <재현 가능한 CLI 혹은 커밋 해시>

## 요약
<3~5줄>

## 조건
- 하드웨어:
- 백엔드 구성:
- 샘플 수:

## 결과
<표/숫자>

## 해석
<요약 지표에 깔리지 않는 관찰·한계>
```

## 원시 데이터 취급

- 원시 벤치 JSON은 `results/`(gitignore·휘발 scratch)에 로컬 보관하고, 리포트에는 **재현 명령**과 **요약 지표**만 남긴다.
- 리포트가 **인용하는 수치**에는 원본 metric JSON의 경로 또는 커밋과 실행 조건을 명시한다. 원본을 보관하지 못하면 그 한계를 밝힌다. 수치를 손으로 만들거나 과거 증거를 덮어쓰지 않는다. 별도 원장 승격이나 옛 훅의 자동 대조는 사용하지 않는다.
- 합성/주입 데이터셋은 `data/`(gitignore 대상)에 로컬 보관. 생성 커맨드를 리포트 상단에 명시한다.

과거 리포트의 `certified/` 경로와 수치는 당시 실험의 기록이다. 현재 남은
JSON은 보관 자료이며 테스트·하네스의 검증 기준으로 사용하지 않는다.
삭제된 과거 원본은
[Git 기록](https://github.com/groovallstar/ner-pipeline/tree/b564d5e02402ba09fb8bc3babbecdc0e945326bf/certified)에서
확인할 수 있다. 새 결과를 이 디렉터리로 승격하거나 자동 출처 선언을 추가하지 않는다.
