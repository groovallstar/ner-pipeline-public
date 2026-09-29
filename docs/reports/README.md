# docs/reports/ — 자유 형식 벤치마크·실험 리포트

GitHub Issue에 매이지 않는 단발성 측정·비교 실험 결과를 **시점 스냅샷**으로 누적 보관한다.

이슈 단위 작업(plan/report)은 `docs/issues/`에, 장기 스펙·규칙은 `docs/specs/`에, 기술 개념·구조·실행 가이드는 `docs/manual/`에 둔다. 이 폴더는 **"언제 어떤 조건에서 어떤 숫자가 나왔는가"** 기록만 담는다.

리포트 작성 기준은 [AGENTS.md](AGENTS.md)를 참조한다.

## 지금 있는 리포트

| 무엇을 재는가 | 파일 |
|---|---|
| LLM 라벨링 품질 (언어별) | `japanese-ner-benchmark.md` · `japanese-ner-pii-benchmark.md` · `vietnamese-ner-benchmark.md` · `vietnamese-ner-pii-benchmark.md` |
| BERT 분류기 (언어별 요약) | `japanese-bert-classifier-benchmark.md` · `korean-bert-classifier-benchmark.md` · `vietnamese-bert-classifier-benchmark.md` · `english-bert-classifier-benchmark.md` |
| BERT 분류기 (엔티티별 진단) | `japanese-bert-classifier-per-entity-diagnosis.md` · `korean-bert-classifier-per-entity-diagnosis.md` |
| BERT 분류기 (동결 히스토리) | `japanese-bert-classifier-history.md` · `vietnamese-bert-classifier-history.md` |
| BERT 분류기 (출하 스펙) | `japanese-bert-classifier-spec.md` · `vietnamese-bert-classifier-spec.md` |
| 데이터 품질 | `vietnamese-ner-silver-quality.md` · `data-regen-canonical10-2026-04.md` |
| 서버 | `language-detection-benchmark.md` · [NER API 기능·부하 테스트](ner-api-functional-load-test.md) |

**언어마다 벌 수가 다르다.** JA 는 넷(요약·히스토리·진단·스펙), VI 는 셋(스펙
없는 진단 대신 silver 품질), KO 는 둘, EN 은 하나다. 나뉜 이유는 측정을 몇 번
반복했는지이지 정책이 아니다 — 여러 번 잰 언어일수록 동결분이 갈라져 나왔다.

시점 접미사가 없는 LLM 라벨링 리포트 넷은 그 뒤 작업이 분류기로 옮겨
가면서 갱신이 멈췄다. 다시 측정할 때는 [파일 규칙](AGENTS.md#파일-규칙)에
따라 갱신하거나 동결한다.

과거 리포트의 `certified/` 경로와 수치는 당시 실험의 기록이다. 그 디렉터리는
저장소에서 삭제했으므로 지금 체크아웃에는 없고, 원본 JSON 은
[삭제 전 Git 기록](https://github.com/groovallstar/ner-pipeline/tree/a60cb813d0be19a1ccb414c660c83c6407bb50a5/certified)에서 확인할 수 있다.
