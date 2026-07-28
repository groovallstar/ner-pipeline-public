# certified/ — 커밋된 결과 원장

리포트·이슈가 **인용하는 수치의 유일한 출처**다. 검사 게이트(결정적 층)의
인용 대조는 이 폴더의 `*.json` 만 카탈로그로 삼는다.

## 무엇이 여기 들어오나

- 채택된 실험의 **metric JSON 만** — `pooled_metrics.json`, fold 의
  `metrics.json`, `thresholds.json` 등. 예측 덤프·체크포인트는 넣지 않는다
  (모델 가중치 `*.pt/pth/bin/safetensors` 는 `.gitignore` 전역 패턴으로
  이 폴더에서도 자동 제외).
- `results/`(gitignore·휘발 scratch)에서 **verbatim 복사**해 커밋한다.
  복사는 실제 실행 산출물을 그대로 옮기는 것이라 값을 새로 만들지 않는다.
- 경로 관례: `certified/classifier/{lang}/<run-slug>/...` — scratch 구조를
  그대로 반영해 어느 실행에서 왔는지 드러낸다.

## 승격 (promotion)

실험이 끝나고 그 수치를 리포트가 인용하기로 정하면, scratch 의 metric JSON 을
이 폴더로 복사해 커밋한다:

```bash
mkdir -p certified/classifier/ko/<run-slug>
cp results/classifier/ko/<run-slug>/pooled_metrics.json \
   certified/classifier/ko/<run-slug>/
git add certified/classifier/ko/<run-slug> && git commit
```

## 인용 — 출처 선언

리포트·이슈의 표 위에 출처를 선언하면 게이트가 **그 파일 안에서만** 대조한다.

```markdown
<!-- certified: classifier/ko/<run-slug>/pooled_metrics.json -->

| 타입 | P | R | F1 | support |
```

경로는 이 폴더 기준 상대경로이며 디렉토리도 된다. 표 바로 앞 선언이 우선이고,
없으면 그 문서의 모든 선언이 합쳐 쓰인다(문서 머리에 한 번만 쓰는 형태).

선언한 경로가 아직 승격돼 있지 않으면 차단된다 — 인용하기 전에 승격하라는 뜻이다.

선언을 아예 안 하면 원장 전체를 뒤진다. 그러면 무관한 값과 우연히 일치해도
통과하고, 원장이 쌓일수록 심해진다. fold 파일까지 승격한 실행이 하나만 있어도
3자리 인용의 3할이 그냥 지나간다. 선언을 붙이는 편이 낫다.

## 잠금은 두 겹

1. `settings.json` 이 AI 의 **Write/Edit(손저작)** 을 막는다 — 실수·직접
   위조 차단. 잠금 대상은 metric JSON 이고, 이 README 같은 설명 문서는
   빠진다. 단 이건 speed-bump 이지 기계 보증이 아니다(복사 자체는
   subprocess 라 가능하다).
2. 진짜 앵커는 **git diff 를 PR 리뷰·판단 층 반박자가 보는 것** — 승격된
   값이 실제 실행에서 나왔는지 사람이 확인한다. 불변성·이력은 git 이 준다.
