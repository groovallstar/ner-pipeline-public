# preserved/ — 잃으면 다시 못 만드는 실행 산출물

`results/` 도 `certified/` 도 아닌 것이 여기 온다. 셋의 차이는 **잃어도 되나**와
**인용 근거인가** 두 축이다.

| 폴더 | 버전 관리 | 무엇 | 잃으면 |
|---|---|---|---|
| `results/` | gitignore·휘발 | 실행 scratch 전부 (체크포인트·로그·예측·metric) | 재학습으로 복구 — 다만 GPU 시간과 **재현되지 않는 난수**가 든다 |
| `certified/` | 커밋 | 리포트가 **인용하는** metric JSON 만 | 인용의 근거가 사라진다 |
| `preserved/` | 커밋 | 인용 대상은 아니지만 **다시 만들 수 없는** 산출물 | 다음 이슈가 비교 기준을 잃는다 |

## 왜 따로 두나

`certified/` 에 넣으면 안 된다. 인용 대조는 "이 값이 원장 어딘가에 있나" 를 묻는
존재 검사라, 원장이 커질수록 지어낸 값도 무관한 실험의 값과 우연히 맞아 통과한다.
예측 덤프는 정수 수만 개를 원장에 붓는 것이라 대조를 통째로 무력화한다.

`results/` 에 두면 안 된다. 그 폴더는 휘발 scratch 로 선언돼 있어 정리 대상이고,
실제로 이슈 #201 의 fold 예측 덤프가 그렇게 사라져 다음 이슈가 base 팔을 **재학습**
해야 했다. 재학습한 base 는 옛 팔과 같은 수가 아니므로 옛 certified 수치와 나란히
놓을 수 없다 — 잃은 것은 파일이 아니라 비교 가능성이다.

## 무엇이 들어오나

**재채점에 필요한 최소분만.** 전체 예측 덤프(`test_predictions.json`)는 문장 원문과
gold 를 함께 담아 fold 당 2 MB 지만, 비순환 재채점이 쓰는 것은 예측 span 뿐이다.
원문과 gold 는 gold 파일에 이미 있고, 오히려 그쪽이 정본이다.

```
preserved/classifier/{lang}/<run-slug>/fold{N}/pred_spans.json
  {"<row id>": [["<type>", start, end], ...], ...}
```

체크포인트·모델 가중치는 넣지 않는다(`.gitignore` 전역 패턴이 막는다).

## 승격

```bash
python3 - <<'PY'
import json, pathlib
SRC = pathlib.Path("results/classifier/ko/<run-slug>")
DST = pathlib.Path("preserved/classifier/ko/<run-slug>")
for fold in range(10):
    rows = json.loads((SRC / f"fold{fold}/test_predictions.json").read_text(encoding="utf-8"))
    slim = {r["id"]: [[s["type"], s["start"], s["end"]] for s in r["pred_spans"]] for r in rows}
    out = DST / f"fold{fold}"; out.mkdir(parents=True, exist_ok=True)
    (out / "pred_spans.json").write_text(
        json.dumps(slim, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8")
PY
```

## 지금 있는 것

| 경로 | 무엇 | 왜 남겼나 |
|---|---|---|
| `classifier/ko/issue202-axis1-base/` | 축1-복합 회수의 base 팔 10-fold 예측 span (25,989 행) | 회수 후 gold 로 **같은 예측을 다시 채점**해야 비순환 Δ 가 나온다. 이 팔을 잃으면 base 를 또 재학습해야 하고, 그러면 다음 이슈도 같은 자리에서 같은 것을 잃는다 |
