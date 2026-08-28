# issue-221: 영문(en) NER 배포 패키지 — roberta-base 체크포인트를 /data/ner/en 에 출하

- Issue: https://github.com/groovallstar/ner-pipeline/issues/221
- PR: https://github.com/groovallstar/ner-pipeline/pull/226
- 브랜치: `feat/issue-221-en-deploy-model`
- 승인일: 2026-08-25

## 배경 (왜)

**영문 백본은 정해졌는데 그것으로 만든 체크포인트가 어디에도 없다.** #215 가
base 급 6종을 3-seed 로 재고 `roberta-base` 를 baseline 으로 확정했지만, 그
이슈가 남긴 것은 metric JSON 뿐이다. 가중치는 `results/`(gitignore·휘발)에
있었고 지금은 지워졌다.

ja·vi 는 `/data/ner/{lang}/` 에 배포 패키지(`model/`·`data/`·`metrics.json`·
`MODEL_CARD.md`)가 앉아 있고, `eval_{ja,vi}_ner_test` 와 REST 서버가 그 경로를
로드한다. en 은 그 자리가 비어 있어서 **영문 추론을 하려면 매번 다시 학습해야
하고, "이 수치는 어느 체크포인트에서 나왔나"를 가리킬 대상이 없다.**

## 목적

`roberta-base` 배포 체크포인트를 만들어 ja·vi 와 동형 구조로 `/data/ner/en/` 에
출하하고, 배포 추론 경로를 ja·vi 와 같은 모양으로 갖춘다.

## 범위

- 포함:
  - 포장 스크립트 `src/ner/scripts/build_en_ner_prod.py`
  - 배포 추론 `src/ner/scripts/eval_en_ner_test.py` + `.sh`
  - `/data/ner/en/{model/, data/{train,valid,test}.jsonl, metrics.json, MODEL_CARD.md}`
  - 패키지 검사 pytest + `certified/classifier/en/deploy-trainseed42/` 승격
  - `docs/manual/pipeline/4-classification.md` §8·§9, `src/ner/scripts/CLAUDE.md`,
    `src/ner/classifier/CLAUDE.md`
- 제외:
  - **정답표 경계 결함 수정** — 조사 중 드러났고 #225 로 분리했다. 당시에는
    ja·vi 배포 모델도 같은 노출을 안은 채 운영 중이라 en 만 막아둘 이유가 없다고
    봤으나, #225 가 확인한 바로는 둘 다 이 경로를 타지 않는다 — ja 는 slow
    토크나이저라 `_encode_ja` 로, vi 배포 모델은 PhoBERT 라 `_encode_phobert`
    로 간다. 노출된 것은 en 뿐이다
  - 서버 서빙(`SUPPORTED_LANGS`·`detect.py`·`translate`) — `detect.py` 는 라틴
    스크립트에 고유 코드포인트가 없는 en 을 의도적으로 `unsupported` 로 둔다.
    양성 감지 원칙을 어떻게 할지가 선행 결정이고 ja·vi 기존 동작에 닿는다
  - confidence threshold — vi 동형으로 raw 출력
  - 백본 재선정·하이퍼파라미터 sweep — #215 가 확정했다
  - 기준 파일(`data_utils.py`·`kfold_pool.py`·`metrics/`·`validity/`)

## 설계

### 학습과 포장의 분리

포장 스크립트는 **학습을 하지 않는다.** `python -m ner.classifier` 가 낸 run
디렉토리를 읽어 배포 레이아웃으로 옮기기만 한다.

vi 판례(`build_vi_ner_prod.py`, 소진 후 폐기)는 분할·인코딩·학습·평가를
스크립트 안에서 전부 했다. en 에서 그 형태를 따르지 않은 것은 **배포
체크포인트가 CLI 의 산물이어야 원장 수치와 같은 코드 경로 위에 놓이기
때문이다.** 포장 스크립트가 학습까지 하면, 그 결과를 벤치마크 원장과 견주는
일이 서로 다른 경로를 비교하는 것이 된다.

```bash
# 1) 학습 — #215 seed42 조건 그대로
python -m ner.classifier --lang en --group-key orig \
    --seed 42 --train-seed 42 --precision bf16 \
    --output-dir results/classifier/en/deploy-trainseed42

# 2) 포장
python src/ner/scripts/build_en_ner_prod.py \
    --run-dir results/classifier/en/deploy-trainseed42
```

### 분할 재유도와 그 검사

학습 CLI 는 분할 JSONL 을 저장하지 않는다. 포장 단계에서 같은 인자로 다시
유도하는데(`split_train_valid_test` 는 결정적), 결정적이라는 사실을 믿고
넘어가지 않고 유도한 크기를 run 의 `metrics.json` 기록과 대조해 어긋나면
중단한다. **실제로 학습에 쓰이지 않은 분할을 배포 데이터로 적어 두는 것이
여기서 가능한 가장 조용한 실패**이기 때문이다. 데이터 지문도 함께 본다 —
크기만 보면 "다른 파일을 같은 비율로 잘랐다"를 못 잡는다.

### test 홀드아웃 크기 — ja·vi 와 다른 자리

폴더 구조는 ja·vi 동형이지만 홀드아웃 크기는 다르다. ja·vi 는 test 가 100·101
문장인데, en gold 에 그대로 적용하면 EVT 가 약 1행·PROD 가 약 2행 들어가 가장
어려운 두 타입이 측정 불능이 된다.

| 타입 | 보유 행 | 비중 |
|---|---:|---:|
| PII 4종(EMAIL/PHONE/CREDIT_CARD/ID_NUM) | 22,529~24,105 | 30% 내외 |
| PER / LOC / ORG | 15,180 / 14,649 / 12,697 | 17~20% |
| DAT | 11,384 | 15% |
| PROD | 1,754 | 2.3% |
| EVT | 856 | 1.1% |

ja·vi 는 원본이 작아(5,270·37,706행) 100문장이 불가피했지만 en 은 76,378행이라
줄일 이유가 없다. #215 와 같은 `test_ratio=0.1`(7,637행)을 그대로 쓴다.

## 현 상태 (fact)

- `--lang en` 배선은 #215 가 이미 넣었다(`test_lang_wiring.py` 가 지킨다).
- 배포 체크포인트는 `train_seed=42` 단일 draw 다. 원장의 backbone-bench
  3-seed 와 데이터·분할·설정이 같아 나란히 놓을 수 있다.
- **EMAIL 이 seed 뽑기를 탄다.** 같은 설정 3-seed 에서 strict F1 이 크게 갈린다.
  원인은 미상이다 — 정답표 경계 결함(#225)으로 봤으나 그 이슈의 원장 전수 대조가
  귀속을 반증했다(같은 결함을 안은 백본 5종 중 4종은 진폭이 없다). 이 이슈는
  그 결함을 안고 출하했고, 결함 자체는 #225 에서 수정됐다.

## 결정 로그 (append-only)

- 2026-08-25: 포장 스크립트를 vi 판례와 달리 학습 없는 packager 로 설계. 배포
  체크포인트가 CLI 산물이어야 원장과 같은 코드 경로 위에 놓인다.
- 2026-08-25: test 홀드아웃을 ja·vi 의 100문장이 아니라 #215 와 같은
  `test_ratio=0.1` 로 결정. 100문장이면 EVT·PROD support 가 사실상 0 이 된다.
- 2026-08-26: **수락 기준 3(재현 대조)을 철회했다.** "양쪽 시드가 고정이니
  차이는 GPU 비결정성뿐"이라는 전제가 거짓이었다 — best 에포크 선택이 valid
  `eval_loss` 기준이라 계단 함수이고, 예측 span 복원이 라벨 생성과 같은 offset
  배열을 써서 채점 경로도 완전히 결정적이지 않다. 대신 프로비넌스 정합 · 같은
  자 · 붕괴 검출 바닥 셋으로 바꿨다. 결과를 본 뒤의 기준 변경이므로 그 사실을
  이슈 댓글에도 남겼다.
- 2026-08-26: EMAIL 진폭의 원인을 정답표 경계 결함으로 규명하고 **#225 로
  분리**했다. 결함을 이 이슈로 흡수하는 안도 검토했으나, 기준 파일 변경 +
  ko·vi 재측정이 따라와 배포가 무기한 밀린다. ja·vi 도 같은 노출을 안은 채
  운영 중이므로 en 만 막지 않는다.
- 2026-08-26: 3-seed median 출하를 검토하다 중단했다. 결함이 남은 채로 draw 를
  고르는 것은 증상 중에서 고르는 일이라, #225 수정 후에 하는 편이 낫다.

## 구현 결과

| 무엇 | 어디 |
|---|---|
| 포장 스크립트 | `src/ner/scripts/build_en_ner_prod.py` — run 을 읽어 `model/`(+tokenizer 동봉)·`data/`·`metrics.json`·`MODEL_CARD.md` 생성. 학습 안 함. `--note` 로 체크포인트별 운영 주의를 카드에 싣는다 |
| 배포 추론 | `src/ner/scripts/eval_en_ner_test.py` + `.sh` — 평가 대상(`--limit`)과 화면 표시(`--show`)를 분리해 받는다. en test 는 7,637행이라 하나로 묶으면 전수 채점과 육안 확인이 서로를 막는다 |
| 배포 번들 | `/data/ner/en/` — ja·vi 동형. `thresholds.json` 없음(vi 와 같이 raw) |
| 검사 | `tests/ner/classifier/test_en_deploy_package.py`(8) · `tests/ner/scripts/test_eval_en_ner_test.py`(6) |
| 원장 | `certified/classifier/en/deploy-trainseed42/metrics.json` |

> 포장 스크립트는 이후 #232(KO 배포·서빙)에서 `build_ner_prod.py` 로 이름이
> 바뀌고 언어를 run 의 `metrics.json` 에서 읽게 됐다. 위 명령의 파일명은 이
> 이슈 시점의 것이다.

**`model/` 에 tokenizer 를 동봉하는 이유가 실증됐다.** `Trainer.save_model()` 은
토크나이저를 남기지 않는데, 그 경로로 `AutoTokenizer.from_pretrained` 를 부르면
**예외가 아니라 망가진 토크나이저가 조용히 돌아온다** — 조사 중 실제로 겪었고
예측이 전량 비는 것으로 나타났다. 그래서 자립 로드 테스트는 로드만 보지 않고
추론까지 태운다.

## 검증

- 테스트: `uv run pytest -q --ignore=tests/server` → 790 passed. `uv run ruff check src/ tests/` → 통과
- 배포 스크립트 실측: `./src/ner/scripts/eval_en_ner_test.sh --limit 300 --show 4`
  — 모델+토크나이저 로드 0.646초, 추론 300문장 1.177초(문장당 3.9ms)

<!-- certified: classifier/en -->

배포 체크포인트 test 성능(7,637문장, strict span):

| | P | R | F1 | support |
|---|---:|---:|---:|---:|
| 전체(10종) | 0.8713 | 0.9267 | 0.8982 | 17092 |

NER 5종 strict F1 을 원장 backbone-bench 3-seed 와 나란히 둔다. 같은 데이터·
분할이라 비교가 성립하지만, **재현이 아니라 같은 레시피의 또 다른 draw** 다.

| 타입 | 배포본 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|---:|
| PER | 0.9417 | 0.9431 | 0.9511 | 0.9474 |
| LOC | 0.8713 | 0.8732 | 0.8821 | 0.8817 |
| ORG | 0.8233 | 0.8367 | 0.8432 | 0.8327 |
| PROD | 0.5385 | 0.5756 | 0.6651 | 0.6402 |
| EVT | 0.7246 | 0.7308 | 0.7434 | 0.7512 |
| EMAIL | 0.8480 | 0.9853 | 0.7305 | 0.9839 |

**배포본은 알려진 네 draw 중 NER-5 가 가장 낮다.** EMAIL 은 원장 안에서도
0.7305~0.9853 로 갈린다. 이 진폭을 #225 가 다룰 것으로 봤으나, 그 이슈가 원장
전수를 대조해 경계 결함으로의 귀속을 반증했다 — 진폭의 원인은 아직 미상이다.
배포 카드 §운영 주의에는 진폭 사실 자체를 적었다.

## 관련 커밋

- `6b3bcc7`: 포장·배포 추론 스크립트 + 검사 14종 + 원장 승격 + 문서 갱신

## 후속 작업

- **en 재학습·재배포** — #225 가 정답표 경계 결함을 고쳤고, 지금 배포 패키지는
  그 결함이 있는 라벨로 학습돼 새 정렬과 불일치한다. 재학습 전에는 수정된 코드와
  함께 서비스하면 안 된다.
- 서버 서빙(`SUPPORTED_LANGS`·`detect.py`) — `detect.py` 의 양성 감지 원칙을
  어떻게 할지가 선행 결정이다.
