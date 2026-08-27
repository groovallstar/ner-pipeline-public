# issue-232: 한국어 NER 배포 패키지와 REST API 서빙

- Issue: https://github.com/groovallstar/ner-pipeline/issues/232
- PR: https://github.com/groovallstar/ner-pipeline/pull/233
- 브랜치: `feat/issue-232-ko-rest-api`
- 승인일: 2026-08-27

## 배경 (왜)

**한국어는 canonical 10종 평면이 완성돼 있고 k-fold 로 성능도 재 뒀는데, 그
실력을 쓸 수 있는 자리가 없었다.**

- `certified/classifier/ko/` 에 10-fold pooled 원장이 셋 앉아 있다. 교차검증
  추정치라 **배포할 체크포인트 하나를 가리키지 않는다.**
- `/data/ner/{ja,vi,en}/` 에는 배포 패키지가 있지만 `/data/ner/ko/` 는 비어
  있었다. 한국어 추론을 하려면 매번 다시 학습해야 하고, "이 수치는 어느
  체크포인트에서 나왔나"를 가리킬 대상이 없었다.
- REST 서버의 `SUPPORTED_LANGS` 는 `('ja','vi')` 였다. 한글 텍스트는 `detect.py`
  에 감지기가 없어 `unsupported` 로 떨어졌다.

#221(en 배포)이 서버 서빙을 범위 밖에 뒀던 이유는 **en 이 라틴 스크립트라 양성
감지가 불가능**해서였다. 한글은 결정적 스크립트 신호라 그 장애물이 없어 —
`DETECTORS` 에 한 줄이다 — ko 는 배포와 서빙을 한 이슈로 닫을 수 있었다.

## 목적

`/data/ner/ko/` 배포 패키지를 ja·vi·en 동형으로 출하하고, REST API 가 한국어를
ja·vi 와 같은 계약으로 서빙한다.

## 범위

- 포함: 배포 패키지(학습·포장·원장 승격) · 포장 스크립트 언어 파라미터화 ·
  배포 추론 스크립트 · 서빙(`SUPPORTED_LANGS`·`detect.py`·OpenAPI·웹 UI) ·
  테스트 · 문서
- 제외
  - **`/v1/translate` 의 ko 번역** — 이 엔드포인트는 웹 데모의 "한국어로 번역"
    이라 ko 입력은 대상이 없다. `lang:"ko"` 는 400 이고 UI 는 버튼을 감춘다.
  - **confidence threshold** — vi·en 동형으로 raw 출력. `thresholds.json` 없이
    출하하고 `inference.py` 의 graceful 폴백이 정상 경로다.
  - **`chunking.py` 문장 분할** — `_SENT_RE` 가 ASCII 마침표를 경계로 안 잡아 긴
    한국어는 단어 경계 폴백(`_char_windows`)을 탄다. **vi 가 지금 그대로 도는
    경로**라 ko 만 고칠 이유가 없고, 고치면 ja·vi 청킹 동작이 함께 바뀐다.
  - 백본 재선정·하이퍼파라미터 sweep — `koelectra-base-v3-discriminator` 기본값
  - 기준 파일(`canonical-entity-schema.md`·`data_utils.py`·`kfold_pool.py`·
    `metrics/`·`validity/`) — 건드리지 않았다. 라벨 인벤토리가 언어 공통이라
    `build_label_maps` 도 그대로다.

## 설계

### 학습·포장 조건

```bash
# 1) 학습 — 원장 k-fold 와 같은 group-key
python -m ner.classifier --lang ko --group-key id \
    --seed 42 --train-seed 42 \
    --output-dir results/classifier/ko/deploy-trainseed42

# 2) 포장
python src/ner/scripts/build_ner_prod.py \
    --run-dir results/classifier/ko/deploy-trainseed42 --lang ko
```

인코딩은 토크나이저 capability 로 분기하므로(koelectra=fast → `_encode_vi`
경로) ko 전용 분기가 필요 없다. 서버의 `_load_tokenizer` 도 ja 만 slow 특례라
ko 는 기존 fast 경로로 들어간다.

### 포장 스크립트를 복사하지 않고 파라미터화한 이유

`build_en_ner_prod.py` 330줄 중 언어에 딸린 것은 lang 가드·기본 out-dir·
MODEL_CARD 문구 정도다. 그대로 복사해 `build_ko_ner_prod.py` 를 만들면 **분할
재유도·지문 대조·누출 가드 같은 안전장치가 두 벌이 되고, 한쪽만 고쳐지는 순간
조용히 갈린다.**

대신 `--lang` 을 run 의 `metrics.json` 에서 읽어 out-dir·카드 문구를 정하게 바꾸고
이름을 `build_ner_prod.py` 로 옮겼다. 카드의 토크나이저 줄도 언어별 표가 아니라
**실제로 동봉된 토크나이저**에서 만든다(`describe_tokenizer`·`tokenizer_files`)
— 백본을 바꿨을 때 카드만 옛말이 되는 일을 막는다.

en 배포 run 디렉토리는 이미 지워져(`results/` 는 휘발) 재포장으로 검증할 수
없으므로, **합성 run 디렉토리 픽스처 pytest** 로 en·ko 두 모양을 모두 태운다 —
원본 en 스크립트에 없던 집행 주체가 여기서 생겼다.

### 원장이 배포 run 자신인 것

en 은 백본 벤치(`backbone-bench/roberta_base_seed42`)라는 선행 원장이 있어
패키지를 그것과 대조했다. ko 는 단일 분할 원장이 없다 — 있는 것은 10-fold
pooled 뿐이고 자가 다르다. 그래서 **배포 run 의 `metrics.json` 자체를 원장으로
승격**하고, 패키지 검사는 *재현*이 아니라 *프로비넌스 정합*(패키지가 자기가
나온 run 과 어긋나지 않는가)을 본다. #221 의 테스트도 이미 재현 대조를 하지
않으므로 판정 성격은 같다.

### 감지기 순서

`DETECTORS` 에 `(_has_hangul, 'ko')` 를 가나 다음·vi 앞에 넣었다. 셋의 신호는
서로소라 순서가 결과를 바꾸지 않지만, 코드 스위칭(한 문장에 둘 이상)에서는
위쪽이 이긴다 — 가나+한글은 ja. 스크립트 신호가 부호 신호보다 강하다는 기존
원칙 그대로다.

한글은 음절(가–힣)만 보지 않고 자모·호환자모·확장 블록까지 잡는다. 음절만
보면 자모 단독 표기(`ㅋㅋㅋ`·옛한글)를 놓친다.

**한자만 있는 텍스트는 종전대로 `unsupported` 다.** ja·ko 가 한자를 공유하므로
한자의 존재는 어느 쪽도 가리키지 않는다 — 가나·한글이라는 *배타적* 스크립트만
신호로 쓰는 이유가 이것이고, 이 경로에 회귀는 없다.

### 번역 대상 언어와 NER 지원 언어의 분리

`/v1/translate` 는 `SUPPORTED_LANGS` 가 아니라 `TRANSLATABLE_LANGS` 를 본다.
목록을 따로 두지 않고 `LANG_NAME`(프롬프트에 박히는 언어 이름)의 키에서
유도했다 — 이름이 곧 자격이라서다. 목록을 별도로 두면 한쪽만 늘어나 조용히
어긋나고, **그때 나타나는 증상은 에러가 아니라 원문이 그대로 '번역'으로
돌아오는 것**이다.

웹 UI 는 ko 결과에서 번역 버튼을 감춘다. 회색으로 비활성만 시키면 "백엔드가
죽었나"로 읽히기 때문이다.

## 구현 결과

| 무엇 | 어디 |
|---|---|
| 포장 스크립트 | `src/ner/scripts/build_ner_prod.py` — `build_en_ner_prod.py` 를 rename + 언어 파라미터화. 언어는 run 의 `lang` 에서 읽고 `--lang` 은 대조용 |
| 배포 추론 | `src/ner/scripts/eval_ko_ner_test.py` + `.sh` — en 판과 같이 평가 대상(`--limit`)과 화면 표시(`--show`)를 분리 |
| 배포 번들 | `/data/ner/ko/` — ja·vi·en 동형. `thresholds.json` 없음 |
| 원장 | `certified/classifier/ko/deploy-trainseed42/metrics.json` |
| 감지 | `src/server/detect.py` — `_HANGUL_RANGES` + `_has_hangul`, `DETECTORS` 에 등록. 구간 검사를 `_in_ranges` 로 공통화 |
| 서빙 계약 | `src/server/config.py`(`SUPPORTED_LANGS`) · `app.py`(OpenAPI 문구·예제·`/v1/translate` 의 `TRANSLATABLE_LANGS` 분기) · `translate.py`(`LANG_NAME` 공개 + `TRANSLATABLE_LANGS` 유도) |
| 웹 UI | `src/server/static/index.html` — 셀렉터에 ko, 미지원 안내에 한자 사례, `syncTranslate()` 로 ko 결과에서 버튼 감춤 |
| 검사 | `test_ko_deploy_package.py`(10) · `test_build_ner_prod.py`(17) · `test_detect.py`(+5) · `test_api.py`(+3) · `test_translate.py`(+2) |

## 검증

### 수락 기준

| # | 기준 | 결과 |
|---|---|---|
| AC1 | 패키지가 원장 run 과 verbatim 정합 · 분할 크기 일치 · 누출 0 | ✅ `test_ko_deploy_package.py` 통과 |
| AC2 | 배포 run test strict overall F1 ≥ 0.85 | ✅ 0.9153 |
| AC3 | 포장 스크립트 파라미터화가 en·ko 두 모양을 태움 | ✅ `test_build_ner_prod.py` 17건 |
| AC4 | `/v1/ner` 한글 자동감지·명시 ko 수용·혼합 배치 1:1·한자만 unsupported | ✅ `test_detect.py`·`test_api.py` + 실서버 확인 |
| AC5 | `/v1/translate` 가 `lang:"ko"` 를 400 거절 | ✅ `test_translate.py` |
| AC6 | 문서 + OpenAPI 문구에 ko 반영 | ✅ 문서 8종 · `test_api.py` 의 OpenAPI 문구 검사 |

### 배포 체크포인트 성능

<!-- certified: classifier/ko/deploy-trainseed42/metrics.json -->

test 2,598문장 · strict span · 임계값 미적용(raw).

| 구분 | P | R | F1 | support |
|---|---:|---:|---:|---:|
| 전체(10종) | 0.8980 | 0.9333 | 0.9153 | 7887 |

| 타입 | P | R | F1 | support |
|---|---:|---:|---:|---:|
| PER | 0.9268 | 0.9219 | 0.9243 | 1881 |
| LOC | 0.8113 | 0.8843 | 0.8462 | 778 |
| ORG | 0.7564 | 0.9135 | 0.8276 | 289 |
| PROD | 0.6429 | 0.7714 | 0.7013 | 385 |
| EVT | 0.5670 | 0.7017 | 0.6272 | 181 |
| DAT | 0.8218 | 0.8800 | 0.8499 | 975 |

PII 4종(EMAIL·PHONE·ID_NUM·CREDIT_CARD)은 합성 주입이라 포화된다(모두 F1
0.997 이상) — 모델 실력의 지표로 읽지 않는다.

10-fold pooled 원장(0.9203, `issue202-axis1-head`)과 가까운 자리다. 다만 자가
달라(교차검증 추정 vs 단일 홀드아웃) 나란히 놓은 것이 아니라 붕괴가 아님을
확인하는 용도다.

### 실서버 확인

3개 언어 모델을 모두 로드한 서버에서:

- `GET /health` → `status: ok`, ja·ko·vi 모두 `loaded: true`
- ko 자동감지 → `{"lang":"ko", entities: PER·DAT·LOC·EVT}`
- ja+ko+vi 혼합 배치 → 입력 순서·lang 1:1 보존
- 한자만(`東京都千代田区`) → `{"lang":"unsupported","entities":[]}`
- `python -m server.scripts.example_client` → DEMO PASS
- `eval_ko_ner_test.py` → run 의 metrics 와 같은 값(P 0.8980 / R 0.9333 /
  F1 0.9153)을 배포 패키지 경로에서 재현

## 결정 로그

- 2026-08-27: 포장 스크립트를 언어별로 복사하지 않고 `build_ner_prod.py` 로
  파라미터화. 안전장치가 여러 벌이 되는 것을 피하는 대가로 합성 run 픽스처
  테스트를 새로 세웠다.
- 2026-08-27: ko 원장을 배포 run 자신으로 삼음. 단일 분할 선행 원장이 없고,
  10-fold pooled 와 대조하는 것은 자가 다른 비교라 혼동을 만든다.
- 2026-08-27: `chunking.py` 를 범위 밖에 둠. 긴 한국어가 단어 경계 폴백을 타는
  것은 vi 가 이미 도는 경로와 같아, ko 만 고칠 근거가 없다.
