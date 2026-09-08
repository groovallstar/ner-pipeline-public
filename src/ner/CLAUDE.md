# src/ner/ — 소스 코드 가이드

## 모듈 구조

패키지 루트: `src/ner/`. import는 `from ner.<모듈>.xxx import Xxx` 형태.

### labelers/

언어별 LLM NER 라벨러 + 공용 유틸(BIO 정렬·태그 정규화·span 매칭). ko·ja·vi·en
네 언어 모두 vLLM 백엔드를 쓴다. en 만 용도가 다르다 — 원천 OntoNotes5 가 사람
gold 라 재라벨할 대상이 없어, PII 주입 결과의 교차 검증에만 쓰고 ja·vi 가 가진
`dataset_loader` 를 두지 않는다. 파일별 역할: `src/ner/labelers/CLAUDE.md`.

### llm_eval/

LLM 라벨링 벤치마크의 오케스트레이션·채점·리포트. KO 는 BIO, JA·VI 는
offset-span 으로 같은 러너가 eval_mode 를 갈라 돈다. CLI 는
`python -m ner.llm_eval`. 파일별 역할: `src/ner/llm_eval/CLAUDE.md`.

### augmenters/

학습 데이터 증강. 세 서브모듈의 관심사가 다르다 — `pii/` 는 네 언어 공통 합성
PII 주입기, `wikiann_vi/` 는 WikiANN-vi 3종을 canonical NER 5종으로 재라벨,
`ontonotes_en/` 은 OntoNotes5 영문 18종을 canonical NER 5종 + DAT 로 변환한다.
파일별 역할: `src/ner/augmenters/CLAUDE.md`.

### classifier/

ja·vi·ko·en canonical 10종 평면 BERT 토큰 분류 파인튜닝. CLI 는
`python -m ner.classifier --lang {ja,vi,ko,en}` 이며 `--group-key orig` 로
누출-free group K-fold 를 돌린다. 파일별 역할: `src/ner/classifier/CLAUDE.md`.

**이 패키지의 두 파일은 기준 파일이다** — `data_utils.py`·`kfold_pool.py` 는
분할을 정하므로 고치면 커밋이 잠긴다(루트 `CLAUDE.md` §하네스).

### metrics/

span/BIO 메트릭 공용 구현 (classifier·llm_eval 공유). 별도 하위 문서가 없어
파일 표를 여기 둔다.

| 파일 | 역할 |
|------|------|
| `bio_metrics.py` | `MetricsCalculator` — static 메서드 3종의 채점 경로. `compute_seqeval`(seqeval 기반 BIO 레벨) · `compute_span_f1`(BIO → char-offset span 추출 후 exact match, KLUE 방식) · `compute_span_match`(BIO 없이 텍스트 span 직접 비교, exact + 포함관계 relaxed). 뒤 둘은 seqeval 을 쓰지 않는 별도 알고리즘이다 |
| `span_metrics.py` | `compute_offset_span_f1`(strict — `(start, end, type)` 정확 매칭) · `compute_offset_span_f1_relaxed`(SemEval'13 Partial 매칭) |

**패키지 전체가 기준 파일이다** — 채점규칙을 쥐고 있어 어느 `.py` 를 고쳐도
커밋이 잠긴다.

### validity/

K-fold 실험 비교 유효성 게이트(재학습 0회). `fold*/metrics.json`·
`pooled_metrics.json` 만 읽어 두 실험을 비교해도 되는지, 개선이 노이즈인지
실측인지 판정한다. `metrics/`(한 run 의 F1 계산)를 한 줄도 import 하지 않는
다른 관심사라 최상위 패키지로 분리했다. CLI 는
`python -m ner.validity {std,repro,compare}`. 파일별 역할:
`src/ner/validity/CLAUDE.md`.

어떤 검증 부분집합을 통과해야 실험이 끝난 것인지의 조합·우선순위는 이 패키지가
정하지 않는다. 그것은 채점규칙을 바꾸는 일이라 기준 파일에 걸리고, 반박자가
누출·조작을 점검한다. **이 패키지도 전체가 기준 파일이다.**

### scripts/

배포 추론 스크립트와 일회성 감사 도구. 배포 추론은 ja·vi·ko·en 각 한 벌
(`eval_<lang>_ner_test.py` + 동명 `.sh` uv 래퍼)이며, 학습 없이 고정 test 와
저장된 임계값으로 추론·채점만 한다. 파일별 역할: `src/ner/scripts/CLAUDE.md`.

## 라벨 스키마

- **KO·JA·VI 공통**: canonical 10종 평면 = NER 5종(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`)
- **KO**: NER 5종 + `DAT` 은 KLUE 유래(`PROD/EVT` 는 LLM 재라벨 증분, TI/QT 드롭), PII 4종(`EMAIL/PHONE/ID_NUM/CREDIT_CARD`)은 합성 주입
- **EN**: NER 5종 + `DAT` 은 OntoNotes5 유래(재라벨 없음 — 원천이 `PRODUCT`·`WORK_OF_ART`·`EVENT` 를 이미 갖고 있다. 18종 중 9종 드롭), PII 4종은 합성 주입 — KO 와 같은 구조다. LOC/ORG 경계는 canonical §3 인프라 규칙대로 **개별 구조물은 `ORG`, 여러 지점을 잇는 경로는 `LOC`** 이며 원본 `FAC` 를 표면별 판정 표로 가른다. PII 는 미국 단일 체계다
- 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## 코딩 컨벤션

- `PYTHONPATH` **설정·주입 금지** — uv editable install이 `.pth`로 `src/`를 자동 등록
- import: `from ner.labelers.xxx import Xxx` (src 접두어 없이)
- 패키지 관리자: **UV** (`uv sync` 또는 `uv pip install -e .`)
- 타입 힌트, `logging` 표준 라이브러리 사용
- **주석/문서 언어**: print/log/예외·argparse help는 영문, docstring·인라인 주석은 한국어
- 상세 규칙: `docs/specs/coding-conventions.md`
