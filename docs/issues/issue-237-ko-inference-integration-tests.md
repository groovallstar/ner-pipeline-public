# issue-237: REST API ko 실모델 통합 테스트

- Issue: https://github.com/groovallstar/ner-pipeline/issues/237
- PR: 없음 — `test` 갈래라 `develop` 직접 커밋(루트 `CLAUDE.md` §워크플로)
- 브랜치: 없음(같은 이유)
- 승인일: 2026-08-28

## 배경 (왜)

`tests/server/test_inference_integration.py` 는 실모델을 로드해 offset 정합·청크
경계·배치 등가·정규화를 검증하는데, **커밋 시점까지 커버 언어가 ja·vi 뿐이었다.**
#232 로 ko 서빙이 들어왔지만 ko 쪽에 있던 것은 stub registry 계약 테스트
(`test_api.py`)와 감지 테스트(`test_detect.py`)뿐이라, 실모델 추론 경로에는 회귀
안전망이 없었다.

REST API 로 ko 를 실제로 태워 보면 ja·vi 와 같은 계약으로 돈다. 그러나 그 확인은
사람이 한 번 돌린 결과라, ko 토크나이저·모델·청킹이 바뀌면 **아무 신호 없이
깨진다** — 계약 테스트는 stub 이라 모델을 안 부르므로 초록불이 그대로 유지된다.

ko 는 특히 청킹 경로가 ja 와 다르다. `chunking._SENT_RE` 의 경계 문자에 ASCII
마침표가 없어 한국어 장문은 문장 분할이 아니라 단어 경계 폴백(`_char_windows`)으로
쪼개진다. #232 가 근거를 대고 범위 밖에 둔 경로인데, 그 경로에서 gold 엔티티가
청크 경계를 가로지르지 않는다는 보장을 ja·vi 테스트는 주지 못한다.

## 목적

`test_inference_integration.py` 가 ko 를 ja·vi 와 같은 축으로 태워, ko 실모델
경로의 회귀가 사람 손이 아니라 pytest 로 잡히게 한다.

## 범위

- 포함: `tests/server/test_inference_integration.py` · `src/server/CLAUDE.md` 의
  검증 섹션
- 제외
  - **src 런타임** — 이 이슈는 검사만 늘린다. 서버 코드는 손대지 않았다.
  - **`chunking.py` 문장 분할 개선** — 한국어가 단어 경계 폴백을 타는 것은 #232
    가 범위 밖에 둔 상태 그대로다. 여기서는 그 경로가 엔티티를 자르지 않는다는
    *사실을 고정*할 뿐 경로를 바꾸지 않는다.
  - **ko 모델 정확도** — 실측에서 `LG전자`·`삼성전자 본사` 의 ORG 미검출을 봤으나
    계약이 아니라 성능이고, 원장의 ORG precision 과 방향이 같다.
  - **기준 파일** — 정답·채점규칙·분할 어느 것도 건드리지 않았다.

## 구현 결과

ja·vi 가 이미 서 있던 일곱 축에 ko 케이스를 하나씩 붙이고, 혼합 배치 테스트를
두 언어에서 세 언어로 넓혔다. 테스트 함수 15개 → 23개.

| 축 | 무엇을 고정하나 |
|---|---|
| offset 정합 | 반환 span 이 원문 슬라이스와 글자 그대로 같다 |
| thresholds 부재 | ko 는 `thresholds.json` 없이 출하돼 raw 로 돈다(vi 와 같은 경로) |
| 장문 청크 | 단어 경계 폴백으로 쪼개도 후반 청크 엔티티가 회수되고 offset 이 맞다 |
| 청크 배치 parity | 여러 청크를 한 forward 로 태운 결과가 청크별 단건과 같다 |
| 청크 경계 | test gold 엔티티 7,887개 전수가 청크 하나에 온전히 들어간다 |
| `predict_many` | cross-text 배치가 단건 순차와 같다 |
| NFD 입력 | 자모 분해형 입력이 NFC 와 같은 엔티티를 낸다 |
| 배포 parity | 서버 predict 로 잰 F1 이 배포 `metrics.json` 과 일치한다 |

**NFD 축이 ko 에서 vi 보다 무겁다.** 한글 음절은 분해되면 초·중·종성으로 갈려
길이가 배 이상 늘어난다(`김` 한 자 → 세 자). 정규화가 빠지면 토크나이저가 자모를
낱개로 보고 offset 도 분해형 기준이 되어 원문 슬라이스와 어긋난다.

**배포 parity 는 재현이 아니라 프로비넌스 정합이다.** ko 원장은 배포 run 자신이라
(#232 의 결정), 이 대조가 답하는 물음은 "패키지가 자기가 나온 run 과 어긋나지
않는가" 다. ja 는 thresholds 가 있어 abstention on·off 두 점을 보지만 ko 는 raw
하나(`overall_strict`)를 본다.

헬퍼 `_ja_overall_f1` 은 실제로 언어 무관이라 `_overall_f1` 로 이름만 바꿔 두
언어가 공유한다.

## 검증

### 수락 기준

| # | 기준 | 결과 |
|---|---|---|
| AC1 | ko 가 ja·vi 대칭 축 일곱을 모두 덮는다 | ✅ ko 케이스 8건 통과 |
| AC2 | ko parity 가 배포 `metrics.json` 의 `overall_strict` 와 abs 1e-6 일치(P·R·F1·support) | ✅ 통과 |
| AC3 | 혼합 배치가 ja·vi·ko 3언어 인터리브로 각 항목 1:1 | ✅ 통과 |
| AC4 | 모델 부재 시 에러가 아니라 skip | ✅ `model_root` 를 없는 경로로 바꿔 23건 전부 skip 확인 |
| AC5 | `uv run pytest tests/server/` 전체 통과 | ✅ 169 passed · 3 skipped |

3 skipped 는 NLLB 실가중치를 요구하는 번역 테스트로 ko 와 무관하다.

### 명령·결과

```
uv run pytest tests/server/test_inference_integration.py   # 23 passed (33s)
uv run pytest tests/server/                                # 169 passed, 3 skipped
```

### REST API 실측

이슈 등록의 계기가 된 실서버 확인. `/data/ner/{ja,ko,vi}` 를 모두 로드한 서버에
36건을 태워 전부 통과했다 — 세 언어 자동감지·명시 lang·혼합 배치 순서 보존,
`{lang, entities}` 스키마와 span 4키 동일, 400·413·422·404 에러 봉투, 단건과
배치의 결정성 일치, 1,680자 ko 장문의 offset 정합, OpenAPI·웹 UI 노출면. 번역을
켠 서버에서는 `lang:"ko"` 가 400 으로 거절되고 같은 요청의 ja 는 백엔드 부재로
503 이 되어, NER 지원 언어와 번역 대상 언어가 갈려 있다는 계약도 확인했다.

`python -m server.scripts.example_client` 는 DEMO PASS(429 부하 셰딩 포함).

## 결정 로그

- 2026-08-28: ko 통합 테스트를 별도 파일로 만들지 않고 기존
  `test_inference_integration.py` 에 넣었다. 축이 같으므로 파일을 가르면 헬퍼가
  두 벌이 되고 한쪽만 고쳐질 때 언어별로 검사 강도가 조용히 갈린다.
- 2026-08-28: ko parity 를 test 전수(2,598문장)로 뒀다. 단건 추론 2,598회가
  17.6초라 스위트 전체가 40초대에 머물러, 표본으로 줄여 얻을 이득이 없었다.
- 2026-08-28: 청크 경계 축을 넣기 전에 먼저 재 봤다 — ko 단어 경계 폴백에서
  gold 7,887개 중 경계를 가로지르는 것이 0이었다. 실패할 축을 테스트로 굳히는
  일을 피하려는 확인이었고, 통과했으므로 그대로 고정했다.
