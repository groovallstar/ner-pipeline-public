# issue-238: gold 파일명을 origin.jsonl 로 통일

- Issue: https://github.com/groovallstar/ner-pipeline/issues/238
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-238-single-gold-origin-jsonl`
- 승인일: 2026-08-28

## 배경 (왜)

**언어마다 gold 파일 이름이 갈려 있었고, 그 탓에 ja 기본 경로가 이미 깨져
있었다.**

`DEFAULT_DATA['ja']` 는 `data/stockmark/pii_all.jsonl` 을 가리켰는데 그 파일이
없다. 실재하는 것은 `data/stockmark/origin.jsonl` 이다. `python -m ner.classifier
--lang ja` 를 기본값으로 돌리면 입력을 못 찾는 상태였다. 착수 시점 분포는 ja·vi
가 `origin.jsonl`, ko·en 이 `pii_all.jsonl` 이었고, 네 파일 모두 PII 주입까지
끝난 canonical 10종 완성본으로 성격이 같은데 이름만 둘로 갈렸다.

두 이름은 원래 파이프라인 단계를 뜻했다 — `origin.jsonl`(주입 전) →
`ner.augmenters.pii` → `pii_all.jsonl`(주입 후). 그런데 vi 가 #133 에서 **"단일
출처"** 를 이유로 주입 후 파일을 `origin.jsonl` 로 리네임했고(근거:
`docs/reports/vietnamese-bert-classifier-history.md`), ja 도 같은 상태가 됐다.
그 결과 같은 성격의 파일이 언어마다 다른 이름으로 불렸다.

주입 전 원본은 ko·ja·vi 어디에도 남아 있지 않다. 디스크에 두는 것은 언어마다
gold 하나뿐이므로, 단계를 구분하던 이름 규칙은 이미 지킬 대상이 없었다.

## 목적

네 언어의 gold 파일명을 `origin.jsonl` 하나로 통일하고 코드·설정·문서 참조를
실재와 맞춘다. ja 의 기본 경로 깨짐이 함께 해소된다.

## 범위

- 포함: `src/ner/classifier/{__main__,error_analysis}.py` · 모듈 `CLAUDE.md` ·
  `src/ner/labelers/ko/data/` 의 설정 JSON 2종 · `tests/` 의 경로 참조 ·
  `docs/manual/**` · `docs/reports/**` 의 **실행 가능한 명령줄** · ko 배포
  패키지와 그 원장 사본의 `metrics.json` 의 `data_path`
- 제외
  - **`docs/issues/**`** — 과거 스냅샷이라 소급 수정하지 않는다.
  - **`docs/reports/**` 의 역사 서술** — 그 시점 사실 기록이다. 지금 실행하면
    실패하는 명령줄만 포함에 뒀다.
  - **en·ja 배포 패키지의 `data_path`** — ko 와 같은 결함이지만 이 브랜치는
    ko 만 다뤘다. 처리 조건은 §후속 작업 에 적었다.
  - **패키지 폴더명**(`{lang}_ner_prod_seed*`)·**ja 의 `data/` 쪽 토크나이저
    누락** — 별건으로 두기로 결정했다.
  - **기준 파일** — 라벨·채점·분할 규칙을 건드리지 않았다.

## 구현 결과

`data/klue/pii_all.jsonl` 과 `data/ontonotes_en/pii_all.jsonl` 을
`origin.jsonl` 로 옮기고, 그 이름을 가리키던 참조를 코드 5곳·설정 2곳·테스트
4곳·문서 13곳에서 맞췄다.

| 무엇 | 어디 |
|---|---|
| 기본 경로 | `classifier/__main__.py` 의 `DEFAULT_DATA` 4종 · `error_analysis.py` 의 같은 dict |
| 설정 | `ko_locorg_ledger.json`(`gold_path`) · `evt_holiday_prereg.json`(`gold`) |
| 검사 | `test_lang_wiring.py` — 이름 규칙 고정 + **실재 확인** 2건 추가 |
| 문서 | 모듈 `CLAUDE.md` 2종 · `docs/manual/` 3종 · `docs/reports/` 명령줄 7건 |

### 주입 명령의 입력을 자리표시자로 바꾼 이유

`2-augmentation.md` 의 ko 생성 명령은 `origin.jsonl` 을 읽어 `pii_all.jsonl` 을
쓰는 형태였다. 통일 후에는 **입력과 출력이 같은 경로**가 되어 그대로 두면
거짓말이 된다. 주입 전 파일은 보관하지 않으므로 입력을 `<주입 전 KLUE gold>`
자리표시자로 적고, 다시 돌리려면 상류에서 새로 만들어야 한다는 사실을 본문에
남겼다. 출력도 `origin.new.jsonl` 로 받아 검토 후 승격하는 형태로 바꿨다 —
입력과 출력이 같은 경로면 실패한 실행이 gold 를 덮어쓴다.

### 검사에 실재 확인을 넣은 이유

종전 `test_lang_wiring.py` 는 경로를 **문자열로만** 봤다. 그래서 ja 기본값이
없는 파일을 가리키는 동안에도 초록불이었다. 파일 존재를 보는 케이스를 넣어
문자열과 디스크를 잇는다. `data/` 는 gitignore 라 CI 에 없으므로 그때는 skip
한다.

## 검증

| # | 기준 | 결과 |
|---|---|---|
| AC1 | 네 데이터셋이 각각 `origin.jsonl` 하나를 gold 로 둔다 | ✅ klue 25,989 · stockmark 5,270 · wikiann_vi 37,706 · ontonotes_en 76,378 |
| AC2 | `DEFAULT_DATA` 네 값이 실재 파일을 가리킨다 | ✅ `test_lang_wiring.py` 12 passed (skip 없이) |
| AC3 | `src/**`·설정 JSON 에 `pii_all` 참조 0 | ✅ grep 결과 없음 |
| AC4 | 주입 명령이 입출력 같은 이름 모순을 안 남긴다 | ✅ 자리표시자 + 임시 출력 승격으로 정정 |
| AC5 | 전체 테스트 통과 | ✅ 1,107 passed · 3 skipped |

3 skipped 는 NLLB 실가중치를 요구하는 번역 테스트다.

```
uv run pytest tests/   # 1107 passed, 3 skipped (78s)
```

## 결정 로그

- 2026-08-28: `origin.jsonl` 쪽으로 통일. 두 이름 중 `pii_all.jsonl` 이 파이프라인
  단계와 맞는 이름이지만, vi 가 #133 에서 이미 "단일 출처"를 이유로 반대 방향을
  택했고 ja 도 그 상태였다. 언어마다 gold 를 하나만 두는 지금 구조에서는 단계를
  구분하는 이름이 가리킬 대상이 없다.
- 2026-08-28: `docs/reports/` 는 명령줄만 고치고 역사 서술은 남겼다. 벤치마크
  리포트의 "데이터: `pii_all.jsonl` 25,989행" 같은 문장은 그 시점 측정 조건의
  기록이라, 소급해 바꾸면 당시 실행과 다르게 읽힌다.
- 2026-08-28: 배포 패키지의 `metrics.json` 은 건드리지 않기로 했다. `data_path`
  는 학습이 무엇을 읽었는지에 대한 기록이고, 파일이 나중에 개명됐다는 이유로
  고치면 실행 기록의 위조가 된다고 봤다.
- 2026-08-28: 위 결정을 뒤집어 ko 의 `data_path` 를 `origin.jsonl` 로 고쳤다.
  "학습 당시 기록" 이라는 분류가 틀렸다. 배포 패키지는 보관물이 아니라 소비되고
  재생성되는 산출물이다 — `test_ko_parity_baseline_raw` 가 서버 predict 의 raw
  F1 을 이 파일의 `overall_strict` 와 맞추고, `build_ner_prod.py` 는 `data_path`
  를 읽어 분할을 다시 유도한다. 개명 뒤 그 재유도가 `dataset not found` 로 멈춰
  있었으므로, 죽은 경로를 남기는 것은 기록을 지키는 일이 아니라 위조를 막는 검사
  하나를 끄는 일이었다. 내용이 같다는 것은 같은 파일 안의 지문
  `a6aaa31ba099c9fd` 이 보증하므로, 바뀐 것은 측정값이 아니라 포인터뿐이다.
  원장 사본도 함께 고쳤다 — `test_package_is_the_ledger_run_verbatim` 이 패키지와
  원장을 글자까지 같게 요구해 한쪽만 고칠 수 없다.
- 2026-08-28: ko 원장(`certified/classifier/ko/**`)을 배포 실행 하나로 줄이려다
  되돌렸다. 인용 대조의 표면을 줄여 검출력을 올린다는 계산이었는데, 그 원장은
  죽은 기록이 아니라 **살아 있는 검사의 기준**이었다 —
  `test_ko_evt_axis1_audit` 이 `issue202-axis1-*` 의 sha256 을 사전등록과
  대조하고, `test_ko_evt_holiday_audit` 이 이슈 문서의 support 를
  `certified/classifier/ko/**` 의 pooled 전량과 맞춘다. 실제로 지워 태워보니
  3건이 실패하고 2건이 **조용히 skip 됐다** — `test_ko_split_audit` 의 두 검사가
  `issue201-r2-principle/split_audit.json` 에 `skipif` 로 걸려 있어, 없으면
  빨간불 대신 "audit artifact not promoted yet" 으로 지나간다. 실패보다 이쪽이
  나쁘다. 다섯을 확인하고 복원했다.
  **원장을 줄일 때는 문서 참조뿐 아니라 `tests/` 참조를 먼저 봐야 하고, 실패만
  세면 조용히 빠지는 검사를 놓친다.**
- 2026-09-10: KO gold 감사 장치를 폐기했다. `src/ner/labelers/ko/data/` 의 원장·
  사전등록 JSON 21 개와 `ko_dat_audit.py`·`ko_evt_axis1_audit.py`·
  `ko_evt_holiday_audit.py`, 그리고 그것들을 gold 에 대조하던 테스트가 함께
  없어졌다. 이 문서가 그 경로를 인용한 자리는 그때의 기록이며 지금은 파일이 없다.
  수치와 판정 결과는 손대지 않았다

## 후속 작업

- `docs/manual/pipeline/3-verification.md:243` 의 재현 명령이 `--data-jsonl`
  이라는 **실재하지 않는 플래그**를 쓴다(실제는 `--data`). 이번 이름 통일과
  무관하게 이미 stale 이라 범위 밖에 뒀다.
- en·ja 배포 패키지의 `data_path` 도 ko 와 같은 이유로 죽어 있다. en 은
  `data/ontonotes_en/pii_all.jsonl` 을 가리키고 지문 `39cb0f9e9c16f265` 이 현재
  `origin.jsonl` 과 일치하므로 ko 와 똑같이 고칠 수 있다. ja 는
  `data/stockmark/pii_all_phonediv.jsonl` 을 가리키는데, 그 run 이 지문을
  기록하지 않아(`data_fingerprint` 가 `null`) 지금 `origin.jsonl` 과 같은
  내용이라는 근거가 행 수(5,270)뿐이다. ja 는 이름 대조가 아니라 재학습이나
  지문 재계산이 앞서야 한다.
