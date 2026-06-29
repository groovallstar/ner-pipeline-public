# issue-147: NER REST API 긴 입력·배치 요청 처리 견고화

- Issue: https://github.com/groovallstar/ner_pipeline/issues/147
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-147-server-longinput-batch`
- 승인일: 2026-06-24

## 목적

REST API(`src/server/`)의 긴 입력·배치 요청 처리를 견고화한다. (1) chunk
추론을 한 배치 forward 로 묶어 multi-chunk·배치 요청을 가속하고, (2) 긴
입력에서 엔티티 recall·offset 이 보존됨을 회귀 가드로 고정하며, (3) 요청당
작업량 상한을 둔다. 모델 `max_length` 는 측정 결과 변경하지 않는다.

## 범위

- 포함: chunk 배치 추론(`inference.py`), recall·offset 가드 테스트, 문장-
  병합 offset 버그 수정(`chunking.py`), 요청-크기 정책(`config.py`·
  `app.py` — `MAX_TOTAL_CHARS` 합산 가드 + size 에러 413), 문서 갱신.
- 제외: 모델 `max_length` 변경(불변, §결정 로그), cross-text 요청 단위
  배치(후속 가능), 배포 인프라.

## 결정 로그 (append-only)

- 2026-06-24: 모델 `max_length=256` **불변**. 측정 근거 — 토큰/문장 분포
  ja p99=141·max=257, vi p99=68·max=212(256 이 99.98%+ 커버). 모델 천장
  ja BERT 512, vi PhoBERT 258(실효 256)=하드캡. 학습 `--max-length 256`.
  256 = PhoBERT 천장 ∧ 학습 운영점 ∧ p99 의 1.8배라 바꿀 품질 이유 없음.
- 2026-06-24: chunk 배치 추론을 **intra-text 한정**으로 국소화(회귀면
  최소). encode_row 가 각 chunk 를 max_length 패딩하므로 `[K, max_length]`
  한 forward 로 스택. multi-chunk ~1.6–2.0x 가속(GPU 측정, illustrative),
  단일 문장(1 chunk)은 배치 차원 1 = 기존과 동일. cross-text 배치는 보류.
- 2026-06-24: size 한도 의미 정정 — text·batch·합산 초과는 `413`(Payload
  Too Large), 잘못된 요청(lang·text/texts 택일)은 `400` 유지. 배치 char
  합산 가드 `MAX_TOTAL_CHARS`(기본 100000) 신설 — per-item 한도만으로는
  막지 못하는 64×20000 요청당 작업량 폭주를 차단.
- 2026-06-24: 회귀 가드 작성 중 `split_for_length` 문장-병합 경로의 offset
  버그 발견·수정 — `_sentences` 가 스킵한 공백/개행이 `cur+sent` 병합에서
  누락돼 `text[base:base+len]==sub` 불변식이 깨지던 것을 원문 슬라이스
  누적(`text[cur_start:start+len(sent)]`)으로 보존.
- 2026-06-24: 결과-시점 refuter 3라운드 — R1 PASS + parity 과대주장 1건,
  R2 parity 회귀 테스트 추가로 해소, R3 chunking 수정 검증. 누적 PASS.

## 구현 결과

- `inference.py`: `_infer_spans`(단건) → `_infer_chunks`(배치). 모든 chunk
  를 `[K, max_length]` 한 forward 로 추론하고 per-chunk offset(base 가산)
  병합. 임계값은 병합 후·canonical 변환 전 적용(parity 위치 불변).
- `chunking.py`: 문장-병합을 원문 슬라이스 누적으로 — offset 불변식 보존.
- `config.py`·`app.py`: `MAX_TOTAL_CHARS` 노브 + 배치 합산 가드, size 에러
  413 전환(잘못된 요청 400 유지).
- 테스트: 배치-vs-단건 parity(ja·vi), gold 엔티티 청크경계 비-가로지름
  전수, 장문 후반 청크 recall, 문장-병합 offset, 합산 가드 413.
- 문서: `src/server/CLAUDE.md`(배치 경로·`MAX_TOTAL_CHARS`·413),
  `issue-141` 알려진 한계의 recall 항목 검증-종결.

## 검증

- 테스트: `uv run pytest tests/server/ -m "not live"` → **40 passed**(계약
  +단위+실모델 통합), live 3 skip-guard(`/data` 의존). ruff clean.
- parity: 배치 multi-chunk == chunk별 단건 forward(label·offset·표면형
  정확, score 1e-5) — `test_{ja,vi}_batched_chunks_match_per_chunk`.
- latency(GPU, illustrative): multi-chunk ~1.6–2.0x, 단일 문장 불변.
- 결과-시점 refuter(격리 Opus): R1 `350c89accae8`·R2 `490f3f737e97`·R3
  `fa96496d8727` 모두 **PASS**(`.omc/state/refuter/`). mutation 으로 가드
  식별력 확인.

## 관련 커밋

- `078910d`: 구현 + 가드 테스트 + chunking 수정 + 문서(단일 feat 커밋)

## 후속 작업

- vi parity: vi `thresholds.json` 생성(별도 이슈, `area:classifier`) 후 vi
  operating-point parity 검증 추가.
- cross-text 요청 단위 배치: 배치 엔드포인트의 텍스트 간 chunk 까지 풀링
  하면 추가 가속 여지. 필요 시 별도 이슈.
