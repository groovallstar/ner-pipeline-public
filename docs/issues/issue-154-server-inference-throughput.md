# issue-154: 서버 추론 처리량 향상(bf16+배치) + 동시성 과부하 제어

- Issue: https://github.com/groovallstar/ner_pipeline/issues/154
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-154-server-inference-throughput`
- 승인일: 2026-06-29

## 목적

ja·vi NER REST API 의 배치 추론 처리량을 높이고(bf16 + cross-text 배치
forward) 단일 GPU 과부하를 제어한다 — 단건 지연·정밀도를 회귀시키지 않음을
게이트로 보증한다.

## 범위

- 포함: bf16 autocast(배치 한정), 배치 엔드포인트 언어별 forward 묶음, 전역
  동시성 세마포어, 처리량·parity 벤치 하네스, 테스트·문서.
- 제외: 동적 배칭(동시 단건 코얼레싱), int8/TensorRT/멀티-GPU, 배포 인프라.

## 성공 기준 (AC)

- [x] AC1 처리량: 출시(bf16+배치) ≥ 2x 동결 baseline(fp32+순차), ja·vi.
- [x] AC2 parity: production 경로 결정-불일치 ≤ max(2, 0.5%×support), 축분해.
- [x] AC3 과부하: 전역 세마포어 + in-flight 불변식 결정적 계약 테스트.
- [x] AC4 무회귀: 기존 green + 이질 배치(혼합 길이·언어) 배치=순차 등가.
- [x] AC5 문서·측정 정합: ruff+테스트 green, CLAUDE.md, 리포트↔results.

## 현 상태 (fact)

- 처리량(A6000·reps80): bf16 배치 vi 4.18x·ja 3.57x vs baseline(fp32 순차
  vi 105.51 / ja 133.54 texts/s). 배칭만 1.8~2.3x, bf16 이 배치 위 추가.
- parity(production·결정적): bf16 배치 vs fp32 순차 — vi 변경 0, ja DAT
  2·PER 2(threshold floor 2 경계 통과).
- 단건(B=1)은 fp32 로 단건 지연을 baseline 과 동일하게(불변) 유지하고, bf16
  은 처리량 이득이 입증된 배치(B>1)에만 적용한다.
- 상세 측정: `docs/reports/server-inference-throughput.md`.

## 결정 로그 (append-only)

- 2026-06-29: 설계 = 단순 배치(요청내) + bf16 + 세마포어. 동적 배칭 보류.
- 2026-06-29: 정의-시점 반박자 반영 — parity 를 production 경로·결정-불일치
  카운트·출시 구성으로, baseline 동결, 과부하를 in-flight 불변식으로 재정의.
- 2026-06-29: bf16 처리량 이득은 배치(B>1)에서만 발현(작은 [1,256] 단건은
  tensor-core 이득 미미) → batch-size 분기로 단건은 fp32(지연 baseline 불변),
  배치만 bf16. "출시=bf16+배치, 단건=fp32" 확정.
- 2026-06-29: baseline 측정이 GPU DVFS 로 ±15% 출렁 → reps 80 정상상태
  수렴으로 안정화(±1%). clock 고정·연속배수 불요.
- 2026-06-29: 서버 미배포·파이프라인 미통합(실 트래픽 0) 확인. 처리량
  최적화는 실수요 없는 선제(scale-ready)임을 인지하고 진행(사용자 결정).
  세마포어 기본값은 측정 결정 불가 → 합리적 기본 + env 조정.

## 검증

- 테스트: `uv run pytest tests/server/ -m "not live"` → 46 passed(이질
  배치·동시성 계약 포함), ruff green.
- 메트릭: 처리량·parity 상세 표 → `docs/reports/server-inference-throughput.md`
  (수치 ↔ `results/throughput_*.json`·`results/parity_*.json` 정합).

## 관련 커밋

- `641c36b`: 측정 하네스(throughput/bench·parity)
- `71c7566`: bf16 배치 추론 + cross-text 배치 forward 엔드포인트
- `1e0d0cd`: 전역 동시성 세마포어(과부하 429) + 계약 테스트
- 문서·리포트 + refuter 측정 정정: 본 커밋
