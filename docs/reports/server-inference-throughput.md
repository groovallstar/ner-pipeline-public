# 서버 추론 처리량·정밀도 리포트 (bf16 배치 + 동시성)

ja·vi NER REST API 서버(`src/server/`)의 배치 추론 처리량 최적화(bf16 +
cross-text 배치 forward)와 동시성 과부하 제어 측정. 관련 이슈: #154.

> **후속 정정(historical)**: 이 리포트가 동결했던 "bf16 배치 default" 결론은
> 이후 철회됐다 — bf16 은 배치 크기에 따라 span 이 갈려(비결정) fp32 운영점과
> 어긋나므로, 서빙을 **fp32 전용**으로 고정하고 `NER_SERVER_PRECISION` 옵션·
> autocast 분기를 제거했다. 아래 처리량·동시성 수치는 유효하나(참고), 정밀도
> 선택 결론만 무효다. 현행 서빙 동작은 `docs/manual/rest-api-spec.md` 참조.

## 측정 환경 (동결)

- HW: RTX A6000 (유휴 GPU 격리 — `CUDA_VISIBLE_DEVICES`).
- 입력: 배포 test set 전체(ja 100문장 / vi 101문장), 파일 순서 고정.
- 경로: production end-to-end `predict`/`predict_many`(apply_threshold=True,
  tokenize→forward→decode→threshold). ja 임계 적용, vi 임계 부재→raw.
- 프로토콜: warmup 10 + reps 80 median, `torch.cuda.synchronize()`, 배치
  B=32. 스크립트 `src/server/scripts/throughput/{bench,parity}.py`.
- 단위 texts/sec. 수치 ↔ `results/throughput_*.json`·`results/parity_*.json`.

> **측정 방법론**: 단건 순차는 forward(~7ms)가 짧고 간헐적이라 GPU clock 이
> idle(210MHz)에 머물러 짧은 측정(reps 5~20)이 ±15% 출렁인다. reps 80 으로
> 정상상태에 수렴시켜 동결한다(±1%). clock 고정·연속배수 불요.

## 처리량 — 3축 분해 (texts/sec)

| 구성 | vi | ja |
|---|---|---|
| fp32 순차 (baseline) | 105.51 | 133.54 |
| fp32 배치 (배칭만) | 240.03 (2.27x) | 242.10 (1.81x) |
| **bf16 배치 (출시)** | **441.52 (4.18x)** | **476.54 (3.57x)** |

- **배칭이 주 이득**(1.8~2.3x): 여러 텍스트를 `[N,256]` 한 forward 로 묶어
  GPU 가 병렬 처리. **bf16 은 배치 위 추가 레버**(~1.6~2.0x): 큰 행렬에서
  tensor core 활용. 합쳐 ≥2x 게이트(vi 211 / ja 267) 초과.
- 단건(B=1)은 fp32 로 둬 **단건 지연을 baseline 과 동일하게(불변)** 유지하고,
  bf16 autocast 는 이득이 입증된 배치(B>1) forward 에만 켠다 — 작은 `[1,256]`
  행렬은 tensor-core 이득이 작아 bf16 의 처리량 이점이 배치에서만 나타난다.

## 정밀도 parity — 출시 구성 게이트

production 경로(apply_threshold=True) 결정-불일치 카운트. 게이트: 엔티티별 변경
span ≤ max(2, 0.5%×support). bf16 run-to-run 결정적(3-run 대칭차 0).

| 축 | vi 변경 | ja 변경 | micro-F1(진단) |
|---|---|---|---|
| **bf16 배치 vs fp32 순차 (출시 게이트)** | **0** | **DAT 2 · PER 2** | vi 1.0 / ja 0.9954 |

- **vi 완전 일치**(변경 0). **ja 게이트 통과** — DAT(support 25)·PER
  (support 135) 각 2건이 threshold floor 2 로 **경계 통과**. 결정적이라 run
  변동이 아닌 fp32↔bf16 고정 차이다(경계임을 명시).
- 배치 커널 드리프트: fp32 라도 배치 크기에 따라 GPU matmul 커널이 달라
  score 가 ~1e-7 흔들리나 결정(label·offset)은 불변 — 배치=순차 등가 테스트로
  고정.

## 동시성 과부하 제어

- 전역 세마포어 `NER_SERVER_MAX_CONCURRENCY`(기본 8) + 대기 큐
  `MAX_QUEUE`(32) + 획득 타임아웃 `ACQUIRE_TIMEOUT_S`(10s) → 초과 시 429.
  in-flight ≤ N 불변식·max=N 포화·큐/타임아웃 거절을 결정적 계약 테스트로
  단언(`tests/server/test_concurrency.py`).
- 기본값은 실 트래픽 부재로 측정 결정 불가 → 합리적 기본 + env 조정.

## 결론

출시(bf16 배치) 처리량 vi 4.18x·ja 3.57x(≥2x 게이트 통과), production parity
게이트 통과(vi 0 · ja 경계 4건, 결정적), 단건 지연 불변(fp32 유지). 단
**서버는 현재 미배포·파이프라인 미통합(실 트래픽 0)** — 본 최적화는
scale-ready 선제 구현이며, 실수요 발생 시 세마포어 기본값 등을 재측정한다.
