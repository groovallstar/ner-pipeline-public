# issue-106 — VI classifier 추론 비용 비교

- GitHub: #106 (area:classifier)
- 브랜치: `feat/issue-106-vi-classifier-inference-cost`
- 선행: `docs/reports/vietnamese-bert-classifier-benchmark.md` §요약 #4

## 목적

accuracy 동률(phobert 0.9461 ≈ xlm-r-base 0.9459) 상황에서 배포 선택을 가르는
비용 축을 정량화한다. accuracy는 #103 확정값 인용, 본 이슈는 **비용만 측정**.

## 핵심 관찰 (trade-off 양면성)

- pyvi 단어분절 비용은 `encode_dataset`(전처리)에 있고 forward에 안 잡힌다
  (`evaluate_model.capture_timing`은 `train_eval.py:176-212` forward만 계측).
  → encode 시간을 별도 계측해야 PhoBERT 실비용이 드러난다.
- 모델 크기 역전: fold0 `best/model.safetensors` xlm-r-base **1.1GB** vs phobert
  **0.54GB** (xlm-r 250K vocab 임베딩). → GPU 메모리·로드는 phobert 유리 가능.
- 상충(phobert: 전처리 불리·메모리 유리)이라 측정으로 결판.

## 계획 (체크박스 = 추적 단위, 커밋 단위 아님)

- [ ] `src/ner/scripts/bench_vi_inference_cost.py` — 5모델 × 비용지표 측정
  - 동일 fold0 test(7675문장, `split_kfold_stratified` seed=42, 모델 무관 결정적)
  - 토크나이저 로드는 학습 경로 미러(`use_fast=True`→phobert slow fallback→pyvi)
  - 모델은 로컬 fold0 `best/` 재사용(재학습 없음), float32(eval 정밀도)
  - GPU 1개 pin(`CUDA_VISIBLE_DEVICES=0`), `synchronize()`로 타이밍 정합
- [ ] 측정 지표: tok_load_s, model_load_s, encode_s/sent(**pyvi 분리축**),
      throughput(batch=32, sent/s), latency(batch=1, ms/sent median),
      peak GPU mem(MB), param count, disk size(MB)
- [ ] 결과 JSON → 리포트 "배포 비용" 섹션(accuracy 표 옆 cost 표)
- [ ] pytest green + ruff clean + refuter PASS

## 측정 방법 메모

- encode_s: `encode_dataset(test_rows, ...)` 전체 시간 → /7675 = per-sent.
  phobert만 pyvi가 여기 포함, fast-tok 모델은 Rust 토크나이저.
- throughput: features 배치(32) forward 루프, `synchronize` 후 wall-clock.
- latency: 첫 L=200문장 batch=1, warmup 후 문장별 synchronize→측정→median.
- peak mem: `reset_peak_memory_stats()` 후 `max_memory_allocated()` (weights+활성).

## 구현 결과

- `src/ner/scripts/bench_vi_inference_cost.py` 추가 — fold0 test 7,675문장,
  float32, A6000 1장(`CUDA_VISIBLE_DEVICES=0`), batch=32 throughput / batch=1
  latency(200문장 median). 결과 JSON:
  `results/classifier/vi/canonical5fold/inference_cost.json`(gitignore).

| 모델 | params | peak mem | encode/sent | lat b=1 | thru b=32 | F1(#103) |
|---|---:|---:|---:|---:|---:|---:|
| phobert-base-v2 | 134M | 789MB | 1.21ms(pyvi) | 6.03ms | 345/s | 0.9461 |
| xlm-roberta-base | 277M | 1,335MB | 0.22ms | 7.54ms | 343/s | 0.9459 |
| mmbert-base | 307M | 1,470MB | 0.22ms | 20.1ms | 209/s | 0.9395 |
| cafebert | 559M | 2,496MB | 0.22ms | 15.3ms | 101/s | 0.9367 |
| xlm-roberta-large | 559M | 2,496MB | 0.21ms | 15.3ms | 101/s | 0.8364 |

- **결론**: 하드웨어 비용 전부 phobert 우위(메모리 41%↓·단문 레이턴시 20%↓·
  params 절반, throughput 동률). pyvi +1ms는 무시 가능 — end-to-end로도
  phobert(7.24ms) < xlm-r-base(7.76ms). 남는 trade-off는 운영축뿐
  (pyvi 의존성·fast 토크나이저 부재·char 정렬 0.996).
- cafebert ≡ xlm-r-large는 동일 아키텍처(XLM-R-large 550M)라 비용 프로파일 일치.
- 리포트 `docs/reports/vietnamese-bert-classifier-benchmark.md`에 §배포 비용 추가.

## 검증

- [x] `pytest tests/ner/classifier/ -q` green (52 passed)
- [x] `ruff check` clean (스크립트)
- [x] refuter 게이트 PASS (`b299f21a14d8` — 숫자 일치 + 결론 재계산 검증)
