---
name: perf-measure
description: Measure-first performance optimization for NER benchmarks and labelers. Use when LLM throughput/latency matters, when tuning vLLM batch/context settings, or when a change is suspected to have regressed speed. Never optimize without measurement.
metadata:
  source: https://github.com/addyosmani/agent-skills/tree/main/skills/performance-optimization
  adaptedFor: ner-pipeline
---

# Performance Measurement (NER 파이프라인 맞춤)

**원칙**: 측정 없는 최적화 = 추측. 추측은 복잡도만 늘린다.

## 워크플로

```
MEASURE → IDENTIFY → FIX → VERIFY → GUARD
```

### 1. MEASURE — 베이스라인 수립
- 소규모 재현 가능한 벤치마크 확정 (고정 `--max-samples`, 고정 모델, 고정 시드)
- 수집 지표:
  - wall-clock time, tokens/s
  - GPU util / VRAM (`nvidia-smi dmon`)
  - 입·출력 토큰 분포 (평균/최대)
  - 정확도 지표 (F1 등) — 속도 최적화가 품질을 희생하지 않았는지 확인

### 2. IDENTIFY — 실제 병목 찾기
- 프로파일 대상: I/O(데이터 로딩) vs 프롬프트 구성 vs LLM 호출 vs 후처리(BIO 정렬) vs 평가
- 간단한 타이밍 로그 → `time.perf_counter()` 구간 측정
- vLLM 서버: 서버 로그의 배치·큐잉 지표 확인

### 3. FIX — 병목만 건드린다
- 주요 튜닝 축:
  - vLLM: `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, 양자화
  - 클라이언트: 동시성, 배치 그룹 크기, 샘플링 파라미터
  - 프롬프트: few-shot 수, 예시 길이
- **한 번에 한 변수만**. 변경 전후 지표 기록.

### 4. VERIFY — 재측정
- 동일 베이스라인 조건에서 재실행
- 품질 지표도 함께 확인 (F1/precision/recall)

### 5. GUARD — 회귀 방지
- 결과 JSON 보존 (`results/`) + 변경 요약을 `docs/guides/` 또는 `docs/reports/`에 기록
- 임계값 이탈 시 경보할 수 있는 체크 추가 고려

## 하지 말 것

- 증거 없이 코드 리팩터링으로 "빠르게 만들기"
- 여러 변경을 한 번에 적용
- 벤치마크 N을 바꾸면서 비교

상위 개념: `docs/wiki/concepts/agent-skills.md`.
