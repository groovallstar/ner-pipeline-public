"""웹 데모 번역 엔진 벤치마크 하네스.

NTREX-128 유래 JA/VI 평가셋(길이 층화 + 합성 PII 주입)으로 온프렘 LLM
번역 후보들을 프로덕션 마스킹-복원 경로(`server.translate`)에 그대로 태워
자동지표(sentinel 생존·PII 원문 보존·문장당 지연) + 중립 LLM-judge
(음차·뜻전달)로 비교한다.

- `build_eval_set` : NTREX 원문 → 층화 표본 + PII 주입 → `data/eval_set.jsonl`
- `run_bench`      : 후보 엔진 스윕 → `results/translate_bench/` 산출
"""
