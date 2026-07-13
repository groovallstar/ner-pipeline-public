"""K-fold 실험 결과의 비교 유효성 게이트 (학습 재실행 0회).

기존 K-fold 산출물(`fold{N}/metrics.json`·`pooled_metrics.json`)만 읽어 두
실험을 비교해도 되는지, 개선이 노이즈인지 실측인지 판정한다. 세 축 + 통합:

- `comparability` — 같은 자(RULER: lang·data_fingerprint·kfold·group_key·
  seed·stratify)로 쟀나. 지문이 경로를 대체해 gold 를 고치면 비교 거부.
- `leakage` — 신뢰 근거로 센 pooled 누출 카운터가 0 인가.
- `noise` — pooled Δ vs σ_repro/σ_fold 밴드 + fold 방향 일관성(조이기 전용).
- `gate.compare` — 셋을 묶어 PASS|FAIL|INVALID|INCONCLUSIVE.

이 `__init__` 은 축 모듈의 공개 이름을 그대로 re-export 한다 — import 경로
`from ner.metrics.variance import compare` 와 CLI `python -m ner.metrics.variance`
는 분리 전과 동일하다(무동작 refactor).
"""
from ner.metrics.variance.comparability import (
    RULER_FIELDS,
    check_comparable,
    run_config,
)
from ner.metrics.variance.gate import compare
from ner.metrics.variance.leakage import (
    TRUSTED_LEAK_BASIS,
    leakage,
    leakage_dups,
)
from ner.metrics.variance.noise import (
    MIN_CONSISTENCY_FRAC,
    fold_paired_deltas,
    fold_std,
    load_sigma_map,
    repro_std,
    write_sigma_repro,
)
from ner.metrics.variance._common import (
    load_fold_metrics,
    load_pooled_metrics,
)

__all__ = [
    "RULER_FIELDS",
    "TRUSTED_LEAK_BASIS",
    "MIN_CONSISTENCY_FRAC",
    "check_comparable",
    "run_config",
    "leakage",
    "leakage_dups",
    "fold_std",
    "fold_paired_deltas",
    "repro_std",
    "write_sigma_repro",
    "load_sigma_map",
    "load_fold_metrics",
    "load_pooled_metrics",
    "compare",
]
