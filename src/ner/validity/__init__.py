"""K-fold 실험 결과의 비교 유효성 게이트 (학습 재실행 0회).

기존 K-fold 산출물(`fold{N}/metrics.json`·`pooled_metrics.json`)만 읽어 두
실험을 비교해도 되는지, 개선이 노이즈인지 실측인지 판정한다. 세 축 + 통합:

- `comparability` — 같은 자(RULER: lang·data_fingerprint·kfold·group_key·
  seed·stratify)로 쟀나. 지문이 경로를 대체해 gold 를 고치면 비교 거부.
- `leakage` — 신뢰 근거로 센 pooled 누출 카운터가 0 인가.
- `variance` — pooled Δ vs σ_repro/σ_fold 밴드 + fold 방향 일관성(조이기 전용).
- `gate.compare` — 셋을 묶어 PASS|FAIL|INVALID|INCONCLUSIVE.

이 `__init__` 은 축 모듈의 공개 이름을 그대로 re-export 한다 —
`from ner.validity import compare`·CLI `python -m ner.validity`. metrics/variance
서브패키지에서 승격됐으며, compare() 의 verdict 는 이동 전과 byte-동일하다 —
golden 특성화 테스트가 이를 잠근다.

세 검증은 각자 독립 함수로 호출 가능하다. 다만 어떤 검증 부분집합을 통과해야
'실험 완료'인지의 조합·우선순위·선언은 이 패키지의 몫이 아니다 — 그것은
채점규칙(자)을 바꾸는 '나중 하네스'가 정의하며, 그때 정의-시점 반박자를 건다.
특히 `variance` 의 paired Δ 는 comparability 가 성립할 때만 유효하므로, 하네스는
variance 를 comparability 없이 조합해선 안 된다.
"""
from ner.validity.comparability import (
    RULER_FIELDS,
    check_comparable,
    run_config,
)
from ner.validity.gate import compare
from ner.validity.leakage import (
    TRUSTED_LEAK_BASIS,
    leakage,
    leakage_dups,
)
from ner.validity.variance import (
    MIN_CONSISTENCY_FRAC,
    fold_paired_deltas,
    fold_std,
    load_sigma_map,
    repro_std,
    write_sigma_repro,
)
from ner.validity._common import (
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
