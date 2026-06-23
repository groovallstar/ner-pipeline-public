"""신뢰도 임계값 graceful 로딩·적용 — 서버 자족 구현.

classifier 의 임계값 모듈에 의존하지 않고 thresholds.json 형식만 contract
로 쓴다. 형식:

    {"thresholds": {"ORG": 0.74, "LOC": 0.66, ...}, "meta": {...}}

span 의 `score`(decode 의 conf_mean)가 해당 type 임계값 미만이면 제거한다.
임계값에 없는 type, 또는 score 키가 없는 span 은 통과시킨다(graceful).
파일이 없으면 빈 dict 를 반환해 abstention 미적용(raw)으로 동작한다.
"""

import json
import os
from typing import Dict, List


def load_thresholds(path: str) -> Dict[str, float]:
    """thresholds.json → {type: threshold}. 파일 없으면 빈 dict(raw)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    return {k: float(v) for k, v in payload.get('thresholds', {}).items()}


def apply_thresholds(spans: List[dict],
                     thresholds: Dict[str, float]) -> List[dict]:
    """score < thresholds[label] 인 span 을 제거한 새 리스트를 반환.

    thresholds 가 비었으면 입력을 그대로 통과시킨다. span 은 canonical 형식
    ({label, start_char, end_char, text, score})을 가정한다.
    """
    if not thresholds:
        return spans
    out: List[dict] = []
    for s in spans:
        thr = thresholds.get(s.get('label'))
        if thr is not None and s.get('score', 1.0) < thr:
            continue
        out.append(s)
    return out
