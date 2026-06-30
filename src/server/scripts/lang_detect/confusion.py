"""혼동행렬 평가 하네스 — 감지기 출력 × gold expected 라벨.

coverage(규칙이 자기 코퍼스를 얼마나 덮나)는 규칙 술어의 재진술이라 쓰지
않는다. 대신 gold expected 라벨에 대한 클래스별 precision/recall 과 버킷별
recall(예측==expected 비율)을 산출한다 — falsifiable·non-gameable.
"""

import json
from collections import defaultdict

CLASSES = ('ja', 'vi', 'unsupported')


def load_gold(path):
    """gold.jsonl 을 행 리스트로 읽는다."""
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def evaluate(detector, rows):
    """detector 를 각 행에 적용해 혼동행렬·클래스 P/R·버킷 recall 산출.

    confusion[expected][predicted]=count. 반환 dict: confusion,
    per_class(precision/recall/tp/fp/fn), per_bucket(total/hit/recall/
    pred_dist).
    """
    confusion = {e: defaultdict(int) for e in CLASSES}
    b_total = defaultdict(int)
    b_hit = defaultdict(int)
    b_pred = defaultdict(lambda: defaultdict(int))
    for row in rows:
        expected = row['expected']
        pred = detector(row['text'])
        confusion[expected][pred] += 1
        bucket = row['bucket']
        b_total[bucket] += 1
        b_pred[bucket][pred] += 1
        if pred == expected:
            b_hit[bucket] += 1
    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fn = sum(v for p, v in confusion[c].items() if p != c)
        fp = sum(confusion[e][c] for e in CLASSES if e != c)
        per_class[c] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
        }
    per_bucket = {
        b: {'total': b_total[b], 'hit': b_hit[b],
            'recall': b_hit[b] / b_total[b] if b_total[b] else None,
            'pred_dist': dict(b_pred[b])}
        for b in b_total
    }
    return {
        'confusion': {e: dict(confusion[e]) for e in CLASSES},
        'per_class': per_class,
        'per_bucket': per_bucket,
    }
