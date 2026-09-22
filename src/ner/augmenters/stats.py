"""생성된 PII 데이터셋 통계."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from ner.augmenters.schema import Record


def compute_stats(records: Iterable[Record]) -> dict[str, Any]:
    """라벨별 빈도, 문자 커버리지, PII 없는 샘플 비율을 계산한다."""
    records = list(records)
    total = len(records)
    label_counts: Counter[str] = Counter()
    label_chars: Counter[str] = Counter()
    total_chars = 0
    no_pii_samples = 0

    for rec in records:
        total_chars += len(rec.text)
        if not rec.entities:
            no_pii_samples += 1
            continue
        for ent in rec.entities:
            label_counts[ent.label] += 1
            label_chars[ent.label] += (ent.end_char - ent.start_char)

    per_label: dict[str, dict[str, Any]] = {}
    for label, cnt in label_counts.items():
        chars = label_chars[label]
        per_label[label] = {
            'count': cnt,
            'char_coverage': chars,
            'char_ratio': (chars / total_chars) if total_chars else 0.0,
        }

    return {
        'total_samples': total,
        'total_chars': total_chars,
        'samples_no_pii': no_pii_samples,
        'samples_no_pii_ratio': (
            no_pii_samples / total if total else 0.0
        ),
        'per_label': per_label,
    }
