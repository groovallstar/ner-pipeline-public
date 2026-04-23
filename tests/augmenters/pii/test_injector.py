"""PIIInjector 동작 검증."""
from __future__ import annotations

from collections import Counter

from augmenters.pii.config import InjectionConfig
from augmenters.pii.injector import PIIInjector
from augmenters.pii.schema import Entity, Record


def _make_record() -> Record:
    text = '山田太郎は東京都に住んでいる'
    ents = [
        Entity(label='PER', start_char=0, end_char=4, text='山田太郎'),
        Entity(label='LOC', start_char=5, end_char=8, text='東京都'),
    ]
    return Record(text=text, entities=ents, id='r1')


def test_spans_consistent_after_injection():
    cfg = InjectionConfig(lang='ja', seed=1)
    inj = PIIInjector(cfg)
    for i in range(50):
        rec = _make_record()
        out = inj.inject(rec)
        for ent in out.entities:
            assert out.text[ent.start_char:ent.end_char] == ent.text, (
                f'iter={i} label={ent.label}'
            )


def test_density_distribution_respected():
    # 1 PII 고정 분포 → 항상 원본 엔티티 2개 + 신규 1개
    cfg = InjectionConfig(
        lang='ja',
        density={0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
        seed=7,
    )
    inj = PIIInjector(cfg)
    for _ in range(20):
        out = inj.inject(_make_record())
        assert len(out.entities) == 3  # 원본 2 + 신규 1


def test_density_distribution_approx():
    cfg = InjectionConfig(
        lang='ja',
        density={0: 0.2, 1: 0.4, 2: 0.3, 3: 0.1},
        seed=42,
    )
    inj = PIIInjector(cfg)
    counts: Counter[int] = Counter()
    n = 2000
    for _ in range(n):
        out = inj.inject(Record(text='テスト', entities=[]))
        counts[len(out.entities)] += 1
    # 각 버킷이 기대 ±10%p 이내
    for k, expected in cfg.density.items():
        observed = counts[k] / n
        assert abs(observed - expected) < 0.05, (k, observed, expected)


def test_seed_determinism():
    cfg1 = InjectionConfig(lang='ja', seed=123)
    cfg2 = InjectionConfig(lang='ja', seed=123)
    a = list(PIIInjector(cfg1).inject_dataset(
        [_make_record() for _ in range(5)]
    ))
    b = list(PIIInjector(cfg2).inject_dataset(
        [_make_record() for _ in range(5)]
    ))
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_label_merge_name_to_jinmei():
    cfg = InjectionConfig(
        lang='ja',
        density={0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
        pii_labels=['NAME'],
        seed=2,
    )
    inj = PIIInjector(cfg)
    out = inj.inject(Record(text='テスト', entities=[]))
    labels = [e.label for e in out.entities]
    assert 'PER' in labels
    assert 'NAME' not in labels
