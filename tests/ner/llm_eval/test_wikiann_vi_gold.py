"""WikiANN-vi gold 추출 단위 테스트.

HF 데이터셋 로딩을 monkeypatch 로 가짜 객체로 대체해, 토큰·BIO 입력에
대한 변환 정확성과 PER/LOC/ORG 필터링을 검증한다.
"""
from __future__ import annotations

from typing import List

import pytest

from ner.llm_eval import wikiann_vi_gold


class _FakeFeature:
    """ClassLabel 흉내 — int2str 만 지원."""
    def __init__(self, mapping: dict):
        self._mapping = mapping

    def int2str(self, idx):
        return self._mapping[idx]


class _FakeDataset:
    def __init__(self, rows: List[dict], feature_map: dict):
        self._rows = rows
        self.features = {
            'ner_tags': type('F', (), {'feature': _FakeFeature(feature_map)})(),
        }

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        sub = [self._rows[i] for i in indices]
        new = _FakeDataset.__new__(_FakeDataset)
        new._rows = sub
        new.features = self.features
        return new


def _patch_loader(monkeypatch, fake_dataset):
    monkeypatch.setattr(
        wikiann_vi_gold,
        'load_dataset',
        lambda *a, **kw: fake_dataset,
    )
    monkeypatch.setattr(
        wikiann_vi_gold, 'ClassLabel',
        type(fake_dataset.features['ner_tags'].feature),
    )


def test_basic_per_loc_org(monkeypatch):
    rows = [
        {
            'tokens': ['Nguyễn', 'Văn', 'A', 'sống', 'ở', 'Hà', 'Nội'],
            'ner_tags': [1, 2, 2, 0, 0, 3, 4],
        },
    ]
    fmap = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
    _patch_loader(monkeypatch, _FakeDataset(rows, fmap))

    records = wikiann_vi_gold.load_wikiann_vi_gold('test')
    assert len(records) == 1
    rec = records[0]
    assert rec['id'] == '0'
    assert rec['text'] == 'Nguyễn Văn A sống ở Hà Nội'
    assert len(rec['gold_spans']) == 2
    types = sorted(s['type'] for s in rec['gold_spans'])
    assert types == ['LOC', 'PER']


def test_diacritic_offsets(monkeypatch):
    """베트남어 diacritic 토큰의 multi-byte offset 정확성 점검."""
    rows = [
        {
            'tokens': ['Bình', 'Dương', 'là', 'tỉnh'],
            'ner_tags': [3, 4, 0, 0],
        },
    ]
    fmap = {0: 'O', 3: 'B-LOC', 4: 'I-LOC'}
    _patch_loader(monkeypatch, _FakeDataset(rows, fmap))

    records = wikiann_vi_gold.load_wikiann_vi_gold('test')
    span = records[0]['gold_spans'][0]
    text = records[0]['text']
    assert span['type'] == 'LOC'
    # text 의 슬라이스가 정확한 표면형을 가리켜야 함
    assert text[span['start']:span['end']] == 'Bình Dương'


def test_filters_unknown_types(monkeypatch):
    """PER/LOC/ORG 외 타입은 안전망 필터로 제거된다."""
    rows = [
        {
            'tokens': ['X', 'Y'],
            'ner_tags': [5, 6],
        },
    ]
    fmap = {5: 'B-MISC', 6: 'I-MISC'}
    _patch_loader(monkeypatch, _FakeDataset(rows, fmap))

    records = wikiann_vi_gold.load_wikiann_vi_gold('test')
    assert records[0]['gold_spans'] == []


def test_max_samples(monkeypatch):
    rows = [
        {'tokens': ['a'], 'ner_tags': [0]},
        {'tokens': ['b'], 'ner_tags': [0]},
        {'tokens': ['c'], 'ner_tags': [0]},
    ]
    fmap = {0: 'O'}
    _patch_loader(monkeypatch, _FakeDataset(rows, fmap))

    records = wikiann_vi_gold.load_wikiann_vi_gold('test', max_samples=2)
    assert len(records) == 2
    assert [r['id'] for r in records] == ['0', '1']


def test_invalid_lengths_raises(monkeypatch):
    """tokens·tags 길이 mismatch 시 bio_to_offset_spans 가 raise."""
    rows = [{'tokens': ['a', 'b'], 'ner_tags': [0]}]
    fmap = {0: 'O'}
    _patch_loader(monkeypatch, _FakeDataset(rows, fmap))

    with pytest.raises(ValueError):
        wikiann_vi_gold.load_wikiann_vi_gold('test')
