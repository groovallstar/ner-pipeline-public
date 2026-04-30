"""JSONL → JapaneseDatasetLoader.load_local 왕복 통합 테스트."""
from __future__ import annotations

import json
from pathlib import Path

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.injector import PIIInjector
from ner.augmenters.pii.schema import Entity, Record


def test_load_local_roundtrip(tmp_path: Path):
    recs = [
        Record(
            text='山田太郎は東京都に住んでいる',
            entities=[
                Entity(label='PER', start_char=0, end_char=4,
                       text='山田太郎'),
                Entity(label='LOC', start_char=5, end_char=8,
                       text='東京都'),
            ],
            id='r1',
        ),
        Record(text='テスト文', entities=[], id='r2'),
    ]
    cfg = InjectionConfig(lang='ja', seed=9)
    injector = PIIInjector(cfg)
    out_path = tmp_path / 'out.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in injector.inject_dataset(recs):
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')

    from ner.labelers.ja import JapaneseDatasetLoader
    loaded = JapaneseDatasetLoader.load_local(str(out_path))
    assert len(loaded) == 2
    for rec in loaded:
        assert set(rec.keys()) == {'id', 'text', 'gold_spans'}
        for sp in rec['gold_spans']:
            assert set(sp.keys()) == {'text', 'type', 'start', 'end'}
            assert rec['text'][sp['start']:sp['end']] == sp['text']
