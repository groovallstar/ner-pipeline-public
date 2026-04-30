"""PII 주입 대상 데이터셋 로더 (Stockmark / JSONL / HF Hub)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ner.augmenters.pii.schema import Entity, Record


def load_stockmark(
    split: str = 'train',
    max_samples: Optional[int] = None,
) -> list[Record]:
    """Stockmark canonical 덤프를 Record 리스트로 로드한다.

    이슈 #21 이후 로더는 `data/stockmark/{train,test}.jsonl` canonical
    덤프를 직접 읽는다. 분할 seed·비율 파라미터는 더 이상 런타임에서
    지정하지 않으며, 덤프 시점에 고정된다.
    """
    from ner.labelers.ja.dataset_loader import JapaneseDatasetLoader
    loader = JapaneseDatasetLoader()
    raw = loader.load(
        split=split,
        max_samples=max_samples,
    )
    records: list[Record] = []
    for row in raw:
        ents = [
            Entity(
                label=g['type'],
                start_char=int(g['start']),
                end_char=int(g['end']),
                text=g['text'],
            )
            for g in row.get('gold_spans', [])
        ]
        records.append(Record(
            text=row['text'], entities=ents, id=row.get('id')
        ))
    return records


def load_jsonl(
    path: str | Path,
    max_samples: Optional[int] = None,
) -> list[Record]:
    """`{text, entities}` 스키마의 JSONL 파일을 로드한다."""
    records: list[Record] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(Record.from_dict(data))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def load_hf(
    name: str,
    split: str = 'train',
    text_field: str = 'text',
    entities_field: str = 'entities',
    label_key: str = 'type',
    start_key: str = 'start',
    end_key: str = 'end',
    text_key: str = 'name',
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> list[Record]:
    """HF dataset을 Record 리스트로 읽어온다.

    각 row가 `{text_field: str, entities_field: list[dict]}` 스키마일 때
    동작한다. 엔티티 필드 키는 파라미터로 조정 가능.
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split, cache_dir=cache_dir)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    records: list[Record] = []
    for i, row in enumerate(ds):
        text = row[text_field]
        ents_raw = row.get(entities_field, []) or []
        ents: list[Entity] = []
        for e in ents_raw:
            # span: [start, end] 또는 start/end 별도 키를 지원한다.
            if 'span' in e and isinstance(e['span'], (list, tuple)):
                s, ee = int(e['span'][0]), int(e['span'][1])
            else:
                s, ee = int(e[start_key]), int(e[end_key])
            label = e.get(label_key, 'MISC')
            et = e.get(text_key, text[s:ee])
            ents.append(Entity(
                label=label, start_char=s, end_char=ee, text=et
            ))
        records.append(Record(text=text, entities=ents, id=str(i)))
    return records
