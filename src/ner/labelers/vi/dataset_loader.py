"""WikiANN-vi canonical 덤프 로더.

기본 split 파일(`data/wikiann_vi/{train,valid,test}.jsonl`)을 읽어
`{id, text, gold_spans:[{text, type, start, end}]}` 스키마로 반환한다(폴백 없음).
입력은 silver 재라벨과 PII 주입을 거쳐 materialize 된 최종 덤프이며 `entities`
스키마(`{label, start_char, end_char, text}`)다. 임의 경로의 같은 스키마 JSONL 은
`load_local` 로 읽는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union


class VietnameseDatasetLoader:
    """canonical WikiANN-vi 덤프를 gold_spans 레코드로 반환한다."""

    DEFAULT_PATH: dict = {
        'train': Path('data/wikiann_vi/train.jsonl'),
        'validation': Path('data/wikiann_vi/valid.jsonl'),
        'test': Path('data/wikiann_vi/test.jsonl'),
    }

    def load(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        """WikiANN-vi 기본 split 덤프를 gold_spans 레코드로 로드한다.

        Args:
            split: 'train' | 'validation' | 'test'.
            max_samples: 반환할 레코드 수 상한.

        파일이 없으면 `FileNotFoundError`. 폴백 없음.
        """
        if split not in self.DEFAULT_PATH:
            raise ValueError(
                f'unknown split {split!r}; expected one of '
                f'{list(self.DEFAULT_PATH)}'
            )
        target = self.DEFAULT_PATH[split]
        if not target.exists():
            raise FileNotFoundError(
                f'WikiANN-vi dump not found: {target}. Read an existing '
                'canonical JSONL with load_local() instead.'
            )
        return self.load_local(target, max_samples=max_samples)

    @staticmethod
    def load_local(
        path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        """임의 JSONL을 labelers/vi 스키마로 읽는다.

        입력은 PII 증강 산출물과 같은 `entities` 스키마다.
        """
        records: List[dict] = []
        with open(Path(path), encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                gold_spans = [
                    {
                        'text': ent.get('text', ''),
                        'type': ent.get(
                            'label', ent.get('type', '')
                        ),
                        'start': int(ent.get(
                            'start_char', ent.get('start', 0)
                        )),
                        'end': int(ent.get(
                            'end_char', ent.get('end', 0)
                        )),
                    }
                    for ent in data.get('entities', [])
                ]
                records.append({
                    'id': str(data.get('id', i)),
                    'text': data['text'],
                    'gold_spans': gold_spans,
                })
                if (max_samples is not None
                        and len(records) >= max_samples):
                    break
        return records
