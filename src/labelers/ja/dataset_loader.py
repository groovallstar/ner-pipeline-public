"""Stockmark NER canonical 덤프 로더.

이슈 #21에서 canonical 5종으로 축소된 JSONL 덤프
(`data/stockmark/{train,test}.jsonl`)를 그대로 읽는다. HF 원본 로딩과
라벨 매핑은 본 모듈의 책임이 아니다 — 원본 재덤프가 필요한 경우는 1회성
마이그레이션 도구의 영역이다. 폴백·자동 매핑 없음.

출력 스키마: `{id, text, gold_spans:[{text, type, start, end}]}`
— `gold_spans[*].type`은 canonical 문자열을 그대로 담는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union


class JapaneseDatasetLoader:
    """canonical Stockmark 덤프를 gold_spans 레코드로 반환한다."""

    DEFAULT_PATH: dict = {
        'train': Path('data/stockmark/train.jsonl'),
        'test': Path('data/stockmark/test.jsonl'),
    }

    def load(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> List[dict]:
        """canonical Stockmark 덤프를 로드한다.

        Args:
            split: 'train' 또는 'test'. `path` 지정 시 무시한다.
            max_samples: 반환할 레코드 수 상한.
            path: 임의 JSONL 경로를 지정해 로드. split 대신 사용.

        파일이 없으면 `FileNotFoundError`. 폴백이나 HF 자동 로딩은 없다.
        """
        if path is not None:
            target = Path(path)
        elif split in self.DEFAULT_PATH:
            target = self.DEFAULT_PATH[split]
        else:
            raise ValueError(
                f'unknown split {split!r}; expected one of '
                f'{list(self.DEFAULT_PATH)}'
            )

        if not target.exists():
            raise FileNotFoundError(
                f'Stockmark canonical dump not found: {target}. '
                'Generate the canonical dump via a 1회성 augmenters '
                'migration pipeline before running the loader.'
            )
        return _read_records(target, max_samples)

    @staticmethod
    def load_local(
        path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        """임의 경로 JSONL을 동일 스키마로 읽는다.

        `load(path=...)`와 동등. 기존 호출 호환용으로 보존한다.
        """
        return _read_records(Path(path), max_samples)


def _read_records(
    path: Path, max_samples: Optional[int],
) -> List[dict]:
    """JSONL을 gold_spans 레코드 리스트로 반환한다.

    입력 스키마:
        `{id, text, entities:[{label, start_char, end_char, text}]}`
    출력 스키마:
        `{id, text, gold_spans:[{text, type, start, end}]}`

    entities 필드의 키는 `label|type`, `start_char|start`,
    `end_char|end` 두 표기 모두 허용한다(PII 증강 산출물 호환).
    """
    records: List[dict] = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            gold_spans = [
                {
                    'text': ent.get('text', ''),
                    'type': ent.get('label', ent.get('type', '')),
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
            if max_samples is not None and len(records) >= max_samples:
                break
    return records
