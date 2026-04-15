"""PII 주입 결과를 표현하는 데이터 클래스.

labelers/ 호환 JSONL 스키마 `{text, entities:[{label, start_char, end_char,
text}]}`와 동일하다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Entity:
    """문자 오프셋 기반 NER 엔티티."""
    label: str
    start_char: int
    end_char: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Entity':
        return cls(
            label=d['label'],
            start_char=int(d['start_char']),
            end_char=int(d['end_char']),
            text=d['text'],
        )


@dataclass
class Record:
    """JSONL 한 줄 = 텍스트 + 엔티티 목록."""
    text: str
    entities: list[Entity] = field(default_factory=list)
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'text': self.text,
            'entities': [e.to_dict() for e in self.entities],
        }
        if self.id is not None:
            out['id'] = self.id
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Record':
        ents = [Entity.from_dict(e) for e in d.get('entities', [])]
        return cls(text=d['text'], entities=ents, id=d.get('id'))
