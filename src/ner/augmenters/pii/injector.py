"""PII 주입 핵심 로직.

기존 NER 레코드의 텍스트 끝(또는 문장 말미)에 자연스러운 커넥터 문구와 함께
PII를 1~N개 주입하고, 원본 엔티티 오프셋과 신규 PII 오프셋을 정확히
재계산한다.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.generators.base import generate_pii
from ner.augmenters.pii.schema import Entity, Record


def apply_label_merge(entity: Entity, rules: dict[str, str]) -> Entity:
    """단일 엔티티에 병합 규칙을 적용한 새 엔티티를 반환한다.

    `NAME → PER`, `ADDRESS → LOC` 등 내부 생성 토큰을 canonical 라벨로
    무조건 변환한다 (#17 Phase 1에서 `ADDRESS` 병합 조건부 로직 제거).
    """
    new_label = rules.get(entity.label, entity.label)
    if new_label == entity.label:
        return entity
    return Entity(
        label=new_label,
        start_char=entity.start_char,
        end_char=entity.end_char,
        text=entity.text,
    )


def merge_entities(
    entities: list[Entity], rules: dict[str, str],
) -> list[Entity]:
    """엔티티 리스트에 병합 규칙을 적용한다."""
    return [apply_label_merge(e, rules) for e in entities]

# 언어별 자연스러운 커넥터 문구.
_JA_CONNECTORS: dict[str, str] = {
    'NAME': '担当者：{v}。',
    'PHONE': '連絡先：{v}。',
    'ADDRESS': '住所：{v}。',
    'DAT': '日付：{v}。',
    'ID_NUM': 'ID番号：{v}。',
    'EMAIL': 'メール：{v}。',
    'CREDIT_CARD': 'カード番号：{v}。',
}

_VI_CONNECTORS: dict[str, str] = {
    'NAME': ' Liên hệ: {v}.',
    'PHONE': ' SĐT: {v}.',
    'ADDRESS': ' Địa chỉ: {v}.',
    'DAT': ' Ngày: {v}.',
    'ID_NUM': ' CCCD: {v}.',
    'EMAIL': ' Email: {v}.',
    'CREDIT_CARD': ' Thẻ: {v}.',
}


class PIIInjector:
    """NER 레코드에 합성 PII를 주입한다."""

    def __init__(self, config: InjectionConfig) -> None:
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        self._connectors = (
            _JA_CONNECTORS if config.lang == 'ja' else _VI_CONNECTORS
        )
        self._labels = config.effective_labels()

    def _sample_n(self) -> int:
        """density 분포에서 주입할 PII 개수를 샘플링한다."""
        items = sorted(self.config.density.items())
        ks = [k for k, _ in items]
        ws = [v for _, v in items]
        return self.rng.choices(ks, weights=ws, k=1)[0]

    def _pick_labels(self, n: int) -> list[str]:
        """n 개 라벨을 중복 없이 선택한다 (가능 범위 내)."""
        if n <= 0:
            return []
        pool = list(self._labels)
        if n >= len(pool):
            self.rng.shuffle(pool)
            return pool
        return self.rng.sample(pool, n)

    def inject(self, record: Record) -> Record:
        """단일 레코드에 PII를 주입하고 새 Record를 반환한다."""
        n = self._sample_n()
        labels = self._pick_labels(n)

        # 텍스트 끝에 주입 (suffix). 문장 종결 부호가 없으면 보강.
        base_text = record.text
        suffix_separator = ''
        if base_text and not base_text.endswith(('。', '.', '!', '?', '！', '？')):
            suffix_separator = '。' if self.config.lang == 'ja' else '.'

        new_text = base_text + suffix_separator
        # 기존 엔티티 offset은 base_text를 건드리지 않으므로 유지된다.
        new_entities: list[Entity] = [
            Entity(
                label=e.label,
                start_char=e.start_char,
                end_char=e.end_char,
                text=e.text,
            )
            for e in record.entities
        ]

        for label in labels:
            value = generate_pii(label, self.config.lang, self.rng)
            template = self._connectors.get(label, ' {v}')
            # 값이 들어갈 위치를 정확히 계산하기 위해 템플릿을
            # prefix + {v} + suffix 로 분해한다.
            before, after = template.split('{v}')
            insert_start = len(new_text) + len(before)
            insert_end = insert_start + len(value)
            new_text = new_text + before + value + after
            new_entities.append(Entity(
                label=label,
                start_char=insert_start,
                end_char=insert_end,
                text=value,
            ))

        # 라벨 병합 규칙 적용 (NAME -> PER 등).
        merged = merge_entities(new_entities, self.config.label_merge_rules)

        # 오프셋 검증 (실패 시 ValueError).
        for ent in merged:
            if new_text[ent.start_char:ent.end_char] != ent.text:
                raise ValueError(
                    f'span offset mismatch: label={ent.label} '
                    f'expected={ent.text!r} '
                    f'got={new_text[ent.start_char:ent.end_char]!r}'
                )

        return Record(text=new_text, entities=merged, id=record.id)

    def inject_dataset(
        self, records: Iterable[Record]
    ) -> Iterator[Record]:
        """레코드 iterable을 주입 결과로 변환하는 제너레이터."""
        for rec in records:
            yield self.inject(rec)
