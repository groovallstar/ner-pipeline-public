"""PII 주입 결과를 LLM 교차 검증하는 모듈.

주입된 골드 엔티티를 LLM(vLLM 등)이 독립적으로 추출한 예측과 대조하여
confirmed / missed / conflict 로 분류하고, 정책에 따라 레코드를 정제한다.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from augmenters.pii.schema import Entity, Record

logger = logging.getLogger(__name__)


# ── 프로토콜 ─────────────────────────────────────────────────────────────

@runtime_checkable
class SpanLabeler(Protocol):
    """label_spans(text, split=...) -> List[dict] 를 구현하는 라벨러.

    verifier는 문맥 보존을 위해 split=False로 호출한다. 모의 라벨러는
    split 인자를 선택적으로 수용하면 된다.
    """

    def label_spans(self, text: str) -> list[dict]: ...


# ── 정책 / 결과 ─────────────────────────────────────────────────────────

class VerifyPolicy(Enum):
    DROP_SPAN = 'drop_span'
    DROP_RECORD = 'drop_record'
    KEEP_ALL = 'keep_all'


@dataclass
class VerifyResult:
    """단일 레코드 검증 결과."""
    record: Record
    confirmed: list[Entity] = field(default_factory=list)
    missed: list[Entity] = field(default_factory=list)
    conflicts: list[tuple[Entity, str]] = field(default_factory=list)
    dropped: bool = False


# ── 매칭 헬퍼 ────────────────────────────────────────────────────────────

_PARTIAL_RATIO_THRESHOLD = 0.7


def _text_match(gold_text: str, pred_text: str) -> bool:
    """정확 매칭 또는 길이 비율 임계값을 충족하는 부분 포함 매칭."""
    g = gold_text.strip()
    p = pred_text.strip()
    if not g or not p:
        return False
    if g == p:
        return True
    if g in p or p in g:
        ratio = min(len(g), len(p)) / max(len(g), len(p))
        return ratio >= _PARTIAL_RATIO_THRESHOLD
    return False


def _find_pred(
    gold: Entity, preds: list[dict], used: set[int],
) -> tuple[int, dict] | None:
    """골드 엔티티에 매칭되는 예측을 찾는다 (정확 우선, 부분 후순위)."""
    # 1차: 정확 매칭
    for i, p in enumerate(preds):
        if i in used:
            continue
        if p.get('text', '').strip() == gold.text.strip():
            return i, p
    # 2차: 부분 매칭
    for i, p in enumerate(preds):
        if i in used:
            continue
        if _text_match(gold.text, p.get('text', '')):
            return i, p
    return None


# ── 검증기 ───────────────────────────────────────────────────────────────

class PIIVerifier:
    """주입된 NER 레코드를 LLM 교차 검증으로 정제한다."""

    def __init__(
        self,
        labeler: SpanLabeler,
        policy: VerifyPolicy = VerifyPolicy.DROP_SPAN,
    ) -> None:
        self._labeler = labeler
        self.policy = policy

    def _call_labeler(self, text: str) -> list[dict]:
        """split=False로 호출하되, 미지원 라벨러는 단순 호출로 fallback."""
        try:
            return self._labeler.label_spans(text, split=False)
        except TypeError:
            return self._labeler.label_spans(text)

    def verify(self, record: Record) -> VerifyResult:
        """단일 레코드를 검증한다. 문맥 보존을 위해 split=False 선호."""
        preds = self._call_labeler(record.text)

        confirmed: list[Entity] = []
        missed: list[Entity] = []
        conflicts: list[tuple[Entity, str]] = []
        used: set[int] = set()

        for gold_ent in record.entities:
            match = _find_pred(gold_ent, preds, used)
            if match is None:
                missed.append(gold_ent)
                continue
            idx, pred = match
            used.add(idx)
            pred_type = pred.get('type', '')
            if pred_type == gold_ent.label:
                confirmed.append(gold_ent)
            else:
                conflicts.append((gold_ent, pred_type))

        return self._apply_policy(
            record, confirmed, missed, conflicts,
        )

    def verify_dataset(
        self, records: Iterable[Record],
    ) -> tuple[list[Record], dict[str, Any]]:
        """레코드 목록을 검증하고 (정제된 레코드, 리포트)를 반환한다."""
        kept: list[Record] = []
        total = 0
        confirmed_count = 0
        missed_count = 0
        conflict_count = 0
        dropped_count = 0
        per_label: dict[str, dict[str, int]] = {}

        for rec in records:
            total += 1
            result = self.verify(rec)
            confirmed_count += len(result.confirmed)
            missed_count += len(result.missed)
            conflict_count += len(result.conflicts)
            if result.dropped:
                dropped_count += 1
            else:
                kept.append(result.record)
            for ent in result.confirmed:
                _inc(per_label, ent.label, 'confirmed')
            for ent in result.missed:
                _inc(per_label, ent.label, 'missed')
            for ent, _ in result.conflicts:
                _inc(per_label, ent.label, 'conflict')

        report: dict[str, Any] = {
            'total': total,
            'confirmed_count': confirmed_count,
            'missed_count': missed_count,
            'conflict_count': conflict_count,
            'dropped_count': dropped_count,
            'kept_count': len(kept),
            'per_label': dict(per_label),
        }
        return kept, report

    # ── 내부 ─────────────────────────────────────────────────────────

    def _apply_policy(
        self,
        record: Record,
        confirmed: list[Entity],
        missed: list[Entity],
        conflicts: list[tuple[Entity, str]],
    ) -> VerifyResult:
        has_issues = bool(missed or conflicts)

        if self.policy == VerifyPolicy.DROP_RECORD and has_issues:
            return VerifyResult(
                record=record,
                confirmed=confirmed,
                missed=missed,
                conflicts=conflicts,
                dropped=True,
            )

        if self.policy == VerifyPolicy.DROP_SPAN and has_issues:
            confirmed_keys = {
                (e.label, e.start_char, e.end_char, e.text)
                for e in confirmed
            }
            kept_entities = [
                e for e in record.entities
                if (e.label, e.start_char, e.end_char, e.text)
                in confirmed_keys
            ]
            new_record = Record(
                text=record.text,
                entities=kept_entities,
                id=record.id,
            )
            return VerifyResult(
                record=new_record,
                confirmed=confirmed,
                missed=missed,
                conflicts=conflicts,
                dropped=False,
            )

        return VerifyResult(
            record=record,
            confirmed=confirmed,
            missed=missed,
            conflicts=conflicts,
            dropped=False,
        )


def _inc(
    d: dict[str, dict[str, int]], label: str, key: str,
) -> None:
    if label not in d:
        d[label] = {}
    d[label][key] = d[label].get(key, 0) + 1
