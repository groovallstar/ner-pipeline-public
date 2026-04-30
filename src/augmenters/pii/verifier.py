"""PII 주입 결과를 LLM 교차 검증하는 모듈.

주입된 골드 엔티티를 LLM(vLLM 등)이 독립적으로 추출한 예측과 대조하여
confirmed / missed / conflict 로 분류하고, 정책에 따라 레코드를 정제한다.
"""
from __future__ import annotations

import asyncio
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
    split 인자를 선택적으로 수용하면 된다. async 경로(`alabel_spans`)를
    구현한 라벨러는 batch 검증 시 단일 event loop 에서 재사용되어
    AsyncOpenAI connection pool 의 'Event loop is closed' retry 폭주를
    회피한다.
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

    async def _acall_labeler(self, text: str) -> list[dict]:
        """async 라벨러를 우선 시도하고, sync 라벨러로 graceful fallback."""
        alabel = getattr(self._labeler, 'alabel_spans', None)
        if alabel is not None:
            try:
                return await alabel(text, split=False)
            except TypeError:
                return await alabel(text)
        return self._call_labeler(text)

    def _classify(
        self, record: Record, preds: list[dict],
    ) -> VerifyResult:
        """gold 엔티티와 예측을 매칭하여 confirmed/missed/conflict 분류."""
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

        return self._apply_policy(record, confirmed, missed, conflicts)

    def verify(self, record: Record) -> VerifyResult:
        """단일 레코드를 검증한다. 문맥 보존을 위해 split=False 선호."""
        preds = self._call_labeler(record.text)
        return self._classify(record, preds)

    async def averify(self, record: Record) -> VerifyResult:
        """verify() 의 async 버전. 외부 event loop 안에서 호출 가능."""
        preds = await self._acall_labeler(record.text)
        return self._classify(record, preds)

    def verify_dataset(
        self, records: Iterable[Record],
    ) -> tuple[list[Record], dict[str, Any]]:
        """레코드 목록을 검증하고 (정제된 레코드, 리포트)를 반환한다.

        async 라벨러(`alabel_spans` 보유)가 주입된 경우 단일 event loop
        에서 batch 처리하여 connection pool 재사용을 보장한다 (sync
        wrapper 의 record 마다 `asyncio.run` 패턴이 유발하던 'Event
        loop is closed' retry 폭주 회피).
        """
        records = list(records)
        if hasattr(self._labeler, 'alabel_spans'):
            results = asyncio.run(self._gather_verify(records))
        else:
            results = [self.verify(r) for r in records]
        return self._aggregate(records, results)

    async def averify_dataset(
        self, records: Iterable[Record],
    ) -> tuple[list[Record], dict[str, Any]]:
        """verify_dataset() 의 async 버전."""
        records = list(records)
        results = await self._gather_verify(records)
        return self._aggregate(records, results)

    # ── 내부 ─────────────────────────────────────────────────────────

    async def _gather_verify(
        self, records: list[Record],
    ) -> list[VerifyResult]:
        """모든 레코드를 단일 event loop 에서 병렬 검증한다.

        라벨러 자체의 동시성 제한(예: BaseVllmLabeler 의 semaphore)이
        실제 동시 HTTP 요청 수를 제어하므로 task 는 모두 한 번에 생성한다.
        """
        return await asyncio.gather(
            *(self.averify(r) for r in records)
        )

    def _aggregate(
        self,
        records: list[Record],
        results: list[VerifyResult],
    ) -> tuple[list[Record], dict[str, Any]]:
        """검증 결과를 집계해 (정제된 레코드, 리포트)를 만든다."""
        kept: list[Record] = []
        confirmed_count = 0
        missed_count = 0
        conflict_count = 0
        dropped_count = 0
        per_label: dict[str, dict[str, int]] = {}
        conflict_details: list[dict[str, Any]] = []
        missed_details: list[dict[str, Any]] = []

        for rec, result in zip(records, results):
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
                missed_details.append({
                    'record_id': rec.id,
                    'gold_label': ent.label,
                    'gold_text': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                })
            for ent, pred_label in result.conflicts:
                _inc(per_label, ent.label, 'conflict')
                conflict_details.append({
                    'record_id': rec.id,
                    'gold_label': ent.label,
                    'pred_label': pred_label,
                    'gold_text': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                })

        report: dict[str, Any] = {
            'total': len(records),
            'confirmed_count': confirmed_count,
            'missed_count': missed_count,
            'conflict_count': conflict_count,
            'dropped_count': dropped_count,
            'kept_count': len(kept),
            'per_label': dict(per_label),
            'conflict_details': conflict_details,
            'missed_details': missed_details,
        }
        return kept, report

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
