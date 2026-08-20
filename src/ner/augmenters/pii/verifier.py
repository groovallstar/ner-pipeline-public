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

from ner.augmenters.pii.schema import Entity, Record

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
    # `verify_labels` 밖이라 판정 대상이 아니었던 엔티티. 정책이 건드리지
    # 않고 그대로 통과한다.
    exempt: list[Entity] = field(default_factory=list)


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
        verify_labels: Iterable[str] | None = None,
    ) -> None:
        """
        Args:
            verify_labels: 판정 대상으로 삼을 라벨. `None`(기본)이면 레코드의
                모든 엔티티가 대상이라 기존 동작과 같다. 값을 주면 그 라벨만
                정책이 버릴 수 있고 나머지는 무조건 통과한다.

        **주입한 PII 와 원본 gold 는 증거의 성격이 다르다.** 주입값은 우리가
        무엇을 어디에 넣었는지 알고 있어 "LLM 이 못 찾았다" 가 곧 "주입이
        어긋났다" 는 신호다. 반면 원본 gold 는 사람이 붙인 정답이라 LLM 이
        못 찾은 것은 **LLM 에 대한 증거**지 gold 에 대한 증거가 아니다.
        구분 없이 `drop_span` 을 걸면 검증 모델의 recall 부족이 사람 주석을
        지운다 — 이 인자를 만들게 한 EN 실측에서 gold NER 8,199 span 중
        2,199 개(27%)가 그렇게 사라졌고 `DAT` 은 54% 가 날아갔다(그 실행의
        값이며 현 산출물로는 재현되지 않는다).
        """
        self._labeler = labeler
        self.policy = policy
        self.verify_labels = (
            frozenset(verify_labels) if verify_labels is not None else None
        )

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
        exempt: list[Entity] = []
        used: set[int] = set()

        for gold_ent in record.entities:
            # 범위 밖 엔티티도 매칭은 돌려 예측을 소비시킨다 — 안 그러면
            # 남은 예측이 범위 안 엔티티에 잘못 붙을 수 있다.
            match = _find_pred(gold_ent, preds, used)
            if match is not None:
                used.add(match[0])
            if (
                self.verify_labels is not None
                and gold_ent.label not in self.verify_labels
            ):
                exempt.append(gold_ent)
                continue
            if match is None:
                missed.append(gold_ent)
                continue
            pred_type = match[1].get('type', '')
            if pred_type == gold_ent.label:
                confirmed.append(gold_ent)
            else:
                conflicts.append((gold_ent, pred_type))

        return self._apply_policy(
            record, confirmed, missed, conflicts, exempt,
        )

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
        exempt_count = 0
        per_label: dict[str, dict[str, int]] = {}
        conflict_details: list[dict[str, Any]] = []
        missed_details: list[dict[str, Any]] = []

        for rec, result in zip(records, results):
            confirmed_count += len(result.confirmed)
            missed_count += len(result.missed)
            conflict_count += len(result.conflicts)
            exempt_count += len(result.exempt)
            if result.dropped:
                dropped_count += 1
            else:
                kept.append(result.record)
            for ent in result.confirmed:
                _inc(per_label, ent.label, 'confirmed')
            for ent in result.exempt:
                _inc(per_label, ent.label, 'exempt')
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
            # 판정 범위 밖이라 정책이 건드리지 않은 엔티티 수.
            # `verify_labels` 를 안 주면 항상 0 이다.
            'exempt_count': exempt_count,
            'verify_labels': (
                sorted(self.verify_labels)
                if self.verify_labels is not None else None
            ),
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
        exempt: list[Entity] | None = None,
    ) -> VerifyResult:
        exempt = exempt or []
        has_issues = bool(missed or conflicts)

        if self.policy == VerifyPolicy.DROP_RECORD and has_issues:
            return VerifyResult(
                record=record,
                confirmed=confirmed,
                missed=missed,
                conflicts=conflicts,
                dropped=True,
                exempt=exempt,
            )

        if self.policy == VerifyPolicy.DROP_SPAN and has_issues:
            keep_keys = {
                (e.label, e.start_char, e.end_char, e.text)
                for e in (*confirmed, *exempt)
            }
            kept_entities = [
                e for e in record.entities
                if (e.label, e.start_char, e.end_char, e.text) in keep_keys
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
                exempt=exempt,
            )

        return VerifyResult(
            record=record,
            confirmed=confirmed,
            missed=missed,
            conflicts=conflicts,
            dropped=False,
            exempt=exempt,
        )


def _inc(
    d: dict[str, dict[str, int]], label: str, key: str,
) -> None:
    if label not in d:
        d[label] = {}
    d[label][key] = d[label].get(key, 0) + 1
