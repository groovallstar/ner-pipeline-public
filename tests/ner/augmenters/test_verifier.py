"""PIIVerifier 단위 테스트 — TDD red phase."""
from __future__ import annotations

import asyncio

from ner.augmenters.schema import Entity, Record
from ner.augmenters.verifier import (
    PIIVerifier,
    VerifyPolicy,
)


# ── 모의 라벨러 ──────────────────────────────────────────────────────────

class FakeLabeler:
    """label_spans 결과를 미리 지정하는 모의 라벨러."""

    def __init__(self, spans: list[dict]) -> None:
        self._spans = spans

    def label_spans(self, text: str) -> list[dict]:
        return list(self._spans)


class FakeAsyncLabeler:
    """alabel_spans 만 가진 async 라벨러 (vLLM 라벨러 대용)."""

    def __init__(self, preds_map: dict[str, list[dict]]) -> None:
        self._preds_map = preds_map
        self.calls = 0

    async def alabel_spans(
        self, text: str, split: bool = True,
    ) -> list[dict]:
        self.calls += 1
        return list(self._preds_map.get(text, []))


# ── 헬퍼 ─────────────────────────────────────────────────────────────────

def _ent(label: str, start: int, end: int, text: str) -> Entity:
    return Entity(label=label, start_char=start, end_char=end, text=text)


def _rec(text: str, entities: list[Entity], rid: str = '0') -> Record:
    return Record(text=text, entities=entities, id=rid)


# ── 단일 레코드 검증 ─────────────────────────────────────────────────────

class TestVerifySingleRecord:
    """verify() 메서드 시나리오."""

    def test_all_confirmed(self):
        """골드와 예측이 완전 일치하면 모두 confirmed."""
        gold = [
            _ent('PHONE', 10, 23, '090-1234-5678'),
            _ent('PER', 0, 4, '山田太郎'),
        ]
        preds = [
            {'text': '090-1234-5678', 'type': 'PHONE'},
            {'text': '山田太郎', 'type': 'PER'},
        ]
        rec = _rec('山田太郎の連絡先：090-1234-5678。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.confirmed) == 2
        assert len(result.missed) == 0
        assert len(result.conflicts) == 0
        assert not result.dropped
        assert len(result.record.entities) == 2

    def test_missed_entity(self):
        """예측에 없는 골드 엔티티는 missed."""
        gold = [
            _ent('EMAIL', 5, 22, 'taro@example.com'),
            _ent('PHONE', 25, 38, '090-1234-5678'),
        ]
        preds = [{'text': 'taro@example.com', 'type': 'EMAIL'}]
        rec = _rec(
            'メール：taro@example.com。連絡先：090-1234-5678。', gold,
        )
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.confirmed) == 1
        assert len(result.missed) == 1
        assert result.missed[0].label == 'PHONE'

    def test_conflict_label_mismatch(self):
        """텍스트는 같지만 라벨이 다르면 conflict."""
        gold = [_ent('LOC', 0, 3, '東京都')]
        preds = [{'text': '東京都', 'type': 'ORG'}]
        rec = _rec('東京都にある施設。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.confirmed) == 0
        assert len(result.conflicts) == 1
        gold_ent, pred_type = result.conflicts[0]
        assert gold_ent.label == 'LOC'
        assert pred_type == 'ORG'

    def test_no_gold_entities(self):
        """골드 엔티티가 없으면 빈 결과."""
        rec = _rec('普通の文章です。', [])
        v = PIIVerifier(FakeLabeler([]), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.confirmed) == 0
        assert len(result.missed) == 0
        assert len(result.conflicts) == 0
        assert not result.dropped


# ── 정책 테스트 ──────────────────────────────────────────────────────────

class TestDropSpanPolicy:
    """DROP_SPAN 정책: missed/conflict 엔티티를 레코드에서 제거."""

    def test_removes_missed_span(self):
        gold = [
            _ent('PHONE', 10, 23, '090-1234-5678'),
            _ent('EMAIL', 25, 41, 'a@example.com'),
        ]
        preds = [{'text': '090-1234-5678', 'type': 'PHONE'}]
        rec = _rec('dummy text 090-1234-5678。a@example.com。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.record.entities) == 1
        assert result.record.entities[0].label == 'PHONE'
        assert not result.dropped

    def test_removes_conflict_span(self):
        gold = [
            _ent('LOC', 0, 3, '東京都'),
            _ent('PHONE', 10, 23, '090-1234-5678'),
        ]
        preds = [
            {'text': '東京都', 'type': 'ORG'},
            {'text': '090-1234-5678', 'type': 'PHONE'},
        ]
        rec = _rec('東京都の連絡先：090-1234-5678。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)

        assert len(result.record.entities) == 1
        assert result.record.entities[0].label == 'PHONE'


class TestDropRecordPolicy:
    """DROP_RECORD 정책: missed/conflict가 있으면 레코드 전체 드롭."""

    def test_drops_record_on_missed(self):
        gold = [_ent('PHONE', 0, 13, '090-1234-5678')]
        preds = []
        rec = _rec('090-1234-5678。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_RECORD)
        result = v.verify(rec)

        assert result.dropped

    def test_keeps_record_when_all_confirmed(self):
        gold = [_ent('PHONE', 0, 13, '090-1234-5678')]
        preds = [{'text': '090-1234-5678', 'type': 'PHONE'}]
        rec = _rec('090-1234-5678。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_RECORD)
        result = v.verify(rec)

        assert not result.dropped


class TestKeepAllPolicy:
    """KEEP_ALL 정책: 모든 엔티티 유지, 플래그만 기록."""

    def test_keeps_all_entities(self):
        gold = [
            _ent('PHONE', 0, 13, '090-1234-5678'),
            _ent('EMAIL', 15, 31, 'a@example.com'),
        ]
        preds = [{'text': '090-1234-5678', 'type': 'PHONE'}]
        rec = _rec('090-1234-5678。a@example.com。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.KEEP_ALL)
        result = v.verify(rec)

        assert len(result.record.entities) == 2
        assert len(result.missed) == 1
        assert not result.dropped


# ── 데이터셋 검증 ────────────────────────────────────────────────────────

class TestVerifyDataset:
    """verify_dataset() 배치 검증 + 리포트."""

    def test_dataset_report_counts(self):
        recs = [
            _rec('090-1234-5678。', [_ent('PHONE', 0, 13, '090-1234-5678')]),
            _rec('a@ex.com。', [_ent('EMAIL', 0, 8, 'a@ex.com')]),
            _rec('普通の文。', []),
        ]
        preds_map = {
            '090-1234-5678。': [
                {'text': '090-1234-5678', 'type': 'PHONE'},
            ],
            'a@ex.com。': [],
            '普通の文。': [],
        }

        class MapLabeler:
            def label_spans(self, text: str) -> list[dict]:
                return preds_map.get(text, [])

        v = PIIVerifier(MapLabeler(), policy=VerifyPolicy.DROP_SPAN)
        kept, report = v.verify_dataset(recs)

        assert len(kept) == 3
        assert report['total'] == 3
        assert report['confirmed_count'] == 1
        assert report['missed_count'] == 1
        assert report['conflict_count'] == 0

    def test_dataset_drop_record_filters(self):
        recs = [
            _rec('090-1234-5678。', [_ent('PHONE', 0, 13, '090-1234-5678')]),
            _rec('a@ex.com。', [_ent('EMAIL', 0, 8, 'a@ex.com')]),
        ]

        class ConfirmAllLabeler:
            def label_spans(self, text: str) -> list[dict]:
                if '090' in text:
                    return [{'text': '090-1234-5678', 'type': 'PHONE'}]
                return []

        v = PIIVerifier(
            ConfirmAllLabeler(), policy=VerifyPolicy.DROP_RECORD,
        )
        kept, report = v.verify_dataset(recs)

        assert len(kept) == 1
        assert kept[0].text == '090-1234-5678。'
        assert report['dropped_count'] == 1


# ── 부분 매칭 ────────────────────────────────────────────────────────────

class TestPartialTextMatch:
    """LLM이 텍스트를 약간 다르게 추출한 경우 부분 매칭."""

    def test_pred_contains_gold_text(self):
        """예측이 골드를 포함하면 매칭 (예: ADDRESS 끝에 공백)."""
        gold = [_ent('ADDRESS', 0, 15, '東京都千代田区1丁目2-3')]
        preds = [
            {'text': '東京都千代田区1丁目2-3', 'type': 'ADDRESS'},
        ]
        rec = _rec('東京都千代田区1丁目2-3に住む。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)
        assert len(result.confirmed) == 1

    def test_gold_contains_pred_text(self):
        """골드가 예측을 포함하면 매칭 (예: LLM이 짧게 추출)."""
        gold = [_ent('ADDRESS', 0, 18, '東京都千代田区1丁目2-3 ビル7F')]
        preds = [
            {'text': '東京都千代田区1丁目2-3', 'type': 'ADDRESS'},
        ]
        rec = _rec('東京都千代田区1丁目2-3 ビル7Fに住む。', gold)
        v = PIIVerifier(FakeLabeler(preds), policy=VerifyPolicy.DROP_SPAN)
        result = v.verify(rec)
        assert len(result.confirmed) == 1


# ── 리포트 디테일 ─────────────────────────────────────────────────────────

class TestReportDetails:
    """verify_dataset 리포트의 conflict_details/missed_details 누적."""

    def test_collects_conflict_and_missed_details(self):
        recs = [
            _rec(
                '東京都の連絡先：a@b.com。',
                [
                    _ent('LOC', 0, 3, '東京都'),
                    _ent('EMAIL', 9, 16, 'a@b.com'),
                ],
                rid='r1',
            ),
            _rec(
                '090-1111-2222。',
                [_ent('PHONE', 0, 13, '090-1111-2222')],
                rid='r2',
            ),
        ]
        preds_map = {
            '東京都の連絡先：a@b.com。': [
                {'text': '東京都', 'type': 'ORG'},
            ],
            '090-1111-2222。': [],
        }

        class MapLabeler:
            def label_spans(self, text: str) -> list[dict]:
                return preds_map.get(text, [])

        v = PIIVerifier(MapLabeler(), policy=VerifyPolicy.DROP_SPAN)
        _, report = v.verify_dataset(recs)

        assert report['conflict_count'] == 1
        assert report['missed_count'] == 2
        # conflict: LOC -> ORG (r1)
        assert report['conflict_details'] == [{
            'record_id': 'r1',
            'gold_label': 'LOC',
            'pred_label': 'ORG',
            'gold_text': '東京都',
            'start_char': 0,
            'end_char': 3,
        }]
        missed_labels = sorted(
            d['gold_label'] for d in report['missed_details']
        )
        assert missed_labels == ['EMAIL', 'PHONE']


# ── async batch 경로 ─────────────────────────────────────────────────────

class TestAsyncBatchPath:
    """alabel_spans 보유 라벨러는 단일 event loop batch 경로로 처리."""

    def test_uses_async_path_when_available(self):
        """alabel_spans 가 있으면 verify_dataset 가 async batch 로 호출."""
        recs = [
            _rec('a。', [_ent('PHONE', 0, 1, 'a')], rid='r0'),
            _rec('b。', [_ent('EMAIL', 0, 1, 'b')], rid='r1'),
        ]
        preds_map = {
            'a。': [{'text': 'a', 'type': 'PHONE'}],
            'b。': [{'text': 'b', 'type': 'EMAIL'}],
        }
        labeler = FakeAsyncLabeler(preds_map)
        v = PIIVerifier(labeler, policy=VerifyPolicy.DROP_SPAN)
        kept, report = v.verify_dataset(recs)

        assert labeler.calls == 2
        assert len(kept) == 2
        assert report['confirmed_count'] == 2
        assert report['conflict_count'] == 0

    def test_averify_dataset_runs_inside_external_loop(self):
        """averify_dataset 는 외부 event loop 안에서도 호출 가능."""
        recs = [_rec('x。', [_ent('PHONE', 0, 1, 'x')], rid='r0')]
        labeler = FakeAsyncLabeler({'x。': [{'text': 'x', 'type': 'PHONE'}]})
        v = PIIVerifier(labeler, policy=VerifyPolicy.DROP_SPAN)

        async def runner():
            return await v.averify_dataset(recs)

        kept, report = asyncio.run(runner())
        assert len(kept) == 1
        assert report['confirmed_count'] == 1
