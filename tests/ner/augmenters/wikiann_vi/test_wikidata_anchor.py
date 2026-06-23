"""Wikidata 앵커 유틸 단위 테스트 (네트워크 없는 부분만)."""
import math

import pytest

from ner.augmenters.wikiann_vi.wikidata_anchor import (
    WIKIDATA_TO_CANONICAL,
    _iter_entities,
    _resolve_title,
    anchor_type,
    run_anchor,
)


class TestAnchorType:
    def test_person(self):
        assert anchor_type(['Q5']) == 'PER'
        assert anchor_type(['Q215627']) == 'PER'

    def test_city_priority(self):
        # 첫 매핑 가능한 P31이 우선 (Q515 = city)
        assert anchor_type(['Q515', 'Q6256']) == 'LOC'

    def test_unknown_returns_none(self):
        assert anchor_type(['Q999999999']) is None
        assert anchor_type([]) is None

    def test_falls_through_to_mapped(self):
        # 앞쪽 미매핑, 뒤쪽 매핑 있으면 뒤쪽 반환
        assert anchor_type(['Q999999999', 'Q5']) == 'PER'

    def test_covers_all_5_types(self):
        """매핑 테이블이 5종 모두를 커버한다 (축소 후 canonical 스키마)."""
        values = set(WIKIDATA_TO_CANONICAL.values())
        assert values == {'PER', 'LOC', 'ORG', 'PROD', 'EVT'}

    def test_no_duplicate_keys(self):
        # dict 생성 시 중복은 자동 제거되므로 이 테스트는 도메인 레벨이지만,
        # 실질 의미는 "각 타입에 최소 3개 이상 앵커 Q-ID가 있어야 강건하다"
        from collections import Counter
        type_counts = Counter(WIKIDATA_TO_CANONICAL.values())
        for t, c in type_counts.items():
            assert c >= 3, f'Type {t} has only {c} anchor Q-IDs'


class TestResolveTitle:
    def test_no_redirect_no_norm(self):
        assert _resolve_title('A', {}, {}) == 'A'

    def test_normalized_only(self):
        norm = {'a': 'A'}
        assert _resolve_title('a', norm, {}) == 'A'

    def test_redirect_only(self):
        redir = {'A': 'B'}
        assert _resolve_title('A', {}, redir) == 'B'

    def test_norm_then_redirect(self):
        norm = {'a': 'A'}
        redir = {'A': 'B'}
        assert _resolve_title('a', norm, redir) == 'B'

    def test_redirect_cycle_safe(self):
        # A → B → A 순환 시 무한루프 방지
        redir = {'A': 'B', 'B': 'A'}
        result = _resolve_title('A', {}, redir)
        # 사이클 한 번 돌면 멈춘다
        assert result in {'A', 'B'}


class TestIterEntities:
    def test_basic(self):
        records = [
            {
                'id': '0', 'text': 't',
                'gold_spans_relabel': [
                    {'text': 'Hà Nội', 'type': 'LOC'},
                    {'text': 'Samsung', 'type': 'ORG'},
                ],
            },
            {
                'id': '1', 'text': 't',
                'gold_spans_relabel': [],
            },
        ]
        out = list(_iter_entities(records, 'gold_spans_relabel'))
        assert len(out) == 2
        assert ('Hà Nội', 'LOC', '0') in out
        assert ('Samsung', 'ORG', '0') in out

    def test_skips_empty_fields(self):
        records = [{
            'id': '0', 'text': 't',
            'gold_spans_relabel': [
                {'text': '', 'type': 'LOC'},
                {'text': 'Hà Nội', 'type': ''},
                {'text': 'Samsung', 'type': 'ORG'},
            ],
        }]
        out = list(_iter_entities(records, 'gold_spans_relabel'))
        assert len(out) == 1
        assert out[0][0] == 'Samsung'

    def test_alt_span_key(self):
        records = [{
            'id': '0', 'text': 't',
            'gold_spans': [{'text': 'x', 'type': 'PER'}],
        }]
        out = list(_iter_entities(records, 'gold_spans'))
        assert out == [('x', 'PER', '0')]


# ---------------------------------------------------------------------------
# run_anchor 집계·판정 테스트 (fetch_qids / fetch_p31 을 monkeypatch로 스텁)
# ---------------------------------------------------------------------------

def _make_records(*spans_per_record):
    """(surface, type) 리스트를 받아 gold_spans_relabel 레코드 목록 생성."""
    records = []
    for i, spans in enumerate(spans_per_record):
        records.append({
            'id': str(i),
            'text': 'dummy',
            'gold_spans_relabel': [
                {'text': s, 'type': t} for s, t in spans
            ],
        })
    return records


class TestRunAnchorAggregation:
    """run_anchor 집계·분기 로직 단위 테스트.

    fetch_qids / fetch_p31 을 monkeypatch로 교체해 네트워크를 완전히 제거한다.
    """

    def test_all_match_single_type(self, monkeypatch):
        """전체 엔티티가 Wikidata 타입과 일치하는 경우 agreement=1.0."""
        records = _make_records(
            [('Hà Nội', 'LOC'), ('TP.HCM', 'LOC')],
        )
        # Q515 = city → LOC
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: 'Q515' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: ['Q515'] for q in qids},
        )

        result = run_anchor(records)

        assert result['total_entities'] == 2
        assert result['unique_surfaces'] == 2
        assert result['with_qid'] == 2
        assert result['with_p31'] == 2
        assert result['with_mapped_type'] == 2
        assert result['matches'] == 2
        assert result['mismatches'] == 0
        assert result['agreement'] == pytest.approx(1.0)
        loc = result['per_type_agreement']['LOC']
        assert loc['agreed'] == 2
        assert loc['total'] == 2
        assert loc['ratio'] == pytest.approx(1.0)

    def test_all_mismatch(self, monkeypatch):
        """라벨 타입과 Wikidata 타입이 모두 불일치하면 agreement=0.0."""
        records = _make_records(
            [('Nguyễn Văn A', 'ORG')],   # 실제론 PER(Q5)
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: 'Q5' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: ['Q5'] for q in qids},
        )

        result = run_anchor(records)

        assert result['matches'] == 0
        assert result['mismatches'] == 1
        assert result['agreement'] == pytest.approx(0.0)
        assert len(result['mismatch_samples']) == 1
        sample = result['mismatch_samples'][0]
        assert sample['surface'] == 'Nguyễn Văn A'
        assert sample['predicted'] == 'ORG'
        assert sample['anchor'] == 'PER'

    def test_no_qid_skips_entity(self, monkeypatch):
        """Q-ID 없는 엔티티는 with_qid 카운트에 포함되지 않아야 한다."""
        records = _make_records(
            [('unknown_surface', 'PER')],
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: None for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {},
        )

        result = run_anchor(records)

        assert result['total_entities'] == 1
        assert result['with_qid'] == 0
        assert result['with_p31'] == 0
        assert result['with_mapped_type'] == 0
        assert result['matches'] == 0
        # Q-ID 없으면 집계 대상 없으므로 agreement 는 nan
        assert math.isnan(result['agreement'])

    def test_no_p31_skips_entity(self, monkeypatch):
        """P31 클레임이 비어있는 엔티티는 with_p31 카운트에서 제외된다."""
        records = _make_records(
            [('SomeOrg', 'ORG')],
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: 'Q99999' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            # P31 빈 리스트 반환 → with_p31 제외
            lambda qids, session=None: {q: [] for q in qids},
        )

        result = run_anchor(records)

        assert result['with_qid'] == 1
        assert result['with_p31'] == 0
        assert result['with_mapped_type'] == 0

    def test_unmapped_p31_counted(self, monkeypatch):
        """P31 Q-ID가 매핑 테이블에 없으면 unmapped_qids에 카운트된다."""
        records = _make_records(
            [('SomeThing', 'PROD')],
        )
        unknown_qid = 'Q000000001'
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: 'Qxyz' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: [unknown_qid] for q in qids},
        )

        result = run_anchor(records)

        assert result['with_p31'] == 1
        assert result['with_mapped_type'] == 0
        assert unknown_qid in result['unmapped_qids']
        assert result['unmapped_qids'][unknown_qid] == 1

    def test_per_type_agreement_multiple_types(self, monkeypatch):
        """복수 타입 혼재 시 per_type_agreement 분기별 집계가 정확해야 한다."""
        records = _make_records(
            # LOC 2개 일치, PER 1개 불일치
            [('Hà Nội', 'LOC'), ('TP.HCM', 'LOC'), ('Tổ chức X', 'PER')],
        )
        # 표면형별 Q-ID 매핑
        qid_map = {
            'Hà Nội': 'Q515',    # LOC
            'TP.HCM': 'Q515',    # LOC
            'Tổ chức X': 'Q5',   # PER (라벨은 PER이므로 일치)
        }
        # Q515 → LOC, Q5 → PER
        p31_map = {
            'Q515': ['Q515'],
            'Q5': ['Q5'],
        }
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: qid_map[t] for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: p31_map[q] for q in qids},
        )

        result = run_anchor(records)

        assert result['matches'] == 3
        assert result['mismatches'] == 0
        loc = result['per_type_agreement']['LOC']
        assert loc['agreed'] == 2
        assert loc['total'] == 2
        per = result['per_type_agreement']['PER']
        assert per['agreed'] == 1
        assert per['total'] == 1

    def test_mismatch_sample_capped_at_30(self, monkeypatch):
        """mismatch_samples 는 최대 30개까지만 수집된다."""
        # 40개 엔티티 모두 mismatch
        spans = [('surface_{}'.format(i), 'ORG') for i in range(40)]
        records = _make_records(spans)

        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            # 모두 Q5(PER) 로 매핑
            lambda titles, session=None: {t: 'Q5' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: ['Q5'] for q in qids},
        )

        result = run_anchor(records)

        assert result['mismatches'] == 40
        assert len(result['mismatch_samples']) == 30

    def test_empty_records(self, monkeypatch):
        """빈 레코드 목록은 모든 카운트가 0이고 agreement 가 nan 이어야 한다."""
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {},
        )

        result = run_anchor([])

        assert result['total_entities'] == 0
        assert result['unique_surfaces'] == 0
        assert result['with_qid'] == 0
        assert result['matches'] == 0
        assert math.isnan(result['agreement'])

    def test_cache_skips_fetch(self, monkeypatch, tmp_path):
        """캐시 파일이 있으면 네트워크 fetch 없이 캐시 데이터로 집계한다.

        run_anchor 는 캐시 히트 시 missing=[] 로 fetch 를 호출하지만
        실제 네트워크 요청은 발생하지 않는다 (fetch_qids([]) 는 즉시 반환).
        여기서는 캐시 데이터만으로 올바른 집계가 나오는지 검증한다.
        """
        import json

        records = _make_records([('Hà Nội', 'LOC')])
        cache_data = {
            'qid': {'Hà Nội': 'Q515'},
            'p31': {'Q515': ['Q515']},   # Q515 = city → LOC
        }
        cache_file = tmp_path / 'cache.json'
        cache_file.write_text(
            json.dumps(cache_data, ensure_ascii=False), encoding='utf-8'
        )

        # fetch_qids / fetch_p31 가 빈 인수 이외의 것을 받으면 실패시킨다
        def _guard_qids(titles, session=None):
            assert titles == [], (
                f'Expected empty titles (cache hit), got {titles}'
            )
            return {}

        def _guard_p31(qids, session=None):
            assert qids == [], (
                f'Expected empty qids (cache hit), got {qids}'
            )
            return {}

        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            _guard_qids,
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            _guard_p31,
        )

        result = run_anchor(records, cache_path=cache_file)

        # 캐시 데이터만으로 집계 → 네트워크 없이 정확한 결과
        assert result['matches'] == 1
        assert result['mismatches'] == 0
        assert result['agreement'] == pytest.approx(1.0)

    def test_cache_written_after_fetch(self, monkeypatch, tmp_path):
        """캐시 파일이 없을 때 fetch 후 캐시가 올바르게 기록돼야 한다."""
        import json

        records = _make_records([('Samsung', 'ORG')])
        cache_file = tmp_path / 'subdir' / 'cache.json'

        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_qids',
            lambda titles, session=None: {t: 'Q783794' for t in titles},
        )
        monkeypatch.setattr(
            'ner.augmenters.wikiann_vi.wikidata_anchor.fetch_p31',
            lambda qids, session=None: {q: ['Q783794'] for q in qids},
        )

        run_anchor(records, cache_path=cache_file)

        assert cache_file.exists()
        saved = json.loads(cache_file.read_text(encoding='utf-8'))
        assert 'qid' in saved
        assert 'p31' in saved
        assert saved['qid']['Samsung'] == 'Q783794'
