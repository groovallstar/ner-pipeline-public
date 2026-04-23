"""Wikidata 앵커 유틸 단위 테스트 (네트워크 없는 부분만)."""
from augmenters.wikiann_vi.wikidata_anchor import (
    WIKIDATA_TO_CANONICAL,
    _iter_entities,
    _resolve_title,
    anchor_type,
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

    def test_covers_all_8_types(self):
        """매핑 테이블이 8종 모두를 커버한다."""
        values = set(WIKIDATA_TO_CANONICAL.values())
        assert values == {
            'PER', 'CORP', 'LOC', 'FAC',
            'PROD', 'EVT', 'POL', 'ORG',
        }

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
                'gold_spans_8type': [
                    {'text': 'Hà Nội', 'type': 'LOC'},
                    {'text': 'Samsung', 'type': 'CORP'},
                ],
            },
            {
                'id': '1', 'text': 't',
                'gold_spans_8type': [],
            },
        ]
        out = list(_iter_entities(records, 'gold_spans_8type'))
        assert len(out) == 2
        assert ('Hà Nội', 'LOC', '0') in out
        assert ('Samsung', 'CORP', '0') in out

    def test_skips_empty_fields(self):
        records = [{
            'id': '0', 'text': 't',
            'gold_spans_8type': [
                {'text': '', 'type': 'LOC'},
                {'text': 'Hà Nội', 'type': ''},
                {'text': 'Samsung', 'type': 'CORP'},
            ],
        }]
        out = list(_iter_entities(records, 'gold_spans_8type'))
        assert len(out) == 1
        assert out[0][0] == 'Samsung'

    def test_alt_span_key(self):
        records = [{
            'id': '0', 'text': 't',
            'gold_spans': [{'text': 'x', 'type': 'PER'}],
        }]
        out = list(_iter_entities(records, 'gold_spans'))
        assert out == [('x', 'PER', '0')]
