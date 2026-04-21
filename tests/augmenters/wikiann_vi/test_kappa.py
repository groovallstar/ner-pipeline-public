"""WikiANN-vi cross-model kappa 유틸 단위 테스트."""
import math

import pytest

from augmenters.wikiann_vi.kappa import (
    cohen_kappa,
    compute,
    confusion_matrix,
    per_type_agreement,
)


class TestCohenKappa:
    def test_perfect_agreement(self):
        a = ['人名', '地名', '人名', 'O']
        b = ['人名', '地名', '人名', 'O']
        r = cohen_kappa(a, b)
        assert r['kappa'] == 1.0
        assert r['po'] == 1.0

    def test_complete_disagreement(self):
        a = ['人名', '地名', '人名', '地名']
        b = ['地名', '人名', '地名', '人名']
        r = cohen_kappa(a, b)
        assert r['po'] == 0.0
        # pe = 0.5, kappa = (0 - 0.5)/(1 - 0.5) = -1.0
        assert r['kappa'] == pytest.approx(-1.0)

    def test_random_agreement_kappa_zero(self):
        a = ['A', 'A', 'B', 'B']
        b = ['A', 'B', 'A', 'B']
        r = cohen_kappa(a, b)
        # po = 0.5, pe = 0.5 → kappa = 0
        assert r['po'] == 0.5
        assert r['kappa'] == pytest.approx(0.0, abs=1e-9)

    def test_empty(self):
        r = cohen_kappa([], [])
        assert math.isnan(r['kappa'])
        assert r['n'] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa(['A'], ['A', 'B'])


class TestPerTypeAgreement:
    def test_excludes_O_class(self):
        a = ['人名', '人名', 'O', 'O']
        b = ['人名', 'O', '地名', 'O']
        r = per_type_agreement(a, b)
        assert '人名' in r
        assert 'O' not in r
        assert r['人名']['agreed'] == 1
        assert r['人名']['total_a'] == 2
        assert r['人名']['ratio'] == 0.5

    def test_one_side_only(self):
        a = ['人名', '地名']
        b = ['O', 'O']
        r = per_type_agreement(a, b)
        assert r['人名']['ratio'] == 0.0
        assert r['地名']['ratio'] == 0.0


class TestConfusionMatrix:
    def test_basic(self):
        a = ['人名', '人名', '地名']
        b = ['人名', '地名', '地名']
        m = confusion_matrix(a, b)
        assert m['人名']['人名'] == 1
        assert m['人名']['地名'] == 1
        assert m['地名']['地名'] == 1


class TestComputeFromRecords:
    def _mk(self, rid: str, spans):
        return {
            'id': rid,
            'text': 't',
            'gold_spans_8type': [
                {
                    'text': 'x',
                    'type': t,
                    'start': s,
                    'end': e,
                }
                for (s, e, t) in spans
            ],
        }

    def test_identical_records(self):
        a = [self._mk('0', [(0, 2, '人名'), (5, 8, '地名')])]
        b = [self._mk('0', [(0, 2, '人名'), (5, 8, '地名')])]
        r = compute(a, b)
        assert r['common_records'] == 1
        assert r['paired_spans'] == 2
        assert r['kappa'] == 1.0

    def test_one_side_extra_span(self):
        """A가 추가 span을 가지면 B쪽은 O로 포함 → 불일치 1건."""
        a = [self._mk('0', [(0, 2, '人名'), (5, 8, '地名')])]
        b = [self._mk('0', [(0, 2, '人名')])]
        r = compute(a, b)
        assert r['paired_spans'] == 2
        assert r['po'] == 0.5  # 1 match / 2 paired
        # 人名 완전 일치, 地名은 A만
        per = r['per_type_agreement']
        assert per['人名']['ratio'] == 1.0
        assert per['地名']['ratio'] == 0.0

    def test_type_disagreement_same_offset(self):
        """동일 offset에 서로 다른 타입 → 불일치."""
        a = [self._mk('0', [(0, 2, '人名')])]
        b = [self._mk('0', [(0, 2, '地名')])]
        r = compute(a, b)
        assert r['paired_spans'] == 1
        assert r['po'] == 0.0

    def test_disjoint_ids_ignored(self):
        """공통 id가 없으면 paired_spans = 0."""
        a = [self._mk('0', [(0, 2, '人名')])]
        b = [self._mk('1', [(0, 2, '人名')])]
        r = compute(a, b)
        assert r['common_records'] == 0
        assert r['paired_spans'] == 0
