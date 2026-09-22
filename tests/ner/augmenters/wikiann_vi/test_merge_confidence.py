"""merge_confidence 단위 테스트."""
import pytest

from ner.augmenters.wikiann_vi.merge_confidence import (
    MERGE_POLICY,
    _is_evt_legit,
    categorize_spans,
    merge_records,
    select_spans,
)


def _span(text, type_, start, end):
    return {'text': text, 'type': type_, 'start': start, 'end': end}


class TestCategorizeSpans:
    def test_both_agree(self):
        a = [_span('X', 'PER', 0, 1)]
        b = [_span('X', 'PER', 0, 1)]
        result = categorize_spans(a, b)
        assert len(result) == 1
        assert result[0]['confidence'] == 'high'
        assert result[0]['source'] == 'both'

    def test_both_disagree_type(self):
        a = [_span('X', 'ORG', 0, 1)]
        b = [_span('X', 'LOC', 0, 1)]
        result = categorize_spans(a, b)
        assert len(result) == 1
        assert result[0]['confidence'] == 'conflict'
        assert result[0]['source'] == 'both_disagree'
        assert result[0]['type'] == 'ORG'  # Gemma primary
        assert result[0]['gemma_type'] == 'ORG'
        assert result[0]['qwen_type'] == 'LOC'

    def test_gemma_only(self):
        a = [_span('X', 'PER', 0, 1)]
        b = []
        result = categorize_spans(a, b)
        assert len(result) == 1
        assert result[0]['confidence'] == 'medium_recall'
        assert result[0]['source'] == 'gemma_only'

    def test_qwen_only(self):
        a = []
        b = [_span('X', 'PER', 0, 1)]
        result = categorize_spans(a, b)
        assert len(result) == 1
        assert result[0]['confidence'] == 'medium_prec'
        assert result[0]['source'] == 'qwen_only'

    def test_mixed(self):
        a = [
            _span('A', 'PER', 0, 1),       # both agree
            _span('B', 'ORG', 2, 3),      # disagree
            _span('C', 'LOC', 4, 5),       # gemma_only
        ]
        b = [
            _span('A', 'PER', 0, 1),
            _span('B', 'LOC', 2, 3),
            _span('D', 'LOC', 6, 7),       # qwen_only
        ]
        result = categorize_spans(a, b)
        by_text = {r['text']: r for r in result}
        assert by_text['A']['confidence'] == 'high'
        assert by_text['B']['confidence'] == 'conflict'
        assert by_text['C']['confidence'] == 'medium_recall'
        assert by_text['D']['confidence'] == 'medium_prec'

    def test_sort_by_start(self):
        a = [_span('Z', 'LOC', 10, 12)]
        b = [_span('A', 'PER', 0, 1)]
        result = categorize_spans(a, b)
        assert [r['text'] for r in result] == ['A', 'Z']


class TestEvtLegit:
    """§3 legit 카테고리 패턴 게이트."""

    @pytest.mark.parametrize('text', [
        'Cúp bóng đá châu Á 2007',          # 연도대회
        'World Cup 2022',
        'UEFA Champions League 2007-08',
        'Công ước Genève',                   # 조약
        'Hiệp định Paris',
        'Hội nghị cấp cao Đông Á',
        'Chiến tranh Nga-Ba Tư',             # 전쟁
        'Trận Ngọc Hồi - Đống Đa',
        'Bão Haiyan',                        # 재해
        'Vụ đánh bom xe lửa tại Madrid',
        'Phong trào Cần Vương',              # 봉기
        'Bầu cử Duma Quốc gia năm 2007',
    ])
    def test_legit_matches(self, text):
        assert _is_evt_legit(text) is True

    @pytest.mark.parametrize('text', [
        'Minh Trị Duy tân',                  # 다년 process — 정당 drop
        'Chiến Quốc',                        # 시대
        'Cải cách Taika',
        'Đại hội Thể thao châu Á',           # 연도 없는 정기대회 → ORG
        'Galaxy S24',                        # PROD
        'Hà Nội',                            # LOC
        '',                                  # 빈 문자열
    ])
    def test_nonlegit_no_match(self, text):
        assert _is_evt_legit(text) is False


class TestSelectSpans:
    """선택 규칙 — EVT single-model legit 구제, PROD gemma_only 완화."""

    def _span(self, type_, conf, text):
        return {'type': type_, 'confidence': conf, 'text': text}

    def test_rescues_single_model_legit_evt(self):
        spans = [
            self._span('EVT', 'high', 'Chiến tranh Việt Nam'),
            self._span('EVT', 'medium_recall', 'Cúp bóng đá châu Á 2007'),
            self._span('EVT', 'medium_prec', 'Công ước Genève'),
            self._span('EVT', 'medium_recall', 'Minh Trị Duy tân'),
            self._span('EVT', 'conflict', 'Trận X 2008'),
            self._span('LOC', 'medium_recall', 'Hà Nội'),
            self._span('PER', 'medium_prec', 'Nam'),
            self._span('PER', 'conflict', 'Lan'),
        ]
        kept = {(s['type'], s['confidence'], s['text'])
                for s in select_spans(spans)}
        # EVT high + legit single-model 구제
        assert ('EVT', 'high', 'Chiến tranh Việt Nam') in kept
        assert ('EVT', 'medium_recall', 'Cúp bóng đá châu Á 2007') in kept
        assert ('EVT', 'medium_prec', 'Công ước Genève') in kept
        # 비-legit EVT single-model 은 drop
        assert ('EVT', 'medium_recall', 'Minh Trị Duy tân') not in kept
        # conflict 는 전 type drop
        assert ('EVT', 'conflict', 'Trận X 2008') not in kept
        assert ('PER', 'conflict', 'Lan') not in kept
        # PER/LOC 는 medium_recall 보존, medium_prec drop
        assert ('LOC', 'medium_recall', 'Hà Nội') in kept
        assert ('PER', 'medium_prec', 'Nam') not in kept

    def test_rescues_gemma_only_prod(self):
        spans = [
            self._span('PROD', 'high', 'iPhone 14'),
            self._span('PROD', 'medium_recall', 'Black or White'),
            self._span('PROD', 'medium_prec', 'Thám tử Conan'),
            self._span('PROD', 'conflict', 'X'),
        ]
        kept = {(s['type'], s['confidence'], s['text'])
                for s in select_spans(spans)}
        # PROD high + gemma_only(medium_recall) 구제
        assert ('PROD', 'high', 'iPhone 14') in kept
        assert ('PROD', 'medium_recall', 'Black or White') in kept
        # qwen_only(medium_prec) · conflict PROD 는 drop
        assert ('PROD', 'medium_prec', 'Thám tử Conan') not in kept
        assert ('PROD', 'conflict', 'X') not in kept


class TestMergeRecords:
    def _record(self, rid, spans):
        return {
            'id': rid, 'text': 't',
            'gold_spans_relabel': spans,
            'relabel_model': 'test-model',
        }

    def test_matched_ids(self):
        gemma = [self._record('0', [_span('X', 'PER', 0, 1)])]
        qwen = [self._record('0', [_span('X', 'PER', 0, 1)])]
        out = merge_records(gemma, qwen)
        assert len(out) == 1
        assert out[0]['merge_policy'] == MERGE_POLICY == 'recall_strict_prod'
        assert out[0]['gold_spans_relabel_merged'][0]['confidence'] == 'high'
        assert 'gold_spans_relabel' not in out[0]

    def test_drops_conflict(self):
        gemma = [self._record('0', [_span('X', 'ORG', 0, 1)])]
        qwen = [self._record('0', [_span('X', 'LOC', 0, 1)])]
        out = merge_records(gemma, qwen)
        assert out[0]['gold_spans_relabel_merged'] == []

    def test_disjoint_ids_drop(self):
        gemma = [self._record('0', [_span('X', 'PER', 0, 1)])]
        qwen = [self._record('1', [_span('X', 'PER', 0, 1)])]
        out = merge_records(gemma, qwen)
        assert out == []

    def test_merge_sources(self):
        gemma = [self._record('0', [])]
        gemma[0]['relabel_model'] = 'gemma-A'
        qwen = [self._record('0', [])]
        qwen[0]['relabel_model'] = 'qwen-B'
        out = merge_records(gemma, qwen)
        assert out[0]['merge_sources'] == {'a': 'gemma-A', 'b': 'qwen-B'}
