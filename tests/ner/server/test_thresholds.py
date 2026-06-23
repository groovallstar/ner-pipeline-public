"""임계값 graceful 로딩·적용 단위 테스트."""

import json

from ner.server.thresholds import apply_thresholds, load_thresholds


def test_load_missing_returns_empty(tmp_path):
    """파일 없으면 빈 dict — abstention 미적용(raw)."""
    assert load_thresholds(str(tmp_path / 'nope.json')) == {}


def test_load_parses_thresholds(tmp_path):
    """thresholds.json 의 thresholds 키만 type→float 로 추출."""
    p = tmp_path / 't.json'
    p.write_text(json.dumps(
        {'thresholds': {'ORG': 0.7, 'LOC': 0.6}, 'meta': {'x': 1}}))
    assert load_thresholds(str(p)) == {'ORG': 0.7, 'LOC': 0.6}


def test_apply_empty_passthrough():
    """빈 임계값이면 입력을 그대로 통과."""
    spans = [{'label': 'ORG', 'score': 0.1}]
    assert apply_thresholds(spans, {}) == spans


def test_apply_filters_below_threshold():
    """score < 임계값 span 제거."""
    spans = [{'label': 'ORG', 'score': 0.5}, {'label': 'ORG', 'score': 0.9}]
    assert apply_thresholds(spans, {'ORG': 0.7}) == [
        {'label': 'ORG', 'score': 0.9}]


def test_apply_passes_types_without_threshold():
    """임계값에 없는 type 은 통과(graceful)."""
    spans = [{'label': 'PER', 'score': 0.1}]
    assert apply_thresholds(spans, {'ORG': 0.7}) == spans


def test_apply_passes_span_without_score():
    """score 키 없는 span 은 통과."""
    spans = [{'label': 'ORG'}]
    assert apply_thresholds(spans, {'ORG': 0.7}) == spans
