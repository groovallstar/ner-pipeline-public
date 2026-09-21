"""VI 배포 추론 스크립트의 평가 모수 계약.

배포 vi test 가 101행이던 시절 이 스크립트는 앞 100행만 재도록 박혀 있었다.
번들의 test 가 커지면 그 상수가 조용히 대부분을 버리고도 같은 모양의 수치
표를 내므로, 기본을 전수로 두고 자르는 것은 인자로만 하게 고정한다.
"""
import pytest

from ner.scripts.eval_vi_ner_test import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEST,
    DEFAULT_THRESHOLDS,
    require_abs,
    select_rows,
    show_tagging,
)


def _rows(n):
    return [{'text': f'Câu {i} về Hà Nội.'} for i in range(n)]


def test_relative_path_is_rejected():
    with pytest.raises(SystemExit) as e:
        require_abs('data/wikiann_vi/test.jsonl', 'test data path')
    assert 'absolute path' in str(e.value)


def test_defaults_point_at_the_deployment_layout():
    assert DEFAULT_MODEL_DIR == '/data/ner/vi/model'
    assert DEFAULT_TEST == '/data/ner/vi/data/test.jsonl'
    assert DEFAULT_THRESHOLDS == '/data/ner/vi/thresholds.json'


def test_every_row_is_evaluated_by_default():
    """기본값은 전수 — 번들 test 가 커져도 조용히 버리지 않는다."""
    rows = _rows(3774)
    assert select_rows(rows, 0) == rows


def test_limit_truncates_and_says_so(capsys):
    """자를 때는 몇 건 중 몇 건인지 화면에 남는다."""
    rows = _rows(3774)
    got = select_rows(rows, 100)
    assert len(got) == 100
    out = capsys.readouterr().out
    assert '100' in out and '3774' in out


def test_show_tagging_truncates_and_says_so(capsys):
    """표시는 잘라도 전체 건수는 남는다."""
    rows = _rows(5)
    spans = [[{'type': 'LOC', 'start': 11, 'end': 17}] for _ in rows]
    show_tagging(rows, spans, limit=2)
    out = capsys.readouterr().out
    assert '2문장' in out and '전체 5' in out
    assert out.count('LOC:') == 2
