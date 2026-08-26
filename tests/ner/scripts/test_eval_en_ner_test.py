"""배포 추론 스크립트의 계약 — 경로 규칙과 표시 절단.

모델 없이 확인 가능한 것만 본다. 실제 추론은 패키지가 있어야 하므로
`tests/ner/classifier/test_en_deploy_package.py` 가 맡는다.

**절대경로 강제를 테스트하는 이유.** ja·vi 스크립트와 같은 규칙인데, 상대
경로를 받아주면 어느 cwd 에서 돌렸느냐에 따라 다른 파일을 재고도 같은 수치
표를 낸다 — 조용히 틀리는 쪽이라 실행 전에 크게 실패해야 한다.
"""
import pytest

from ner.scripts.eval_en_ner_test import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEST,
    DEFAULT_THRESHOLDS,
    require_abs,
    show_tagging,
)


def test_relative_path_is_rejected():
    with pytest.raises(SystemExit) as e:
        require_abs('data/ontonotes_en/test.jsonl', 'test data path')
    assert 'absolute path' in str(e.value)


def test_absolute_path_is_accepted():
    assert require_abs('/data/ner/en/data/test.jsonl', 'test data path') is None


def test_defaults_point_at_the_deployment_layout():
    """기본 경로가 배포 레이아웃을 그대로 가리켜야 인자 없이 돈다."""
    assert DEFAULT_MODEL_DIR == '/data/ner/en/model'
    assert DEFAULT_TEST == '/data/ner/en/data/test.jsonl'
    assert DEFAULT_THRESHOLDS == '/data/ner/en/thresholds.json'


def _rows(n):
    return [{'text': f'Sentence {i} about Seoul.'} for i in range(n)]


def test_show_tagging_truncates_and_says_so(capsys):
    """표시는 잘라도, 전체가 몇 건인지 화면에 남아야 한다."""
    rows = _rows(5)
    spans = [[{'type': 'LOC', 'start': 18, 'end': 23}] for _ in rows]
    show_tagging(rows, spans, limit=2)
    out = capsys.readouterr().out
    assert '2문장' in out and '전체 5' in out
    assert out.count('LOC:') == 2


def test_show_tagging_zero_limit_shows_all(capsys):
    rows = _rows(3)
    spans = [[] for _ in rows]
    show_tagging(rows, spans, limit=0)
    out = capsys.readouterr().out
    assert '3문장' in out
    assert '전체' not in out
    assert out.count('(없음)') == 3
