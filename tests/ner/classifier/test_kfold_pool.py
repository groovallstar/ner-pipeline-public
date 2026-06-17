"""classifier.kfold_pool 의 pooled span F1 / 무결성 검증 단위 테스트."""

import json
import os

import pytest

from ner.classifier.kfold_pool import pool_fold_predictions


def _write_fold(fold_dir: str, records: list) -> None:
    """fold dir 에 test_predictions.json 작성."""
    os.makedirs(fold_dir, exist_ok=True)
    path = os.path.join(fold_dir, 'test_predictions.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f)


def test_pool_fold_predictions_f1(tmp_path):
    """두 fold pooling 시 strict/relaxed F1 이 손계산 값과 일치.

    fold0: gold/pred 모두 [PER 0-3] → exact match.
    fold1: gold [LOC 0-5], pred [LOC 0-4] → strict miss, relaxed partial.
    strict: TP=1, gold=2, pred=2 → F1=0.5.
    relaxed: COR=1, PAR=1, POS=2, ACT=2 → score=1.5 → F1=0.75.
    """
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    _write_fold(fold0, [{
        'id': 's0',
        'text': 'sentence zero',
        'gold_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
        'pred_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
    }])
    _write_fold(fold1, [{
        'id': 's1',
        'text': 'sentence one',
        'gold_spans': [{'type': 'LOC', 'start': 0, 'end': 5}],
        'pred_spans': [{'type': 'LOC', 'start': 0, 'end': 4}],
    }])

    result = pool_fold_predictions([fold0, fold1])
    assert result['n_folds'] == 2
    assert result['n_sentences'] == 2
    assert result['strict']['overall']['f1'] == pytest.approx(0.5)
    assert result['relaxed']['overall']['f1'] == pytest.approx(0.75)


def test_pool_fold_predictions_duplicate_text(tmp_path):
    """fold 간 중복 text 가 있으면 ValueError."""
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    rec = [{
        'id': 'a',
        'text': 'same sentence',
        'gold_spans': [],
        'pred_spans': [],
    }]
    # id 는 달라도 text 가 같으면 에러
    rec2 = [{
        'id': 'b',
        'text': 'same sentence',
        'gold_spans': [],
        'pred_spans': [],
    }]
    _write_fold(fold0, rec)
    _write_fold(fold1, rec2)
    with pytest.raises(ValueError):
        pool_fold_predictions([fold0, fold1])


def test_pool_cross_fold_orig_leak_raises(tmp_path):
    """같은 orig 가 두 fold 의 test 에 걸치면 ValueError (주입텍스트는 달라도)."""
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    _write_fold(fold0, [{
        'id': 'a', 'text': 'injected one', 'orig': 'same original',
        'gold_spans': [], 'pred_spans': [],
    }])
    _write_fold(fold1, [{
        'id': 'b', 'text': 'injected two', 'orig': 'same original',
        'gold_spans': [], 'pred_spans': [],
    }])
    with pytest.raises(ValueError):
        pool_fold_predictions([fold0, fold1])


def test_pool_same_orig_within_fold_ok(tmp_path):
    """같은 orig 가 한 fold 안에서 여러 번(group 정상)이면 누출 아님."""
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    _write_fold(fold0, [
        {'id': 'a', 'text': 'inj a', 'orig': 'g1',
         'gold_spans': [], 'pred_spans': []},
        {'id': 'b', 'text': 'inj b', 'orig': 'g1',
         'gold_spans': [], 'pred_spans': []},
    ])
    _write_fold(fold1, [
        {'id': 'c', 'text': 'inj c', 'orig': 'g2',
         'gold_spans': [], 'pred_spans': []},
    ])
    result = pool_fold_predictions([fold0, fold1])
    assert result['n_sentences'] == 3
    assert result['cross_fold_orig_dups'] == 0


def test_pool_allow_cross_fold_leak_counts(tmp_path):
    """require_no_leak=False 면 누출에 ValueError 대신 카운트만."""
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    _write_fold(fold0, [{
        'id': 'a', 'text': 'inj one', 'orig': 'shared orig',
        'gold_spans': [], 'pred_spans': [],
    }])
    _write_fold(fold1, [{
        'id': 'b', 'text': 'inj two', 'orig': 'shared orig',
        'gold_spans': [], 'pred_spans': [],
    }])
    result = pool_fold_predictions([fold0, fold1], require_no_leak=False)
    assert result['n_sentences'] == 2
    assert result['cross_fold_orig_dups'] == 1


def test_pool_fold_predictions_duplicate_id_allowed(tmp_path):
    """id 가 중복돼도 text 가 다르면 정상 동작 (데이터셋 id 비고유 허용)."""
    fold0 = str(tmp_path / 'fold0')
    fold1 = str(tmp_path / 'fold1')
    _write_fold(fold0, [{
        'id': 'shared',
        'text': 'first sentence',
        'gold_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
        'pred_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
    }])
    _write_fold(fold1, [{
        'id': 'shared',
        'text': 'second sentence',
        'gold_spans': [{'type': 'LOC', 'start': 0, 'end': 5}],
        'pred_spans': [{'type': 'LOC', 'start': 0, 'end': 5}],
    }])
    result = pool_fold_predictions([fold0, fold1])
    assert result['n_sentences'] == 2
    assert result['strict']['overall']['f1'] == pytest.approx(1.0)
