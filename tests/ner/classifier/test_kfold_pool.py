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
    """레거시 경로: 그룹 키 선언이 없으면 id 중복은 판정 기준이 아니다.

    id 의 뜻은 데이터셋마다 다르다(KO=원본 인덱스, VI=행 번호, JA=원문 파생
    다행 공유). 그룹 키로 쓸지는 데이터가 선언해야 하며, 선언이 없으면
    orig → text 순으로 물러선다.
    """
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


# --- 선언된 그룹 키 기반 누출 검증 ---


def _rec(sid, text, *, group=None, group_key=None, orig=None):
    """test_predictions.json 레코드 하나 (span 은 완전 일치로 고정)."""
    rec = {
        'id': sid,
        'text': text,
        'gold_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
        'pred_spans': [{'type': 'PER', 'start': 0, 'end': 3}],
    }
    if group is not None:
        rec['group'] = group
    if group_key is not None:
        rec['group_key'] = group_key
    if orig is not None:
        rec['orig'] = orig
    return rec


def test_group_basis_detects_leak_when_text_cannot(tmp_path):
    """형제 문장이 재작성돼 글자가 달라도, 선언된 그룹 값으로 누출을 잡는다."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'rewritten one', group='root0',
                             group_key='orig')])
    _write_fold(fold1, [_rec('b', 'utterly different', group='root0',
                             group_key='orig')])
    with pytest.raises(ValueError, match='cross-fold sibling leak'):
        pool_fold_predictions([fold0, fold1])

    result = pool_fold_predictions([fold0, fold1], require_no_leak=False)
    assert result['cross_fold_group_dups'] == 1
    assert result['leak_check_basis'] == 'group'
    assert result['group_key'] == 'orig'


def test_group_basis_allows_same_fold_repeat(tmp_path):
    """같은 fold 안의 형제 반복은 group 분할의 정상 동작이라 누출이 아니다."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'one', group='root0', group_key='orig'),
                        _rec('b', 'two', group='root0', group_key='orig')])
    _write_fold(fold1, [_rec('c', 'three', group='root1', group_key='orig')])
    result = pool_fold_predictions([fold0, fold1])
    assert result['cross_fold_group_dups'] == 0
    assert result['leak_check_basis'] == 'group'


def test_group_key_none_reports_unmeasured_not_zero(tmp_path):
    """행 단위 분할을 명시하면 카운터는 0 이 아니라 null(미측정)이다."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'one', group_key='none')])
    _write_fold(fold1, [_rec('b', 'two', group_key='none')])
    result = pool_fold_predictions([fold0, fold1])
    assert result['cross_fold_group_dups'] is None
    assert result['cross_fold_orig_dups'] is None
    assert result['leak_check_basis'] == 'none'
    # 같은 문장이 두 fold 에 있어도 미측정이라 ValueError 를 내지 않는다
    _write_fold(fold1, [_rec('b', 'one', group_key='none')])
    assert pool_fold_predictions([fold0, fold1])['leak_check_basis'] == 'none'


def test_explicit_group_key_none_overrides_recorded(tmp_path):
    """CLI 가 none 을 강제하면 기록된 group 값이 있어도 미측정으로 남는다."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'one', group='root0', group_key='orig')])
    _write_fold(fold1, [_rec('b', 'two', group='root0', group_key='orig')])
    result = pool_fold_predictions([fold0, fold1], group_key=None)
    assert result['cross_fold_group_dups'] is None
    assert result['leak_check_basis'] == 'none'


def test_legacy_predictions_fall_back_to_orig_then_text(tmp_path):
    """group_key 가 없는 옛 예측 파일은 orig → text 순으로 물러선다."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'one', orig='root0')])
    _write_fold(fold1, [_rec('b', 'two', orig='root1')])
    result = pool_fold_predictions([fold0, fold1])
    assert result['leak_check_basis'] == 'orig'
    assert result['cross_fold_group_dups'] == 0

    _write_fold(fold0, [_rec('a', 'one')])
    _write_fold(fold1, [_rec('b', 'two')])
    result = pool_fold_predictions([fold0, fold1])
    assert result['leak_check_basis'] == 'text'


def test_mixed_basis_reports_weakest(tmp_path):
    """근거가 섞이면 가장 약한 것으로 보고한다 (보수적)."""
    fold0, fold1 = str(tmp_path / 'fold0'), str(tmp_path / 'fold1')
    _write_fold(fold0, [_rec('a', 'one', group='root0', group_key='orig')])
    _write_fold(fold1, [_rec('b', 'two', group_key='orig')])  # group 값 없음
    result = pool_fold_predictions([fold0, fold1])
    assert result['leak_check_basis'] == 'text'
