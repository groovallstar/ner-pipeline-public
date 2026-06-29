"""classify_span_errors / aggregate_errors / sample_for_review 단위 테스트.

추론·모델 로딩 경로는 별도 (느린) 통합 테스트로 분리. 본 파일은 순수 함수만.
run_inference 의 tokenizer 선택 분기(use_fast fallback, VI/PhoBERT 경로)는
실모델 없이 mock 으로 결정 로직만 검증한다.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from ner.classifier.error_analysis import (
    NULL,
    aggregate_errors,
    build_confusion_matrix,
    classify_span_errors,
    load_pooled_predictions,
    run_pooled_error_analysis,
    sample_for_review,
    top_errors_by_type,
)


def _g(t, s, e, txt=''):
    return {'type': t, 'start': s, 'end': e, 'text': txt}


def test_classify_exact_match():
    gold = [_g('PER', 0, 3, 'abc')]
    pred = [_g('PER', 0, 3, 'abc')]
    r = classify_span_errors(gold, pred)
    assert r['counts']['exact'] == 1
    assert r['counts']['fn'] == 0
    assert r['counts']['fp'] == 0


def test_classify_boundary_error():
    # type 일치 + overlap 있으나 offset 다름
    gold = [_g('LOC', 0, 5, 'Tokyo')]
    pred = [_g('LOC', 0, 7, 'Tokyo S')]
    r = classify_span_errors(gold, pred)
    assert r['counts']['exact'] == 0
    assert len(r['fn']) == 1
    assert len(r['fp']) == 1
    assert r['fn'][0]['error_class'] == 'BOUNDARY'
    assert r['fp'][0]['error_class'] == 'BOUNDARY'


def test_classify_type_mismatch():
    # offset 동일하나 type 다름 → BOUNDARY 아님 (overlap 있고 type 다름 = TYPE_MISMATCH)
    gold = [_g('ORG', 10, 15)]
    pred = [_g('LOC', 10, 15)]
    r = classify_span_errors(gold, pred)
    assert r['fn'][0]['error_class'] == 'TYPE_MISMATCH'
    assert r['fp'][0]['error_class'] == 'TYPE_MISMATCH'


def test_classify_miss_and_hallucination():
    # gold 1, pred 1 — 둘은 무관 (overlap 0)
    gold = [_g('PER', 0, 3)]
    pred = [_g('LOC', 10, 13)]
    r = classify_span_errors(gold, pred)
    fn_classes = [e['error_class'] for e in r['fn']]
    fp_classes = [e['error_class'] for e in r['fp']]
    assert fn_classes == ['MISS']
    assert fp_classes == ['HALLUCINATION']


def test_classify_priority_exact_over_boundary():
    # 한 gold 가 두 pred 와 매칭 가능: EXACT 우선
    gold = [_g('PER', 0, 3)]
    pred = [_g('PER', 0, 4), _g('PER', 0, 3)]
    r = classify_span_errors(gold, pred)
    # EXACT 1개 + 남은 pred = HALLUCINATION
    assert r['counts']['exact'] == 1
    assert any(e['error_class'] == 'HALLUCINATION' for e in r['fp'])


def test_classify_boundary_picks_largest_overlap():
    # gold 1 + 두 BOUNDARY 후보 → 더 큰 overlap 선택
    gold = [_g('LOC', 0, 10)]
    pred = [_g('LOC', 0, 3), _g('LOC', 0, 9)]
    r = classify_span_errors(gold, pred)
    matched = [e for e in r['fn'] if e['error_class'] == 'BOUNDARY']
    assert len(matched) == 1
    assert matched[0]['matched_pred']['end'] == 9


def test_classify_overlap_diff_type_is_type_mismatch():
    gold = [_g('ORG', 0, 5)]
    pred = [_g('LOC', 2, 8)]
    r = classify_span_errors(gold, pred)
    assert r['fn'][0]['error_class'] == 'TYPE_MISMATCH'


def test_aggregate_counts():
    sr1 = {
        'counts': {'gold': 2, 'pred': 2, 'exact': 1, 'fn': 1, 'fp': 1},
        'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
        'fp': [{'error_class': 'HALLUCINATION', 'pred': _g('LOC', 5, 8)}],
    }
    sr2 = {
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
        'fn': [{'error_class': 'BOUNDARY', 'gold': _g('PER', 0, 3),
                'matched_pred': _g('PER', 0, 4)}],
        'fp': [{'error_class': 'BOUNDARY', 'pred': _g('PER', 0, 4),
                'matched_gold': _g('PER', 0, 3)}],
    }
    agg = aggregate_errors([sr1, sr2])
    assert agg['totals']['gold'] == 3
    assert agg['totals']['exact'] == 1
    assert agg['fn_by_class'] == {'MISS': 1, 'BOUNDARY': 1}
    assert agg['fp_by_class'] == {'HALLUCINATION': 1, 'BOUNDARY': 1}
    assert agg['fn_by_type']['PER'] == {'MISS': 1, 'BOUNDARY': 1}


def test_sample_for_review_deterministic():
    # 충분히 많은 오류 문장 생성
    sentence_results = []
    for i in range(100):
        sentence_results.append({
            'sent_idx': i,
            'text': f'sentence {i}',
            'gold_spans': [_g('PER', 0, 3)],
            'pred_spans': [],
            'exact': [],
            'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
            'fp': [],
            'counts': {'gold': 1, 'pred': 0, 'exact': 0, 'fn': 1, 'fp': 0},
        })
    s1 = sample_for_review(sentence_results, ratio=0.1, seed=42)
    s2 = sample_for_review(sentence_results, ratio=0.1, seed=42)
    assert [x['sent_idx'] for x in s1] == [x['sent_idx'] for x in s2]
    assert 5 <= len(s1) <= 15  # ratio + min_per_type 조합


def test_confusion_matrix_exact_and_boundary():
    # EXACT 1건 + BOUNDARY 1건 모두 대각선 (gold_type == pred_type)
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [{'gold': _g('PER', 0, 3), 'pred': _g('PER', 0, 3)}],
        'fn': [{'error_class': 'BOUNDARY', 'gold': _g('PER', 5, 10),
                'matched_pred': _g('PER', 5, 12)}],
        'fp': [{'error_class': 'BOUNDARY', 'pred': _g('PER', 5, 12),
                'matched_gold': _g('PER', 5, 10)}],
        'counts': {'gold': 2, 'pred': 2, 'exact': 1, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    # EXACT 1 + BOUNDARY 1 = 2 (모두 PER→PER)
    assert m['PER']['PER'] == 2
    assert NULL not in m  # MISS/HALL 없음
    assert 'PER' in m and len(m['PER']) == 1


def test_confusion_matrix_type_mismatch_no_double_count():
    # FN/FP 양쪽에 TYPE_MISMATCH entry 동일 — FN side 만 카운트되어 1번만 잡힘
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'TYPE_MISMATCH', 'gold': _g('ORG', 0, 5),
                'matched_pred': _g('PROD', 0, 5)}],
        'fp': [{'error_class': 'TYPE_MISMATCH', 'pred': _g('PROD', 0, 5),
                'matched_gold': _g('ORG', 0, 5)}],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    assert m['ORG']['PROD'] == 1
    assert m.get('PROD', {}).get('ORG', 0) == 0  # 반대 방향 카운트 없음


def test_confusion_matrix_miss_and_hallucination():
    # MISS → gold→NULL, HALLUCINATION → NULL→pred
    sr = [{
        'sent_idx': 0, 'text': '...',
        'gold_spans': [], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'MISS', 'gold': _g('LOC', 0, 3)}],
        'fp': [{'error_class': 'HALLUCINATION', 'pred': _g('PROD', 10, 13)}],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 1, 'fp': 1},
    }]
    m = build_confusion_matrix(sr)
    assert m['LOC'][NULL] == 1
    assert m[NULL]['PROD'] == 1


def test_top_errors_by_type_surface_frequency_and_context():
    # 'abc' 3번 + 'def' 1번 → 'abc' 빈도 우선
    sr = []
    for i in range(3):
        sr.append({
            'sent_idx': i,
            'text': 'XXX abc YYY',
            'gold_spans': [], 'pred_spans': [],
            'exact': [], 'fn': [],
            'fp': [{
                'error_class': 'HALLUCINATION',
                'pred': {'type': 'PROD', 'start': 4, 'end': 7, 'text': 'abc'},
            }],
            'counts': {'gold': 0, 'pred': 1, 'exact': 0, 'fn': 0, 'fp': 1},
        })
    sr.append({
        'sent_idx': 99,
        'text': 'YYY def ZZZ',
        'gold_spans': [], 'pred_spans': [],
        'exact': [], 'fn': [],
        'fp': [{
            'error_class': 'TYPE_MISMATCH',
            'pred': {'type': 'PROD', 'start': 4, 'end': 7, 'text': 'def'},
            'matched_gold': {'type': 'ORG', 'start': 4, 'end': 7, 'text': 'def'},
        }],
        'counts': {'gold': 1, 'pred': 1, 'exact': 0, 'fn': 0, 'fp': 1},
    })
    top = top_errors_by_type(
        sr, side='fp', type_='PROD', n=10, ctx_chars=4,
    )
    assert len(top) == 4
    assert top[0]['surface'] == 'abc'  # 빈도 3 우선
    assert top[0]['left_ctx'] == 'XXX '  # ctx_chars=4
    assert top[0]['right_ctx'] == ' YYY'
    assert top[0]['counter_type'] == NULL  # HALLUCINATION
    # def 항목은 TYPE_MISMATCH 로 ORG counter
    def_entries = [r for r in top if r['surface'] == 'def']
    assert len(def_entries) == 1
    assert def_entries[0]['counter_type'] == 'ORG'


def test_sample_for_review_skips_clean_sentences():
    # FN/FP 가 없는 문장은 샘플 후보 아님
    clean = {
        'sent_idx': 0,
        'text': 'clean',
        'gold_spans': [], 'pred_spans': [],
        'exact': [], 'fn': [], 'fp': [],
        'counts': {'gold': 0, 'pred': 0, 'exact': 0, 'fn': 0, 'fp': 0},
    }
    err = {
        'sent_idx': 1,
        'text': 'err',
        'gold_spans': [_g('PER', 0, 3)], 'pred_spans': [],
        'exact': [],
        'fn': [{'error_class': 'MISS', 'gold': _g('PER', 0, 3)}],
        'fp': [],
        'counts': {'gold': 1, 'pred': 0, 'exact': 0, 'fn': 1, 'fp': 0},
    }
    s = sample_for_review([clean, err], ratio=1.0, min_per_type=1, seed=0)
    assert all(x['sent_idx'] != 0 for x in s)
    assert any(x['sent_idx'] == 1 for x in s)


def _write_fold(fold_dir, records):
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, 'test_predictions.json'),
              'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)


def test_load_pooled_predictions_concat_and_index(tmp_path):
    # fold0 2문장 + fold1 1문장 → 전역 sent_idx 0,1,2
    f0 = str(tmp_path / 'fold0')
    f1 = str(tmp_path / 'fold1')
    _write_fold(f0, [
        {'text': 'AAA org BBB',
         'gold_spans': [{'type': 'ORG', 'start': 4, 'end': 7}],
         'pred_spans': [{'type': 'ORG', 'start': 4, 'end': 7}]},
        {'text': 'no entities here',
         'gold_spans': [], 'pred_spans': []},
    ])
    _write_fold(f1, [
        {'text': 'CCC loc DDD',
         'gold_spans': [{'type': 'LOC', 'start': 4, 'end': 7}],
         'pred_spans': []},
    ])
    sents = load_pooled_predictions([f0, f1])
    assert [s['sent_idx'] for s in sents] == [0, 1, 2]
    # span 에 text 키 없을 때 문장에서 surface 복원
    assert sents[0]['gold_spans'][0]['text'] == 'org'


def test_load_pooled_predictions_duplicate_text_raises(tmp_path):
    f0 = str(tmp_path / 'fold0')
    f1 = str(tmp_path / 'fold1')
    dup = {'text': 'same sentence',
           'gold_spans': [], 'pred_spans': []}
    _write_fold(f0, [dup])
    _write_fold(f1, [dup])
    with pytest.raises(ValueError, match='duplicate test sentence'):
        load_pooled_predictions([f0, f1])


def test_run_pooled_error_analysis_aggregate(tmp_path):
    # fold0: ORG EXACT 1 + ORG HALLUCINATION 1 / fold1: LOC MISS 1
    f0 = str(tmp_path / 'fold0')
    f1 = str(tmp_path / 'fold1')
    _write_fold(f0, [
        {'text': 'AAA org BBB ZZ',
         'gold_spans': [{'type': 'ORG', 'start': 4, 'end': 7}],
         'pred_spans': [{'type': 'ORG', 'start': 4, 'end': 7},
                        {'type': 'ORG', 'start': 12, 'end': 14}]},
    ])
    _write_fold(f1, [
        {'text': 'CCC loc DDD',
         'gold_spans': [{'type': 'LOC', 'start': 4, 'end': 7}],
         'pred_spans': []},
    ])
    out = str(tmp_path / 'diag')
    res = run_pooled_error_analysis(
        lang='ja', fold_dirs=[f0, f1], output_dir=out,
        with_diagnosis=True, diagnosis_fp_types=['ORG'],
        diagnosis_fn_types=['LOC'],
    )
    agg = res['aggregate']
    assert agg['totals']['exact'] == 1
    assert agg['fp_by_class'].get('HALLUCINATION') == 1
    assert agg['fn_by_class'].get('MISS') == 1
    # pooled suffix 산출물 생성 확인
    assert os.path.exists(os.path.join(out, 'error_analysis_pooled.json'))
    assert os.path.exists(os.path.join(out, 'confusion_matrix_pooled.md'))
    assert os.path.exists(os.path.join(out, 'fp_top_ORG_pooled.md'))
    assert os.path.exists(os.path.join(out, 'fn_top_LOC_pooled.md'))


# ---------------------------------------------------------------------------
# run_inference tokenizer 선택 분기 (VI/PhoBERT + use_fast fallback)
# 실모델·네트워크 없이 분기 결정 로직만 mock 으로 검증.
# ---------------------------------------------------------------------------

def _make_fast_tokenizer(name_or_path='some/fast-model'):
    """is_fast=True 를 가진 가짜 fast tokenizer."""
    tok = MagicMock()
    tok.is_fast = True
    tok.name_or_path = name_or_path
    return tok


def _make_slow_tokenizer(name_or_path='some/slow-model'):
    """is_fast=False, name 에 'phobert' 없는 가짜 slow tokenizer."""
    tok = MagicMock()
    tok.is_fast = False
    tok.name_or_path = name_or_path
    return tok


def _make_phobert_tokenizer():
    """name_or_path 에 'phobert' 를 포함하는 가짜 PhoBERT tokenizer."""
    tok = MagicMock()
    tok.is_fast = False
    tok.name_or_path = 'vinai/phobert-base'
    return tok


def _patch_data_loading(rows=None):
    """load_jsonl + split_train_valid_test 를 가짜로 대체하는 컨텍스트 매니저.

    run_inference 는 tokenizer 선택 전에 데이터 로딩을 수행하므로,
    분기 테스트에서도 파일 I/O 없이 진행하도록 두 함수를 함께 mock 한다.
    """
    if rows is None:
        rows = []
    return patch.multiple(
        'ner.classifier.data_utils',
        load_jsonl=MagicMock(return_value=rows),
        split_train_valid_test=MagicMock(return_value=(rows, rows, rows)),
    )


def test_run_inference_use_fast_fallback():
    """use_fast=True 가 OSError 를 일으킬 때 use_fast=False 로 fallback 한다."""
    slow_tok = _make_slow_tokenizer()
    call_log = []

    def fake_from_pretrained(name, use_fast):
        call_log.append(use_fast)
        if use_fast:
            raise OSError('fast tokenizer not available')
        return slow_tok

    # AutoTokenizer 는 run_inference 내부에서 from transformers import 로 로컬 바인딩
    # → transformers.AutoTokenizer 를 패치해야 한다
    # encode_dataset 에서 중단해 torch/모델 로딩까지 진입하지 않는다
    with _patch_data_loading(), \
         patch('transformers.AutoTokenizer.from_pretrained',
               side_effect=fake_from_pretrained), \
         patch('ner.classifier.data_utils.encode_dataset',
               side_effect=RuntimeError('stop early')):
        from ner.classifier.error_analysis import run_inference
        with pytest.raises(RuntimeError, match='stop early'):
            run_inference(
                lang='vi',
                model_path='/fake/model',
                data_path='/fake/data.jsonl',
                tokenizer_name='some/tokenizer',
                split='test',
            )

    # use_fast=True 먼저 시도, 실패 후 use_fast=False 로 재시도
    assert call_log == [True, False], (
        f'expected [True, False] but got {call_log}'
    )


def test_run_inference_use_fast_success():
    """use_fast=True 가 성공하면 fallback(use_fast=False)은 호출되지 않는다."""
    fast_tok = _make_fast_tokenizer()
    call_log = []

    def fake_from_pretrained(name, use_fast):
        call_log.append(use_fast)
        return fast_tok

    with _patch_data_loading(), \
         patch('transformers.AutoTokenizer.from_pretrained',
               side_effect=fake_from_pretrained), \
         patch('ner.classifier.data_utils.encode_dataset',
               side_effect=RuntimeError('stop early')):
        from ner.classifier.error_analysis import run_inference
        with pytest.raises(RuntimeError, match='stop early'):
            run_inference(
                lang='vi',
                model_path='/fake/model',
                data_path='/fake/data.jsonl',
                tokenizer_name='some/tokenizer',
                split='test',
            )

    # fallback 없이 use_fast=True 단 1회만 호출
    assert call_log == [True], f'expected [True] but got {call_log}'


def test_run_inference_tokenizer_none_raises():
    """tokenizer_name=None 이면 ValueError 를 즉시 발생시킨다."""
    # tokenizer_name=None 검사는 데이터 로딩 이후에 있으므로 I/O 를 mock 한다
    with _patch_data_loading():
        from ner.classifier.error_analysis import run_inference
        with pytest.raises(ValueError, match='tokenizer_name is required'):
            run_inference(
                lang='vi',
                model_path='/fake/model',
                data_path='/fake/data.jsonl',
                tokenizer_name=None,
                split='test',
            )


# ---------------------------------------------------------------------------
# _is_phobert 판별 로직 (data_utils — 네트워크 불필요 순수 함수)
# ---------------------------------------------------------------------------

def test_is_phobert_detects_phobert_in_name_or_path():
    """name_or_path 에 'phobert' 가 포함되면 True."""
    from ner.classifier.data_utils import _is_phobert
    tok = _make_phobert_tokenizer()
    assert _is_phobert(tok) is True


def test_is_phobert_false_for_fast_tokenizer():
    """fast tokenizer 는 name 에 'phobert' 없으면 False."""
    from ner.classifier.data_utils import _is_phobert
    tok = _make_fast_tokenizer('xlm-roberta-base')
    assert _is_phobert(tok) is False


def test_is_phobert_detects_phobert_class_name():
    """클래스 이름에 'phobert' 가 포함되면 True (name_or_path 무관)."""
    from ner.classifier.data_utils import _is_phobert

    class PhobertTokenizer:
        is_fast = False
        name_or_path = 'some/other-model'

    assert _is_phobert(PhobertTokenizer()) is True


# ---------------------------------------------------------------------------
# encode_row 분기 선택 (data_utils — mock 으로 실제 인코딩 생략)
# ---------------------------------------------------------------------------

def _dummy_enc_result():
    """(enc, offsets) 가짜 반환값."""
    enc = {'input_ids': [0], 'attention_mask': [1]}
    offs = [(0, 0)]
    return enc, offs


def test_encode_row_routes_fast_tokenizer_to_encode_vi():
    """is_fast=True 이면 _encode_vi 경로를 탄다."""
    from ner.classifier import data_utils
    tok = _make_fast_tokenizer()
    row = {'text': 'Hà Nội', 'entities': []}

    with patch.object(data_utils, '_encode_vi',
                      return_value=_dummy_enc_result()) as mock_vi, \
         patch.object(data_utils, '_encode_phobert',
                      return_value=_dummy_enc_result()) as mock_ph, \
         patch.object(data_utils, '_encode_ja',
                      return_value=_dummy_enc_result()) as mock_ja:
        label2id = {'O': 0}
        data_utils.encode_row(row, tok, label2id, lang='vi')

    mock_vi.assert_called_once()
    mock_ph.assert_not_called()
    mock_ja.assert_not_called()


def test_encode_row_routes_phobert_to_encode_phobert():
    """is_fast=False + _is_phobert=True 이면 _encode_phobert 경로를 탄다."""
    from ner.classifier import data_utils
    tok = _make_phobert_tokenizer()
    row = {'text': 'Hà Nội', 'entities': []}

    with patch.object(data_utils, '_encode_vi',
                      return_value=_dummy_enc_result()) as mock_vi, \
         patch.object(data_utils, '_encode_phobert',
                      return_value=_dummy_enc_result()) as mock_ph, \
         patch.object(data_utils, '_encode_ja',
                      return_value=_dummy_enc_result()) as mock_ja:
        label2id = {'O': 0}
        data_utils.encode_row(row, tok, label2id, lang='vi')

    mock_ph.assert_called_once()
    mock_vi.assert_not_called()
    mock_ja.assert_not_called()


def test_encode_row_routes_slow_non_phobert_to_encode_ja():
    """is_fast=False + _is_phobert=False 이면 _encode_ja 경로(JA slow)를 탄다."""
    from ner.classifier import data_utils
    tok = _make_slow_tokenizer('tohoku-nlp/bert-base-japanese-v3')
    row = {'text': '東京', 'entities': []}

    with patch.object(data_utils, '_encode_vi',
                      return_value=_dummy_enc_result()) as mock_vi, \
         patch.object(data_utils, '_encode_phobert',
                      return_value=_dummy_enc_result()) as mock_ph, \
         patch.object(data_utils, '_encode_ja',
                      return_value=_dummy_enc_result()) as mock_ja:
        label2id = {'O': 0}
        data_utils.encode_row(row, tok, label2id, lang='ja')

    mock_ja.assert_called_once()
    mock_vi.assert_not_called()
    mock_ph.assert_not_called()
