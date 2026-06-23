"""모델 통합 테스트 — 실제 /data/ner/{ja,vi} 모델 의존(로컬 전용).

/data 는 gitignore 라 CI 에서 재현 불가하므로, 모델 디렉토리가 없으면
`pytest.skip`(약화 아님 — 사유 출력). 핵심은 모델 품질이 아니라 **offset
정합성**: 반환된 어떤 span 도 `text[start_char:end_char] == span['text']`.
"""

import json
import os

import pytest

from ner.classifier.data_utils import load_jsonl
from ner.metrics.span_metrics import compute_offset_span_f1
from ner.server.config import ServerConfig
from ner.server.inference import LangModel

_CONFIG = ServerConfig()
_JA_DIR = _CONFIG.model_dir('ja')
_VI_DIR = _CONFIG.model_dir('vi')
_JA_TEST = os.path.join(_CONFIG.model_root, 'ja', 'data', 'test.jsonl')
_JA_METRICS = os.path.join(_CONFIG.model_root, 'ja', 'metrics.json')
_JA_PARITY_READY = (os.path.isdir(_JA_DIR) and os.path.isfile(_JA_TEST)
                    and os.path.isfile(_JA_METRICS))


def _assert_offsets_consistent(model, text):
    for ent in model.predict(text):
        assert text[ent['start_char']:ent['end_char']] == ent['text']


@pytest.mark.skipif(not os.path.isdir(_JA_DIR),
                    reason=f'ja model dir not present: {_JA_DIR}')
def test_ja_offset_consistency():
    """ja: 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(model, '織田信長は東京都千代田区に住んでいた。')


@pytest.mark.skipif(not os.path.isdir(_VI_DIR),
                    reason=f'vi model dir not present: {_VI_DIR}')
def test_vi_offset_consistency():
    """vi(PhoBERT): 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(model, 'Hà Nội là thủ đô của Việt Nam.')


@pytest.mark.skipif(not os.path.isdir(_VI_DIR),
                    reason=f'vi model dir not present: {_VI_DIR}')
def test_vi_graceful_raw_when_no_thresholds():
    """vi 는 thresholds.json 부재 → raw(abstention 미적용)로 동작."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    assert model.has_thresholds is False


@pytest.mark.skipif(not os.path.isdir(_JA_DIR),
                    reason=f'ja model dir not present: {_JA_DIR}')
def test_ja_long_input_chunk_offsets():
    """max_length 초과 입력: chunk 분할 후에도 span offset 이 원문과 일치.

    chunk 가 실제로 쪼개지는지 확인한 뒤(>1), 병합된 모든 span 의 글로벌
    offset 정합성(`text[s:e]==surface`)을 검증한다 — auto-chunk offset 보정.
    """
    from ner.server.chunking import split_for_length
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    text = '東京都に住む織田信長は安土城を築いた。' * 40
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1
    ents = model.predict(text)
    assert ents  # 후반 청크에서도 엔티티가 잡혀야 함
    for ent in ents:
        assert text[ent['start_char']:ent['end_char']] == ent['text']


def _ja_overall_f1(model, rows, abstain):
    """ja test rows 에 대한 strict overall F1 — 서버 predict 경로로 계산.

    gold/pred 를 metrics 모듈이 요구하는 {type,start,end} 형식으로 변환.
    """
    gold_list, pred_list = [], []
    for row in rows:
        gold_list.append([
            {'type': e['label'], 'start': e['start_char'],
             'end': e['end_char']}
            for e in row['entities']])
        pred_list.append([
            {'type': s['label'], 'start': s['start_char'],
             'end': s['end_char']}
            for s in model.predict(row['text'], abstain=abstain)])
    return compute_offset_span_f1(gold_list, pred_list)['overall']


@pytest.mark.skipif(not _JA_PARITY_READY,
                    reason='ja model/test/metrics not all present')
def test_ja_parity_operating_point():
    """abstention on → 서버 operating-point F1 이 metrics.json 과 일치.

    eval 스크립트를 import 하지 않고, 기록된 운영점(데이터)과 대조해 parity
    를 검증한다(리팩터 중인 코드에 비의존).
    """
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = load_jsonl(_JA_TEST)
    expected = json.load(open(_JA_METRICS, encoding='utf-8'))[
        'abstention']['overall_operating']
    got = _ja_overall_f1(model, rows, abstain=True)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)
    assert got['precision'] == pytest.approx(expected['precision'], abs=1e-6)
    assert got['recall'] == pytest.approx(expected['recall'], abs=1e-6)


@pytest.mark.skipif(not _JA_PARITY_READY,
                    reason='ja model/test/metrics not all present')
def test_ja_parity_baseline_raw():
    """abstention off → raw baseline F1 이 metrics.json baseline 과 일치."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = load_jsonl(_JA_TEST)
    expected = json.load(open(_JA_METRICS, encoding='utf-8'))[
        'abstention']['overall_baseline']
    got = _ja_overall_f1(model, rows, abstain=False)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)
