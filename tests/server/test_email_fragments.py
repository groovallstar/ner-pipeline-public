"""서버 추론이 EMAIL 조각을 임계값보다 먼저 합치는지 — 실모델 없이 도는 겹.

디코더가 한 주소를 `al` + `42@example.com` 두 조각으로 내는 상황을 stub 으로
고정하고, `predict`·`predict_many` 두 입구가 모두 한 span 을 돌려주는지 본다.
실모델로 같은 축을 재는 겹은 `test_en_email_probe.py` 다(모델 없으면 skip).
"""

from ner.classifier.data_utils import build_label_maps
from server.inference import LangModel

_TEXT = 'Please send the report to alice42@example.com.'
# 손으로 센 오프셋: `alice42@example.com` 은 26..45, 그중 `al` 26..28 과
# `42@example.com` 31..45 가 EMAIL 조각이고 `ice` 28..31 은 O 로 떨어졌다.
_EMAIL = (26, 45)


def _fragments(low=0.9, high=0.95):
    return [{'type': 'EMAIL', 'start': 26, 'end': 28, 'score': low},
            {'type': 'EMAIL', 'start': 31, 'end': 45, 'score': high}]


def _bare_model(fragments, thresholds=None):
    """디코드 결과만 정해 둔 LangModel — 토크나이저·forward 를 건너뛴다."""
    lm = object.__new__(LangModel)
    lm.lang = 'en'
    lm.label2id, lm.id2label = build_label_maps()
    lm.thresholds = thresholds or {}
    lm.has_thresholds = bool(lm.thresholds)
    lm._tokenize = lambda text: ([{}], [[]], [0])
    lm._infer_encoded = lambda feats, offs, bases: [dict(s) for s in fragments]
    lm._forward_feats = lambda feats: ([None], [None])
    lm._decode_chunk = (
        lambda pred, conf, offs, base: [dict(s) for s in fragments])
    return lm


def _emails(spans):
    return [(s['start_char'], s['end_char'], s['text'])
            for s in spans if s['label'] == 'EMAIL']


def test_predict_returns_one_span_for_a_fragmented_address():
    lm = _bare_model(_fragments())
    assert _emails(lm.predict(_TEXT)) == [
        (*_EMAIL, 'alice42@example.com')]


def test_predict_many_returns_one_span_for_a_fragmented_address():
    lm = _bare_model(_fragments())
    [spans] = lm.predict_many([_TEXT])
    assert _emails(spans) == [(*_EMAIL, 'alice42@example.com')]


def test_fragments_merge_before_the_threshold_filters_them():
    """임계값은 합친 span 에 건다 — 학습 평가가 임계값을 fit 한 단위와 같다.

    거르고 나서 합치면 확신 있는 조각(0.95)만 살아 `42@example.com` 이라는
    반쪽 주소가 나간다. 합친 span 은 약한 조각의 score(0.6)를 받아 통째로
    걸러진다.
    """
    lm = _bare_model(_fragments(low=0.6, high=0.95), thresholds={'EMAIL': 0.7})
    assert _emails(lm.predict(_TEXT)) == []
    [spans] = lm.predict_many([_TEXT])
    assert _emails(spans) == []
