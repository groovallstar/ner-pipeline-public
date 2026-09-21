"""학습 평가(`evaluate_model`)가 서버와 같이 EMAIL 조각을 합치는지.

서버만 합치고 평가가 안 합치면, 배포 metrics 가 서버 출력과 다른 디코드로
잰 값이 된다. 모델 로드를 가짜로 바꿔 한 주소를 두 조각으로 내게 하고,
strict F1 이 그 주소를 맞힌 것으로 세는지 본다.
"""

from types import SimpleNamespace

import torch

from ner.classifier import train_eval
from ner.classifier.data_utils import build_label_maps

_TEXT = 'to alice42@example.com.'
# 손으로 센 오프셋: 주소 3..22. 토큰 `al` 3..5 · `ice` 5..8 · `42@example.com`
# 8..22 · `.` 22..23 이고, 앞뒤 (0, 0) 은 special token 이다.
_OFFSETS = [(0, 0), (0, 2), (3, 5), (5, 8), (8, 22), (22, 23), (0, 0)]
_LABELS = ['O', 'O', 'B-EMAIL', 'O', 'B-EMAIL', 'O', 'O']


class _FakeModel:
    """정해 둔 라벨에 큰 logit 을 주는 토큰 분류기."""

    def __init__(self, label_ids, num_labels):
        self._label_ids = label_ids
        self._num_labels = num_labels

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self._num_labels), -10.0)
        for b in range(batch):
            for t, lid in enumerate(self._label_ids):
                logits[b, t, lid] = 10.0
        return SimpleNamespace(logits=logits)


def test_evaluate_model_scores_a_fragmented_address_as_one_hit(monkeypatch):
    label2id, id2label = build_label_maps()
    fake = _FakeModel([label2id[x] for x in _LABELS], len(label2id))
    monkeypatch.setattr(
        train_eval.AutoModelForTokenClassification, 'from_pretrained',
        lambda *args, **kwargs: fake)

    n = len(_OFFSETS)
    res = train_eval.evaluate_model(
        model_path='unused',
        eval_features=[{'input_ids': [0] * n, 'attention_mask': [1] * n}],
        eval_offsets=[_OFFSETS],
        eval_rows=[{'text': _TEXT, 'entities': [
            {'label': 'EMAIL', 'start_char': 3, 'end_char': 22}]}],
        id2label=id2label, return_spans=True)

    assert res['pred_spans_list'] == [[
        {'type': 'EMAIL', 'start': 3, 'end': 22}]]
    assert res['strict']['overall']['f1'] == 1.0
