"""boundary_weights_tensor 단위 테스트."""

import torch

from ner.classifier.data_utils import (
    boundary_weights_tensor,
    class_weights_tensor,
)


def _build_label2id():
    """canonical 10종 + O = 21 라벨."""
    types = ['PER', 'LOC', 'ORG', 'PROD', 'EVT',
             'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD']
    label2id = {'O': 0}
    for i, t in enumerate(types):
        label2id[f'B-{t}'] = 1 + 2 * i
        label2id[f'I-{t}'] = 2 + 2 * i
    return label2id


def test_boundary_weights_shape():
    label2id = _build_label2id()
    w = boundary_weights_tensor(label2id, w_b=1.5, w_i=1.2, w_o=1.0)
    assert w.shape == (21,)
    assert w.dtype == torch.float32


def test_boundary_weights_values():
    label2id = _build_label2id()
    w = boundary_weights_tensor(label2id, w_b=1.5, w_i=1.2, w_o=1.0)
    assert abs(w[label2id['O']].item() - 1.0) < 1e-5
    for label, idx in label2id.items():
        if label.startswith('B-'):
            assert abs(w[idx].item() - 1.5) < 1e-5, label
        elif label.startswith('I-'):
            assert abs(w[idx].item() - 1.2) < 1e-5, label


def test_boundary_weights_default_no_effect():
    label2id = _build_label2id()
    w = boundary_weights_tensor(label2id)
    assert torch.allclose(w, torch.ones(21))


def test_boundary_combined_with_class_weights():
    """class_weights × boundary_weights 의 elementwise 곱."""
    label2id = _build_label2id()
    cw = class_weights_tensor(label2id, w_ner=2.0, w_pii=0.5, w_o=1.0)
    bw = boundary_weights_tensor(label2id, w_b=1.5, w_i=1.0, w_o=1.0)
    combined = cw * bw

    def _close(actual, expected):
        assert abs(actual - expected) < 1e-5
    _close(combined[label2id['B-PER']].item(), 3.0)
    _close(combined[label2id['I-PER']].item(), 2.0)
    _close(combined[label2id['B-EMAIL']].item(), 0.75)
    _close(combined[label2id['I-EMAIL']].item(), 0.5)
    _close(combined[label2id['O']].item(), 1.0)
