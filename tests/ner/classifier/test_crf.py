"""BertCRFForTokenClassification 단위 테스트.

다운로드 회피를 위해 BertConfig 로 작은 BERT 를 inline 인스턴스화 후
임시 디렉토리에 저장 → BertCRFForTokenClassification.from_pretrained 로 로드.
"""

import os
import tempfile

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification

from ner.classifier.train_eval import BertCRFForTokenClassification


LABEL2ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@pytest.fixture(scope='module')
def tiny_model_path():
    """모듈당 1회 — 작은 BERT (vocab 1000, hidden 32, 2 layer) 를
    임시 디렉토리에 저장한다."""
    config = BertConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = BertForTokenClassification(config)
    tmp = tempfile.mkdtemp()
    model.save_pretrained(tmp)
    yield tmp


@pytest.fixture(scope='module')
def crf_model(tiny_model_path):
    """모듈당 1회 CRF wrapper 인스턴스."""
    return BertCRFForTokenClassification(
        tiny_model_path, num_labels=len(LABEL2ID),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )


def test_forward_with_labels_returns_loss_and_logits(crf_model):
    """forward(labels=...) 는 {'loss', 'logits'} 반환."""
    input_ids = torch.tensor(
        [[10, 20, 30, 40, 50]], dtype=torch.long
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.long)

    out = crf_model(input_ids, attention_mask, labels=labels)

    assert 'loss' in out
    assert 'logits' in out
    assert out['loss'].requires_grad
    assert torch.isfinite(out['loss'])
    assert out['logits'].shape == (1, 5, 5)


def test_forward_without_labels_returns_logits_only(crf_model):
    """forward(labels=None) 는 logits 만 반환."""
    input_ids = torch.tensor([[10, 20, 30, 50]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

    out = crf_model(input_ids, attention_mask)

    assert 'logits' in out
    assert 'loss' not in out
    assert out['logits'].shape == (1, 4, 5)


def test_decode_returns_variable_length_lists(crf_model):
    """Viterbi decode 는 attention_mask 합 길이의 가변 시퀀스를 반환."""
    input_ids = torch.tensor([
        [10, 20, 30, 40, 50, 0],
        [10, 20, 30, 40, 60, 70],
    ], dtype=torch.long)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.long)

    preds = crf_model.decode(input_ids, attention_mask)

    assert len(preds) == 2
    assert len(preds[0]) == 5
    assert len(preds[1]) == 6
    for tag in preds[0] + preds[1]:
        assert 0 <= tag <= 4


def test_ignore_index_in_loss(crf_model):
    """-100 위치는 mask 적용으로 loss 에 영향 없이 finite 유지."""
    input_ids = torch.tensor([[10, 20, 30, 50, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    labels = torch.tensor([[0, 1, 2, -100, -100]], dtype=torch.long)

    out = crf_model(input_ids, attention_mask, labels=labels)

    assert torch.isfinite(out['loss'])


def test_save_load_round_trip(crf_model):
    """save_pretrained → from_pretrained 후 CRF transition matrix 보존."""
    with tempfile.TemporaryDirectory() as tmp:
        crf_model.save_pretrained(tmp)

        assert os.path.exists(os.path.join(tmp, 'crf.pt'))
        assert os.path.exists(os.path.join(tmp, 'config.json'))

        loaded = BertCRFForTokenClassification.from_pretrained(
            tmp, num_labels=5,
            id2label=ID2LABEL, label2id=LABEL2ID,
        )

        assert torch.allclose(
            loaded.crf.transitions.detach(),
            crf_model.crf.transitions.detach(),
        )
        assert torch.allclose(
            loaded.crf.start_transitions.detach(),
            crf_model.crf.start_transitions.detach(),
        )
        assert torch.allclose(
            loaded.crf.end_transitions.detach(),
            crf_model.crf.end_transitions.detach(),
        )


def test_bio_consistency_via_decode(crf_model):
    """학습되지 않은 CRF 도 decode 결과의 길이가 mask 와 일치."""
    input_ids = torch.tensor(
        [[10, 20, 30, 40, 50, 60, 70]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
    )

    preds = crf_model.decode(input_ids, attention_mask)

    assert len(preds) == 1
    assert len(preds[0]) == 7
