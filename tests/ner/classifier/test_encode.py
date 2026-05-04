"""classifier 의 encode_row 가 실제 토크나이저로 정상 동작하는지 검증.

JA(slow tokenizer) 와 VI(fast tokenizer) 의 char-offset 정렬이 entity span
복원 가능한 BIO 라벨을 산출하는지 round-trip 으로 확인.
"""

import pytest

from ner.classifier.data_utils import (
    build_label_maps,
    decode_bio_to_spans,
    encode_row,
)


@pytest.fixture(scope='module')
def label_maps():
    return build_label_maps()


@pytest.fixture(scope='module')
def vi_tokenizer():
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)


@pytest.fixture(scope='module')
def ja_tokenizer():
    pytest.importorskip('fugashi')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        'tohoku-nlp/bert-base-japanese-v3', use_fast=False
    )


def _has_label(labels, label_id):
    return any(l == label_id for l in labels)


def test_encode_vi_email_roundtrip(vi_tokenizer, label_maps):
    """VI: EMAIL 엔티티의 BIO 라벨을 디코드하면 원래 char span 이 복원되어야 한다."""
    label2id, id2label = label_maps
    text = 'Liên hệ: a@b.com là email công ty.'
    entities = [{
        'label': 'EMAIL',
        'start_char': text.index('a@b.com'),
        'end_char': text.index('a@b.com') + len('a@b.com'),
        'text': 'a@b.com',
    }]
    row = {'text': text, 'entities': entities, 'id': 't1'}
    feat, offs = encode_row(row, vi_tokenizer, label2id, 'vi', max_length=64)

    # 라벨에 EMAIL 이 적어도 한 번 등장
    assert _has_label(feat['labels'], label2id['B-EMAIL'])

    # decode 시 원래 entity span 이 (거의) 복원
    spans = decode_bio_to_spans(feat['labels'], offs, id2label)
    email_spans = [s for s in spans if s['type'] == 'EMAIL']
    assert len(email_spans) == 1
    s = email_spans[0]
    # fast tokenizer 정렬은 char 정확 — 동일 span 기대
    assert s['start'] == entities[0]['start_char']
    assert s['end'] == entities[0]['end_char']


def test_encode_ja_email_roundtrip(ja_tokenizer, label_maps):
    """JA: slow tokenizer 의 greedy match 로 EMAIL 엔티티가 BIO 에 정렬되는지."""
    label2id, id2label = label_maps
    email = 'opacgpos58@gmail.com'
    text = f'お問い合わせは {email} まで。'
    entities = [{
        'label': 'EMAIL',
        'start_char': text.index(email),
        'end_char': text.index(email) + len(email),
        'text': email,
    }]
    row = {'text': text, 'entities': entities, 'id': 't2'}
    feat, offs = encode_row(row, ja_tokenizer, label2id, 'ja', max_length=128)

    # 라벨에 EMAIL 이 적어도 한 번 등장
    assert _has_label(feat['labels'], label2id['B-EMAIL'])

    # decode → EMAIL span 복원 (subword 그래뉼래리티로 인해 start/end 가 entity 범위 내에 위치)
    spans = decode_bio_to_spans(feat['labels'], offs, id2label)
    email_spans = [s for s in spans if s['type'] == 'EMAIL']
    assert len(email_spans) >= 1
    es = email_spans[0]
    # 디코드 span 이 entity 범위 내에 안전히 들어가야 함
    assert entities[0]['start_char'] <= es['start']
    assert es['end'] <= entities[0]['end_char']


def test_encode_no_entities(vi_tokenizer, label_maps):
    """엔티티 0 개 row 도 정상 인코드 (전부 O)."""
    label2id, _ = label_maps
    row = {'text': 'Đây là câu không có entity.', 'entities': [], 'id': 't3'}
    feat, offs = encode_row(row, vi_tokenizer, label2id, 'vi', max_length=32)
    o_id = label2id['O']
    # special token 은 -100, 그 외 모두 O
    non_special = [l for l in feat['labels'] if l != -100]
    assert all(l == o_id for l in non_special)
    # input_ids 길이 = max_length (padding)
    assert len(feat['input_ids']) == 32
    assert len(feat['attention_mask']) == 32
    assert len(feat['labels']) == 32
