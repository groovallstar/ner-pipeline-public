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


@pytest.fixture(scope='module')
def deberta_tokenizer():
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        'microsoft/mdeberta-v3-base', use_fast=True
    )


def _has_label(labels, label_id):
    return any(lid == label_id for lid in labels)


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


def test_encode_deberta_multiword_entity_roundtrip(deberta_tokenizer, label_maps):
    """DeBERTa-V3(SentencePiece) fast tokenizer 는 ▁ 토큰 offset 에 선행 공백을
    포함시켜, 0번이 아닌 위치에서 시작하는 다단어 entity 의 첫 토큰이 경계
    밖으로 밀려 라벨이 누락된다(과거 F1 0.0 붕괴 원인). _encode_vi 의 공백
    trim 으로 char span 이 정확히 복원되어야 한다 (회귀 가드)."""
    label2id, id2label = label_maps
    text = 'Liên hệ: Nguyễn Văn An nhé.'
    name = 'Nguyễn Văn An'
    start = text.index(name)
    entities = [{
        'label': 'PER',
        'start_char': start,
        'end_char': start + len(name),
        'text': name,
    }]
    row = {'text': text, 'entities': entities, 'id': 'd1'}
    feat, offs = encode_row(row, deberta_tokenizer, label2id, 'vi', max_length=64)

    assert _has_label(feat['labels'], label2id['B-PER'])
    # trim 후 어떤 토큰 offset 도 선행/후행 공백을 포함하지 않아야 한다
    for s, e in offs:
        if s == e:
            continue
        assert not text[s].isspace()
        assert not text[e - 1].isspace()
    # decode 시 다단어 PER span 이 char 정확히 복원
    spans = decode_bio_to_spans(feat['labels'], offs, id2label)
    per_spans = [s for s in spans if s['type'] == 'PER']
    assert len(per_spans) == 1
    assert (per_spans[0]['start'], per_spans[0]['end']) == (
        entities[0]['start_char'], entities[0]['end_char']
    )


def test_encode_deberta_numeric_pii_trailing_period(deberta_tokenizer, label_maps):
    """다국어 SentencePiece 는 숫자형 entity 끝과 문장부호를 한 토큰으로 병합
    한다(예: '568.'). 후행 `.`/`,` trim 으로 경계 직전 숫자형 PII 가 정확히
    복원되어야 한다 (회귀 가드)."""
    label2id, id2label = label_maps
    num = '836394225258'
    text = f'ID: {num}.'
    start = text.index(num)
    entities = [{
        'label': 'ID_NUM',
        'start_char': start,
        'end_char': start + len(num),
        'text': num,
    }]
    row = {'text': text, 'entities': entities, 'id': 'd2'}
    feat, offs = encode_row(row, deberta_tokenizer, label2id, 'vi', max_length=32)

    spans = decode_bio_to_spans(feat['labels'], offs, id2label)
    id_spans = [s for s in spans if s['type'] == 'ID_NUM']
    assert len(id_spans) == 1
    # 끝이 마침표를 포함하지 않고 숫자 경계에서 정확히 끝나야 함
    assert (id_spans[0]['start'], id_spans[0]['end']) == (
        entities[0]['start_char'], entities[0]['end_char']
    )


@pytest.fixture(scope='module')
def phobert_tokenizer():
    pytest.importorskip('pyvi')
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('vinai/phobert-base-v2')


def test_encode_phobert_wordseg_roundtrip(phobert_tokenizer, label_maps):
    """PhoBERT(slow, 단어분절 전제): pyvi 분절 + 단어 char span 정렬로
    다단어 entity 가 원문 char span 으로 정확히 복원되어야 한다."""
    label2id, id2label = label_maps
    text = 'Liên hệ: Nguyễn Văn An tại Hà Nội.'
    name = 'Nguyễn Văn An'
    start = text.index(name)
    entities = [{
        'label': 'PER',
        'start_char': start,
        'end_char': start + len(name),
        'text': name,
    }]
    row = {'text': text, 'entities': entities, 'id': 'p1'}
    feat, offs = encode_row(row, phobert_tokenizer, label2id, 'vi', max_length=64)

    assert not phobert_tokenizer.is_fast  # slow 경로 확인
    assert _has_label(feat['labels'], label2id['B-PER'])
    spans = decode_bio_to_spans(feat['labels'], offs, id2label)
    per_spans = [s for s in spans if s['type'] == 'PER']
    assert len(per_spans) == 1
    assert (per_spans[0]['start'], per_spans[0]['end']) == (
        entities[0]['start_char'], entities[0]['end_char']
    )


def test_encode_no_entities(vi_tokenizer, label_maps):
    """엔티티 0 개 row 도 정상 인코드 (전부 O)."""
    label2id, _ = label_maps
    row = {'text': 'Đây là câu không có entity.', 'entities': [], 'id': 't3'}
    feat, offs = encode_row(row, vi_tokenizer, label2id, 'vi', max_length=32)
    o_id = label2id['O']
    # special token 은 -100, 그 외 모두 O
    non_special = [lid for lid in feat['labels'] if lid != -100]
    assert all(lid == o_id for lid in non_special)
    # input_ids 길이 = max_length (padding)
    assert len(feat['input_ids']) == 32
    assert len(feat['attention_mask']) == 32
    assert len(feat['labels']) == 32
