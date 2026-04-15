"""label_merger 단위 테스트."""
from augmenters.pii.injector import (
    apply_label_merge,
    is_simple_place,
    merge_entities,
)
from augmenters.pii.schema import Entity


def test_name_to_jinmei():
    e = Entity(label='NAME', start_char=0, end_char=3, text='太郎')
    r = apply_label_merge(e, {'NAME': '人名'})
    assert r.label == '人名'
    assert r.start_char == 0 and r.end_char == 3


def test_address_simple_place_mapped():
    e = Entity(label='ADDRESS', start_char=0, end_char=3, text='東京')
    r = apply_label_merge(e, {'ADDRESS': '地名'})
    assert r.label == '地名'


def test_address_complex_kept():
    addr = '東京都 新宿区 1-2-3'
    e = Entity(label='ADDRESS', start_char=0, end_char=len(addr), text=addr)
    r = apply_label_merge(e, {'ADDRESS': '地名'})
    assert r.label == 'ADDRESS'


def test_merge_entities_passthrough_unknown_labels():
    ents = [
        Entity(label='PHONE', start_char=0, end_char=3, text='090'),
        Entity(label='NAME', start_char=4, end_char=6, text='花子'),
    ]
    out = merge_entities(ents, {'NAME': '人名'})
    assert out[0].label == 'PHONE'
    assert out[1].label == '人名'


def test_is_simple_place():
    assert is_simple_place('東京')
    assert not is_simple_place('東京都 新宿区')
    assert not is_simple_place('東京,渋谷')
    assert not is_simple_place('')
