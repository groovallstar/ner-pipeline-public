"""label_merger 단위 테스트."""
from ner.augmenters.pii.injector import apply_label_merge, merge_entities
from ner.augmenters.pii.schema import Entity


def test_name_to_per():
    e = Entity(label='NAME', start_char=0, end_char=3, text='太郎')
    r = apply_label_merge(e, {'NAME': 'PER'})
    assert r.label == 'PER'
    assert r.start_char == 0 and r.end_char == 3


def test_address_simple_mapped_to_loc():
    # 단일 토큰 주소도 LOC로 병합된다 (#17 Phase 1에서 무조건 병합).
    e = Entity(label='ADDRESS', start_char=0, end_char=3, text='東京')
    r = apply_label_merge(e, {'ADDRESS': 'LOC'})
    assert r.label == 'LOC'


def test_address_complex_also_mapped_to_loc():
    # 복합 주소도 무조건 LOC. 이전에는 단일 토큰만 병합했으나 #17 Phase 1
    # 이후 canonical에서 ADDRESS 라벨 자체를 제거하므로 일괄 흡수한다.
    addr = '東京都 新宿区 1-2-3'
    e = Entity(label='ADDRESS', start_char=0, end_char=len(addr), text=addr)
    r = apply_label_merge(e, {'ADDRESS': 'LOC'})
    assert r.label == 'LOC'


def test_merge_entities_passthrough_unknown_labels():
    ents = [
        Entity(label='PHONE', start_char=0, end_char=3, text='090'),
        Entity(label='NAME', start_char=4, end_char=6, text='花子'),
    ]
    out = merge_entities(ents, {'NAME': 'PER'})
    assert out[0].label == 'PHONE'
    assert out[1].label == 'PER'


def test_dat_label_passthrough():
    # DAT는 canonical 라벨이라 병합되지 않고 그대로 유지된다.
    e = Entity(label='DAT', start_char=0, end_char=10, text='1985年4月3日')
    r = apply_label_merge(e, {'NAME': 'PER', 'ADDRESS': 'LOC'})
    assert r.label == 'DAT'
