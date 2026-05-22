"""build_extra 순수 함수 단위 테스트 (분배·검증·leak-free)."""

from ner.augmenters.ja.domain_mine import build_extra as be


def _rec(text, status, spans):
    return {'text': text, 'anchor_status': status, 'merged_spans': spans}


def _prod(text, s, e):
    return {'text': text[s:e], 'type': 'PROD', 'start': s, 'end': e,
            'confidence': 'high', 'source': 'both'}


def test_confirmed_goes_to_test_anchor_only_to_train():
    recs = [
        _rec('『A』は本だ', 'confirmed', [_prod('『A』は本だ', 1, 2)]),
        _rec('Bという曲がある', 'anchor_only', [_prod('Bという曲がある', 0, 1)]),
    ]
    test, train = be.build_extra(recs)
    assert len(test) == 1 and len(train) == 1
    assert test[0]['id'].endswith('test-0')
    assert train[0]['id'].endswith('train-0')


def test_conflict_dropped():
    recs = [_rec('株式会社Xは企業', 'conflict',
                 [{'text': '株式会社X', 'type': 'ORG', 'start': 0,
                   'end': 5, 'confidence': 'high', 'source': 'both'}])]
    test, train = be.build_extra(recs)
    assert test == [] and train == []


def test_entities_use_contract_keys():
    recs = [_rec('『A』は本だ', 'confirmed', [_prod('『A』は本だ', 1, 2)])]
    test, _ = be.build_extra(recs)
    ent = test[0]['entities'][0]
    assert set(ent) == {'label', 'start_char', 'end_char', 'text'}
    assert ent['label'] == 'PROD'


def test_leak_free_same_text_prefers_test():
    # 같은 문장이 confirmed(test) + anchor_only(train) 양쪽에 → test 만 유지
    text = '同じ文がある場合'
    recs = [
        _rec(text, 'confirmed', [_prod(text, 0, 2)]),
        _rec(text, 'anchor_only', [_prod(text, 0, 2)]),
    ]
    test, train = be.build_extra(recs)
    assert len(test) == 1 and len(train) == 0


def test_invalid_offset_record_skipped():
    # text mismatch (offset 손상) 레코드는 검증서 탈락
    bad = _rec('東京で会う', 'confirmed',
               [{'text': '東京', 'type': 'PROD', 'start': 1, 'end': 3,
                 'confidence': 'high', 'source': 'both'}])
    test, train = be.build_extra([bad])
    assert test == [] and train == []


def test_prod_count_helper():
    recs = [_rec('『A』は本だ', 'confirmed', [_prod('『A』は本だ', 1, 2)])]
    test, _ = be.build_extra(recs)
    assert be._prod_count(test) == 1


def test_confirmed_split_feeds_train():
    # confirmed_test_frac=0.5 → confirmed 절반은 train 으로 주입
    recs = [_rec(f'確認文{i}である長め', 'confirmed',
                 [_prod(f'確認文{i}である長め', 0, 3)]) for i in range(4)]
    recs.append(_rec('anchorのみ文だよ', 'anchor_only',
                     [_prod('anchorのみ文だよ', 0, 6)]))
    test, train = be.build_extra(recs, confirmed_test_frac=0.5)
    # 4 confirmed → 2 test + 2 train, anchor_only 1 → train. train=3
    assert len(test) == 2
    assert len(train) == 3
    # leak-free: test 와 train 문장 비중복
    tset = {r['text'] for r in test}
    trset = {r['text'] for r in train}
    assert tset.isdisjoint(trset)


def test_confirmed_split_deterministic():
    recs = [_rec(f'確認文{i}である長め', 'confirmed',
                 [_prod(f'確認文{i}である長め', 0, 3)]) for i in range(6)]
    a = be.build_extra(recs, confirmed_test_frac=0.5, seed=7)
    b = be.build_extra(recs, confirmed_test_frac=0.5, seed=7)
    assert [r['text'] for r in a[0]] == [r['text'] for r in b[0]]


def test_train_internal_dedup():
    text = '繰り返す文章だよ'
    recs = [
        _rec(text, 'anchor_only', [_prod(text, 0, 2)]),
        _rec(text, 'anchor_only', [_prod(text, 0, 2)]),
    ]
    _, train = be.build_extra(recs)
    assert len(train) == 1
