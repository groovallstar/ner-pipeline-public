"""ja.prod_seed.seed_selector 단위 테스트.

도메인 분류, train+valid index 선별, LONG-pool 제외, jsonl 생성 통합까지
모두 cover. classifier 학습 경로는 별도 통합 테스트로 분리.
"""

import json

from ner.augmenters.ja.prod_seed.seed_selector import (
    categorize_prod_surface,
    oversample_to_jsonl,
    select_domain_seed_indices,
    select_long_seed_indices,
)


def _prod_row(text, prod_surface):
    """text 안에서 prod_surface 위치를 찾아 PROD entity row 를 만든다."""
    start = text.find(prod_surface)
    assert start >= 0, f'prod_surface={prod_surface!r} not in text={text!r}'
    return {
        'text': text,
        'entities': [
            {
                'label': 'PROD',
                'start_char': start,
                'end_char': start + len(prod_surface),
                'text': prod_surface,
            },
        ],
    }


def test_categorize_prod_surface_law():
    assert categorize_prod_surface('郵政法案') == 'law'
    assert categorize_prod_surface('連邦制定法') == 'law'
    assert categorize_prod_surface('共産党宣言') == 'law'


def test_categorize_prod_surface_book():
    assert categorize_prod_surface('米子商工案内') == 'book'
    assert categorize_prod_surface('国史大辞典') == 'book'
    assert categorize_prod_surface('経済学批判') == 'book'


def test_categorize_prod_surface_food_transit_music():
    assert categorize_prod_surface('食べ比べセット') == 'food'
    assert categorize_prod_surface('東急百貨店エコポイントカード') == 'transit_card'
    assert categorize_prod_surface('みんなのうた') == 'music_work'


def test_categorize_prod_surface_misc_returns_none():
    assert categorize_prod_surface('機動戦士ガンダム') is None
    assert categorize_prod_surface('iPhone') is None
    assert categorize_prod_surface('Android') is None


def test_select_domain_seed_indices_excludes_test_split():
    # 12 rows — seed=42, valid=0.166, test=0.166 → test 2 / valid 2 / train 8
    rows = [_prod_row(f'文{i} 郵政法案', '郵政法案') for i in range(12)]
    selected = select_domain_seed_indices(
        rows, valid_ratio=1 / 6, test_ratio=1 / 6, seed=42,
    )
    # train+valid = 10 row 모두 도메인 매칭 → 10건
    assert len(selected) == 10
    # test split 2건은 빠져야 함
    assert len(rows) - len(selected) == 2


def test_select_domain_seed_indices_ignores_non_prod_labels():
    # PROD 가 아닌 ORG entity 만 있는 row 는 선별 안 됨
    rows = [
        {'text': '法案を提出', 'entities': [
            {'label': 'ORG', 'start_char': 0, 'end_char': 2, 'text': '法案'},
        ]},
        _prod_row('郵政法案を', '郵政法案'),
    ]
    selected = select_domain_seed_indices(
        rows, valid_ratio=0.0, test_ratio=0.0, seed=42,
    )
    # 두 row 모두 train+valid 이지만 첫째는 PROD 없음 → 둘째만
    assert selected == {1}


def test_select_long_seed_indices_excludes_famous_media():
    rows = [
        _prod_row('文A 機動戦士ガンダム を見る', '機動戦士ガンダム'),
        _prod_row('文B 詩人野口雨情ここにて眠る を', '詩人野口雨情ここにて眠る'),
        _prod_row('文C 短い', '短い'),  # length < min
    ]
    selected = select_long_seed_indices(
        rows,
        min_length=6,
        valid_ratio=0.0,
        test_ratio=0.0,
        seed=42,
    )
    # ガンダム = famous-media keyword → 제외
    # 詩人...眠る = len 12, 도메인 미매칭, famous 키워드 없음 → 채택
    # 短い = len 2 < min_length → 제외
    assert selected == {1}


def test_select_long_seed_indices_skips_domain_matches():
    # 도메인 매칭은 domain pool 이 담당 — long pool 에서는 중복 방지로 제외
    rows = [_prod_row('文 郵政法案', '郵政法案')]
    long_only = select_long_seed_indices(
        rows, min_length=4, valid_ratio=0.0, test_ratio=0.0, seed=42,
    )
    assert long_only == set()  # 도메인 (law) 매칭 → 제외


def test_oversample_to_jsonl_extra_only_domain(tmp_path):
    rows = (
        [_prod_row(f'文{i} 郵政法案', '郵政法案') for i in range(5)]
        + [_prod_row('文5 機動戦士ガンダム', '機動戦士ガンダム')]  # 도메인 미매칭
    )
    input_path = tmp_path / 'input.jsonl'
    with open(input_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    output_path = tmp_path / 'out.jsonl'
    stats = oversample_to_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        oversample=3,
        use_domain=True,
        use_long=False,
        extra_only=True,
        valid_ratio=0.0,
        test_ratio=0.0,
    )
    # 도메인 매칭 5건 × N=3 = 15 extra
    assert stats['domain_seed_sentences'] == 5
    assert stats['long_seed_sentences'] == 0
    assert stats['candidates'] == 5
    assert stats['extra_rows'] == 15
    assert stats['base_extra_rows'] == 0
    assert stats['output_rows'] == 15

    with open(output_path, encoding='utf-8') as fp:
        out_rows = [json.loads(line) for line in fp]
    assert len(out_rows) == 15
    # 모든 출력 row 는 法案 surface 를 가진 도메인 매칭 row 여야 함
    for r in out_rows:
        assert any(e['text'].endswith('法案') for e in r['entities'])


def test_oversample_to_jsonl_with_base_extra_prefix(tmp_path):
    rows = [_prod_row('文 郵政法案', '郵政法案')]
    input_path = tmp_path / 'input.jsonl'
    with open(input_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # base extra = S7N2 neg 같은 prefix 시뮬레이션 (4 row)
    base_extra = tmp_path / 'base_extra.jsonl'
    with open(base_extra, 'w', encoding='utf-8') as f:
        for i in range(4):
            f.write(json.dumps(
                {'text': f'neg{i}', 'entities': []},
                ensure_ascii=False,
            ) + '\n')

    output_path = tmp_path / 'out.jsonl'
    stats = oversample_to_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        oversample=2,
        use_domain=True,
        base_extra_path=str(base_extra),
        extra_only=True,
        valid_ratio=0.0,
        test_ratio=0.0,
    )
    # base 4 + (1 도메인 × N=2) = 6
    assert stats['base_extra_rows'] == 4
    assert stats['extra_rows'] == 2
    assert stats['output_rows'] == 6
