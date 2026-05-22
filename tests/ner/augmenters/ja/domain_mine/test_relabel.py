"""relabel 순수 함수 단위 테스트 (LLM 호출 없음)."""

from ner.augmenters.ja.domain_mine import relabel


def test_assign_offsets_basic():
    text = '東京で会う'
    spans = [{'text': '東京', 'type': 'LOC'}]
    out = relabel.assign_offsets(text, spans)
    assert out == [{'text': '東京', 'type': 'LOC', 'start': 0, 'end': 2}]


def test_assign_offsets_skips_absent_surface():
    text = '東京で会う'
    spans = [{'text': '大阪', 'type': 'LOC'}]
    assert relabel.assign_offsets(text, spans) == []


def test_assign_offsets_multi_occurrence_non_overlapping():
    text = 'SuicaとSuica'
    spans = [{'text': 'Suica', 'type': 'PROD'},
             {'text': 'Suica', 'type': 'PROD'}]
    out = relabel.assign_offsets(text, spans)
    assert [(s['start'], s['end']) for s in out] == [(0, 5), (6, 11)]


def test_assign_offsets_drops_empty_fields():
    text = 'abc'
    spans = [{'text': '', 'type': 'LOC'}, {'text': 'a', 'type': ''}]
    assert relabel.assign_offsets(text, spans) == []


def test_apply_anchor_confirmed_when_consensus_prod():
    # 합의가 anchor 와 겹치는 PROD → confirmed, 합의 그대로 유지
    text = 'Suicaは便利だ'
    spans = [{'text': 'Suica', 'type': 'PROD', 'start': 0, 'end': 5,
              'confidence': 'high', 'source': 'both'}]
    out, status = relabel.apply_anchor(spans, [(0, 5)], text)
    assert status == 'confirmed'
    assert out[0]['type'] == 'PROD' and out[0]['confidence'] == 'high'


def test_apply_anchor_conflict_when_consensus_non_prod():
    # 株式会社シベール=ORG 케이스: 합의가 non-PROD → conflict (anchor 거짓)
    text = '株式会社シベールは企業だ'
    spans = [{'text': '株式会社シベール', 'type': 'ORG', 'start': 0,
              'end': 8, 'confidence': 'high', 'source': 'both'}]
    out, status = relabel.apply_anchor(spans, [(5, 8)], text)
    assert status == 'conflict'
    # 합의를 덮지 않는다 (ORG 유지, PROD 위조 안 함)
    assert all(s['type'] != 'PROD' for s in out)


def test_apply_anchor_anchor_only_when_consensus_silent():
    # 두 모델 다 놓침 → anchor_only (소스만 보증, FN 회복 후보)
    text = '『ブウタン』は漫画だ'
    out, status = relabel.apply_anchor([], [(1, 5)], text)
    assert status == 'anchor_only'
    assert len(out) == 1
    assert out[0]['type'] == 'PROD'
    assert out[0]['confidence'] == 'anchor_only'


def test_apply_anchor_keeps_wider_consensus_prod_boundary():
    # 합의 PROD 가 anchor 보다 넓으면 그대로 유지 (경계 보정)
    text = 'シベールの日曜日は映画だ'
    spans = [{'text': 'シベールの日曜日', 'type': 'PROD', 'start': 0,
              'end': 8, 'confidence': 'high', 'source': 'both'}]
    out, status = relabel.apply_anchor(spans, [(0, 4)], text)
    assert status == 'confirmed'
    assert len(out) == 1 and out[0]['end'] == 8


def test_apply_anchor_sorts_by_offset():
    text = '日本でSuicaを使う長い文章'
    spans = [{'text': 'Suica', 'type': 'PROD', 'start': 3, 'end': 8,
              'confidence': 'high', 'source': 'both'}]
    out, status = relabel.apply_anchor(spans, [(0, 2)], text)
    assert [s['start'] for s in out] == [0, 3]
