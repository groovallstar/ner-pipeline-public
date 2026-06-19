"""apply_silver_gap 단위 테스트 — gemma-검증 silver-갭 additive 삽입."""

from ner.augmenters.wikiann_vi.audit_vi_prod_evt import apply_silver_gap


def _case(sid, text, start, end, silver, model='EVT', correct='EVT',
          direction='FP', ambiguous=False):
    return {
        'sentence_id': sid, 'text': text, 'start': start, 'end': end,
        'surface': text[start:end], 'silver_label': silver,
        'model_label': model, 'direction': direction,
        'adj': {'correct': correct, 'ambiguous': ambiguous, 'rule': 'r'},
    }


def _gold(sid, text, entities=None):
    return {'id': sid, 'text': text, 'orig': sid, 'entities': entities or []}


class TestApplySilverGap:
    def test_additive_insert_preserves_orig(self):
        text = 'Chiến thắng Kỷ dậu là sự kiện'
        gold = [_gold('s1', text, [])]
        adj = [_case('s1', text, 0, 11, silver='O')]
        rows, added, skipped = apply_silver_gap(adj, gold, 'EVT')
        assert len(added) == 1
        ents = rows[0]['entities']
        assert len(ents) == 1
        assert ents[0]['label'] == 'EVT'
        assert ents[0]['start_char'] == 0 and ents[0]['end_char'] == 11
        assert rows[0]['orig'] == 's1'  # 원본 필드 보존
        assert skipped == {'surface': 0, 'overlap': 0}

    def test_additive_only_excludes_cross_type(self):
        text = 'Đại hội đại biểu toàn quốc họp'
        gold = [_gold('s1', text, [{'label': 'ORG', 'start_char': 0,
                                    'end_char': 7, 'text': 'Đại hội'}])]
        # silver=ORG (cross-type) — additive_only 면 제외
        adj = [_case('s1', text, 15, 22, silver='ORG')]
        rows, added, _ = apply_silver_gap(adj, gold, 'EVT')
        assert added == []
        assert len(rows[0]['entities']) == 1  # ORG 그대로

    def test_include_cross_type_recovers_org(self):
        text = 'Đại hội đại biểu toàn quốc họp'
        gold = [_gold('s1', text, [])]
        adj = [_case('s1', text, 15, 22, silver='ORG')]
        rows, added, _ = apply_silver_gap(adj, gold, 'EVT',
                                          additive_only=False)
        assert len(added) == 1
        assert rows[0]['entities'][0]['label'] == 'EVT'

    def test_dedup_same_span(self):
        text = 'World Cup 2022 diễn ra'
        gold = [_gold('s1', text, [])]
        adj = [_case('s1', text, 0, 14, silver='O'),
               _case('s1', text, 0, 14, silver='O')]
        rows, added, _ = apply_silver_gap(adj, gold, 'EVT')
        assert len(added) == 1
        assert len(rows[0]['entities']) == 1

    def test_overlap_skipped(self):
        text = 'World Cup 2022 diễn ra'
        gold = [_gold('s1', text, [{'label': 'ORG', 'start_char': 0,
                                    'end_char': 9, 'text': 'World Cup'}])]
        adj = [_case('s1', text, 0, 14, silver='O')]
        rows, added, skipped = apply_silver_gap(adj, gold, 'EVT')
        assert added == []
        assert skipped['overlap'] == 1
        assert len(rows[0]['entities']) == 1  # 기존 ORG 불변

    def test_surface_mismatch_skipped(self):
        text = 'World Cup 2022 diễn ra'
        case = _case('s1', text, 0, 14, silver='O')
        case['surface'] = 'WRONG SURFACE'  # text[0:14] 와 불일치
        rows, added, skipped = apply_silver_gap([case], [_gold('s1', text)],
                                                'EVT')
        assert added == []
        assert skipped['surface'] == 1

    def test_excludes_non_fp_and_ambiguous(self):
        text = 'World Cup 2022 diễn ra'
        gold = [_gold('s1', text, [])]
        adj = [
            _case('s1', text, 0, 14, silver='O', direction='FN'),
            _case('s1', text, 0, 14, silver='O', ambiguous=True),
            _case('s1', text, 0, 14, silver='O', correct='ORG'),
            _case('s1', text, 0, 14, silver='O', model='PROD'),
        ]
        rows, added, _ = apply_silver_gap(adj, gold, 'EVT')
        assert added == []

    def test_row_without_case_unchanged(self):
        text = 'Không có sự kiện ở đây'
        gold = [_gold('s9', text, [{'label': 'PER', 'start_char': 0,
                                    'end_char': 6, 'text': 'Không'}])]
        rows, added, _ = apply_silver_gap([], gold, 'EVT')
        assert added == []
        assert rows[0] == gold[0]
