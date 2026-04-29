"""WikiANN-vi relabel_8type 단위 테스트 (LLM 호출 없는 부분만)."""
from augmenters.wikiann_vi.relabel_8type import (
    Relabeler,
    match_offsets,
    parse_spans,
)


class TestParseSpans:
    def test_plain_list(self):
        raw = '[{"text": "Hà Nội", "type": "LOC"}]'
        assert parse_spans(raw) == [{'text': 'Hà Nội', 'type': 'LOC'}]

    def test_list_with_think_prefix(self):
        raw = (
            '<think>some reasoning here</think>'
            '[{"text": "Samsung", "type": "ORG"}]'
        )
        assert parse_spans(raw) == [
            {'text': 'Samsung', 'type': 'ORG'},
        ]

    def test_single_dict(self):
        raw = '{"text": "Hồ Chí Minh", "type": "PER"}'
        assert parse_spans(raw) == [
            {'text': 'Hồ Chí Minh', 'type': 'PER'},
        ]

    def test_wrapped_dict(self):
        raw = '{"entities": [{"text": "Hà Nội", "type": "LOC"}]}'
        assert parse_spans(raw) == [{'text': 'Hà Nội', 'type': 'LOC'}]

    def test_regex_fallback(self):
        raw = (
            'Here is the answer: [{"text": "Vịnh Hạ Long", '
            '"type": "LOC"}] end.'
        )
        result = parse_spans(raw)
        assert result == [{'text': 'Vịnh Hạ Long', 'type': 'LOC'}]

    def test_empty_or_invalid(self):
        assert parse_spans('') == []
        assert parse_spans('no json here') == []
        assert parse_spans('{"bad": null}') == []

    def test_empty_list(self):
        assert parse_spans('[]') == []


class TestMatchOffsets:
    def test_basic_single_span(self):
        text = 'Hà Nội là thủ đô của Việt Nam.'
        spans = [
            {'text': 'Hà Nội', 'type': 'LOC'},
            {'text': 'Việt Nam', 'type': 'LOC'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 2
        assert result[0] == {
            'text': 'Hà Nội', 'type': 'LOC', 'start': 0, 'end': 6,
        }
        assert result[1]['text'] == 'Việt Nam'
        assert text[result[1]['start']:result[1]['end']] == 'Việt Nam'

    def test_duplicate_surface_form(self):
        """동일 표면형이 두 번 나오면 각각 다른 오프셋 매칭."""
        text = 'Hà Nội và Hà Nội là cùng tên.'
        spans = [
            {'text': 'Hà Nội', 'type': 'LOC'},
            {'text': 'Hà Nội', 'type': 'LOC'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 2
        starts = sorted(r['start'] for r in result)
        assert starts == [0, text.find('Hà Nội', 1)]

    def test_substring_safety(self):
        """긴 엔티티를 먼저 매칭해 부분 문자열 충돌을 방지한다."""
        text = 'Đại học Quốc gia Hà Nội tọa lạc tại Hà Nội.'
        spans = [
            {'text': 'Hà Nội', 'type': 'LOC'},
            {
                'text': 'Đại học Quốc gia Hà Nội',
                'type': 'ORG',
            },
        ]
        result = match_offsets(text, spans)
        assert len(result) == 2
        by_text = {r['text']: r for r in result}
        assert by_text['Đại học Quốc gia Hà Nội']['start'] == 0
        assert by_text['Hà Nội']['start'] > len(
            'Đại học Quốc gia Hà Nội'
        )

    def test_not_found_dropped(self):
        text = 'Không có tên nào.'
        spans = [{'text': 'Samsung', 'type': 'ORG'}]
        assert match_offsets(text, spans) == []

    def test_empty_inputs(self):
        assert match_offsets('', []) == []
        assert match_offsets('abc', []) == []
        assert match_offsets('', [{'text': 'x', 'type': 'y'}]) == []

    def test_non_dict_spans_skipped(self):
        """LLM 이 dict 대신 str/int 등 잘못된 타입 섞어 반환할 때 안전 폴백."""
        text = 'Hà Nội là thủ đô.'
        spans = [
            'Hà Nội',                                  # 잘못된 타입 (str)
            {'text': 'Hà Nội', 'type': 'LOC'},         # 정상
            42,                                        # 잘못된 타입 (int)
            None,                                      # 잘못된 타입 (None)
        ]
        result = match_offsets(text, spans)
        assert len(result) == 1
        assert result[0]['type'] == 'LOC'

    def test_missing_fields_skipped(self):
        text = 'Hà Nội.'
        spans = [
            {'text': '', 'type': 'LOC'},
            {'text': 'Hà Nội', 'type': ''},
            {'text': 'Hà Nội', 'type': 'LOC'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 1
        assert result[0]['text'] == 'Hà Nội'

    def test_preserves_original_order(self):
        """LLM 출력 원래 순서를 유지한다."""
        text = 'Đảng Cộng sản Việt Nam và Bộ Giáo dục hợp tác.'
        spans = [
            {'text': 'Đảng Cộng sản Việt Nam', 'type': 'ORG'},
            {'text': 'Bộ Giáo dục', 'type': 'ORG'},
        ]
        result = match_offsets(text, spans)
        assert [r['text'] for r in result] == [
            'Đảng Cộng sản Việt Nam', 'Bộ Giáo dục',
        ]


class TestRelabelerBatchParse:
    def test_parse_batch_dict(self):
        raw = (
            '{"0": [{"text": "Hà Nội", "type": "LOC"}], '
            '"1": [{"text": "Samsung", "type": "ORG"}]}'
        )
        result = Relabeler._parse_batch(raw)
        assert 0 in result and 1 in result
        assert result[0] == [{'text': 'Hà Nội', 'type': 'LOC'}]
        assert result[1] == [{'text': 'Samsung', 'type': 'ORG'}]

    def test_parse_batch_with_think(self):
        raw = (
            '<think>processing</think>\n'
            '{"0": [{"text": "x", "type": "PER"}], "2": []}'
        )
        result = Relabeler._parse_batch(raw)
        assert result.get(0) == [{'text': 'x', 'type': 'PER'}]
        assert result.get(2) == []

    def test_parse_batch_regex_fallback(self):
        raw = 'The output is: {"0": [{"text": "a", "type": "b"}]} end.'
        result = Relabeler._parse_batch(raw)
        assert 0 in result
        assert result[0] == [{'text': 'a', 'type': 'b'}]

    def test_parse_batch_invalid(self):
        assert Relabeler._parse_batch('') == {}
        assert Relabeler._parse_batch('not json') == {}
        # 리스트 형태는 SINGLE 포맷이므로 BATCH 파싱에서는 빈 dict
        assert Relabeler._parse_batch('[1,2,3]') == {}

    def test_parse_batch_skips_non_int_keys(self):
        raw = '{"abc": [{"text": "x", "type": "y"}], "0": []}'
        result = Relabeler._parse_batch(raw)
        assert 'abc' not in result
        assert 0 in result
        assert result[0] == []

    def test_build_batch_format(self):
        r = Relabeler(batch_size=3)
        prompt = r._build_batch(['câu 1', 'câu 2'])
        assert '0: câu 1' in prompt
        assert '1: câu 2' in prompt
