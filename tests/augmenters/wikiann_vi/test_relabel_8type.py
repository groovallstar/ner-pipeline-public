"""WikiANN-vi relabel_8type 단위 테스트 (LLM 호출 없는 부분만)."""
from augmenters.wikiann_vi.relabel_8type import (
    match_offsets,
    parse_spans,
)


class TestParseSpans:
    def test_plain_list(self):
        raw = '[{"text": "Hà Nội", "type": "地名"}]'
        assert parse_spans(raw) == [{'text': 'Hà Nội', 'type': '地名'}]

    def test_list_with_think_prefix(self):
        raw = (
            '<think>some reasoning here</think>'
            '[{"text": "Samsung", "type": "法人名"}]'
        )
        assert parse_spans(raw) == [
            {'text': 'Samsung', 'type': '法人名'},
        ]

    def test_single_dict(self):
        raw = '{"text": "Hồ Chí Minh", "type": "人名"}'
        assert parse_spans(raw) == [
            {'text': 'Hồ Chí Minh', 'type': '人名'},
        ]

    def test_wrapped_dict(self):
        raw = '{"entities": [{"text": "Hà Nội", "type": "地名"}]}'
        assert parse_spans(raw) == [{'text': 'Hà Nội', 'type': '地名'}]

    def test_regex_fallback(self):
        raw = (
            'Here is the answer: [{"text": "Vịnh Hạ Long", '
            '"type": "地名"}] end.'
        )
        result = parse_spans(raw)
        assert result == [{'text': 'Vịnh Hạ Long', 'type': '地名'}]

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
            {'text': 'Hà Nội', 'type': '地名'},
            {'text': 'Việt Nam', 'type': '地名'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 2
        assert result[0] == {
            'text': 'Hà Nội', 'type': '地名', 'start': 0, 'end': 6,
        }
        assert result[1]['text'] == 'Việt Nam'
        assert text[result[1]['start']:result[1]['end']] == 'Việt Nam'

    def test_duplicate_surface_form(self):
        """동일 표면형이 두 번 나오면 각각 다른 오프셋 매칭."""
        text = 'Hà Nội và Hà Nội là cùng tên.'
        spans = [
            {'text': 'Hà Nội', 'type': '地名'},
            {'text': 'Hà Nội', 'type': '地名'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 2
        starts = sorted(r['start'] for r in result)
        assert starts == [0, text.find('Hà Nội', 1)]

    def test_substring_safety(self):
        """긴 엔티티를 먼저 매칭해 부분 문자열 충돌을 방지한다."""
        text = 'Đại học Quốc gia Hà Nội tọa lạc tại Hà Nội.'
        spans = [
            {'text': 'Hà Nội', 'type': '地名'},
            {
                'text': 'Đại học Quốc gia Hà Nội',
                'type': 'その他の組織名',
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
        spans = [{'text': 'Samsung', 'type': '法人名'}]
        assert match_offsets(text, spans) == []

    def test_empty_inputs(self):
        assert match_offsets('', []) == []
        assert match_offsets('abc', []) == []
        assert match_offsets('', [{'text': 'x', 'type': 'y'}]) == []

    def test_missing_fields_skipped(self):
        text = 'Hà Nội.'
        spans = [
            {'text': '', 'type': '地名'},
            {'text': 'Hà Nội', 'type': ''},
            {'text': 'Hà Nội', 'type': '地名'},
        ]
        result = match_offsets(text, spans)
        assert len(result) == 1
        assert result[0]['text'] == 'Hà Nội'

    def test_preserves_original_order(self):
        """LLM 출력 원래 순서를 유지한다."""
        text = 'Đảng Cộng sản Việt Nam và Bộ Giáo dục hợp tác.'
        spans = [
            {'text': 'Đảng Cộng sản Việt Nam', 'type': '政治的組織名'},
            {'text': 'Bộ Giáo dục', 'type': '政治的組織名'},
        ]
        result = match_offsets(text, spans)
        assert [r['text'] for r in result] == [
            'Đảng Cộng sản Việt Nam', 'Bộ Giáo dục',
        ]
