"""Unit tests for Japanese NER span matcher."""
from ner.labelers.ja.span_matcher import match_spans


class TestMatchSpans:
    def test_perfect_match(self):
        text = "東京は首都"
        spans = [{"text": "東京", "type": "地名"}]
        result = match_spans(text, spans)
        assert len(result) == 1
        assert result[0] == {"text": "東京", "type": "地名", "start": 0, "end": 2}

    def test_duplicate_entity_text(self):
        text = "東京から東京へ"
        spans = [
            {"text": "東京", "type": "地名"},
            {"text": "東京", "type": "地名"},
        ]
        result = match_spans(text, spans)
        assert len(result) == 2
        starts = sorted(r["start"] for r in result)
        assert starts == [0, 4]

    def test_substring_entities(self):
        """東京都 should NOT consume 東京's position."""
        text = "東京都は東京の中心"
        spans = [
            {"text": "東京都", "type": "地名"},
            {"text": "東京", "type": "地名"},
        ]
        result = match_spans(text, spans)
        assert len(result) == 2
        by_text = {r["text"]: r for r in result}
        assert by_text["東京都"]["start"] == 0
        assert by_text["東京都"]["end"] == 3
        assert by_text["東京"]["start"] == 4
        assert by_text["東京"]["end"] == 6

    def test_entity_not_found(self):
        text = "大阪は都市"
        spans = [{"text": "京都", "type": "地名"}]
        result = match_spans(text, spans)
        assert len(result) == 0

    def test_empty_input(self):
        assert match_spans("", []) == []
        assert match_spans("テスト", []) == []
        assert match_spans("", [{"text": "a", "type": "b"}]) == []

    def test_multiple_entity_types(self):
        text = "織田信長は安土城を建てた"
        spans = [
            {"text": "織田信長", "type": "人名"},
            {"text": "安土城", "type": "施設名"},
        ]
        result = match_spans(text, spans)
        assert len(result) == 2
        assert result[0]["type"] == "人名"
        assert result[0]["start"] == 0
        assert result[1]["type"] == "施設名"
        assert result[1]["start"] == 5
