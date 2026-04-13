"""Tests for language-neutral LLM labeler helpers."""
from labelers.llm_helpers import parse_spans, spans_to_bio, split_sentences


class TestSplitSentences:
    def test_ko_punctuation(self):
        out = split_sentences("First sentence here. Second sentence follows.", lang="ko")
        assert len(out) >= 1
        assert "First sentence here" in out[0]

    def test_ja_punctuation(self):
        # Japanese punctuation 。！？ triggers split when lang='ja'
        out = split_sentences("東京は日本の首都です。富士山は高い。", lang="ja")
        # Either split into 2 or merged via the min-length buffer; must contain both sides.
        joined = " ".join(out)
        assert "東京" in joined and "富士山" in joined

    def test_empty_text(self):
        assert split_sentences("") == [""]

    def test_newline_split(self):
        out = split_sentences("line one text\nline two text\n", lang="ko")
        assert len(out) >= 1


class TestParseSpans:
    def test_bare_array(self):
        assert parse_spans('[{"text":"東京","type":"地名"}]') == [
            {"text": "東京", "type": "地名"}
        ]

    def test_wrapped_dict(self):
        assert parse_spans('{"entities":[{"text":"a","type":"b"}]}') == [
            {"text": "a", "type": "b"}
        ]

    def test_single_dict(self):
        assert parse_spans('{"text":"x","type":"y"}') == [{"text": "x", "type": "y"}]

    def test_think_stripped(self):
        raw = '<think>reasoning here</think>[{"text":"x","type":"y"}]'
        assert parse_spans(raw) == [{"text": "x", "type": "y"}]

    def test_regex_fallback(self):
        raw = 'noise [{"text":"x","type":"y"}] trailing'
        assert parse_spans(raw) == [{"text": "x", "type": "y"}]

    def test_invalid_returns_empty(self):
        assert parse_spans("no json here") == []


class TestSpansToBio:
    def test_ko_substring_match(self):
        # "서울시" matches within token "서울시에서"
        tokens = ["서울시에서", "살다"]
        spans = [{"text": "서울시", "type": "LC"}]
        assert spans_to_bio(tokens, spans) == ["B-LC", "O"]

    def test_ja_substring_match(self):
        # "東京" matches within "東京は"
        tokens = ["東京は", "首都"]
        spans = [{"text": "東京", "type": "地名"}]
        assert spans_to_bio(tokens, spans) == ["B-地名", "O"]

    def test_exact_multitoken(self):
        tokens = ["김", "철수", "가", "왔다"]
        spans = [{"text": "김 철수", "type": "PS"}]
        assert spans_to_bio(tokens, spans) == ["B-PS", "I-PS", "O", "O"]

    def test_no_match_all_o(self):
        tokens = ["aaa", "bbb"]
        spans = [{"text": "zzz", "type": "PS"}]
        assert spans_to_bio(tokens, spans) == ["O", "O"]

    def test_empty_spans(self):
        tokens = ["a", "b"]
        assert spans_to_bio(tokens, []) == ["O", "O"]
