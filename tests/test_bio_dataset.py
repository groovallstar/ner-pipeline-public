"""Tests for labelers.bio_dataset module.

Unit tests use fixtures only (no network/disk dependency).
Smoke tests require HF cache and are skipped when absent.
"""

import dataclasses
import json
import os

import pytest

from labelers.bio_dataset import (
    DatasetNotFoundError,
    DatasetSpec,
    REGISTRY,
    TokenUnit,
    extract_spans,
    load,
    normalize_bio_variant,
)


# ---------------------------------------------------------------------------
# TokenUnit enum
# ---------------------------------------------------------------------------

class TestTokenUnit:
    def test_members(self):
        members = {m.name for m in TokenUnit}
        assert members == {"SYLLABLE", "WORD", "MORPHEME"}

    def test_values_are_lowercase(self):
        assert TokenUnit.SYLLABLE.value == "syllable"
        assert TokenUnit.WORD.value == "word"
        assert TokenUnit.MORPHEME.value == "morpheme"

    def test_str_contains_value(self):
        # str(Enum) varies by Python version; .value is always the raw string
        assert TokenUnit.SYLLABLE.value == "syllable"
        assert TokenUnit.SYLLABLE == "syllable"  # str mixin comparison


# ---------------------------------------------------------------------------
# DatasetSpec dataclass
# ---------------------------------------------------------------------------

class TestDatasetSpec:
    @pytest.fixture()
    def spec(self):
        return DatasetSpec(
            name="test", lang="ko", unit=TokenUnit.SYLLABLE,
            bio_variant="iob2", joiner="",
        )

    def test_frozen(self, spec):
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"

    def test_hashable(self, spec):
        d = {spec: 1}
        assert d[spec] == 1

    def test_config_defaults_to_none(self, spec):
        assert spec.config is None


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_klue_entry(self):
        s = REGISTRY["klue"]
        assert s.lang == "ko"
        assert s.unit == TokenUnit.SYLLABLE
        assert s.bio_variant == "iob2"
        assert s.joiner == ""
        assert s.config == "ner"

    def test_kmou_entry(self):
        s = REGISTRY["nlp-kmu/kor_ner"]
        assert s.lang == "ko"
        assert s.unit == TokenUnit.MORPHEME
        assert s.bio_variant == "kmou_i_alone"
        assert s.joiner == " "

    def test_all_korean(self):
        for name, spec in REGISTRY.items():
            assert spec.lang == "ko", f"{name} has lang={spec.lang}"


# ---------------------------------------------------------------------------
# normalize_bio_variant
# ---------------------------------------------------------------------------

class TestNormalizeBioVariant:
    def test_iob2_identity(self):
        tags = ["B-PS", "I-PS", "O"]
        result = normalize_bio_variant(tags, "iob2")
        assert result == ["B-PS", "I-PS", "O"]
        assert result is not tags  # should be a copy

    def test_kmou_single_entity(self):
        assert normalize_bio_variant(
            ["B_PS", "I", "O"], "kmou_i_alone"
        ) == ["B-PS", "I-PS", "O"]

    def test_kmou_multiple_entities(self):
        assert normalize_bio_variant(
            ["B_OG", "I", "O", "B_LC", "I", "O"], "kmou_i_alone"
        ) == ["B-OG", "I-OG", "O", "B-LC", "I-LC", "O"]

    def test_kmou_dangling_i(self):
        assert normalize_bio_variant(
            ["O", "I", "B_LC"], "kmou_i_alone"
        ) == ["O", "O", "B-LC"]

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="nonexistent"):
            normalize_bio_variant(["O"], "nonexistent")


# ---------------------------------------------------------------------------
# extract_spans
# ---------------------------------------------------------------------------

class TestExtractSpans:
    def test_syllable_with_space_token(self):
        tokens = ["경", "찰", "은", " ", "박", "씨", "는"]
        tags = ["B-OG", "I-OG", "O", "O", "B-PS", "O", "O"]
        result = extract_spans(tokens, tags, joiner="")
        assert result == [
            {"text": "경찰", "type": "OG"},
            {"text": "박", "type": "PS"},
        ]

    def test_word_level(self):
        tokens = ["Apple", "Inc.", "acquired", "Beats", "Electronics", "."]
        tags = ["B-OG", "I-OG", "O", "B-OG", "I-OG", "O"]
        result = extract_spans(tokens, tags, joiner=" ")
        assert result == [
            {"text": "Apple Inc.", "type": "OG"},
            {"text": "Beats Electronics", "type": "OG"},
        ]

    def test_type_switch_without_o(self):
        tokens = ["김", "철", "수", "서", "울"]
        tags = ["B-PS", "I-PS", "I-PS", "B-LC", "I-LC"]
        result = extract_spans(tokens, tags, joiner="")
        assert result == [
            {"text": "김철수", "type": "PS"},
            {"text": "서울", "type": "LC"},
        ]

    def test_whitespace_tokens_excluded(self):
        tokens = ["서", "울", " ", "타", "워"]
        tags = ["B-LC", "I-LC", "I-LC", "I-LC", "I-LC"]
        result = extract_spans(tokens, tags, joiner="")
        assert len(result) == 1
        assert result[0]["text"] == "서울타워"

    def test_trailing_entity(self):
        tokens = ["서", "울"]
        tags = ["B-LC", "I-LC"]
        result = extract_spans(tokens, tags, joiner="")
        assert result == [{"text": "서울", "type": "LC"}]

    def test_empty_input(self):
        assert extract_spans([], [], joiner="") == []


# ---------------------------------------------------------------------------
# load — error path (unit test, no cache needed)
# ---------------------------------------------------------------------------

class TestLoadFromJsonl:
    """Unit tests for JSONL fallback path (no HF cache needed)."""

    def test_sentence_field_from_jsonl(self, tmp_path):
        """JSONL fallback produces sentence field using spec joiner."""
        jsonl_file = tmp_path / "validation.jsonl"
        record = {
            "id": "0",
            "tokens": ["경", "찰", "은", " ", "출", "동"],
            "ner_tags": ["B-OG", "I-OG", "O", "O", "O", "O"],
        }
        jsonl_file.write_text(json.dumps(record, ensure_ascii=False) + "\n")

        # Monkey-patch the JSONL path to point to tmp_path
        import labelers.bio_dataset as bio_mod
        _original_load = bio_mod.load  # noqa: F841 (kept for potential teardown)

        def patched_load(spec_name, split, max_samples=None):
            from pathlib import Path
            spec = bio_mod.REGISTRY[spec_name]
            return bio_mod._load_from_jsonl(Path(jsonl_file), spec, max_samples)

        results = patched_load("klue", "validation")
        assert len(results) == 1
        r = results[0]
        assert "sentence" in r
        assert r["sentence"] == "".join(r["tokens"])
        assert r["spans"] == [{"text": "경찰", "type": "OG"}]

    def test_jsonl_max_samples(self, tmp_path):
        """JSONL fallback respects max_samples."""
        jsonl_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(20):
            lines.append(json.dumps({
                "tokens": ["서", "울"], "ner_tags": ["B-LC", "I-LC"],
            }, ensure_ascii=False))
        jsonl_file.write_text("\n".join(lines) + "\n")

        import labelers.bio_dataset as bio_mod
        from pathlib import Path
        spec = bio_mod.REGISTRY["klue"]
        results = bio_mod._load_from_jsonl(Path(jsonl_file), spec, max_samples=5)
        assert len(results) == 5


class TestLoadErrors:
    def test_unknown_spec_raises(self):
        with pytest.raises(DatasetNotFoundError):
            load("not-in-registry", "train")


# ---------------------------------------------------------------------------
# Smoke tests — real data (skipped when cache absent)
# ---------------------------------------------------------------------------

KLUE_CACHE = "/work/.huggingface/datasets/klue___klue"
KMOU_CACHE = "/work/.huggingface/datasets/nlp-kmu___kor_ner"

KLUE_ENTITY_TYPES = {"PS", "LC", "OG", "DT", "TI", "QT"}
KMOU_ENTITY_TYPES = {"PS", "LC", "OG", "DT", "TI"}


@pytest.mark.integration
class TestSmokeKLUE:
    @pytest.fixture(autouse=True)
    def _require_cache(self):
        if not os.path.isdir(KLUE_CACHE):
            pytest.skip("KLUE HF cache not present")

    def test_load_returns_valid_records(self):
        records = load("klue", "validation", max_samples=5)
        assert len(records) <= 5
        assert len(records) > 0
        for r in records:
            assert set(r.keys()) == {"id", "tokens", "bio_tags", "spans", "sentence"}
            for tag in r["bio_tags"]:
                assert tag == "O" or tag.startswith("B-") or tag.startswith("I-")
            assert r["sentence"] == "".join(r["tokens"])

    def test_at_least_one_span(self):
        records = load("klue", "validation", max_samples=5)
        has_spans = any(len(r["spans"]) > 0 for r in records)
        assert has_spans, "Expected at least one record with non-empty spans"

    def test_entity_types_in_range(self):
        records = load("klue", "validation", max_samples=5)
        for r in records:
            for span in r["spans"]:
                assert span["type"] in KLUE_ENTITY_TYPES, f"unexpected type: {span['type']}"


@pytest.mark.integration
class TestSmokeKMOU:
    @pytest.fixture(autouse=True)
    def _require_cache(self):
        if not os.path.isdir(KMOU_CACHE):
            pytest.skip("KMOU HF cache not present")

    def test_load_returns_iob2_normalized(self):
        records = load("nlp-kmu/kor_ner", "validation", max_samples=5)
        assert len(records) > 0
        for r in records:
            assert set(r.keys()) == {"id", "tokens", "bio_tags", "spans", "sentence"}
            for tag in r["bio_tags"]:
                assert tag != "I", "bare 'I' tag found — normalization failed"
                assert "_" not in tag or tag == "O", f"underscore tag found: {tag}"
            assert r["sentence"] == " ".join(r["tokens"])

    def test_at_least_one_span(self):
        records = load("nlp-kmu/kor_ner", "validation", max_samples=5)
        has_spans = any(len(r["spans"]) > 0 for r in records)
        assert has_spans, "Expected at least one record with non-empty spans"

    def test_entity_types_in_range(self):
        records = load("nlp-kmu/kor_ner", "validation", max_samples=5)
        for r in records:
            for span in r["spans"]:
                assert span["type"] in KMOU_ENTITY_TYPES, f"unexpected type: {span['type']}"
