"""BIO token-sequence NER dataset spec registry, loader, and span extractor.

Implements the tokenization-unit × BIO-variant × span extraction process
documented in docs/ko/bio-span-process.md §2~§4.

Public API:
    TokenUnit, DatasetSpec, REGISTRY,
    DatasetNotFoundError,
    normalize_bio_variant, extract_spans, load
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from datasets import ClassLabel, load_dataset
from datasets.exceptions import DatasetNotFoundError as HFDatasetNotFoundError


class DatasetNotFoundError(Exception):
    """Raised when a dataset spec is not in the REGISTRY or HF load fails."""


class TokenUnit(str, Enum):
    """Tokenization granularity of a NER dataset."""
    SYLLABLE = "syllable"   # KLUE: character-level with explicit space tokens
    WORD = "word"           # WikiANN-style: whitespace-split words
    MORPHEME = "morpheme"   # KMOU/Stockmark: morphological analysis units


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable metadata for a BIO token-sequence NER dataset."""
    name: str               # HuggingFace dataset name
    lang: str               # language code (e.g. "ko")
    unit: TokenUnit
    bio_variant: str        # "iob2" | "kmou_i_alone"
    joiner: str             # token joiner for span text reconstruction
    config: Optional[str] = None  # HF datasets config name


# ---------------------------------------------------------------------------
# Registry — initial entries: Korean only
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetSpec] = {
    "klue": DatasetSpec(
        name="klue",
        lang="ko",
        unit=TokenUnit.SYLLABLE,
        bio_variant="iob2",
        joiner="",
        config="ner",
    ),
    # nlp-kmu/kor_ner is the HF-cached version of github.com/kmounlp/NER.
    # Morpheme-level tokens (particles split from stems).
    # ClassLabel names: ["I", "O", "B_OG", "B_TI", "B_LC", "B_DT", "B_PS"]
    "nlp-kmu/kor_ner": DatasetSpec(
        name="nlp-kmu/kor_ner",
        lang="ko",
        unit=TokenUnit.MORPHEME,
        bio_variant="kmou_i_alone",
        joiner=" ",
        config=None,
    ),
}


# ---------------------------------------------------------------------------
# BIO variant normalization
# ---------------------------------------------------------------------------

def normalize_bio_variant(tags: List[str], variant: str) -> List[str]:
    """Convert dataset-specific BIO tags to standard IOB2.

    Supported variants:
        "iob2"          — identity (already standard)
        "kmou_i_alone"  — B_TYPE → B-TYPE, standalone I → I-{prev_type}
    """
    if variant == "iob2":
        return list(tags)

    if variant == "kmou_i_alone":
        out: List[str] = []
        current_type: Optional[str] = None
        for tag in tags:
            if tag == "O":
                out.append("O")
                current_type = None
            elif tag.startswith("B_"):
                entity_type = tag[2:]
                out.append(f"B-{entity_type}")
                current_type = entity_type
            elif tag == "I":
                if current_type is not None:
                    out.append(f"I-{current_type}")
                else:
                    out.append("O")
            elif tag.startswith("B-"):
                out.append(tag)
                current_type = tag[2:]
            elif tag.startswith("I-"):
                out.append(tag)
            else:
                out.append("O")
                current_type = None
        return out

    raise ValueError(f"unknown bio_variant: {variant!r}")


# ---------------------------------------------------------------------------
# Span extraction from IOB2 tags
# ---------------------------------------------------------------------------

def extract_spans(
    tokens: List[str],
    bio_tags: List[str],
    joiner: str,
) -> List[dict]:
    """Extract entity spans from IOB2-tagged token sequence.

    Args:
        tokens: Token list (may contain whitespace-only tokens for syllable-level).
        bio_tags: IOB2 tags aligned 1:1 with tokens.
        joiner: String used to join tokens within a span ("" for syllable, " " for word/morpheme).

    Returns:
        List of {"text": str, "type": str} dicts.
    """
    spans: List[dict] = []
    current_chars: List[str] = []
    current_type: str = ""

    for tok, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            # Flush any in-progress span
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            # Start new span
            current_type = tag[2:]
            current_chars = [tok] if tok.strip() else []
        elif tag.startswith("I-") and current_type and tag[2:] == current_type:
            # Continue current span
            if tok.strip():
                current_chars.append(tok)
        else:
            # O or mismatching I-: flush and reset
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            current_chars = []
            current_type = ""

    # Flush trailing span
    if current_chars and current_type:
        spans.append({"text": joiner.join(current_chars), "type": current_type})

    return spans


# ---------------------------------------------------------------------------
# End-to-end loader
# ---------------------------------------------------------------------------

def load(
    spec_name: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """Load a BIO NER dataset end-to-end: HF load → IOB2 normalize → span extract.

    Args:
        spec_name: Key in REGISTRY (e.g. "klue", "nlp-kmu/kor_ner").
        split: Dataset split (e.g. "train", "validation", "test").
        max_samples: Limit the number of records returned.

    Returns:
        List of records: {"id": str, "tokens": list[str], "bio_tags": list[str],
                          "spans": list[{"text": str, "type": str}]}

    Raises:
        DatasetNotFoundError: If spec_name is not in REGISTRY or HF load fails.
    """
    try:
        spec = REGISTRY[spec_name]
    except KeyError:
        raise DatasetNotFoundError(f"unknown spec: {spec_name!r}")

    # Try JSONL fallback first (avoids Python 3.13 + datasets incompatibility)
    jsonl_path = Path("/data/ner") / spec.name.replace("/", "_") / f"{split}.jsonl"
    if jsonl_path.exists():
        return _load_from_jsonl(jsonl_path, spec, max_samples)

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/work/.huggingface/datasets")

    try:
        hf_dataset = load_dataset(
            spec.name,
            spec.config,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
    except (HFDatasetNotFoundError, FileNotFoundError, ValueError, TypeError) as exc:
        raise DatasetNotFoundError(f"HF load failed for {spec.name!r}: {exc}") from exc

    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    # Detect ClassLabel for int → str conversion
    features = hf_dataset.features
    tag_feature = features.get("ner_tags")
    label_feature = None
    if hasattr(tag_feature, "feature"):
        label_feature = tag_feature.feature
    elif isinstance(tag_feature, ClassLabel):
        label_feature = tag_feature

    records: List[dict] = []
    for i, row in enumerate(hf_dataset):
        tokens = list(row["tokens"])
        raw_tags = row["ner_tags"]

        # Decode int tags to strings via ClassLabel when available
        if label_feature is not None and isinstance(label_feature, ClassLabel):
            raw_tags_str = [label_feature.int2str(t) for t in raw_tags]
        else:
            raw_tags_str = [str(t) for t in raw_tags]

        bio_tags = normalize_bio_variant(raw_tags_str, spec.bio_variant)
        spans = extract_spans(tokens, bio_tags, spec.joiner)

        records.append({
            "id": str(i),
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": spans,
            "sentence": spec.joiner.join(tokens),
        })

    return records


def _load_from_jsonl(
    path: Path,
    spec: DatasetSpec,
    max_samples: Optional[int],
) -> List[dict]:
    """Load pre-exported JSONL records and apply BIO normalization + span extraction."""
    raw_records: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))
            if max_samples is not None and len(raw_records) >= max_samples:
                break

    records: List[dict] = []
    for i, row in enumerate(raw_records):
        tokens = row.get("tokens", [])
        raw_tags_str = row.get("ner_tags", [])
        bio_tags = normalize_bio_variant(raw_tags_str, spec.bio_variant)
        spans = extract_spans(tokens, bio_tags, spec.joiner)
        records.append({
            "id": row.get("id", str(i)),
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": spans,
            "sentence": spec.joiner.join(tokens),
        })

    return records
