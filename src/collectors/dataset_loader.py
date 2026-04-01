import json
import os
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, ClassLabel
from datasets.exceptions import DatasetNotFoundError as HFDatasetNotFoundError


class NERRecord(dict):
    """Standard NER record: tokens, ner_tags, id."""
    tokens: List[str]
    ner_tags: List[str]
    id: str


class DatasetNotFoundError(Exception):
    pass


# Columns differ across datasets; defaults cover most cases.
DATASET_COLUMN_MAP: dict[str, dict[str, str]] = {
    "klue": {"tokens": "tokens", "ner_tags": "ner_tags"},
    "kmounlp/NER": {"tokens": "words", "ner_tags": "ner"},
}
_DEFAULT_COLUMN_MAP = {"tokens": "tokens", "ner_tags": "ner_tags"}


class DatasetLoader:
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", "/work/.huggingface/datasets"
        )

    def load(
        self,
        name: str,
        config: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        try:
            hf_dataset = load_dataset(
                name,
                config,
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=False,
            )
        except (HFDatasetNotFoundError, FileNotFoundError, ValueError) as e:
            raise DatasetNotFoundError(f"Dataset '{name}' not found: {e}") from e

        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        col_map = DATASET_COLUMN_MAP.get(name, _DEFAULT_COLUMN_MAP)
        return self._to_ner_records(hf_dataset, col_map)

    def _to_ner_records(self, hf_dataset, col_map: dict[str, str]) -> List[dict]:
        tokens_col = col_map["tokens"]
        tags_col = col_map["ner_tags"]

        # Detect ClassLabel feature for automatic int → str conversion
        features = hf_dataset.features
        tag_feature = features.get(tags_col)
        label_feature = None
        if hasattr(tag_feature, "feature"):
            label_feature = tag_feature.feature  # Sequence(ClassLabel)
        elif isinstance(tag_feature, ClassLabel):
            label_feature = tag_feature

        records = []
        for i, row in enumerate(hf_dataset):
            tokens = row[tokens_col]
            raw_tags = row[tags_col]

            if label_feature is not None and isinstance(label_feature, ClassLabel):
                ner_tags = [label_feature.int2str(t) for t in raw_tags]
            else:
                ner_tags = [str(t) for t in raw_tags]

            records.append({"tokens": tokens, "ner_tags": ner_tags, "id": str(i)})

        return records


if __name__ == "__main__":
    DATASET = "klue"
    CONFIG = "ner"
    SPLITS = ["train", "validation"]
    OUTPUT_DIR = "/data/ner"
    MAX_SAMPLES = None  # None = 전체, 정수 지정 시 해당 수만큼만

    loader = DatasetLoader()
    out_base = Path(OUTPUT_DIR) / DATASET.replace("/", "_")
    out_base.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"Loading {DATASET}/{CONFIG} [{split}] ...")
        try:
            records = loader.load(DATASET, config=CONFIG, split=split, max_samples=MAX_SAMPLES)
        except DatasetNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        out_path = out_base / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Saved {len(records)} records -> {out_path}")
