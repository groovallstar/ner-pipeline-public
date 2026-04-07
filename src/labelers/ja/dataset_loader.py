"""Stockmark NER Wikipedia dataset loader.

Dataset: stockmark/ner-wikipedia-dataset (HuggingFace Hub)
Format: raw text + character offset spans (not BIO)
Entity types: 人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名

Split strategy: train_test_split(test_size=0.2, seed=42) since only train split exists.
"""
import os
from typing import List, Optional

from datasets import load_dataset


class JapaneseDatasetLoader:
    """Load Stockmark NER Wikipedia dataset with reproducible splits."""

    DATASET_NAME = "stockmark/ner-wikipedia-dataset"

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", "/work/.huggingface/datasets"
        )

    def load(
        self,
        name: Optional[str] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        seed: int = 42,
        test_size: float = 0.2,
    ) -> List[dict]:
        """Load dataset and return records with gold spans.

        Args:
            name: Dataset name (default: stockmark/ner-wikipedia-dataset)
            split: "train" (80%) or "test" (20%). Default "test" for benchmarking.
            max_samples: Limit number of records returned.
            seed: Random seed for reproducible split.
            test_size: Fraction for test split.

        Returns:
            List of records: {"id": str, "text": str, "gold_spans": [{"text", "type", "start", "end"}]}
        """
        dataset_name = name or self.DATASET_NAME
        hf_dataset = load_dataset(
            dataset_name,
            split="train",  # Stockmark only has train
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )

        # Create reproducible train/test split
        splits = hf_dataset.train_test_split(test_size=test_size, seed=seed)
        selected = splits["test"] if split == "test" else splits["train"]

        if max_samples is not None:
            selected = selected.select(range(min(max_samples, len(selected))))

        return self._to_records(selected)

    @staticmethod
    def _to_records(dataset) -> List[dict]:
        """Convert HuggingFace dataset to record format."""
        records = []
        for i, row in enumerate(dataset):
            text = row["text"]
            entities = row.get("entities", [])

            gold_spans = []
            for entity in entities:
                span = entity.get("span", [0, 0])
                gold_spans.append({
                    "text": entity.get("name", ""),
                    "type": entity.get("type", ""),
                    "start": span[0],
                    "end": span[1],
                })

            records.append({
                "id": str(row.get("curid", i)),
                "text": text,
                "gold_spans": gold_spans,
            })

        return records
