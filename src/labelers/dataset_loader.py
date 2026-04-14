import json
import os
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, ClassLabel
from datasets.exceptions import DatasetNotFoundError as HFDatasetNotFoundError


class NERRecord(dict):
    """표준 NER 레코드: tokens, ner_tags, id."""
    tokens: List[str]
    ner_tags: List[str]
    id: str


class DatasetNotFoundError(Exception):
    pass


# 데이터셋마다 컬럼명이 다르다. 기본값은 대부분의 경우를 커버한다.
DATASET_COLUMN_MAP: dict[str, dict[str, str]] = {
    "klue": {"tokens": "tokens", "ner_tags": "ner_tags"},
    "kmounlp/NER": {"tokens": "words", "ner_tags": "ner"},
}
_DEFAULT_COLUMN_MAP = {"tokens": "tokens", "ner_tags": "ner_tags"}


class HFTokenDatasetLoader:
    """HF 데이터셋에서 BIO 태그 (tokens/ner_tags) 형식 레코드를 로드한다."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", "/work/.huggingface/datasets"
        )

    def load(
        self,
        name: Optional[str] = None,
        config: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        if name is None:
            raise ValueError("HFTokenDatasetLoader.load requires 'name'")
        # JSONL 폴백을 먼저 시도한다 (Python 3.13 + datasets 비호환성 회피)
        jsonl_path = Path("/data/ner") / name.replace("/", "_") / f"{split}.jsonl"
        if jsonl_path.exists():
            return self._load_jsonl(jsonl_path, max_samples)

        try:
            hf_dataset = load_dataset(
                name,
                config,
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=False,
            )
        except (HFDatasetNotFoundError, FileNotFoundError, ValueError, TypeError) as e:
            raise DatasetNotFoundError(f"Dataset '{name}' not found: {e}") from e

        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        col_map = DATASET_COLUMN_MAP.get(name, _DEFAULT_COLUMN_MAP)
        return self._to_ner_records(hf_dataset, col_map)

    @staticmethod
    def _load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[dict]:
        """사전 내보낸 JSONL 레코드를 로드한다."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    def _to_ner_records(self, hf_dataset, col_map: dict[str, str]) -> List[dict]:
        tokens_col = col_map["tokens"]
        tags_col = col_map["ner_tags"]

        # 정수 → 문자열 자동 변환을 위해 ClassLabel 피처를 감지한다
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

            record = {"tokens": tokens, "ner_tags": ner_tags, "id": str(i)}
            # 원문 sentence가 있으면 보존한다 (예: KLUE NER)
            if "sentence" in row:
                record["sentence"] = row["sentence"]
            records.append(record)

        return records


# 하위 호환 별칭 (리팩터 이전 이름).
DatasetLoader = HFTokenDatasetLoader


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
