"""Stockmark NER Wikipedia 데이터셋 로더.

데이터셋: stockmark/ner-wikipedia-dataset (HuggingFace Hub)
형식: 원시 텍스트 + 문자 오프셋 span (BIO 아님)
엔티티 타입: 人名, 法人名, 地名, 施設名, 製品名, イベント名, 政治的組織名, その他の組織名

분할 전략: train 분할만 존재하므로 train_test_split(test_size=0.2, seed=42) 사용.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset


class JapaneseDatasetLoader:
    """재현 가능한 분할로 Stockmark NER Wikipedia 데이터셋을 로드한다."""

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
        """데이터셋을 로드하여 gold span이 포함된 레코드를 반환한다.

        Args:
            name: 데이터셋 이름 (기본값: stockmark/ner-wikipedia-dataset)
            split: "train" (80%) 또는 "test" (20%). 벤치마크 기본값은 "test".
            max_samples: 반환할 레코드 수 제한.
            seed: 재현 가능한 분할을 위한 랜덤 시드.
            test_size: 테스트 분할 비율.

        Returns:
            레코드 리스트: {"id": str, "text": str, "gold_spans": [{"text", "type", "start", "end"}]}
        """
        dataset_name = name or self.DATASET_NAME
        hf_dataset = load_dataset(
            dataset_name,
            split="train",  # Stockmark는 train 분할만 존재한다
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )

        # 재현 가능한 train/test 분할을 생성한다
        splits = hf_dataset.train_test_split(test_size=test_size, seed=seed)
        selected = splits["test"] if split == "test" else splits["train"]

        if max_samples is not None:
            selected = selected.select(range(min(max_samples, len(selected))))

        return self._to_records(selected)

    @staticmethod
    def load_local(
        path: str, max_samples: Optional[int] = None
    ) -> List[dict]:
        """로컬 JSONL 파일(PII 주입 결과 등)을 load()와 동일한 스키마로 읽는다.

        입력 스키마: {text, entities:[{label,start_char,end_char,text}]}
        출력 스키마: {id, text, gold_spans:[{text,type,start,end}]}
        """
        records: List[dict] = []
        p = Path(path)
        with open(p, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data['text']
                gold_spans = []
                for ent in data.get('entities', []):
                    gold_spans.append({
                        'text': ent.get('text', ''),
                        'type': ent.get('label', ent.get('type', '')),
                        'start': int(ent.get(
                            'start_char', ent.get('start', 0)
                        )),
                        'end': int(ent.get(
                            'end_char', ent.get('end', 0)
                        )),
                    })
                records.append({
                    'id': str(data.get('id', i)),
                    'text': text,
                    'gold_spans': gold_spans,
                })
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    @staticmethod
    def _to_records(dataset) -> List[dict]:
        """HuggingFace 데이터셋을 레코드 형식으로 변환한다."""
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
