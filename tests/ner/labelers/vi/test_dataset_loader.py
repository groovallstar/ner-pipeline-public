"""WikiANN-vi dataset_loader 테스트.

`load()`는 기본 split 덤프를, `load_local()`은 임의 경로의 같은 스키마 JSONL 을
읽는다.
"""
import json
from pathlib import Path

import pytest

from ner.labelers.vi.dataset_loader import VietnameseDatasetLoader


class TestLoadLocal:
    def test_load_local_jsonl(self, tmp_path: Path):
        path = tmp_path / 'sample.jsonl'
        rows = [
            {
                'text': 'Hà Nội là thủ đô.',
                'entities': [{
                    'text': 'Hà Nội',
                    'label': 'LOC',
                    'start_char': 0,
                    'end_char': 6,
                }],
            },
            {
                'text': 'Samsung.',
                'entities': [{
                    'text': 'Samsung',
                    'type': 'ORG',
                    'start': 0,
                    'end': 7,
                }],
            },
        ]
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        records = VietnameseDatasetLoader.load_local(str(path))
        assert len(records) == 2
        assert records[0]['text'] == 'Hà Nội là thủ đô.'
        assert records[0]['gold_spans'][0] == {
            'text': 'Hà Nội',
            'type': 'LOC',
            'start': 0,
            'end': 6,
        }
        assert records[1]['gold_spans'][0]['type'] == 'ORG'

    def test_load_local_max_samples(self, tmp_path: Path):
        path = tmp_path / 'multi.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(5):
                f.write(json.dumps({
                    'text': f'Sample {i}',
                    'entities': [],
                }) + '\n')
        records = VietnameseDatasetLoader.load_local(
            str(path), max_samples=3
        )
        assert len(records) == 3


class TestLoadDefaultSplit:
    """기본 split 덤프 로딩."""

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown split'):
            VietnameseDatasetLoader().load(split='bogus')

    def test_missing_file_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setitem(
            VietnameseDatasetLoader.DEFAULT_PATH, 'test',
            tmp_path / 'absent.jsonl',
        )
        with pytest.raises(FileNotFoundError, match='not found'):
            VietnameseDatasetLoader().load(split='test')

    def test_default_paths_exposed(self) -> None:
        for split in ('train', 'validation', 'test'):
            assert split in VietnameseDatasetLoader.DEFAULT_PATH

    def test_default_split_reads_entities(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """기본 split 파일은 entities 스키마 → load_local 경로로 읽힌다."""
        path = tmp_path / 'test.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'id': '0',
                'text': 'Hà Nội là thủ đô.',
                'entities': [{
                    'text': 'Hà Nội', 'label': 'LOC',
                    'start_char': 0, 'end_char': 6,
                }],
            }, ensure_ascii=False) + '\n')
        monkeypatch.setitem(
            VietnameseDatasetLoader.DEFAULT_PATH, 'test', path,
        )
        records = VietnameseDatasetLoader().load(split='test')
        assert len(records) == 1
        assert records[0]['gold_spans'][0] == {
            'text': 'Hà Nội', 'type': 'LOC', 'start': 0, 'end': 6,
        }
