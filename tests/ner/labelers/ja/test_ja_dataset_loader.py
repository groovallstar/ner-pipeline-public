"""Stockmark JA canonical 덤프 로더 테스트.

HF 원본 로딩과 JA→canonical 매핑은 본 모듈에서 제거됐다.
테스트는 canonical JSONL 덤프 스키마 기반으로 load/load_local 왕복만
검증한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner.labelers.ja.dataset_loader import JapaneseDatasetLoader


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _sample_records() -> list[dict]:
    return [
        {
            'id': '1',
            'text': 'トヨタ自動車は東京で新型プリウスを発表した。',
            'entities': [
                {'label': 'ORG', 'start_char': 0, 'end_char': 6,
                 'text': 'トヨタ自動車'},
                {'label': 'LOC', 'start_char': 7, 'end_char': 9,
                 'text': '東京'},
                {'label': 'PROD', 'start_char': 12, 'end_char': 16,
                 'text': 'プリウス'},
            ],
        },
        {
            'id': '2',
            'text': '1985年4月3日に生まれた。',
            'entities': [
                {'label': 'DAT', 'start_char': 0, 'end_char': 9,
                 'text': '1985年4月3日'},
            ],
        },
    ]


class TestLoadFromPath:
    def test_load_with_explicit_path(self, tmp_path: Path) -> None:
        path = tmp_path / 'sample.jsonl'
        _write_jsonl(path, _sample_records())
        records = JapaneseDatasetLoader().load(path=path)
        assert len(records) == 2
        assert records[0]['id'] == '1'
        assert records[0]['text'] == 'トヨタ自動車は東京で新型プリウスを発表した。'
        types = [g['type'] for g in records[0]['gold_spans']]
        assert types == ['ORG', 'LOC', 'PROD']

    def test_max_samples(self, tmp_path: Path) -> None:
        path = tmp_path / 'sample.jsonl'
        _write_jsonl(path, _sample_records())
        records = JapaneseDatasetLoader().load(
            path=path, max_samples=1,
        )
        assert len(records) == 1

    def test_schema_conversion(self, tmp_path: Path) -> None:
        path = tmp_path / 'sample.jsonl'
        _write_jsonl(path, _sample_records())
        records = JapaneseDatasetLoader().load(path=path)
        span = records[0]['gold_spans'][0]
        # entities.label -> gold_spans.type, start_char -> start
        assert span == {
            'text': 'トヨタ自動車',
            'type': 'ORG',
            'start': 0,
            'end': 6,
        }

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown split'):
            JapaneseDatasetLoader().load(split='bogus')

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / 'absent.jsonl'
        with pytest.raises(FileNotFoundError, match='not found'):
            JapaneseDatasetLoader().load(path=target)


class TestLoadLocal:
    """`load_local(path)`은 `load(path=...)`의 명시적 별칭."""

    def test_load_local_matches_load(self, tmp_path: Path) -> None:
        path = tmp_path / 'sample.jsonl'
        _write_jsonl(path, _sample_records())
        via_load = JapaneseDatasetLoader().load(path=path)
        via_local = JapaneseDatasetLoader.load_local(path)
        assert via_load == via_local

    def test_legacy_key_aliases(self, tmp_path: Path) -> None:
        """entities 내부가 start/end/type 표기여도 수용한다."""
        path = tmp_path / 'alt.jsonl'
        records = [{
            'id': '9',
            'text': 'テスト',
            'entities': [
                {'type': 'PER', 'start': 0, 'end': 3, 'text': 'テスト'},
            ],
        }]
        _write_jsonl(path, records)
        out = JapaneseDatasetLoader.load_local(path)
        assert out[0]['gold_spans'][0] == {
            'text': 'テスト', 'type': 'PER', 'start': 0, 'end': 3,
        }


class TestDefaultPaths:
    def test_default_paths_exposed(self) -> None:
        """CLI 등이 기본 경로를 조회할 수 있도록 노출된다."""
        assert 'train' in JapaneseDatasetLoader.DEFAULT_PATH
        assert 'test' in JapaneseDatasetLoader.DEFAULT_PATH
