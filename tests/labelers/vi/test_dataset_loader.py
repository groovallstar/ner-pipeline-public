"""WikiANN-vi dataset_loader 테스트.

이슈 #21 이후 `load()`는 canonical 덤프 JSONL을 읽는다. HF 원본 로딩은
본 모듈에서 제거됐다. BIO↔offset span 유틸은 재라벨 파이프라인
(augmenters/wikiann_vi) 용도로 유지된다.
"""
import json
from pathlib import Path

import pytest

from labelers.vi.dataset_loader import (
    VietnameseDatasetLoader,
    bio_to_offset_spans,
    offset_spans_to_bio,
)


class TestBioToOffsetSpans:
    def test_single_entity(self):
        tokens = [
            'Nguyễn', 'Xuân', 'Phúc', 'đã', 'đến', 'Đà', 'Nẵng',
        ]
        tags = [
            'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC',
        ]
        text, spans = bio_to_offset_spans(tokens, tags)
        assert text == 'Nguyễn Xuân Phúc đã đến Đà Nẵng'
        assert len(spans) == 2
        assert spans[0] == {
            'text': 'Nguyễn Xuân Phúc',
            'type': 'PER',
            'start': 0,
            'end': 16,
        }
        assert spans[1] == {
            'text': 'Đà Nẵng',
            'type': 'LOC',
            'start': 24,
            'end': 31,
        }

    def test_no_entity(self):
        tokens = ['không', 'có', 'thực', 'thể']
        tags = ['O', 'O', 'O', 'O']
        text, spans = bio_to_offset_spans(tokens, tags)
        assert text == 'không có thực thể'
        assert spans == []

    def test_consecutive_entities_same_type(self):
        tokens = ['Hà', 'Nội', 'Đà', 'Nẵng']
        tags = ['B-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
        text, spans = bio_to_offset_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0]['text'] == 'Hà Nội'
        assert spans[1]['text'] == 'Đà Nẵng'
        assert text[spans[0]['start']:spans[0]['end']] == 'Hà Nội'
        assert text[spans[1]['start']:spans[1]['end']] == 'Đà Nẵng'

    def test_different_types_adjacent(self):
        tokens = ['Ông', 'Lee', 'tại', 'Samsung']
        tags = ['O', 'B-PER', 'O', 'B-ORG']
        text, spans = bio_to_offset_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0]['type'] == 'PER'
        assert spans[0]['text'] == 'Lee'
        assert spans[1]['type'] == 'ORG'
        assert spans[1]['text'] == 'Samsung'

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bio_to_offset_spans(['a', 'b'], ['O'])


class TestRoundTrip:
    def test_round_trip_basic(self):
        tokens = [
            'Ông', 'Lee', 'đến', 'Hà', 'Nội', 'gặp', 'Samsung',
        ]
        tags = ['O', 'B-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-ORG']
        _, spans = bio_to_offset_spans(tokens, tags)
        restored = offset_spans_to_bio(tokens, spans)
        assert restored == tags

    def test_round_trip_all_o(self):
        tokens = ['một', 'hai', 'ba']
        tags = ['O', 'O', 'O']
        _, spans = bio_to_offset_spans(tokens, tags)
        restored = offset_spans_to_bio(tokens, spans)
        assert restored == tags

    def test_round_trip_unicode(self):
        tokens = ['Thủ', 'đô', 'Việt', 'Nam', 'là', 'Hà', 'Nội']
        tags = [
            'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC',
        ]
        _, spans = bio_to_offset_spans(tokens, tags)
        restored = offset_spans_to_bio(tokens, spans)
        assert restored == tags

    def test_round_trip_consecutive_same_type(self):
        tokens = ['Hà', 'Nội', 'Đà', 'Nẵng']
        tags = ['B-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
        _, spans = bio_to_offset_spans(tokens, tags)
        restored = offset_spans_to_bio(tokens, spans)
        assert restored == tags


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


class TestLoadCanonicalDump:
    """canonical WikiANN-vi 덤프(`gold_spans_8type_merged`) 로딩."""

    def _write(self, path: Path, span_key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for i, t in enumerate(['Hà Nội là thủ đô.', 'Samsung.']):
                rec = {'id': str(i), 'text': t}
                rec[span_key] = [
                    {'text': 'Hà Nội', 'type': 'LOC', 'start': 0, 'end': 6}
                    if i == 0 else
                    {'text': 'Samsung', 'type': 'ORG',
                     'start': 0, 'end': 7}
                ]
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def test_load_with_default_span_key(self, tmp_path: Path) -> None:
        path = tmp_path / 'dump.jsonl'
        self._write(path, span_key='gold_spans_8type_merged')
        records = VietnameseDatasetLoader().load(path=path)
        assert len(records) == 2
        assert records[0]['gold_spans'][0]['type'] == 'LOC'
        assert records[1]['gold_spans'][0]['type'] == 'ORG'

    def test_load_with_custom_span_key(self, tmp_path: Path) -> None:
        """3종 `gold_spans` 필드도 `span_key`로 선택 가능."""
        path = tmp_path / 'dump3.jsonl'
        self._write(path, span_key='gold_spans')
        records = VietnameseDatasetLoader().load(
            path=path, span_key='gold_spans',
        )
        assert records[0]['gold_spans'][0]['text'] == 'Hà Nội'

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown split'):
            VietnameseDatasetLoader().load(split='bogus')

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / 'absent.jsonl'
        with pytest.raises(FileNotFoundError, match='not found'):
            VietnameseDatasetLoader().load(path=target)

    def test_default_paths_exposed(self) -> None:
        for split in ('train', 'validation', 'test'):
            assert split in VietnameseDatasetLoader.DEFAULT_PATH
