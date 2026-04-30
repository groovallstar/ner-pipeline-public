"""vi_silver_quality 단위 테스트 — 합성 silver vs 합성 gold 로 F1 검증."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval import vi_silver_quality


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def test_extract_silver_spans_filters_non_3type():
    rec = {
        'gold_spans_8type': [
            {'type': 'PER', 'start': 0, 'end': 3, 'text': 'abc'},
            {'type': 'PROD', 'start': 5, 'end': 8, 'text': 'xyz'},
            {'type': 'ORG', 'start': 10, 'end': 12, 'text': 'oo'},
        ],
    }
    spans = vi_silver_quality._extract_silver_spans(rec)
    types = [s['type'] for s in spans]
    assert types == ['PER', 'ORG']


def test_extract_silver_spans_prefers_merged_key():
    rec = {
        'gold_spans_8type': [{'type': 'PER', 'start': 0, 'end': 1, 'text': 'a'}],
        'gold_spans_8type_merged': [
            {'type': 'LOC', 'start': 5, 'end': 6, 'text': 'b'},
        ],
    }
    spans = vi_silver_quality._extract_silver_spans(rec)
    assert len(spans) == 1
    assert spans[0]['type'] == 'LOC'


def test_evaluate_split_perfect_match(tmp_path, monkeypatch):
    silver = [
        {
            'id': '0', 'text': 'A B C',
            'gold_spans_8type': [
                {'type': 'PER', 'start': 0, 'end': 1, 'text': 'A'},
                {'type': 'LOC', 'start': 4, 'end': 5, 'text': 'C'},
            ],
        },
        {
            'id': '1', 'text': 'D E',
            'gold_spans_8type': [
                {'type': 'ORG', 'start': 0, 'end': 1, 'text': 'D'},
            ],
        },
    ]
    silver_path = tmp_path / 'gemma_test.jsonl'
    _write_jsonl(silver_path, silver)

    fake_gold = [
        {'id': '0', 'text': 'A B C', 'gold_spans': [
            {'type': 'PER', 'start': 0, 'end': 1, 'text': 'A'},
            {'type': 'LOC', 'start': 4, 'end': 5, 'text': 'C'},
        ]},
        {'id': '1', 'text': 'D E', 'gold_spans': [
            {'type': 'ORG', 'start': 0, 'end': 1, 'text': 'D'},
        ]},
    ]
    monkeypatch.setattr(
        vi_silver_quality, 'load_wikiann_vi_gold',
        lambda *a, **kw: fake_gold,
    )

    result = vi_silver_quality.evaluate_split(silver_path, 'test')
    assert result['matched_records'] == 2
    assert result['overall']['f1'] == pytest.approx(1.0)
    assert result['per_entity']['PER']['f1'] == pytest.approx(1.0)
    assert result['per_entity']['LOC']['support'] == 1
    assert result['per_entity']['ORG']['support'] == 1


def test_evaluate_split_partial_match(tmp_path, monkeypatch):
    """silver 가 1개 빼먹고 1개 잘못 라벨한 케이스."""
    silver = [
        {
            'id': '0', 'text': 'A B C',
            'gold_spans_8type': [
                # PER 맞춤, LOC 빠짐, 가짜 ORG 추가
                {'type': 'PER', 'start': 0, 'end': 1, 'text': 'A'},
                {'type': 'ORG', 'start': 2, 'end': 3, 'text': 'B'},
            ],
        },
    ]
    silver_path = tmp_path / 'gemma_test.jsonl'
    _write_jsonl(silver_path, silver)

    fake_gold = [
        {'id': '0', 'text': 'A B C', 'gold_spans': [
            {'type': 'PER', 'start': 0, 'end': 1, 'text': 'A'},
            {'type': 'LOC', 'start': 4, 'end': 5, 'text': 'C'},
        ]},
    ]
    monkeypatch.setattr(
        vi_silver_quality, 'load_wikiann_vi_gold',
        lambda *a, **kw: fake_gold,
    )

    result = vi_silver_quality.evaluate_split(silver_path, 'test')
    # gold 2건 / pred 2건 / TP 1건 (PER) → P=R=F1=0.5
    assert result['overall']['precision'] == pytest.approx(0.5)
    assert result['overall']['recall'] == pytest.approx(0.5)
    assert result['overall']['f1'] == pytest.approx(0.5)
    # LOC 은 silver 에 없음 → recall=0
    assert result['per_entity']['LOC']['recall'] == 0.0


def test_evaluate_silver_set_skips_missing(tmp_path, monkeypatch, caplog):
    """silver 파일이 없는 split 은 경고 로그 후 결과에서 빠진다."""
    silver_path = tmp_path / 'gemma_test.jsonl'
    _write_jsonl(silver_path, [
        {'id': '0', 'text': 'a',
         'gold_spans_8type': [{'type': 'PER', 'start': 0, 'end': 1, 'text': 'a'}]},
    ])
    monkeypatch.setattr(
        vi_silver_quality, 'load_wikiann_vi_gold',
        lambda *a, **kw: [
            {'id': '0', 'text': 'a',
             'gold_spans': [{'type': 'PER', 'start': 0, 'end': 1, 'text': 'a'}]},
        ],
    )
    results = vi_silver_quality.evaluate_silver_set(
        tmp_path, 'gemma', ['test', 'validation'],
    )
    assert set(results.keys()) == {'test'}
