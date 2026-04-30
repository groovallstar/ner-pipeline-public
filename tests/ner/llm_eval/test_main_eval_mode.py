"""llm_eval CLI eval_mode 디스패치 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner.llm_eval.__main__ import _eval_mode_for_lang
from ner.llm_eval.benchmark_runner import BenchmarkRunner


class _StubLabeler:
    """gold_spans 와 동일한 pred 를 반환하는 스텁."""

    total_prompt_tokens = 0
    total_completion_tokens = 0

    def __init__(self, mapping: dict):
        # text → list of {text, type} 매핑
        self._mapping = mapping

    def label_spans(self, text: str):
        return self._mapping.get(text, [])


@pytest.mark.parametrize('lang,expected', [
    ('ja', 'offset_span'),
    ('vi', 'offset_span'),
    ('ko', 'bio'),
])
def test_eval_mode_for_lang(lang, expected):
    assert _eval_mode_for_lang(lang) == expected


def test_vi_offset_span_path_runs_end_to_end():
    # VI 로컬 PII JSONL 흐름을 시뮬레이션:
    # gold_spans 만 들어있는 레코드 + match_spans 가 처리할 raw span
    records = [
        {
            'id': '0',
            'text': 'Nguyễn Văn A sống ở Hà Nội.',
            'gold_spans': [
                {'text': 'Nguyễn Văn A', 'type': 'PER',
                 'start': 0, 'end': 12},
                {'text': 'Hà Nội', 'type': 'LOC',
                 'start': 20, 'end': 26},
            ],
        },
    ]
    labeler = _StubLabeler({
        'Nguyễn Văn A sống ở Hà Nội.': [
            {'text': 'Nguyễn Văn A', 'type': 'PER'},
            {'text': 'Hà Nội', 'type': 'LOC'},
        ],
    })
    runner = BenchmarkRunner(
        records,
        compute_bertscore=False,
        lang='vi',
        eval_mode='offset_span',
        sample_concurrency=2,
    )
    runner.add_labeler('stub', 'vllm', labeler)
    results = runner.run()
    assert len(results) == 1
    span_f1 = results[0].metrics['span_f1']
    # gold 와 pred 가 동일하므로 micro F1 == 1.0
    assert span_f1['overall']['f1'] == pytest.approx(1.0)


def test_vi_local_jsonl_loads_via_loader(tmp_path: Path):
    # VietnameseDatasetLoader.load_local 이 PII 스키마(entities)를
    # gold_spans 로 변환하는지 확인 — VI offset_span 경로의 입력 계약.
    from ner.labelers.vi.dataset_loader import VietnameseDatasetLoader

    payload = {
        'id': 'r1',
        'text': 'Liên hệ: a@b.com',
        'entities': [
            {'text': 'a@b.com', 'label': 'EMAIL',
             'start_char': 9, 'end_char': 16},
        ],
    }
    p = tmp_path / 'pii.jsonl'
    p.write_text(json.dumps(payload, ensure_ascii=False) + '\n',
                 encoding='utf-8')

    records = VietnameseDatasetLoader.load_local(p)
    assert len(records) == 1
    assert records[0]['gold_spans'] == [
        {'text': 'a@b.com', 'type': 'EMAIL', 'start': 9, 'end': 16},
    ]
