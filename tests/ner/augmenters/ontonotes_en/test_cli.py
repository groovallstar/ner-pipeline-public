"""CLI 주 경로 — 산출물이 언제 남고 언제 안 남는지를 고정한다.

이 파일이 보는 것은 변환의 옳음이 아니라 **CLI 의 계약**이다. 변환이 옳은지는
`test_fixture_invariants.py`(무조건)와 `test_corpus_invariants.py`(원천 있을
때)가 본다. 여기서는 실패했을 때 디스크에 무엇이 남는지만 본다 — 그 구분이
없으면 "대조에 걸렸는데 산출물은 완결돼 보이는" 상태를 아무도 잡지 못한다.

원천(`data/`)이 없어도 도는 무조건 층이다. 골든 픽스처를 원천 파일 이름으로
복사해 CLI 를 실제로 태운다.
"""
import json
import shutil
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.__main__ import SPLIT_FILES, main

FIXTURE = Path(__file__).resolve().parents[2] / 'golden' / 'ontonotes_en'


@pytest.fixture
def raw_dir(tmp_path):
    """골든 픽스처를 원천 배치 그대로 깐다 (train 은 4 파일로 쪼개 배포된다)."""
    source = tmp_path / 'raw'
    source.mkdir()
    shutil.copy(FIXTURE / 'label.json', source / 'label.json')
    for filenames in SPLIT_FILES.values():
        for filename in filenames:
            shutil.copy(FIXTURE / 'sample.json', source / filename)
    return source


def run(raw_dir, out_dir, *extra):
    return main([
        '--raw-dir', str(raw_dir), '--output-dir', str(out_dir), *extra,
    ])


def test_writes_one_file_per_split_plus_meta(raw_dir, tmp_path):
    out = tmp_path / 'out'
    assert run(raw_dir, out) == 0

    assert sorted(p.name for p in out.glob('*.jsonl')) == [
        'test.jsonl', 'train.jsonl', 'valid.jsonl',
    ]
    assert list(out.glob('*.partial')) == []

    meta = json.loads((out / 'conversion_meta.json').read_text('utf-8'))
    # train 은 같은 픽스처 4 벌이므로 valid 의 정확히 4 배여야 한다 — 파일이
    # 하나만 읽히거나 두 번 읽히면 이 배수가 깨진다.
    assert meta['sentence_counts']['train'] == (
        4 * meta['sentence_counts']['valid']
    )
    assert set(meta['splits']) == set(SPLIT_FILES)


def test_leaves_nothing_behind_when_the_token_check_fails(
    raw_dir, tmp_path, monkeypatch,
):
    """대조가 실패하면 `.jsonl` 도 `.partial` 도 남지 않는다.

    실패 조건은 주입한다 — 대조기 자체가 옳은지는 다른 파일이 보고, 여기서
    묻는 것은 "문제가 보고됐을 때 CLI 가 무엇을 남기나" 뿐이다. 곧장 최종
    이름으로 쓰면 비영 종료에도 완결돼 보이는 파일이 남아, 다음 사람이 검사를
    지난 산출물로 읽는다.
    """
    monkeypatch.setattr(
        'ner.augmenters.ontonotes_en.__main__.check_entity_token_match',
        lambda record, tokens, tags, id2label: ['injected mismatch'],
    )
    out = tmp_path / 'out'
    assert run(raw_dir, out) == 1

    assert list(out.glob('*.jsonl')) == []
    assert list(out.glob('*.partial')) == []
    assert not (out / 'conversion_meta.json').exists()


def test_a_repeated_split_is_converted_once(raw_dir, tmp_path):
    """`--splits train train` 이 터지지 않고 한 번만 변환한다."""
    out = tmp_path / 'out'
    assert run(raw_dir, out, '--splits', 'train', 'train') == 0

    assert sorted(p.name for p in out.glob('*.jsonl')) == ['train.jsonl']
    meta = json.loads((out / 'conversion_meta.json').read_text('utf-8'))
    assert meta['splits'] == ['train']


def test_an_exception_midway_leaves_no_finished_looking_file(
    raw_dir, tmp_path, monkeypatch,
):
    """두 번째 split 에서 터져도 첫 split 이 완결된 이름으로 남지 않는다.

    스테이징이 실제로 막는 것이 이것이다. 대조 실패 경로는 스스로 정리하므로
    스테이징이 없어도 뒤끝이 깨끗하지만, 예외·중단은 정리 코드를 못 지난다 —
    그때 최종 이름으로 써 두었으면 반쯤 쓰인 파일이 `train.jsonl` 로 남아
    검사를 지난 산출물과 구별되지 않는다. `.partial` 이면 구별된다.
    """
    real = __import__(
        'ner.augmenters.ontonotes_en.__main__', fromlist=['convert_one_split'],
    ).convert_one_split
    calls = []

    def blow_up_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('injected failure midway')
        return real(*args, **kwargs)

    monkeypatch.setattr(
        'ner.augmenters.ontonotes_en.__main__.convert_one_split',
        blow_up_on_the_second,
    )
    out = tmp_path / 'out'
    with pytest.raises(RuntimeError):
        run(raw_dir, out)

    assert list(out.glob('*.jsonl')) == []
    assert [p.name for p in out.glob('*.partial')] == ['train.jsonl.partial']
