"""병합 CLI 의 계약 — 무엇을 떨어뜨리고 언제 산출물을 안 남기나.

원천(`data/`)이 없어도 도는 무조건 층이다. 실 코퍼스가 골든 넘버를 지키는지는
`test_merged_corpus.py`(데이터 있을 때)가 따로 본다.

여기서 고정하는 계약은 셋이다 — `split` 필드가 사라진다는 것, 무결성 검사에
걸리면 출력 파일이 **아예 생기지 않는다**는 것, 그리고 병합본이
`validate_group_key` 를 통과한다는 것. 마지막이 이 도구의 존재 이유다: 세
파일을 그냥 이어붙이면 `split` 이 "더 강한 그룹 키" 로 잡혀 `--group-key orig`
가 막힌다.
"""
import json

import pytest

from ner.augmenters.ontonotes_en.merge_splits import main
from ner.classifier.data_utils import load_jsonl, validate_group_key


def write_split(directory, split, rows):
    """한 split 파일을 깐다."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{split}.jsonl'
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path


def row(split, index, text, entities=()):
    """검사를 통과하는 최소 행을 만든다."""
    rid = f'en-{split}-{index:06d}'
    return {
        'text': text,
        'entities': [dict(e) for e in entities],
        'id': rid,
        'orig': rid,
        'split': split,
    }


def entity(text, label, start):
    return {
        'label': label,
        'start_char': start,
        'end_char': start + len(text),
        'text': text,
    }


@pytest.fixture
def input_dir(tmp_path):
    """세 split 이 각각 두 행씩 든 정상 입력."""
    base = tmp_path / 'pii'
    write_split(base, 'train', [
        row('train', 0, 'Alice works at Acme.',
            [entity('Alice', 'PER', 0), entity('Acme', 'ORG', 15)]),
        row('train', 1, 'Mail me at a@b.com today.',
            [entity('a@b.com', 'EMAIL', 11)]),
    ])
    write_split(base, 'valid', [
        row('valid', 0, 'Bob visited Paris.',
            [entity('Bob', 'PER', 0), entity('Paris', 'LOC', 12)]),
        row('valid', 1, 'No entities here.'),
    ])
    write_split(base, 'test', [
        row('test', 0, 'The Olympics ran in May.',
            [entity('Olympics', 'EVT', 4), entity('May', 'DAT', 20)]),
        row('test', 1, 'Card 4111111111111111 expired.',
            [entity('4111111111111111', 'CREDIT_CARD', 5)]),
    ])
    return base


def run(input_dir, output):
    return main(['--input-dir', str(input_dir), '--output', str(output)])


def test_merges_every_row_and_drops_split(input_dir, tmp_path):
    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 0

    rows = load_jsonl(str(out))
    assert len(rows) == 6
    assert all('split' not in r for r in rows)
    # 어느 split 이었는지는 접두사에 남는다 — 필드를 떼도 정보를 잃지 않는다.
    assert sorted(r['id'].rsplit('-', 1)[0] for r in rows) == [
        'en-test', 'en-test', 'en-train', 'en-train', 'en-valid', 'en-valid',
    ]


def test_merged_file_passes_group_key_validation(input_dir, tmp_path):
    """이 도구가 존재하는 이유 — 그냥 이어붙이면 여기서 막힌다."""
    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 0
    validate_group_key(load_jsonl(str(out)), 'orig')


def test_split_field_would_block_group_key(input_dir, tmp_path):
    """`split` 을 남긴 채 합치면 그것이 더 강한 키로 잡힌다는 사실을 고정한다.

    이 검사가 깨지면 `split` 을 떨어뜨릴 근거가 사라진 것이므로, 병합 도구의
    설계를 다시 봐야 한다.
    """
    rows = []
    for split in ('train', 'valid', 'test'):
        with (input_dir / f'{split}.jsonl').open(encoding='utf-8') as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    with pytest.raises(ValueError, match='stronger key exists'):
        validate_group_key(rows, 'orig')


def test_offset_mismatch_leaves_no_output(input_dir, tmp_path):
    bad = row('train', 2, 'Alice works at Acme.',
              [entity('Alice', 'PER', 0)])
    bad['entities'][0]['start_char'] = 6  # 오프셋만 어긋뜨린다
    write_split(input_dir, 'train', [bad])

    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 1
    assert not out.exists()
    assert not out.with_suffix('.jsonl.partial').exists()


def test_label_outside_canonical_leaves_no_output(input_dir, tmp_path):
    bad = row('train', 2, 'Alice works at Acme.',
              [entity('Alice', 'NORP', 0)])
    write_split(input_dir, 'train', [bad])

    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 1
    assert not out.exists()


def test_missing_group_key_leaves_no_output(input_dir, tmp_path):
    bad = row('train', 2, 'Alice works at Acme.')
    del bad['orig']
    write_split(input_dir, 'train', [bad])

    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 1
    assert not out.exists()


def test_group_key_shared_across_splits_leaves_no_output(input_dir, tmp_path):
    """split 이 달라도 `orig` 가 같으면 한 그룹이 된다 — 크게 실패해야 한다."""
    shared = row('valid', 0, 'Bob visited Paris.')
    shared['orig'] = 'en-train-000000'
    write_split(input_dir, 'valid', [shared])

    out = tmp_path / 'pii_all.jsonl'
    assert run(input_dir, out) == 1
    assert not out.exists()


def test_missing_input_file_leaves_no_output(tmp_path):
    out = tmp_path / 'pii_all.jsonl'
    assert run(tmp_path / 'absent', out) == 1
    assert not out.exists()
