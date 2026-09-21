"""EMAIL 붙은 문맥 치환 스크립트 수락 검사.

스크립트가 더하는 것은 파일 입출력과 원장뿐이다. 원장의 지문이 실제로 쓴
판본과 맞아야 다음 단계 검사가 디스크 코퍼스를 이 기록에 대 볼 수 있다.
"""
import hashlib
import json

from ner.scripts import rewrite_email_context as rw


def _rows(n):
    rows = []
    for i in range(n):
        text = f'Contact team {i} at user{i}@example.com or call 555-01{i:02d}.'
        email = f'user{i}@example.com'
        phone = f'555-01{i:02d}'
        rows.append({
            'text': text, 'id': f'r{i}', 'orig': f'o{i // 2}',
            'entities': [
                {'label': 'EMAIL', 'start_char': text.index(email),
                 'end_char': text.index(email) + len(email), 'text': email},
                {'label': 'PHONE', 'start_char': text.index(phone),
                 'end_char': text.index(phone) + len(phone), 'text': phone},
            ],
        })
    return rows


def _write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                            for r in rows), encoding='utf-8')


def test_main_writes_the_corpus_and_a_matching_ledger(tmp_path):
    src, dst = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    _write(src, _rows(60))
    assert rw.main(['--input', str(src), '--output', str(dst),
                    '--seed', '5']) == 0

    ledger = json.loads((tmp_path / 'out.jsonl.email-context.json')
                        .read_text(encoding='utf-8'))
    assert ledger['input_sha256'] == hashlib.sha256(
        src.read_bytes()).hexdigest()
    assert ledger['output_sha256'] == hashlib.sha256(
        dst.read_bytes()).hexdigest()
    assert ledger['rows'] == 60 and ledger['email_spans'] == 60
    assert sum(ledger['forms'].values()) == 60
    assert ledger['glued_share']['before'] == 0.0


def test_rows_keep_their_ids_groups_and_surfaces(tmp_path):
    src, dst = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    rows = _rows(60)
    _write(src, rows)
    rw.main(['--input', str(src), '--output', str(dst), '--seed', '5'])
    out = [json.loads(line) for line in dst.read_text(encoding='utf-8')
           .splitlines()]
    assert [(r['id'], r['orig']) for r in out] == [
        (r['id'], r['orig']) for r in rows]
    for before, after in zip(rows, out):
        assert [e['text'] for e in after['entities']] == [
            e['text'] for e in before['entities']]
        for e in after['entities']:
            assert after['text'][e['start_char']:e['end_char']] == e['text']


def test_dry_run_writes_nothing(tmp_path):
    src, dst = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    _write(src, _rows(5))
    rw.main(['--input', str(src), '--output', str(dst), '--dry-run'])
    assert not dst.exists()
    assert not (tmp_path / 'out.jsonl.email-context.json').exists()
