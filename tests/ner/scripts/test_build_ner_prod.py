"""포장 스크립트가 언어에 무관하게 도는가 — 합성 run 으로 en·ko 를 태운다.

이 스크립트는 원래 en 전용 사본이었고, ko 를 더하면서 언어 파라미터화했다.
**사본을 하나 더 만들지 않은 대가로 이 테스트가 필요하다** — 분할 재유도·
지문 대조·누출 가드가 한 벌뿐이라, 그것이 어느 언어에서도 같게 동작한다는
사실을 여기서 고정하지 않으면 한 언어에서만 맞는 코드가 조용히 들어온다.

`/data` 도 HF 허브도 건드리지 않는다. run 디렉토리·데이터셋은 tmp 에 짓고
토크나이저는 stub 으로 갈아끼운다 — 포장 스크립트가 보는 것은 run 이 남긴
숫자와 파일이지 실제 가중치가 아니므로, 그 계약만 있으면 검사가 성립한다.
"""
import json
import os

import pytest

from ner.classifier.data_utils import (
    dataset_fingerprint,
    split_train_valid_test,
)
from ner.scripts import build_ner_prod as bnp


class _StubTokenizer:
    """`save_pretrained` 만 있는 토크나이저 — 카드 문구의 재료."""

    is_fast = True

    def save_pretrained(self, out_dir):
        for name in ('tokenizer_config.json', 'vocab.txt'):
            with open(os.path.join(out_dir, name), 'w',
                      encoding='utf-8') as f:
                f.write('{}')


def _rows(n=60):
    """group_key 로 쓸 `id` 를 가진 최소 canonical 행들."""
    out = []
    for i in range(n):
        text = f'sentence {i} with Alice here'
        start = text.index('Alice')
        out.append({
            'id': f'r{i}',
            'text': text,
            'entities': [{'label': 'PER', 'start_char': start,
                          'end_char': start + len('Alice'), 'text': 'Alice'}],
        })
    return out


def _write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _make_run(tmp_path, lang, model_name, **overrides):
    """합성 run 디렉토리 + 데이터셋을 짓고 run 경로를 돌려준다.

    분할 크기·지문은 실제 `split_train_valid_test`·`dataset_fingerprint` 로
    구한다 — 손으로 적으면 스크립트의 대조가 늘 통과해 검사가 허울이 된다.
    """
    rows = _rows()
    data_path = str(tmp_path / f'{lang}_data.jsonl')
    _write_jsonl(data_path, rows)
    tr, va, te = split_train_valid_test(rows, 0.1, 0.1, 42, group_key='id')

    run_dir = tmp_path / f'run_{lang}'
    (run_dir / 'best').mkdir(parents=True)
    for name in ('config.json', 'model.safetensors'):
        (run_dir / 'best' / name).write_text('{}', encoding='utf-8')

    per = {'precision': 0.9, 'recall': 0.9, 'f1': 0.9, 'support': 60}
    metrics = {
        'lang': lang,
        'model_name': model_name,
        'data_path': data_path,
        'data_fingerprint': dataset_fingerprint(rows),
        'n_rows': len(rows),
        'n_groups': len(rows),
        'group_key': 'id',
        'train_samples': len(tr),
        'valid_samples': len(va),
        'test_samples': len(te),
        'valid_ratio': 0.1,
        'test_ratio': 0.1,
        'seed': 42,
        'train_seed': 42,
        'precision': 'fp16',
        'epochs': 5,
        'batch_size': 16,
        'lr': 5e-05,
        'max_length': 256,
        'metric_for_best': 'eval_loss',
        'kfold': None,
        'curriculum': False,
        'overall_strict': dict(per),
        'per_entity_strict': {'PER': dict(per)},
    }
    metrics.update(overrides)
    (run_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False), encoding='utf-8')
    return str(run_dir), metrics


@pytest.fixture
def stub_tokenizer(monkeypatch):
    """HF 허브를 타지 않도록 토크나이저 로드를 stub 으로 바꾼다."""
    class _Auto:
        @staticmethod
        def from_pretrained(name, use_fast=True):
            return _StubTokenizer()

    monkeypatch.setattr(bnp, 'AutoTokenizer', _Auto)


LANGS = [('ko', 'monologg/koelectra-base-v3-discriminator', '한국어'),
         ('en', 'roberta-base', '영어')]


def _package(tmp_path, monkeypatch, lang, model_name, extra_argv=()):
    """기본 out-dir 유도를 실제로 태우려고 배포 루트를 tmp 로 옮긴다."""
    run_dir, metrics = _make_run(tmp_path, lang, model_name)
    root = tmp_path / 'deploy'
    monkeypatch.setattr(bnp, 'DEPLOY_ROOT', str(root))
    monkeypatch.setattr(
        'sys.argv',
        ['build_ner_prod.py', '--run-dir', run_dir, *extra_argv])
    bnp.main()
    return root / lang, metrics, run_dir


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_package_layout_is_language_independent(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """언어와 무관하게 같은 레이아웃이 나오고 out-dir 이 lang 에서 유도된다."""
    out, metrics, _ = _package(tmp_path, monkeypatch, lang, model_name)

    assert (out / 'model' / 'model.safetensors').exists()
    assert (out / 'model' / 'tokenizer_config.json').exists()
    assert (out / 'MODEL_CARD.md').exists()
    for name in ('train', 'valid', 'test'):
        path = out / 'data' / f'{name}.jsonl'
        got = sum(1 for _ in path.open(encoding='utf-8'))
        assert got == metrics[f'{name}_samples']


@pytest.mark.parametrize('lang,model_name,label', LANGS + [
    ('ja', 'tohoku-nlp/bert-base-japanese-v3', '일본어'),
    ('vi', 'xlm-roberta-base', '베트남어'),
])
def test_model_card_speaks_the_run_language(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """카드 문구가 run 의 언어를 따라간다 — 제목·평가 스크립트·재현 명령."""
    out, _, _ = _package(tmp_path, monkeypatch, lang, model_name)
    card = (out / 'MODEL_CARD.md').read_text(encoding='utf-8')

    assert card.startswith(f'# {lang.upper()} NER production 모델')
    assert f'{label} canonical 10종' in card
    if lang in ('ja', 'vi'):
        assert f'eval_{lang}_ner_test.py' in card
    else:
        assert f'eval_{lang}_ner_test.py' not in card
    assert f'--lang {lang}' in card
    assert 'build_ner_prod.py' in card


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_model_card_describes_the_tokenizer_actually_shipped(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """토크나이저 줄은 언어별 표가 아니라 동봉된 실물에서 나온다."""
    out, _, _ = _package(tmp_path, monkeypatch, lang, model_name)
    card = (out / 'MODEL_CARD.md').read_text(encoding='utf-8')

    assert 'fast(`_StubTokenizer`)' in card
    assert 'tokenizer_config.json' in card
    assert 'vocab.txt' in card


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_metrics_are_copied_verbatim_plus_provenance(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """run 의 수치는 손대지 않고 포장 사실만 덧붙는다."""
    out, metrics, run_dir = _package(tmp_path, monkeypatch, lang, model_name)
    packaged = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))

    assert packaged['deploy_package'] == {
        'source_run_dir': run_dir,
        'leaked_test_groups': 0,
        'split_rederived': True,
    }
    del packaged['deploy_package']
    assert packaged == metrics


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_fingerprint_mismatch_refuses(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """데이터셋이 run 이후 바뀌었으면 포장하지 않는다(자가 바뀐 상태)."""
    run_dir, metrics = _make_run(tmp_path, lang, model_name)
    m = bnp.load_run_metrics(run_dir)
    _write_jsonl(metrics['data_path'], _rows(59))

    with pytest.raises(SystemExit, match='fingerprint mismatch'):
        bnp.rederive_split(m)


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_recorded_split_sizes_must_match(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """재유도한 분할이 run 기록과 다르면 중단 — 안 쓰인 분할 출하 방지."""
    run_dir, _ = _make_run(tmp_path, lang, model_name, train_samples=999)
    m = bnp.load_run_metrics(run_dir)

    with pytest.raises(SystemExit, match='does not match the run'):
        bnp.rederive_split(m)


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_kfold_run_is_not_packageable(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """k-fold run 은 배포할 체크포인트 하나를 가리키지 않는다."""
    run_dir, _ = _make_run(tmp_path, lang, model_name, kfold=10)

    with pytest.raises(SystemExit, match='k-fold'):
        bnp.load_run_metrics(run_dir)


@pytest.mark.parametrize('lang,model_name,label', LANGS)
def test_lang_cross_check_refuses_wrong_run(
        tmp_path, monkeypatch, stub_tokenizer, lang, model_name, label):
    """--lang 이 run 과 어긋나면 다른 언어 자리에 쓰지 않고 멈춘다."""
    run_dir, _ = _make_run(tmp_path, lang, model_name)
    other = 'ja' if lang != 'ja' else 'vi'

    with pytest.raises(SystemExit, match='--lang says'):
        bnp.load_run_metrics(run_dir, expect_lang=other)


def test_unknown_lang_is_refused(tmp_path, monkeypatch, stub_tokenizer):
    """카드 이름표가 없는 언어는 포장 대상이 아니다(문구를 쓸 수 없다)."""
    run_dir, _ = _make_run(tmp_path, 'ko', 'x')
    path = os.path.join(run_dir, 'metrics.json')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    m['lang'] = 'th'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False)

    with pytest.raises(SystemExit, match='packageable languages'):
        bnp.load_run_metrics(run_dir)
