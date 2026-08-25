"""언어 배선 — `--lang` 이 받는 값과 언어별 기본값이 맞물리는가.

배선이 반쪽만 되면 실패가 늦고 애매하게 난다: choices 에만 넣으면 기본값
조회에서 `KeyError` 로 죽고, 기본값에만 넣으면 그 언어를 아예 못 부른다. 양쪽을
한 파일에서 보아 어느 쪽이 빠져도 여기서 걸리게 한다.

`--lang` 이 실제로 받는 값은 일부러 거절당해서 확인한다 — 없는 언어를 주면
argparse 가 인자 파싱 단계에서 끝내며 허용 목록을 그대로 뱉으므로, 학습을
시작하지 않고도 계약을 읽을 수 있다.
"""
import re

import pytest

from ner.classifier.__main__ import DEFAULT_DATA, DEFAULT_MODEL, main

SUPPORTED = ('ja', 'vi', 'ko', 'en')


@pytest.mark.parametrize('lang', SUPPORTED)
def test_every_language_has_defaults(lang):
    assert lang in DEFAULT_DATA
    assert lang in DEFAULT_MODEL


def test_defaults_cover_exactly_the_supported_languages():
    assert set(DEFAULT_DATA) == set(SUPPORTED)
    assert set(DEFAULT_MODEL) == set(SUPPORTED)


def test_en_points_at_the_merged_corpus():
    """en 은 split 별 파일이 아니라 병합본을 본다 — 재분할이 전제이기 때문."""
    assert DEFAULT_DATA['en'] == 'data/ontonotes_en/pii_all.jsonl'


def test_lang_option_accepts_exactly_the_supported_languages(
    monkeypatch, capsys,
):
    monkeypatch.setattr(
        'sys.argv', ['ner.classifier', '--lang', 'xx', '--group-key', 'orig'],
    )
    with pytest.raises(SystemExit):
        main()
    message = capsys.readouterr().err
    assert "invalid choice: 'xx'" in message
    # 허용 목록은 usage 줄의 `--lang {...}` 에 그대로 찍힌다. 거절 문구의
    # 서식은 argparse 판마다 달라 그쪽에 기대지 않는다.
    listed = re.search(r'--lang \{([^}]*)\}', message)
    assert listed, message
    assert set(listed.group(1).split(',')) == set(SUPPORTED)
