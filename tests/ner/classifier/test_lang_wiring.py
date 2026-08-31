"""언어 배선 — `--lang` 이 받는 값과 언어별 기본값이 맞물리는가.

배선이 반쪽만 되면 실패가 늦고 애매하게 난다: choices 에만 넣으면 기본값
조회에서 `KeyError` 로 죽고, 기본값에만 넣으면 그 언어를 아예 못 부른다. 양쪽을
한 파일에서 보아 어느 쪽이 빠져도 여기서 걸리게 한다.

`--lang` 이 실제로 받는 값은 일부러 거절당해서 확인한다 — 없는 언어를 주면
argparse 가 인자 파싱 단계에서 끝내며 허용 목록을 그대로 뱉으므로, 학습을
시작하지 않고도 계약을 읽을 수 있다.
"""
import os
import re

import pytest

from ner.classifier.__main__ import DEFAULT_DATA, DEFAULT_MODEL, main

SUPPORTED = ('ja', 'vi', 'ko', 'en')
# 저장소 루트 — 기본 경로가 루트 기준 상대경로라 여기서 풀어 존재를 본다.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_DATA_ROOT = os.path.join(_REPO, 'data')


@pytest.mark.parametrize('lang', SUPPORTED)
def test_every_language_has_defaults(lang):
    assert lang in DEFAULT_DATA
    assert lang in DEFAULT_MODEL


def test_defaults_cover_exactly_the_supported_languages():
    assert set(DEFAULT_DATA) == set(SUPPORTED)
    assert set(DEFAULT_MODEL) == set(SUPPORTED)


def test_en_points_at_the_merged_corpus():
    """en 은 split 별 파일이 아니라 병합본을 본다 — 재분할이 전제이기 때문."""
    assert DEFAULT_DATA['en'] == 'data/ontonotes_en/origin.jsonl'


def test_every_default_is_named_origin_jsonl():
    """네 언어의 gold 파일명이 `origin.jsonl` 하나로 통일돼 있다.

    언어마다 이름이 갈리면 코드가 가리키는 경로와 실재가 어긋나도 조용하다 —
    실제로 ja 기본값이 없는 파일(`pii_all.jsonl`)을 가리킨 채로 남아 있었고,
    그 언어를 기본값으로 돌릴 때에야 드러났다. 이름 규칙을 문자열로 고정해
    다음에 한쪽만 바뀌는 것을 막는다.
    """
    for lang in SUPPORTED:
        assert DEFAULT_DATA[lang].endswith('/origin.jsonl'), lang


@pytest.mark.skipif(not os.path.isdir(_DATA_ROOT),
                    reason=f'dataset root not present: {_DATA_ROOT}')
@pytest.mark.parametrize('lang', SUPPORTED)
def test_default_data_path_exists(lang):
    """기본 경로가 **실재하는 파일**을 가리킨다.

    `data/` 는 gitignore 라 CI 에 없다 — 그때는 skip 한다(약화 아님, 사유
    출력). 로컬에서는 이 검사가 경로 문자열과 디스크를 잇는 유일한 자리다.
    """
    path = os.path.join(_REPO, DEFAULT_DATA[lang])
    assert os.path.isfile(path), f'{lang}: {DEFAULT_DATA[lang]} is missing'


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
