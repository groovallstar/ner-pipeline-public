"""`--verify vllm` 이 `en` 라벨러를 고르는지 확인한다.

이 배선이 없으면 교차 검증이 `ValueError` 로 끊기고, 주입 산출물은 검증
없이 나간다.
"""
import pytest

from ner.augmenters.__main__ import _build_verify_labeler


def test_en_resolves_to_the_english_labeler():
    from ner.labelers.en.vllm_ner_labeler import VllmNERLabeler

    labeler = _build_verify_labeler(
        lang='en', base_url='http://localhost:1/v1', model='x', concurrency=2,
    )
    assert isinstance(labeler, VllmNERLabeler)
    assert labeler.lang == 'en'


@pytest.mark.parametrize('lang', ['ja', 'vi'])
def test_existing_languages_still_resolve(lang):
    labeler = _build_verify_labeler(
        lang=lang, base_url='http://localhost:1/v1', model='x', concurrency=2,
    )
    assert labeler.lang == lang


def test_unsupported_lang_still_raises():
    with pytest.raises(ValueError):
        _build_verify_labeler(
            lang='de', base_url='http://localhost:1/v1', model='x',
            concurrency=2,
        )
