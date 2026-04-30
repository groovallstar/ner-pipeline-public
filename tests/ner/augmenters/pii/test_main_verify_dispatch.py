"""ner.augmenters.pii.__main__ 의 verify 라벨러 lang 분기 단위 테스트."""
from __future__ import annotations

import pytest

from ner.augmenters.pii import __main__ as pii_main


@pytest.mark.parametrize('lang,expected_module', [
    ('ja', 'ner.labelers.ja.vllm_ner_labeler'),
    ('vi', 'ner.labelers.vi.vllm_ner_labeler'),
])
def test_build_verify_labeler_dispatches_by_lang(
    lang, expected_module,
):
    labeler = pii_main._build_verify_labeler(
        lang=lang,
        base_url='http://localhost:0/v1',
        model='dummy-model',
        concurrency=1,
    )
    # 라벨러 클래스가 기대한 모듈에서 임포트되었는지 확인
    assert type(labeler).__module__ == expected_module


def test_build_verify_labeler_rejects_unknown_lang():
    with pytest.raises(ValueError, match='unsupported lang'):
        pii_main._build_verify_labeler(
            lang='ko', base_url='', model='', concurrency=1,
        )
