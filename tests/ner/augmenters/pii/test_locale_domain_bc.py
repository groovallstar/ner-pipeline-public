"""로케일별 이메일 도메인 확장이 기존 언어를 바꾸지 않는지 확인한다.

`random_email` 에 `domains` 를 더한 것은 additive 여야 한다 — ja·vi·ko 는
로케일 목록을 선언하지 않으므로 공용 `EMAIL_DOMAINS` 를 그대로 쓴다.
"""
import random

from ner.augmenters.pii.generators.base import (
    EMAIL_DOMAINS,
    generate_pii,
    random_email,
)


def test_default_still_draws_from_shared_pool():
    rng = random.Random(1)
    for _ in range(200):
        assert random_email('taro', rng).rsplit('@', 1)[1] in EMAIL_DOMAINS


def test_explicit_domains_are_honoured():
    rng = random.Random(1)
    only = ['example.test']
    for _ in range(50):
        assert random_email('taro', rng, only).endswith('@example.test')


def test_existing_languages_keep_the_shared_pool():
    for lang in ('ja', 'vi', 'ko'):
        rng = random.Random(3)
        for _ in range(200):
            email = generate_pii('EMAIL', lang, rng)
            assert email.rsplit('@', 1)[1] in EMAIL_DOMAINS
