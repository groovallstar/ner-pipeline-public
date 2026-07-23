"""요청 로깅·request-id 미들웨어 테스트 — stub 추론기로 모델 없이 CI 가능.

성공 요청은 DEBUG(기본 INFO 에선 침묵)로만, 거절은 WARNING 으로 남는지와
`X-Request-ID` 에코·생성을 검증한다. caplog 로 레벨·문구·필드를 확인한다.
"""

import logging

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import ServerConfig
from server.inference import ModelUnavailable
from server.request_log import _REASONS

_MW_LOGGER = 'server.request_log'


class StubRegistry:
    """결정적 stub — 첫 단어를 PER span 으로 반환(로그 필드 검증용)."""

    def __init__(self, unavailable=()):
        self._unavailable = set(unavailable)

    def predict(self, text, lang, apply_threshold=True):
        if lang in self._unavailable:
            raise ModelUnavailable(f'model for lang {lang!r} is not loaded')
        first = text.split(' ', 1)[0] if text else ''
        return [] if not first else [{
            'label': 'PER', 'start_char': 0, 'end_char': len(first),
            'text': first, 'score': 1.0}]

    def predict_batch(self, texts, langs, apply_threshold=True):
        return [self.predict(t, lang, apply_threshold)
                for t, lang in zip(texts, langs)]

    def health(self):
        return {'status': 'ok', 'langs': {}}


def _client(registry=None, config=None):
    return TestClient(create_app(registry or StubRegistry(),
                                 config or ServerConfig()))


# --- request-id 헤더 ---

def test_request_id_generated_on_success():
    """응답에 X-Request-ID 가 실린다(요청에 없으면 생성)."""
    r = _client().post('/v1/ner', json={'text': 'Alice x', 'lang': 'vi'})
    assert r.status_code == 200
    assert r.headers.get('x-request-id')  # 비어있지 않은 값


def test_request_id_echoed_when_provided():
    """수신 X-Request-ID 는 그대로 에코된다(추적 연속성)."""
    r = _client().post('/v1/ner', json={'text': 'Alice x', 'lang': 'vi'},
                       headers={'x-request-id': 'trace-42'})
    assert r.headers.get('x-request-id') == 'trace-42'


def test_request_id_on_rejection():
    """거절 응답(413 등)에도 X-Request-ID 가 실린다."""
    cfg = ServerConfig(max_batch=1)
    r = _client(config=cfg).post('/v1/ner', json={
        'texts': ['a', 'b'], 'lang': 'vi'})
    assert r.status_code == 413
    assert r.headers.get('x-request-id')


# --- 성공 요청: DEBUG 에서만, INFO 에선 침묵 ---

def test_success_silent_at_info(caplog):
    """기본 레벨(INFO)에서 성공 요청은 미들웨어 로그를 남기지 않는다."""
    with caplog.at_level(logging.INFO, logger=_MW_LOGGER):
        r = _client().post('/v1/ner', json={'text': 'Alice x', 'lang': 'vi'})
    assert r.status_code == 200
    assert [rec for rec in caplog.records if rec.name == _MW_LOGGER] == []


def test_success_single_logged_at_debug(caplog):
    """DEBUG 에서 단일 성공은 lang·batch·entities·rid 를 한 줄로 남긴다."""
    with caplog.at_level(logging.DEBUG, logger=_MW_LOGGER):
        _client().post('/v1/ner', json={'text': 'Alice x', 'lang': 'vi'})
    line = _one(caplog, _MW_LOGGER)
    assert line.levelno == logging.DEBUG
    msg = line.getMessage()
    assert 'status=200' in msg and 'lang=vi' in msg
    assert 'batch=1' in msg and 'entities=1' in msg and 'rid=' in msg


def test_success_batch_logged_at_debug(caplog):
    """DEBUG 에서 배치 성공은 batch 개수·entities 합·언어 요약을 남긴다."""
    with caplog.at_level(logging.DEBUG, logger=_MW_LOGGER):
        _client().post('/v1/ner', json={
            'texts': ['aa x', 'bb y', 'cc z'], 'lang': 'vi'})
    msg = _one(caplog, _MW_LOGGER).getMessage()
    assert 'batch=3' in msg and 'entities=3' in msg and 'lang=vi' in msg


# --- 거절: WARNING 상시 + reason 태그 ---

def test_reject_bad_request_warns(caplog):
    """400(잘못된 lang) → WARNING reason=bad_request."""
    with caplog.at_level(logging.INFO, logger=_MW_LOGGER):
        r = _client().post('/v1/ner', json={'text': 'a', 'lang': 'ko'})
    assert r.status_code == 400
    line = _one(caplog, _MW_LOGGER)
    assert line.levelno == logging.WARNING
    assert 'rejected' in line.getMessage() and 'reason=bad_request' in \
        line.getMessage()


def test_reject_payload_too_large_warns(caplog):
    """413(배치 초과) → WARNING reason=payload_too_large."""
    cfg = ServerConfig(max_batch=1)
    with caplog.at_level(logging.INFO, logger=_MW_LOGGER):
        _client(config=cfg).post('/v1/ner', json={
            'texts': ['a', 'b'], 'lang': 'vi'})
    assert 'reason=payload_too_large' in _one(caplog, _MW_LOGGER).getMessage()


def test_reject_model_unavailable_warns(caplog):
    """503(미로드 언어) → WARNING reason=model_unavailable."""
    reg = StubRegistry(unavailable=('vi',))
    with caplog.at_level(logging.INFO, logger=_MW_LOGGER):
        r = _client(registry=reg).post('/v1/ner',
                                       json={'text': 'a', 'lang': 'vi'})
    assert r.status_code == 503
    assert 'reason=model_unavailable' in _one(caplog, _MW_LOGGER).getMessage()


def test_reject_unauthorized_warns(caplog):
    """401(API-key 불일치) → WARNING reason=unauthorized."""
    cfg = ServerConfig(api_key='secret')
    with caplog.at_level(logging.INFO, logger=_MW_LOGGER):
        r = _client(config=cfg).post('/v1/ner',
                                     json={'text': 'a', 'lang': 'vi'})
    assert r.status_code == 401
    assert 'reason=unauthorized' in _one(caplog, _MW_LOGGER).getMessage()


def test_reason_table_pins_overload_and_unavailable():
    """상태→사유 매핑 고정(429 는 HTTP 로 재현 어려워 표로 pin)."""
    assert _REASONS[429] == 'overloaded'
    assert _REASONS[503] == 'model_unavailable'


def _one(caplog, name):
    """지정 로거의 유일한 레코드를 돌려준다(정확히 1건 가정)."""
    recs = [rec for rec in caplog.records if rec.name == name]
    assert len(recs) == 1, f'expected 1 record, got {len(recs)}'
    return recs[0]
