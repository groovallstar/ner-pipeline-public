"""예제 클라이언트의 선택적 429 재시도와 기존 응답 반환 계약을 검증한다."""

import json

import pytest
import requests

from server.scripts.example_client import NERClient


def _response(status, retry_after=None):
    """응답 본문을 수신한 requests 응답을 구성한다."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({'status': status}).encode()
    response._content_consumed = True
    if retry_after is not None:
        response.headers['Retry-After'] = retry_after
    return response


def _transport(monkeypatch, responses):
    """HTTP 전송과 대기 경계만 대체하여 실제 클라이언트의 호출 순서를 기록한다."""
    events = []
    pending = iter(responses)

    def post(url, **kwargs):
        events.append(('post', url, kwargs))
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr('server.scripts.example_client.requests.post', post)
    monkeypatch.setattr('time.sleep', lambda seconds: events.append(('sleep', seconds)))
    return events


@pytest.mark.parametrize('batch', [False, True], ids=['single', 'batch'])
def test_retry_preserves_request_and_waits_for_header(monkeypatch, batch):
    """429 후 헤더만큼 기다리고 인증·본문·타임아웃을 그대로 재전송한다."""
    final = _response(200)
    events = _transport(monkeypatch, [_response(429, '2'), final])
    client = NERClient('http://example.test/', api_key='test-key', timeout=5,
                       max_retries=3)
    if batch:
        got = client.ner_batch(['first', 'second'], lang='en')
        body = {'texts': ['first', 'second'], 'lang': 'en'}
    else:
        got = client.ner_single('first', lang='en')
        body = {'text': 'first', 'lang': 'en'}
    request = ('post', 'http://example.test/v1/ner', {
        'json': body, 'headers': {'Content-Type': 'application/json',
                                 'x-api-key': 'test-key'}, 'timeout': 5})
    assert got is final
    assert events == [request, ('sleep', 2), request]


def test_default_returns_429_without_retry(monkeypatch):
    """기본 호출은 기존처럼 첫 429 응답을 그대로 반환한다."""
    first = _response(429, '1')
    events = _transport(monkeypatch, [first])
    assert NERClient().ner_single('text') is first
    assert [event[0] for event in events] == ['post']


def test_retry_limit_returns_last_429_without_extra_sleep(monkeypatch):
    """추가 시도 상한 이후에는 마지막 429를 반환하며 더 기다리지 않는다."""
    final = _response(429, '1')
    events = _transport(monkeypatch, [_response(429, '1'), final])
    assert NERClient(max_retries=1).ner_single('text') is final
    assert [event[0] for event in events] == ['post', 'sleep', 'post']


@pytest.mark.parametrize('header', [None, '', 'invalid', '-1'])
def test_missing_or_invalid_retry_after_waits_one_second(monkeypatch, header):
    """현재 서버의 초 단위 헤더가 없거나 잘못됐으면 1초 후 재시도한다."""
    events = _transport(monkeypatch, [_response(429, header), _response(200)])
    assert NERClient(max_retries=1).ner_single('text').status_code == 200
    assert events[1] == ('sleep', 1)


@pytest.mark.parametrize('status', [200, 400, 413, 422, 500, 503])
def test_other_status_is_not_retried(monkeypatch, status):
    """자동 재시도 범위는 429로 제한한다."""
    first = _response(status)
    events = _transport(monkeypatch, [first])
    assert NERClient(max_retries=3).ner_single('text') is first
    assert [event[0] for event in events] == ['post']


def test_network_error_propagates_without_retry(monkeypatch):
    """서버가 이미 처리 중일 수 있는 통신 오류는 자동 재전송하지 않는다."""
    events = _transport(monkeypatch, [requests.Timeout('timed out')])
    with pytest.raises(requests.Timeout):
        NERClient(max_retries=3).ner_single('text')
    assert [event[0] for event in events] == ['post']


def test_negative_retry_limit_is_rejected():
    """음수 상한 때문에 요청 없이 종료되는 설정을 거절한다."""
    with pytest.raises(ValueError, match='max_retries'):
        NERClient(max_retries=-1)
