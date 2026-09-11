"""실제 ASGI 요청으로 큐 대기 중 연결 종료와 과부하 응답을 검증한다."""

import asyncio
import json

import pytest

import server.app as app_module
from server.app import create_app
from server.concurrency import ConcurrencyGuard
from server.config import ServerConfig


class Registry:
    """실모델 대신 실제로 실행된 추론 입력을 기록한다."""

    def __init__(self):
        self.calls = []

    def predict(self, text, lang):
        self.calls.append(text)
        return []

    def predict_batch(self, texts, langs):
        self.calls.extend(texts)
        return [[] for _ in texts]


def _app_and_guard(monkeypatch, config):
    """앱이 만든 실제 guard를 보관하여 대기 상태를 결정적으로 제어한다."""
    guards = []

    def capture(*args, **kwargs):
        guard = ConcurrencyGuard(*args, **kwargs)
        guards.append(guard)
        return guard

    monkeypatch.setattr(app_module, 'ConcurrencyGuard', capture)
    registry = Registry()
    app = create_app(registry, config)
    return app, guards[0], registry


async def _post(app, body, messages=None):
    """연결 종료 이벤트를 주입할 수 있는 HTTP 요청을 앱 전체에 전달한다."""
    messages = messages if messages is not None else asyncio.Queue()
    messages.put_nowait({'type': 'http.request',
                         'body': json.dumps(body).encode(), 'more_body': False})
    sent = []

    async def send(message):
        sent.append(message)

    await app({'type': 'http', 'asgi': {'version': '3.0'},
               'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
               'path': '/v1/ner', 'raw_path': b'/v1/ner', 'query_string': b'',
               'headers': [(b'content-type', b'application/json')],
               'client': ('127.0.0.1', 1234), 'server': ('test', 80)},
              messages.get, send)
    start = next(m for m in sent if m['type'] == 'http.response.start')
    content = b''.join(m.get('body', b'') for m in sent
                       if m['type'] == 'http.response.body')
    return start, content


@pytest.mark.parametrize('body', [{'text': 'abandoned', 'lang': 'en'},
                                 {'texts': ['abandoned', 'second'], 'lang': 'en'}],
                         ids=['single', 'batch'])
def test_disconnected_waiter_skips_inference_and_releases_slot(monkeypatch, body):
    """대기 중 끊긴 단건·배치는 추론하지 않고 후속 정상 요청에 슬롯을 돌려준다."""
    async def scenario():
        app, guard, registry = _app_and_guard(
            monkeypatch, ServerConfig(max_concurrency=1))
        messages = asyncio.Queue()
        async with guard:
            task = asyncio.create_task(_post(app, body, messages))

            async def wait_until_queued():
                while guard.waiting != 1:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_queued(), 2)
            messages.put_nowait({'type': 'http.disconnect'})
        start, _ = await asyncio.wait_for(task, 2)
        assert registry.calls == []
        assert start['status'] == 499
        assert guard.in_flight == guard.waiting == 0
        start, content = await asyncio.wait_for(
            _post(app, {'text': 'normal', 'lang': 'en'}), 2)
        assert start['status'] == 200
        assert json.loads(content) == {'lang': 'en', 'entities': []}
        assert registry.calls == ['normal']

    asyncio.run(scenario())


@pytest.mark.parametrize('max_queue', [0, 1], ids=['queue-full', 'timeout'])
def test_overloaded_response_advises_retry(monkeypatch, max_queue):
    """큐 만석과 대기 시간 초과 모두 기존 오류 본문과 재시도 헤더를 반환한다."""
    async def scenario():
        app, guard, registry = _app_and_guard(monkeypatch, ServerConfig(
            max_concurrency=1, max_queue=max_queue, acquire_timeout_s=0.01))
        async with guard:
            start, content = await asyncio.wait_for(
                _post(app, {'text': 'normal', 'lang': 'en'}), 2)
        assert start['status'] == 429
        assert dict(start['headers'])[b'retry-after'] == b'1'
        error = json.loads(content)['error']
        assert error['status'] == 429
        expected = ('queue full (>= 0 waiting)' if max_queue == 0
                    else 'acquire timed out (0.01s)')
        assert error['message'] == expected
        spec = app.openapi()
        declared = spec['paths']['/v1/ner']['post']['responses']['429']
        schema = declared['content']['application/json']['schema']
        assert schema['$ref'].endswith('/ErrorResponse')
        assert set(error) == set(spec['components']['schemas']['ErrorBody']['properties'])
        retry_header = declared['headers']['Retry-After']
        assert retry_header['schema']['type'] == 'integer'
        assert int(dict(start['headers'])[b'retry-after']) == retry_header['example']
        assert registry.calls == []
        assert guard.in_flight == guard.waiting == 0

    asyncio.run(scenario())
