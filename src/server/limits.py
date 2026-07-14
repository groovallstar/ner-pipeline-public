"""요청 바디 크기 상한 미들웨어 — 전량 버퍼링 전에 413 으로 거부한다.

max_chars/max_batch 는 Pydantic 파싱 *후* 동작하는 논리 가드라, 그 이전에
바디를 통째로 메모리에 올리는 전송 계층 메모리 고갈(DoS)을 막지 못한다.
이 순수 ASGI 미들웨어는 라우팅·인증보다 앞서 두 경로로 바디를 bound 한다.

- Content-Length 헤더가 상한 초과 → 바디를 읽기 전에 즉시 413.
- 헤더가 없거나(chunked) 실제보다 작게 신고된 경우 → 수신 바이트를 누적해
  상한 초과 시 413. 우회(chunked transfer-encoding)를 막는다.

인증(require_key)보다 앞단이라 API-key on/off 와 무관하게 적용된다 —
비인증 요청도 대용량 바디로 OOM 을 유발할 수 없다.
"""

import json
from typing import Optional


def _content_length(scope) -> Optional[int]:
    """ASGI scope 헤더에서 Content-Length 를 정수로 읽는다(없거나 불량이면 None)."""
    for name, value in scope.get('headers', []):
        if name == b'content-length':
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_413(send, max_bytes: int) -> None:
    """app.py `_error` 와 동일한 `{error:{status,message}}` 봉투로 413 전송."""
    body = json.dumps({
        'error': {
            'status': 413,
            'message': f'request body exceeds max_body_bytes ({max_bytes})',
        }
    }).encode('utf-8')
    await send({
        'type': 'http.response.start',
        'status': 413,
        'headers': [
            (b'content-type', b'application/json'),
            (b'content-length', str(len(body)).encode('ascii')),
        ],
    })
    await send({'type': 'http.response.body', 'body': body})


class BodySizeLimitMiddleware:
    """바디 바이트를 max_bytes 로 bound 하는 순수 ASGI 미들웨어."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        # 1) Content-Length 조기 거부 — 바디를 한 바이트도 읽기 전에 막는다.
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _send_413(send, self.max_bytes)
            return

        # 2) 스트리밍 누적 — chunked/과소신고 우회를 막는다. 상한 초과 시
        # 직접 413 을 보내고 앱에는 disconnect 를 주입해 조용히 unwind 시키며,
        # 이후 앱이 내려는 응답은 guarded_send 로 삼켜 이중 전송을 막는다.
        state = {'total': 0, 'responded': False}

        async def limited_receive():
            message = await receive()
            if message['type'] == 'http.request':
                state['total'] += len(message.get('body', b''))
                if state['total'] > self.max_bytes and not state['responded']:
                    state['responded'] = True
                    await _send_413(send, self.max_bytes)
                    return {'type': 'http.disconnect'}
            return message

        async def guarded_send(message):
            if state['responded']:
                return
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
