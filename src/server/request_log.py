"""요청 로깅 미들웨어 — 요청별 request-id·지연·결과를 한 줄로 남긴다.

로그량을 억제하려고 성공 요청(2xx)은 DEBUG(기본 INFO 레벨에선 침묵)로만
남기고, 거절(4xx·503)만 WARNING 으로 상시 남긴다. 미처리 예외(500)의
트레이스백은 app 의 예외 핸들러(`_unhandled`)가 따로 ERROR 로 남기므로 여기선
관여하지 않는다(요청당 이중 로그 방지).

순수 ASGI 미들웨어라 라우팅·인증 이전에 request-id 를 scope 에 심고 응답
헤더(`X-Request-ID`)로 에코한다. 핸들러는 `request.state.ner_meta` 에
lang·batch·entities 를 채워 이 미들웨어가 성공 로그에 함께 싣게 한다 —
동일 scope 를 공유하므로(순수 ASGI, scope 무복제) 핸들러가 채운 값을
응답 완료 후 여기서 읽는다. 미기록(비-NER 경로·예외)이면 기본 요약만 남긴다.
"""

import logging
import secrets
import time

from starlette.datastructures import MutableHeaders

logger = logging.getLogger(__name__)

# 상태코드 → 거절 사유(로그 태그). 응답 상태만으로 결정된다(핸들러 협조 불필요).
_REASONS = {
    400: 'bad_request',
    401: 'unauthorized',
    413: 'payload_too_large',
    422: 'validation_failed',
    429: 'overloaded',
    503: 'model_unavailable',
}


def _request_id(scope) -> str:
    """수신 `X-Request-ID` 를 재사용하고(추적 연속성), 없으면 8-hex 생성."""
    for name, value in scope.get('headers', []):
        if name == b'x-request-id' and value:
            return value.decode('latin-1')[:64]
    return secrets.token_hex(4)


def _emit(scope, status: int, latency_ms: float, rid: str, meta) -> None:
    """상태코드 계층에 따라 레벨·문구를 갈라 한 줄 남긴다."""
    path = scope.get('path', '')
    if status in _REASONS:
        # 알려진 거절(400/401/413/422/429/503) — 503(모델 미로드)도 5xx 지만
        # 서버 결함이 아니라 거절이라 사유 매핑을 먼저 본다.
        logger.warning('request rejected status=%d reason=%s path=%s rid=%s',
                       status, _REASONS[status], path, rid)
    elif status >= 500:
        # 미처리 서버 결함(500 등). 트레이스백은 app 예외 핸들러가 남기고
        # 여기선 요약만(대개 도달 안 함: 미처리 예외는 send 없이 전파돼 이
        # 경로를 안 탄다). 방어적 기록.
        logger.warning('request failed status=%d latency_ms=%.0f path=%s '
                       'rid=%s', status, latency_ms, path, rid)
    elif status >= 400:
        # 매핑 밖 4xx(404·405 등).
        logger.warning('request rejected status=%d reason=%s path=%s rid=%s',
                       status, 'error', path, rid)
    elif meta:
        logger.debug('request status=%d latency_ms=%.0f lang=%s batch=%d '
                     'entities=%d rid=%s', status, latency_ms, meta['lang'],
                     meta['batch'], meta['entities'], rid)
    else:
        logger.debug('request status=%d latency_ms=%.0f path=%s rid=%s',
                     status, latency_ms, path, rid)


class RequestLogMiddleware:
    """요청별 request-id·지연·결과를 한 줄로 남기는 순수 ASGI 미들웨어."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        rid = _request_id(scope)
        state = scope.setdefault('state', {})
        state['request_id'] = rid
        captured = {'status': 500}
        start = time.perf_counter()

        async def logging_send(message):
            if message['type'] == 'http.response.start':
                captured['status'] = message['status']
                MutableHeaders(scope=message).append('x-request-id', rid)
            await send(message)

        await self.app(scope, receive, logging_send)
        latency_ms = (time.perf_counter() - start) * 1000.0
        _emit(scope, captured['status'], latency_ms, rid, state.get('ner_meta'))
