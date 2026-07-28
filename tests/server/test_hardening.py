"""적대적 하드닝 회귀 테스트 — 바디 크기 상한·에러 봉투·상수시간 키 비교.

공격 표면 테스트에서 확인된 세 결함(전송 계층 메모리 고갈, 봉투 계약 이탈,
비상수시간 키 비교)의 수정을 stub registry 로 모델 없이 고정한다. 실서버·
GPU 불요 — 미들웨어·핸들러·인증 계약만 본다.
"""

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import ServerConfig


class StubRegistry:
    """첫 단어를 PER span 으로 반환. `__boom__` 은 미처리 예외를 유발한다."""

    def predict(self, text, lang, apply_threshold=True):
        if text == '__boom__':
            raise RuntimeError('SECRET=hunter2 internal path=/etc/creds')
        first = text.split(' ', 1)[0] if text else ''
        if not first:
            return []
        return [{'label': 'PER', 'start_char': 0, 'end_char': len(first),
                 'text': first, 'score': 1.0}]

    def predict_batch(self, texts, langs, apply_threshold=True):
        return [self.predict(t, lang, apply_threshold)
                for t, lang in zip(texts, langs)]

    def health(self):
        return {'status': 'ok', 'langs': {}}


def _client(config=None, raise_server_exceptions=True):
    return TestClient(
        create_app(StubRegistry(), config or ServerConfig()),
        raise_server_exceptions=raise_server_exceptions)


# ---------------------------------------------------------------------------
# AC1 — 바디 크기 상한(전송 계층 메모리 고갈 가드)
# ---------------------------------------------------------------------------
def test_body_under_limit_passes():
    """상한 이내 요청은 정상 200 — 미들웨어가 정상 경로를 막지 않는다."""
    cfg = ServerConfig(max_body_bytes=64 * 1024)
    r = _client(cfg).post('/v1/ner', json={'text': 'テスト', 'lang': 'ja'})
    assert r.status_code == 200


def test_content_length_over_limit_413_envelope():
    """Content-Length 가 상한 초과면 바디를 읽기 전에 413 + 봉투."""
    cfg = ServerConfig(max_body_bytes=64 * 1024)
    big = 'x' * (128 * 1024)
    r = _client(cfg).post('/v1/ner', json={'text': big, 'lang': 'ja'})
    assert r.status_code == 413
    err = r.json()['error']
    assert err['status'] == 413
    assert 'max_body_bytes' in err['message']


def test_chunked_over_limit_413():
    """Content-Length 없는 chunked 바디도 스트리밍 누적으로 413(우회 차단).

    Content-Length 만 검사하면 chunked transfer-encoding 으로 우회되므로,
    수신 바이트 누적이 상한을 넘는 순간 거절해야 한다.
    """
    cfg = ServerConfig(max_body_bytes=64 * 1024)

    def gen():
        yield b'{"lang":"ja","text":"' + b'x' * (200 * 1024)
        yield b'"}'

    r = _client(cfg).post('/v1/ner', content=gen(),
                          headers={'content-type': 'application/json'})
    assert r.status_code == 413


def test_body_limit_applies_before_auth():
    """바디 상한은 인증보다 앞단 — API-key 설정·미제시여도 초과 바디는 413.

    인증 실패(401)보다 크기 거절(413)이 먼저여야 비인증 공격자가 대용량
    바디로 메모리를 고갈시킬 수 없다.
    """
    cfg = ServerConfig(max_body_bytes=64 * 1024, api_key='secret')
    big = 'x' * (128 * 1024)
    r = _client(cfg).post('/v1/ner', json={'text': big, 'lang': 'ja'})
    assert r.status_code == 413  # 401 이 아니라 413 이 먼저


# ---------------------------------------------------------------------------
# AC2 — 상수시간 API 키 비교
# ---------------------------------------------------------------------------
def test_api_key_same_length_wrong_rejected():
    """실제 키와 길이가 같은 오답도 거절 — compare_digest 정상 동작."""
    cfg = ServerConfig(api_key='abcdef')
    c = _client(cfg)
    wrong = c.post('/v1/ner', json={'text': 'a', 'lang': 'ja'},
                   headers={'x-api-key': 'abcdeX'})  # 동일 길이·마지막만 상이
    assert wrong.status_code == 401
    right = c.post('/v1/ner', json={'text': 'a', 'lang': 'ja'},
                   headers={'x-api-key': 'abcdef'})
    assert right.status_code == 200


def test_api_key_empty_header_rejected():
    """빈 x-api-key 헤더도 401 — None/'' 을 상수시간 비교가 안전 처리."""
    cfg = ServerConfig(api_key='secret')
    r = _client(cfg).post('/v1/ner', json={'text': 'a', 'lang': 'ja'},
                          headers={'x-api-key': ''})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# AC3 — 에러 봉투 일관성
# ---------------------------------------------------------------------------
def test_pydantic_validation_wrapped_in_envelope():
    """Pydantic 타입 오류도 기본 422 {detail:[...]} 대신 봉투로 통일."""
    c = _client()
    for body in ({'text': 123, 'lang': 'ja'},
                 {'texts': 'not-a-list', 'lang': 'ja'},
                 {'texts': [None], 'lang': 'ja'}):
        r = c.post('/v1/ner', json=body)
        assert r.status_code == 422
        assert r.json()['error']['status'] == 422


def test_unknown_route_404_envelope():
    """라우트 미매칭도 봉투 — 라우터는 starlette 쪽 HTTPException 을 던진다.

    핸들러를 fastapi.HTTPException 에만 걸면 404·405 는 FastAPI 기본 핸들러로
    새어 `{"detail": ...}` 를 낸다. 부모 클래스에 걸어야 전 경로가 통일된다.
    """
    r = _client().post('/v1/nonexistent', json={'text': 'a'})
    assert r.status_code == 404
    assert r.json()['error']['status'] == 404
    assert 'detail' not in r.json()


def test_method_not_allowed_405_envelope_keeps_allow_header():
    """메서드 불일치도 봉투 — 단 405 의 `Allow` 헤더는 보존해야 한다."""
    r = _client().get('/v1/ner')
    assert r.status_code == 405
    assert r.json()['error']['status'] == 405
    assert 'POST' in r.headers['allow']


def test_unhandled_exception_wrapped_no_leak():
    """미처리 모델 예외는 500 봉투로 통일하고 내부 메시지를 노출하지 않는다."""
    r = _client(raise_server_exceptions=False).post(
        '/v1/ner', json={'text': '__boom__', 'lang': 'ja'})
    assert r.status_code == 500
    body = r.text
    assert r.json()['error']['status'] == 500
    assert 'hunter2' not in body
    assert '/etc/creds' not in body
    assert 'Traceback' not in body
