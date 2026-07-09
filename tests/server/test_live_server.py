"""실서버(네트워크) 스모크 — uvicorn 서브프로세스 기동 후 HTTP 검증.

in-process TestClient 와 달리 실제 포트 바인딩·네트워크 경로를 확인한다.
서버 기동이 모델 로드에 의존하므로 `/data` 모델이 없으면 모듈 전체 skip.
`live` 마커가 붙어 `uv run pytest -m live` 로 따로 돌릴 수도 있다.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from server.config import ServerConfig

_JA_DIR = ServerConfig().model_dir('ja')

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.path.isdir(_JA_DIR),
                       reason=f'ja model dir not present: {_JA_DIR}'),
]


def _free_port() -> int:
    """OS 가 할당한 빈 포트를 받아 반환(테스트 동시성·충돌 회피)."""
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope='module')
def base_url():
    """uvicorn 서버를 서브프로세스로 띄우고 준비될 때까지 대기 후 종료."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, '-m', 'server',
         '--host', '127.0.0.1', '--port', str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f'server exited early (code {proc.returncode})')
            try:
                if httpx.get(f'{url}/health', timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise RuntimeError('server did not become ready in time')
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health_live(base_url):
    """실서버 /health 가 200·status ok."""
    r = httpx.get(f'{base_url}/health', timeout=5)
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_ner_single_live(base_url):
    """실서버 단일 추론 — 자동감지 ja + offset 정합성."""
    text = '織田信長は東京都千代田区に住んでいた。'
    r = httpx.post(f'{base_url}/v1/ner', json={'text': text}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'ja'
    # 빈 결과를 통과시키지 않는다 — 아래 offset 루프는 entities 가 비면
    # 0회 돌아 vacuous 하게 참이 된다(모델이 아무것도 못 뽑아도 통과).
    assert body['entities'], 'expected at least one entity'
    for ent in body['entities']:
        assert text[ent['start_char']:ent['end_char']] == ent['text']


def test_bad_lang_live(base_url):
    """실서버 잘못된 lang → 400 + 구조화 에러."""
    r = httpx.post(f'{base_url}/v1/ner',
                   json={'text': 'x', 'lang': 'ko'}, timeout=10)
    assert r.status_code == 400
    assert 'error' in r.json()
