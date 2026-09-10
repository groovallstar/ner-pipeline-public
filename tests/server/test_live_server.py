"""실서버(네트워크) 스모크 — uvicorn 서브프로세스 기동 후 HTTP 검증.

in-process TestClient 와 달리 실제 포트 바인딩·네트워크 경로를 확인한다.
설정한 모델 루트에 지원 언어 모델 디렉터리가 없으면 해당 fixture를 skip.
`live` 마커가 붙어 `uv run pytest -m live` 로 따로 돌릴 수도 있다.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from server.config import SUPPORTED_LANGS, ServerConfig

pytestmark = pytest.mark.live

_UNSUPPORTED_LANG = 'th'
assert _UNSUPPORTED_LANG not in SUPPORTED_LANGS


def _free_port() -> int:
    """OS 가 할당한 빈 포트를 받아 반환(테스트 동시성·충돌 회피)."""
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _missing_models(config):
    """모델 사전 조건을 확인한다."""
    return [config.model_dir(lang) for lang in config.langs
            if not os.path.isdir(config.model_dir(lang))]


def _wait_until_ready(proc, url, log_path, timeout=120):
    """실서버 준비를 기다린다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f'server exited early (code {proc.returncode}); log: {log_path}')
        try:
            response = httpx.get(f'{url}/health', timeout=2, trust_env=False)
            if (response.status_code == 200
                    and response.json().get('status') == 'ok'):
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f'server did not become ready in time; log: {log_path}')


def _stop_server(proc):
    """테스트가 시작한 서버를 종료한다."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope='module')
def base_url(tmp_path_factory):
    """현재 설정으로 기동하고 로그를 보관하며 테스트 소유 프로세스만 종료한다."""
    config = ServerConfig.from_env()
    missing = _missing_models(config)
    if missing:
        pytest.skip('model directories not present: ' + ', '.join(missing))
    port = _free_port()
    log_path = tmp_path_factory.mktemp('live-server') / 'startup.log'
    env = os.environ.copy()
    # 공유 기본 로그 파일을 사용하지 않고 stdout/stderr를 한 파일로 보관한다.
    env['NER_SERVER_LOG_FILE'] = ''
    url = f'http://127.0.0.1:{port}'
    with log_path.open('wb') as log:
        log_path.chmod(0o600)
        proc = subprocess.Popen(
            [sys.executable, '-m', 'server', '--host', '127.0.0.1',
             '--port', str(port), '--model-root', config.model_root],
            env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_until_ready(proc, url, log_path)
            yield url
        finally:
            _stop_server(proc)


@pytest.fixture(scope='module')
def client(base_url):
    """기동에 사용한 인증 설정으로 로컬 서버를 호출한다."""
    key = ServerConfig.from_env().api_key
    headers = {'x-api-key': key} if key else {}
    with httpx.Client(base_url=base_url, headers=headers,
                      timeout=30, trust_env=False) as value:
        yield value


def test_health_live(client):
    """실서버 /health 가 200·status ok."""
    r = client.get('/health', timeout=5)
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_ner_single_live(client):
    """실서버 단일 추론 — 자동감지 ja + offset 정합성."""
    text = '織田信長は東京都千代田区に住んでいた。'
    r = client.post('/v1/ner', json={'text': text}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'ja'
    # 빈 결과를 통과시키지 않는다 — 아래 offset 루프는 entities 가 비면
    # 0회 돌아 vacuous 하게 참이 된다(모델이 아무것도 못 뽑아도 통과).
    assert body['entities'], 'expected at least one entity'
    for ent in body['entities']:
        assert text[ent['start_char']:ent['end_char']] == ent['text']


def test_bad_lang_live(client):
    """실서버 잘못된 lang → 400 + 구조화 에러."""
    r = client.post('/v1/ner',
                   json={'text': 'x', 'lang': _UNSUPPORTED_LANG}, timeout=10)
    assert r.status_code == 400
    assert 'error' in r.json()


@pytest.mark.skipif(os.environ.get('NER_SERVER_TEST_LIVE_TRANSLATE') != '1',
                    reason='set NER_SERVER_TEST_LIVE_TRANSLATE=1 for translation')
@pytest.mark.parametrize('lang,text,phone', [
    ('ja', '電話番号は090-1234-5678です。', '090-1234-5678'),
    ('vi', 'Số điện thoại là 0904-123-456.', '0904-123-456'),
])
def test_ner_to_translation_live(client, lang, text, phone):
    """실모델 NER span을 번역에 연결해 한국어 출력과 PII 보존을 확인한다."""
    assert ServerConfig.from_env().translate_enabled, 'translation must be enabled'
    status = client.get('/v1/translate/status')
    assert status.status_code == 200
    assert status.json()['enabled'] is True
    assert status.json()['available'] is True
    ner = client.post('/v1/ner', json={'text': text, 'lang': lang})
    assert ner.status_code == 200
    spans = ner.json()['entities']
    assert any(s['label'] == 'PHONE' and s['text'] == phone for s in spans)
    response = client.post('/v1/translate', json={
        'text': text, 'lang': ner.json()['lang'], 'spans': spans})
    assert response.status_code == 200
    body = response.json()
    assert body['lang'] == lang
    assert phone in body['translation']
    assert any('가' <= char <= '힣' for char in body['translation'])
    pii_labels = {'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'}
    for span in spans:
        if span['label'] in pii_labels:
            assert span['text'] in body['translation']


def test_en_fallback_live(client):
    """실서버에 영어 문장을 lang 없이 던지면 en 으로 감지되고 결과가 온다.

    모델 사전 조건은 공용 base_url fixture가 현재 설정으로 확인한다.
    """
    text = 'Barack Obama was born in Hawaii in 1961.'
    r = client.post('/v1/ner', json={'text': text}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'en'
    assert body['entities'], 'expected at least one entity'
    for ent in body['entities']:
        assert text[ent['start_char']:ent['end_char']] == ent['text']
