"""실모델 없이 live 검사 도구의 준비성·실패 진단·정리를 검증한다."""

import subprocess
from types import SimpleNamespace

import httpx
import pytest

import test_live_server as live
from server.config import ServerConfig


def test_missing_models_checks_all_languages_and_environment(tmp_path, monkeypatch):
    """JA만 있는 환경을 전체 모델 준비로 오인하지 않는다."""
    (tmp_path / 'ja' / 'model').mkdir(parents=True)
    monkeypatch.setenv('NER_SERVER_MODEL_ROOT', str(tmp_path))
    missing = live._missing_models(ServerConfig.from_env())
    assert missing == [str(tmp_path / 'vi' / 'model'),
                       str(tmp_path / 'ko' / 'model')]


def test_startup_exit_reports_code_and_log_path(tmp_path):
    """기동 실패에는 종료 코드와 로그 위치만 노출한다."""
    log = tmp_path / 'server.log'
    log.write_text('sensitive diagnostic')
    proc = SimpleNamespace(poll=lambda: 17, returncode=17)
    with pytest.raises(RuntimeError) as exc:
        live._wait_until_ready(proc, 'http://127.0.0.1:1', log)
    assert '17' in str(exc.value)
    assert str(log) in str(exc.value)
    assert 'sensitive diagnostic' not in str(exc.value)


@pytest.mark.parametrize('body', [{'status': 'degraded'}, {}])
def test_http_200_without_ready_status_times_out(body, tmp_path, monkeypatch):
    """포트가 열렸어도 모델이 준비되지 않으면 성공하지 않는다."""
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(live.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(live.time, 'sleep', lambda _: None)
    monkeypatch.setattr(live.httpx, 'get',
                        lambda *a, **k: httpx.Response(200, json=body))
    log = tmp_path / 'server.log'
    with pytest.raises(RuntimeError, match='ready') as exc:
        live._wait_until_ready(SimpleNamespace(poll=lambda: None),
                               'http://127.0.0.1:1', log, timeout=1)
    assert str(log) in str(exc.value)


def test_ready_status_returns(tmp_path, monkeypatch):
    """정상 health 응답이면 제한 시간 전에 준비를 마친다."""
    monkeypatch.setattr(live.httpx, 'get', lambda *a, **k:
                        httpx.Response(200, json={'status': 'ok'}))
    live._wait_until_ready(SimpleNamespace(poll=lambda: None),
                           'http://127.0.0.1:1', tmp_path / 'server.log')


def test_forced_stop_waits_for_process_reaping():
    """정상 종료 시간이 지나면 강제 종료 후 자식 프로세스를 회수한다."""
    calls = []

    def wait(timeout=None):
        calls.append(('wait', timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired('server', timeout)

    proc = SimpleNamespace(terminate=lambda: calls.append('terminate'),
                           kill=lambda: calls.append('kill'), wait=wait)
    live._stop_server(proc)
    assert calls == ['terminate', ('wait', 10), 'kill', ('wait', None)]


def test_fixture_keeps_startup_log_and_reaps_failed_child(tmp_path, monkeypatch):
    """서버 대신 실패하는 자식 프로세스로 실제 로그 보관·종료를 확인한다."""
    for lang in ('ja', 'vi', 'ko'):
        (tmp_path / lang / 'model').mkdir(parents=True)
    monkeypatch.setenv('NER_SERVER_MODEL_ROOT', str(tmp_path))
    original_popen = subprocess.Popen
    children = []

    def failing_server(command, **kwargs):
        child = original_popen(
            [live.sys.executable, '-c',
             'import sys; print("startup failure", file=sys.stderr); sys.exit(17)'],
            **kwargs)
        children.append(child)
        child.wait(timeout=10)
        return child

    monkeypatch.setattr(live.subprocess, 'Popen', failing_server)
    fixture = live.base_url.__wrapped__(SimpleNamespace(mktemp=lambda _: tmp_path))
    with pytest.raises(RuntimeError, match='code 17') as exc:
        next(fixture)
    log = tmp_path / 'startup.log'
    assert 'startup failure' in log.read_text()
    assert str(log) in str(exc.value)
    assert log.stat().st_mode & 0o777 == 0o600
    assert children[0].poll() == 17


def test_requested_translation_rejects_disabled_configuration(monkeypatch):
    """번역 검사를 요구했는데 서버 번역이 꺼져 있으면 성공하지 않는다."""
    monkeypatch.setenv('NER_SERVER_TRANSLATE_ENABLED', 'false')
    with pytest.raises(AssertionError, match='translation must be enabled'):
        live.test_ner_to_translation_live(None, 'ja', 'text', '090-1234-5678')
