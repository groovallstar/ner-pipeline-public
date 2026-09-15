"""lifecycle 스크립트가 compose 와 같은 실행 대상을 고르는지 고정한다.

`docker/server/` 의 세 스크립트는 한때 프로젝트·컨테이너·포트를 자기가 정했다.
compose 는 같은 디렉토리의 `.env` 를 읽는데 스크립트는 셸만 읽어서, 같은 설정을
두 곳이 다르게 풀었다 — `.env` 로 포트를 바꾸면 서버는 새 포트에 뜨는데 안내는
8008 을 가리키고, 컨테이너 이름을 바꾸면 `logs.sh` 가 없는 컨테이너를 찾았다.
증상이 에러가 아니라 "안내가 거짓말" 이라 조용하다.

그래서 값을 푸는 곳을 compose 하나로 줄였다. 이 파일은 그 구조가 유지되는지를
본다 — 스크립트가 `-p` 를 만들어 넘기지 않는지, `logs.sh` 가 컨테이너 이름이
아니라 서비스 이름으로 붙는지, health 안내 포트가 compose 가 돌려준 값인지.

검사는 두 층이다.

- `docker` 스텁을 `PATH` 앞에 두고 스크립트를 실제로 돌려 **넘어간 인자**를 본다.
  Docker 데몬도 이미지도 없이 돈다.
- docker CLI 가 있으면 **실제 compose 파일**에 `--env-file` 을 물려 해석 우선순위
  (셸 > env 파일 > compose 파일의 `name:`)가 그대로인지 본다. 스크립트가 이
  우선순위에 기대고 있으므로, compose 쪽이 바뀌면 여기서 걸려야 한다.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SERVER_DIR = _ROOT / 'docker' / 'server'
_COMPOSE_FILE = _SERVER_DIR / 'docker-compose.yml'
_SCRIPTS = ('start.sh', 'stop.sh', 'logs.sh', 'resolve.sh')

# 스텁이 인자를 이어 붙일 때 쓰는 구분자. 인자 안에 나올 수 없는 제어문자라야
# 공백·경로가 든 인자를 그대로 복원할 수 있다.
_ARG_SEP = '\x1f'

# compose 설정을 셸이 아니라 이 파일이 쥐게 하려고 비우는 키.
_ENV_KEYS = ('COMPOSE_PROJECT_NAME', 'NER_SERVER_CONTAINER', 'NER_SERVER_PORT')

_STUB = '''#!/usr/bin/env python3
import os
import sys

argv = sys.argv[1:]
with open(os.environ['DOCKER_STUB_LOG'], 'a', encoding='utf-8') as fh:
    fh.write('\\x1f'.join(argv) + '\\n')
if 'config' in argv:
    sys.stdout.write(os.environ.get('DOCKER_STUB_CONFIG', ''))
'''


def _config(name='ner-ko', container='ner-server-ko', port='9099'):
    """`docker compose config --format json` 이 돌려주는 모양의 최소 판."""
    return {
        'name': name,
        'services': {
            'ner-server': {
                'container_name': container,
                'ports': [{
                    'mode': 'ingress',
                    'target': int(port),
                    'published': port,
                    'protocol': 'tcp',
                }],
            },
        },
    }


@pytest.fixture
def run_script(tmp_path):
    """`PATH` 앞에 docker 스텁을 둔 채 스크립트를 돌리고 호출 인자를 돌려준다."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'docker'
    stub.write_text(_STUB, encoding='utf-8')
    stub.chmod(0o755)
    log = tmp_path / 'calls.log'
    log.write_text('', encoding='utf-8')

    def run(script, *args, config=None):
        env = dict(os.environ)
        env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
        env['DOCKER_STUB_LOG'] = str(log)
        env['DOCKER_STUB_CONFIG'] = '' if config is None else json.dumps(config)
        for key in _ENV_KEYS:
            env.pop(key, None)
        # cwd 를 스크립트 디렉토리 밖에 두어, 스크립트가 자기 위치를 스스로
        # 찾는지까지 함께 본다.
        proc = subprocess.run(
            ['bash', str(_SERVER_DIR / script), *args],
            cwd=str(tmp_path), env=env, capture_output=True, text=True,
        )
        calls = [
            line.split(_ARG_SEP)
            for line in log.read_text(encoding='utf-8').splitlines()
        ]
        log.write_text('', encoding='utf-8')
        return proc, calls

    return run


def test_start_never_builds_a_project_flag(run_script):
    """`-p` 를 스크립트가 만들면 compose 가 푼 프로젝트와 갈릴 수 있다."""
    proc, calls = run_script('start.sh', '--no-build', config=_config())

    assert proc.returncode == 0, proc.stderr
    compose_calls = [call for call in calls if call and call[0] == 'compose']
    assert compose_calls, calls
    for call in compose_calls:
        assert '-p' not in call, call


def test_start_health_hint_uses_the_port_compose_reported(run_script):
    proc, _ = run_script('start.sh', '--no-build', config=_config(port='9099'))

    assert 'localhost:9099/health' in proc.stdout
    assert '8008' not in proc.stdout
    assert 'project=ner-ko' in proc.stdout
    assert 'container=ner-server-ko' in proc.stdout


def test_start_says_nothing_rather_than_guessing_a_port(run_script):
    """해석에 실패했을 때 기본값을 채우면 그것이 다시 거짓 안내가 된다."""
    proc, _ = run_script('start.sh', '--no-build', config=None)

    assert proc.returncode == 0, proc.stderr
    assert '/health' not in proc.stdout
    assert '8008' not in proc.stdout
    assert 'config' in proc.stderr


def test_stop_without_argument_lets_compose_choose(run_script):
    proc, calls = run_script('stop.sh', config=_config(name='ner-ko'))

    assert proc.returncode == 0, proc.stderr
    down_calls = [call for call in calls if 'down' in call]
    assert down_calls, calls
    for call in down_calls:
        assert '-p' not in call, call
    assert 'Stopped: ner-ko' in proc.stdout


def test_stop_with_argument_targets_that_project(run_script):
    proc, calls = run_script('stop.sh', 'ner-alt', config=_config())

    assert proc.returncode == 0, proc.stderr
    down_calls = [call for call in calls if 'down' in call]
    assert down_calls, calls
    for call in down_calls:
        assert call[call.index('-p') + 1] == 'ner-alt', call
    assert 'Stopped: ner-alt' in proc.stdout


def test_logs_attaches_by_service_name_not_container_name(run_script):
    """서비스 이름은 `.env` 로 안 바뀌므로 컨테이너 이름을 알 필요가 없다."""
    proc, calls = run_script('logs.sh', config=_config(container='ner-server-ko'))

    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1, calls
    call = calls[0]
    assert call[0] == 'compose'
    assert 'logs' in call
    assert call[-1] == 'ner-server'
    assert 'ner-server-ko' not in call
    # cwd 가 다른 곳인데도 자기 옆의 compose 파일을 집는다.
    assert str(_COMPOSE_FILE) in call


def test_logs_with_argument_attaches_to_that_container(run_script):
    proc, calls = run_script('logs.sh', 'ner-server-manual', config=_config())

    assert proc.returncode == 0, proc.stderr
    assert calls == [['logs', '-f', 'ner-server-manual']]


@pytest.mark.parametrize('script', _SCRIPTS)
def test_script_parses(script):
    proc = subprocess.run(
        ['bash', '-n', str(_SERVER_DIR / script)], capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stderr


def _require_compose():
    if shutil.which('docker') is None:
        pytest.skip('docker CLI is unavailable')
    probe = subprocess.run(
        ['docker', 'compose', 'version'], capture_output=True, text=True,
    )
    if probe.returncode != 0:
        pytest.skip('docker compose plugin is unavailable')


def _resolved(env_file=None, shell_env=None):
    """실제 compose 파일을 해석해 (프로젝트, 컨테이너, 게시 포트)를 돌려준다."""
    cmd = ['docker', 'compose']
    if env_file is not None:
        cmd += ['--env-file', str(env_file)]
    cmd += ['-f', str(_COMPOSE_FILE), 'config', '--format', 'json']
    env = dict(os.environ)
    for key in _ENV_KEYS:
        env.pop(key, None)
    env.update(shell_env or {})
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads(proc.stdout)
    svc = cfg['services']['ner-server']
    return cfg['name'], svc['container_name'], svc['ports'][0]['published']


def test_compose_defaults_come_from_the_compose_file(tmp_path):
    """스크립트가 기본값을 안 들고 있으므로 `name:` 이 프로젝트 기본값이다."""
    _require_compose()

    env_file = tmp_path / 'empty.env'
    env_file.write_text('', encoding='utf-8')

    assert _resolved(env_file) == ('ner-server', 'ner-server', '8008')


def test_compose_reads_the_env_file(tmp_path):
    _require_compose()
    env_file = tmp_path / 'target.env'
    env_file.write_text(
        'COMPOSE_PROJECT_NAME=ner-ko\n'
        'NER_SERVER_CONTAINER=ner-server-ko\n'
        'NER_SERVER_PORT=9099\n',
        encoding='utf-8',
    )

    assert _resolved(env_file) == ('ner-ko', 'ner-server-ko', '9099')


def test_shell_environment_beats_the_env_file(tmp_path):
    """세 값이 같은 순서로 풀려야 스크립트가 한 대상만 가리킨다."""
    _require_compose()
    env_file = tmp_path / 'target.env'
    env_file.write_text(
        'COMPOSE_PROJECT_NAME=ner-ko\n'
        'NER_SERVER_CONTAINER=ner-server-ko\n'
        'NER_SERVER_PORT=9099\n',
        encoding='utf-8',
    )
    shell_env = {
        'COMPOSE_PROJECT_NAME': 'ner-shell',
        'NER_SERVER_CONTAINER': 'ner-server-shell',
        'NER_SERVER_PORT': '7007',
    }

    assert _resolved(env_file, shell_env) == (
        'ner-shell', 'ner-server-shell', '7007',
    )
