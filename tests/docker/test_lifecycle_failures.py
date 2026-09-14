"""실제 lifecycle 셸의 실패 전파와 후속 Docker 호출을 검사한다."""

import json
import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROJECTS = [
    'vllm-gemma4-31b-awq-8bit', 'vllm-qwen38-27b-w4a16-awq',
]
_STUB = '''#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ['DOCKER_STUB_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if 'config' in args:
    print(json.dumps({'name': 'ner-server', 'services': {'ner-server': {
        'container_name': 'ner-server', 'ports': [{'published': '8008'}],
    }}}))
    sys.exit(0)
project = args[args.index('-p') + 1] if '-p' in args else 'ner-server'
command = 'down' if 'down' in args else 'up'
code = json.loads(os.environ['DOCKER_STUB_FAILURES']).get(
    project + ':' + command, 0,
)
if code:
    print(f'Docker failure: {project} {command} ({code})', file=sys.stderr)
sys.exit(code)
'''


@pytest.fixture
def run_script(tmp_path):
    """외부 Docker만 대체하고 셸의 종료 코드·출력·호출 순서를 돌려준다."""
    stub = tmp_path / 'docker'
    stub.write_text(_STUB, encoding='utf-8')
    stub.chmod(0o755)
    log = tmp_path / 'calls.jsonl'

    def run(script, *args, failures=None):
        env = dict(os.environ)
        env.pop('NER_SERVER_PROJECT', None)
        env.update({
            'PATH': f'{tmp_path}{os.pathsep}{env["PATH"]}',
            'DOCKER_STUB_LOG': str(log),
            'DOCKER_STUB_FAILURES': json.dumps(failures or {}),
        })
        proc = subprocess.run(
            ['bash', str(_ROOT / 'docker' / script), *args],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return proc, calls

    return run


@pytest.mark.parametrize('args,project', [
    ((), 'ner-server'), (('ner-alt',), 'ner-alt'),
])
def test_server_stop_preserves_failure(run_script, args, project):
    proc, calls = run_script(
        'server/stop.sh', *args, failures={f'{project}:down': 17},
    )
    assert proc.returncode == 17
    assert f'Docker failure: {project} down (17)' in proc.stderr
    assert 'Stopped:' not in proc.stdout
    assert len([call for call in calls if 'down' in call]) == 1


@pytest.mark.parametrize('command', ['down', 'up'])
def test_server_start_stops_on_failure(run_script, command):
    proc, calls = run_script(
        'server/start.sh', failures={f'ner-server:{command}': 17},
    )
    assert proc.returncode == 17
    assert f'Docker failure: ner-server {command} (17)' in proc.stderr
    assert '/health' not in proc.stdout
    assert not any('config' in call for call in calls)
    if command == 'down':
        assert not any('up' in call for call in calls)


@pytest.mark.parametrize('args,build', [((), True), (('--no-build',), False)])
def test_server_start_runs_up_after_successful_cleanup(run_script, args, build):
    proc, calls = run_script('server/start.sh', *args)
    assert proc.returncode == 0, proc.stderr
    assert 'down' in calls[0]
    assert 'up' in calls[1]
    assert ('--build' in calls[1]) is build
    assert '/health' in proc.stdout


@pytest.mark.parametrize('args', [(), ('first', 'second')])
@pytest.mark.parametrize('codes', [(0, 0), (17, 0), (0, 23), (17, 23)])
def test_vllm_stop_attempts_every_project_and_reports_failures(
    run_script, args, codes,
):
    projects = list(args) or _DEFAULT_PROJECTS
    failures = {f'{p}:down': code for p, code in zip(projects, codes)}
    proc, calls = run_script('vllm/stop.sh', *args, failures=failures)
    assert proc.returncode == next((code for code in codes if code), 0)
    assert [call[call.index('-p') + 1] for call in calls] == projects
    assert all('down' in call for call in calls)
    for project, code in zip(projects, codes):
        if code:
            assert f'Docker failure: {project} down ({code})' in proc.stderr
            assert f'Stopped: {project}' not in proc.stdout
        else:
            assert f'Stopped: {project}' in proc.stdout
    assert ('All vLLM projects stopped.' in proc.stdout) is (not any(codes))
