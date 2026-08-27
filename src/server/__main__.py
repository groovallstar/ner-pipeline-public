"""`python -m server` — uvicorn 으로 ja/ko/vi NER API 서버를 기동한다.

설정은 `NER_SERVER_*` 환경변수(기본값은 config 참조)를 따르고, host/port 는
커맨드라인으로도 덮어쓸 수 있다. 시작 시 지원 언어 모델을 모두 로드하며, 일부
언어 로드에 실패해도 서버는 떠서 해당 언어 요청에만 503 을 돌려준다.
"""

import argparse
import logging
import logging.handlers
import os

import uvicorn

from server.app import create_app
from server.config import ServerConfig
from server.inference import ModelRegistry
from server.translate import build_translator

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


def _configure_logging(config: ServerConfig) -> None:
    """루트 로거 구성 — stderr + (설정 시) 회전 파일 핸들러.

    파일 로그는 `NER_SERVER_LOG_FILE`(기본 `/tmp/ner-server.log`)에 남기고,
    매주 회전(TimedRotating, 월요일 기준)해 직전 1주치만 보관하고 그보다
    오래된 파일은 자동 삭제한다 — /tmp 에 로그가 무한정 쌓이지 않게 일주일
    단위로 정리. 빈 값이면 stderr 만. 파일 열기 실패(경로 권한 등)는 stderr
    로깅을 유지한 채 경고만 낸다 — 파일 로그 실패가 서버 기동을 막지 않게.
    """
    logging.basicConfig(level=config.log_level.upper(), format=_LOG_FORMAT)
    if not config.log_file:
        return
    try:
        directory = os.path.dirname(config.log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # when='W0' interval=1 → 매주 월요일 회전, backupCount=1 → 직전 1주치
        # 보관 후 자동 삭제(일주일 단위 정리). 크기 기반이 아니라 시간 기반.
        handler = logging.handlers.TimedRotatingFileHandler(
            config.log_file, when='W0', interval=1, backupCount=1,
            encoding='utf-8')
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            'file logging disabled (%s): %s', config.log_file, exc)


def main() -> None:
    config = ServerConfig.from_env()
    p = argparse.ArgumentParser(
        description='Run the ja/ko/vi NER REST API server (uvicorn).')
    p.add_argument('--host', default=config.host, help='Bind host')
    p.add_argument('--port', type=int, default=config.port, help='Bind port')
    p.add_argument('--model-root', default=config.model_root,
                   help='Root dir holding {lang}/model and {lang}/'
                        'thresholds.json')
    args = p.parse_args()
    config.host = args.host
    config.port = args.port
    config.model_root = args.model_root

    _configure_logging(config)

    registry = ModelRegistry.load(config)
    translator = build_translator(config)
    app = create_app(registry, config, translator)
    # access_log=False: 요청당 액세스 로그는 RequestLogMiddleware 가 소유한다
    # (이중 로그 방지). 성공 요청은 DEBUG 라 기본 INFO 에선 조용하다.
    uvicorn.run(app, host=config.host, port=config.port, access_log=False)


if __name__ == '__main__':
    main()
