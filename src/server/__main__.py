"""`python -m server` — uvicorn 으로 ja/vi NER API 서버를 기동한다.

설정은 `NER_SERVER_*` 환경변수(기본값은 config 참조)를 따르고, host/port 는
커맨드라인으로도 덮어쓸 수 있다. 시작 시 두 언어 모델을 모두 로드하며, 일부
언어 로드에 실패해도 서버는 떠서 해당 언어 요청에만 503 을 돌려준다.
"""

import argparse
import logging

import uvicorn

from server.app import create_app
from server.config import ServerConfig
from server.inference import ModelRegistry


def main() -> None:
    config = ServerConfig.from_env()
    p = argparse.ArgumentParser(
        description='Run the ja/vi NER REST API server (uvicorn).')
    p.add_argument('--host', default=config.host, help='Bind host')
    p.add_argument('--port', type=int, default=config.port, help='Bind port')
    p.add_argument('--model-root', default=config.model_root,
                   help='Root dir holding {lang}/model and {lang}/'
                        'thresholds.json')
    args = p.parse_args()
    config.host = args.host
    config.port = args.port
    config.model_root = args.model_root

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    registry = ModelRegistry.load(config)
    app = create_app(registry, config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == '__main__':
    main()
