"""서버 설정 — 환경변수·기본값 모음.

배포 토폴로지(보류)와 분리해, 런타임 동작만 환경변수로 조정한다. 모든
경로는 `model_root/{lang}/...` 배포 레이아웃(`/data/ner/{lang}/model`,
`/data/ner/{lang}/thresholds.json`)을 따른다.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

DEFAULT_MODEL_ROOT = '/data/ner'
SUPPORTED_LANGS: Tuple[str, ...] = ('ja', 'vi')


@dataclass
class ServerConfig:
    """서버 런타임 설정. `from_env()` 로 환경변수에서 구성한다."""

    model_root: str = DEFAULT_MODEL_ROOT
    langs: Tuple[str, ...] = SUPPORTED_LANGS
    precision: str = 'bf16'        # 추론 정밀도(fp32|bf16) — 출시 default
    max_length: int = 256          # 모델 토큰 한도(초과 시 chunk)
    max_chars: int = 20000         # 요청 1건 텍스트 char 상한
    max_batch: int = 64            # 배치 텍스트 개수 상한
    max_total_chars: int = 100000  # 배치 전체 char 합산 상한(작업량 가드)
    api_key: Optional[str] = None  # None 이면 인증 비활성
    host: str = '0.0.0.0'
    port: int = 8000

    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """`NER_SERVER_*` 환경변수에서 설정을 읽는다(미설정 시 기본값)."""
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw else default

        return cls(
            model_root=os.environ.get(
                'NER_SERVER_MODEL_ROOT', DEFAULT_MODEL_ROOT),
            precision=os.environ.get('NER_SERVER_PRECISION', 'bf16'),
            max_length=_int('NER_SERVER_MAX_LENGTH', 256),
            max_chars=_int('NER_SERVER_MAX_CHARS', 20000),
            max_batch=_int('NER_SERVER_MAX_BATCH', 64),
            max_total_chars=_int('NER_SERVER_MAX_TOTAL_CHARS', 100000),
            api_key=os.environ.get('NER_SERVER_API_KEY') or None,
            host=os.environ.get('NER_SERVER_HOST', '0.0.0.0'),
            port=_int('NER_SERVER_PORT', 8000),
        )

    def model_dir(self, lang: str) -> str:
        """언어별 HF 모델 디렉토리 경로."""
        return os.path.join(self.model_root, lang, 'model')

    def thresholds_path(self, lang: str) -> str:
        """언어별 thresholds.json 경로(없을 수 있음 — graceful)."""
        return os.path.join(self.model_root, lang, 'thresholds.json')
