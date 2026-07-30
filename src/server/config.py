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
    max_length: int = 256          # 모델 토큰 한도(초과 시 chunk)
    max_chars: int = 20000         # 요청 1건 텍스트 char 상한
    max_batch: int = 64            # 배치 텍스트 개수 상한
    max_total_chars: int = 100000  # 배치 전체 char 합산 상한(작업량 가드)
    max_body_bytes: int = 2 * 1024 * 1024  # 요청 바디 바이트 상한(전송 가드)
    max_concurrency: int = 8       # 동시 추론 상한(전역 세마포어)
    max_queue: int = 32            # 대기 큐 깊이 상한(초과 시 429)
    acquire_timeout_s: float = 10.0  # 세마포어 대기 타임아웃(초과 429)
    api_key: Optional[str] = None  # None 이면 인증 비활성
    host: str = '0.0.0.0'
    port: int = 8008
    log_level: str = 'INFO'        # 루트 로그 레벨(DEBUG 로 요청별 상세 켬)
    log_file: str = '/tmp/ner-server.log'  # 회전 파일 로그 경로(빈 값=stderr만)
    # 웹 데모 UI용 한국어 번역(additive, 기본 비활성). 활성 시 백엔드를
    # 명시해야 한다 — /v1/ner 코어와는 독립.
    #
    # 백엔드별 키는 **미설정(None)과 설정을 구별해야** 한다. 기본값을 여기서
    # 채워버리면 "안 준 것"과 "그 값을 준 것"이 같아져, 백엔드에 안 맞는 키가
    # 왔는지(기준: 설정만 보고 무엇을 부르는지 안다)를 볼 수 없다. 그래서
    # 백엔드 전용 키는 None 으로 두고, 실효 기본값은 그 값을 아는 쪽이 넣는다 —
    # timeout·device 는 `translate.py`, 반복 억제 하한은 토크나이저를 쥔
    # `translate_nllb.py`(sentinel 토큰 길이에서 계산).
    translate_enabled: bool = False
    translate_backend: str = ''    # 활성 시 필수 — 'llm' | 'nllb'(기본값 없음)
    translate_model: str = ''      # 활성 시 필수(두 백엔드 공용 — "무슨 모델")
    translate_max_concurrency: int = 2  # 번역 동시 in-flight 상한(초과 429)
    # backend=llm 전용
    translate_base_url: Optional[str] = None  # llm 활성 시 필수
    translate_api_key: Optional[str] = None   # OpenAI 호환 키(vLLM 은 불필요)
    translate_timeout_s: Optional[float] = None  # LLM 호출 타임아웃(초)
    # backend=nllb 전용
    translate_device: Optional[str] = None    # 모델을 올릴 device
    translate_no_repeat_ngram: Optional[int] = None  # 반복 억제(0=끔)

    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """`NER_SERVER_*` 환경변수에서 설정을 읽는다(미설정 시 기본값)."""
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw else default

        def _float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw else default

        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw == '':
                return default
            return raw.strip().lower() in ('1', 'true', 'yes', 'on')

        def _opt(name: str) -> Optional[str]:
            """미설정·빈 값이면 None — 설정 여부를 값으로 구별한다."""
            return os.environ.get(name) or None

        def _opt_int(name: str) -> Optional[int]:
            raw = _opt(name)
            return int(raw) if raw is not None else None

        def _opt_float(name: str) -> Optional[float]:
            raw = _opt(name)
            return float(raw) if raw is not None else None

        return cls(
            model_root=os.environ.get(
                'NER_SERVER_MODEL_ROOT', DEFAULT_MODEL_ROOT),
            max_length=_int('NER_SERVER_MAX_LENGTH', 256),
            max_chars=_int('NER_SERVER_MAX_CHARS', 20000),
            max_batch=_int('NER_SERVER_MAX_BATCH', 64),
            max_total_chars=_int('NER_SERVER_MAX_TOTAL_CHARS', 100000),
            max_body_bytes=_int('NER_SERVER_MAX_BODY_BYTES', 2 * 1024 * 1024),
            max_concurrency=_int('NER_SERVER_MAX_CONCURRENCY', 8),
            max_queue=_int('NER_SERVER_MAX_QUEUE', 32),
            acquire_timeout_s=_float('NER_SERVER_ACQUIRE_TIMEOUT_S', 10.0),
            api_key=os.environ.get('NER_SERVER_API_KEY') or None,
            host=os.environ.get('NER_SERVER_HOST', '0.0.0.0'),
            port=_int('NER_SERVER_PORT', 8008),
            log_level=os.environ.get('NER_SERVER_LOG_LEVEL', 'INFO'),
            log_file=os.environ.get(
                'NER_SERVER_LOG_FILE', '/tmp/ner-server.log'),
            translate_enabled=_bool('NER_SERVER_TRANSLATE_ENABLED', False),
            translate_backend=os.environ.get(
                'NER_SERVER_TRANSLATE_BACKEND', '').strip().lower(),
            translate_model=os.environ.get('NER_SERVER_TRANSLATE_MODEL', ''),
            translate_max_concurrency=_int(
                'NER_SERVER_TRANSLATE_MAX_CONCURRENCY', 2),
            translate_base_url=_opt('NER_SERVER_TRANSLATE_BASE_URL'),
            translate_api_key=_opt('NER_SERVER_TRANSLATE_API_KEY'),
            translate_timeout_s=_opt_float('NER_SERVER_TRANSLATE_TIMEOUT_S'),
            translate_device=_opt('NER_SERVER_TRANSLATE_DEVICE'),
            translate_no_repeat_ngram=_opt_int(
                'NER_SERVER_TRANSLATE_NO_REPEAT_NGRAM'),
        )

    def model_dir(self, lang: str) -> str:
        """언어별 HF 모델 디렉토리 경로."""
        return os.path.join(self.model_root, lang, 'model')

    def thresholds_path(self, lang: str) -> str:
        """언어별 thresholds.json 경로(없을 수 있음 — graceful)."""
        return os.path.join(self.model_root, lang, 'thresholds.json')
