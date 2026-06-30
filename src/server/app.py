"""FastAPI 앱 — ja/vi NER 단일·배치 엔드포인트.

요청은 원문 텍스트(단일 `text` 또는 배치 `texts`) + 선택 `lang`. lang 생략
시 텍스트별 자동 감지하고 응답에 감지 결과를 에코한다. 출력 span 은
canonical 형식. 핸들러는 무상태 — 모든 가변 상태는 주입된 registry 안에 있고
요청 처리는 그것을 읽기만 한다.
"""

import logging
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from server.concurrency import ConcurrencyGuard, Overloaded
from server.config import SUPPORTED_LANGS, ServerConfig
from server.detect import UNSUPPORTED, detect_lang
from server.inference import ModelUnavailable

logger = logging.getLogger(__name__)


class NERRequest(BaseModel):
    """NER 요청 — `text` 또는 `texts` 중 정확히 하나 + 선택 `lang`."""

    text: Optional[str] = None
    texts: Optional[List[str]] = None
    lang: Optional[str] = None


class Span(BaseModel):
    """canonical 엔티티 span."""

    label: str
    start_char: int
    end_char: int
    text: str
    score: Optional[float] = None


class SingleResponse(BaseModel):
    """단일 텍스트 응답 — 감지/지정 lang + 엔티티."""

    lang: str
    entities: List[Span]


class BatchItem(BaseModel):
    """배치 항목 — 텍스트별 lang + 엔티티."""

    lang: str
    entities: List[Span]


class BatchResponse(BaseModel):
    """배치 응답 — 입력 순서와 1:1."""

    results: List[BatchItem]


def _error(status: int, message: str) -> JSONResponse:
    """구조화 에러 응답 `{"error": {...}}`."""
    return JSONResponse(
        status_code=status,
        content={'error': {'status': status, 'message': message}},
    )


def create_app(registry, config: Optional[ServerConfig] = None) -> FastAPI:
    """registry(추론기)와 config 로 FastAPI 앱을 구성한다.

    registry 는 `predict(text, lang, abstain)` / `health()` 를 제공하는
    객체면 된다(실모델 또는 테스트 stub). config 미지정 시 기본값.
    """
    config = config or ServerConfig()
    app = FastAPI(title='NER API', version='1')
    guard = ConcurrencyGuard(config.max_concurrency, config.max_queue,
                             config.acquire_timeout_s)

    def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        """env API-key 가 설정된 경우에만 헤더를 검증한다(미설정 시 오픈)."""
        if config.api_key and x_api_key != config.api_key:
            raise HTTPException(
                status_code=401, detail='invalid or missing API key')

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(ModelUnavailable)
    async def _model_unavail(request: Request, exc: ModelUnavailable):
        return _error(503, str(exc))

    @app.exception_handler(Overloaded)
    async def _overloaded(request: Request, exc: Overloaded):
        return _error(429, str(exc))

    def _check_text(text: str) -> None:
        if len(text) > config.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f'text exceeds max_chars ({config.max_chars})')

    def _resolve_lang(text: str, given: Optional[str]) -> str:
        return given if given else detect_lang(text)

    def _infer_one(text: str, given: Optional[str],
                   abstain: bool) -> dict:
        """텍스트 1건 추론 — 미지원이면 모델 호출 없이 빈 결과 반환.

        자동 감지가 `unsupported`(ja·vi 신호 부재)면 200 으로 `{lang:
        'unsupported', entities: []}` 를 돌려준다(에러 아님). 배치에서
        항목별로 호출돼 부분 성공·응답 순서 1:1 을 보장한다. 명시 `lang`
        은 상위에서 SUPPORTED_LANGS 로 검증돼 unsupported 가 될 수 없다.
        """
        _check_text(text)
        lang = _resolve_lang(text, given)
        if lang == UNSUPPORTED:
            return {'lang': lang, 'entities': []}
        return {'lang': lang,
                'entities': registry.predict(text, lang, abstain)}

    @app.post('/v1/ner',
              response_model=Union[SingleResponse, BatchResponse],
              dependencies=[Depends(require_key)])
    async def ner(req: NERRequest, abstain: bool = Query(default=True)):
        """단일(`text`) 또는 배치(`texts`) NER 추론.

        추론은 전역 guard 안에서 run_in_threadpool 로 실행한다 — 동시
        in-flight 를 max_concurrency 로 묶고 과부하(큐/타임아웃 초과)는 429.
        검증·언어감지는 추론이 아니라 guard 밖에서 빠르게 처리한다.
        """
        if (req.text is None) == (req.texts is None):
            raise HTTPException(
                status_code=400,
                detail="provide exactly one of 'text' or 'texts'")
        if req.lang is not None and req.lang not in SUPPORTED_LANGS:
            raise HTTPException(
                status_code=400, detail=f"unsupported lang '{req.lang}'")

        if req.text is not None:
            return _infer_one(req.text, req.lang, abstain)

        assert req.texts is not None  # 위 oneof 검증이 보장 — 타입 narrowing
        if len(req.texts) > config.max_batch:
            raise HTTPException(
                status_code=413,
                detail=f'batch exceeds max_batch ({config.max_batch})')
        total_chars = sum(len(t) for t in req.texts)
        if total_chars > config.max_total_chars:
            raise HTTPException(
                status_code=413,
                detail=(f'batch total chars {total_chars} exceeds '
                        f'max_total_chars ({config.max_total_chars})'))
        results = [_infer_one(t, req.lang, abstain) for t in req.texts]
        return {'results': results}

    @app.get('/health')
    def health():
        """언어별 모델 로드 상태·임계값 존재 보고(인증 없음)."""
        return registry.health()

    return app
