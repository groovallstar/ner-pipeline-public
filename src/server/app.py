"""FastAPI 앱 — ja/vi NER 단일·배치 엔드포인트.

요청은 원문 텍스트(단일 `text` 또는 배치 `texts`) + 선택 `lang`. lang 생략
시 텍스트별 자동 감지하고 응답에 감지 결과를 에코한다. 출력 span 은
canonical 형식. 핸들러는 무상태 — 모든 가변 상태는 주입된 registry 안에 있고
요청 처리는 그것을 읽기만 한다.
"""

import logging
import secrets
from pathlib import Path
from typing import List, Optional, Union

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from server.concurrency import ConcurrencyGuard, Overloaded
from server.config import SUPPORTED_LANGS, ServerConfig
from server.detect import UNSUPPORTED, detect_lang
from server.inference import ModelUnavailable
from server.limits import BodySizeLimitMiddleware
from server.request_log import RequestLogMiddleware
from server.translate import TranslationUnavailable

logger = logging.getLogger(__name__)

# 웹 데모 UI — 임포트 시 1회 읽어 재사용한다. 서버와 동일 출처로 서빙되므로
# 브라우저는 CORS 없이 /v1/ner 를 직접 호출한다(내부 개발·데모용). 자산
# 부재·읽기 실패는 폴백으로 흡수해 코어 API(/v1/ner·/health) 부팅을 막지 않는다.
try:
    _UI_HTML = (
        Path(__file__).resolve().parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')
except OSError:
    logger.warning('demo UI asset not found; serving placeholder at /')
    _UI_HTML = ('<!doctype html><meta charset="utf-8">'
                '<p>NER demo UI is unavailable.</p>')


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


class TranslateRequest(BaseModel):
    """번역 요청 — 원문 `text` + `lang`(ja/vi) + `/v1/ner` 결과 `spans`.

    호출자(웹 UI)가 `/v1/ner` 에서 받은 엔티티를 그대로 넘긴다. 그중 PII
    라벨만 마스킹 대상이며, 나머지는 무시된다(고유명사는 음차).
    """

    text: str
    lang: str
    spans: List[Span] = []


class TranslateResponse(BaseModel):
    """번역 응답 — lang + 한국어 글로스."""

    lang: str
    translation: str


class ErrorBody(BaseModel):
    """에러 봉투 내용 — 상태코드와 사람이 읽는 한 줄 사유."""

    status: int
    message: str


class ErrorResponse(BaseModel):
    """서버가 판정하는 모든 에러의 공통 봉투(`_error` 가 내는 형태).

    OpenAPI 의 422 선언을 이 모델로 덮기 위해 존재한다 — 선언을 안 덮으면
    FastAPI 기본 `HTTPValidationError`(`{"detail": [...]}`)가 남아, 봉투로
    통일된 실제 응답과 기계 계약이 어긋난다.
    """

    error: ErrorBody


def _error(status: int, message: str) -> JSONResponse:
    """구조화 에러 응답 `{"error": {...}}`."""
    return JSONResponse(
        status_code=status,
        content={'error': {'status': status, 'message': message}},
    )


def _lang_summary(langs: List[str]) -> str:
    """배치 로그용 언어 요약 — 중복 제거 후 '+' 결합(예: `ja+vi`)."""
    seen: List[str] = []
    for lang in langs:
        if lang not in seen:
            seen.append(lang)
    return '+'.join(seen) if seen else '-'


# OpenAPI/Swagger 노출용 — 외부 소비자를 위한 사용법만 담는다. 구현 세부는
# 핸들러 docstring 에 두고, Swagger 에는 아래 요약·설명만 노출한다.
_NER_SUMMARY = '텍스트에서 개체명 추출 (일본어·베트남어)'

_NER_DESCRIPTION = (
    '일본어(`ja`)·베트남어(`vi`) 텍스트에서 개체명(인물·장소·조직 등)과 '
    '그 위치를 추출합니다.\n\n'
    '**요청** — `text`(단일 문장) 또는 `texts`(여러 문장 배치) 중 하나를 '
    '보냅니다. 둘 다 넣거나 둘 다 비우면 400 입니다. `lang` 은 선택이며, '
    '생략하면 자동 감지합니다(`ja`·`vi` 외 값은 400).\n\n'
    '**응답** — 개체마다 `label`(종류), `start_char`·`end_char`(원문 글자 '
    '위치, 시작 포함·끝 제외), `text`(해당 글자)를 돌려줍니다. 배치 응답 '
    '`results` 는 입력 순서와 1:1 입니다.\n\n'
    '**참고** — 일본어·베트남어가 아닌 텍스트는 에러가 아니라 '
    '`{"lang":"unsupported","entities":[]}` (200) 로 응답합니다. 여러 줄 '
    '텍스트는 문자열을 직접 잇지 말고 JSON 인코더로 보내세요(개행을 escape '
    '하지 않으면 400/422).'
)

# Swagger "Try it out" 용 실행 가능한 예제 — 각 항목은 text/texts 택일을
# 지켜 그대로 Execute 하면 200 이 나온다(Swagger 에 예제 dropdown 으로 노출).
_NER_BODY_EXAMPLES = {
    'single': {
        'summary': '단일 텍스트 — 언어 자동 감지(일본어)',
        'value': {'text': '織田信長は東京都千代田区に住んでいた。'},
    },
    'batch': {
        'summary': '배치 — 혼합 언어(텍스트별 감지)',
        'value': {
            'texts': ['トヨタは日本の会社です。', 'Hà Nội là thủ đô.'],
        },
    },
    'lang_specified': {
        'summary': '언어 명시(자동 감지 대신 직접 지정)',
        'value': {
            'text': 'アップルは2007年にiPhoneを発売した。',
            'lang': 'ja',
        },
    },
}


def create_app(registry, config: Optional[ServerConfig] = None,
               translator=None) -> FastAPI:
    """registry(추론기)와 config 로 FastAPI 앱을 구성한다.

    registry 는 `predict(text, lang)` / `health()` 를 제공하는
    객체면 된다(실모델 또는 테스트 stub). config 미지정 시 기본값.
    translator 는 `translate(text, lang, spans)` 를 가진 객체(또는 None) —
    None 이면 `/v1/translate` 는 503(번역 비활성)을 낸다.
    """
    config = config or ServerConfig()
    # defaultModelsExpandDepth=-1 → Swagger UI 하단 Schemas(모델 목록) 섹션을
    # 숨긴다(연동 노이즈 제거). 스키마는 openapi.json·엔드포인트엔 그대로 남는다.
    app = FastAPI(
        title='NER API', version='1',
        swagger_ui_parameters={'defaultModelsExpandDepth': -1})
    # 라우팅·인증보다 앞서 바디 바이트를 bound — 전송 계층 메모리 고갈 가드.
    app.add_middleware(BodySizeLimitMiddleware,
                       max_bytes=config.max_body_bytes)
    # 요청 로그 미들웨어는 가장 바깥(마지막 add = 최외곽)에 둔다 — 전송 계층
    # 413 을 포함한 최종 상태·전체 지연을 관측하고 request-id 를 심는다.
    app.add_middleware(RequestLogMiddleware)
    guard = ConcurrencyGuard(config.max_concurrency, config.max_queue,
                             config.acquire_timeout_s)

    def require_key(x_api_key: Optional[str] = Header(
            default=None, include_in_schema=False)) -> None:
        """env API-key 가 설정된 경우에만 헤더를 검증한다(미설정 시 오픈).

        인증 off 가 기본이라 `x-api-key` 헤더는 Swagger/OpenAPI 파라미터에
        노출하지 않는다(검증 로직 자체는 키 설정 시 그대로 동작). 비교는
        secrets.compare_digest 로 상수시간 — 키에 대한 타이밍 사이드채널을
        차단한다(bytes 로 인코딩해 임의 유니코드 입력에도 TypeError 없이).
        """
        if config.api_key:
            provided = (x_api_key or '').encode('utf-8')
            expected = config.api_key.encode('utf-8')
            if not secrets.compare_digest(provided, expected):
                raise HTTPException(
                    status_code=401, detail='invalid or missing API key')

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError):
        # Pydantic 검증 실패도 구조화 봉투로 통일한다(기본 422 {detail:[...]}
        # 대신). 필드 내부 구조는 노출하지 않고 개수만 요약한다.
        return _error(422, f'request validation failed ({len(exc.errors())} '
                           'error(s))')

    @app.exception_handler(ModelUnavailable)
    async def _model_unavail(request: Request, exc: ModelUnavailable):
        return _error(503, str(exc))

    @app.exception_handler(Overloaded)
    async def _overloaded(request: Request, exc: Overloaded):
        return _error(429, str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # 미처리 예외(모델 forward 오류 등)를 봉투로 통일하고 내부 예외
        # 메시지·트레이스백은 응답에 노출하지 않는다(서버 로그에만 기록).
        # request-id 를 실어 요청 로그 미들웨어 라인과 상관지을 수 있게 한다.
        rid = getattr(request.state, 'request_id', '-')
        logger.exception('unhandled error during request rid=%s', rid)
        return _error(500, 'internal server error')

    def _check_text(text: str) -> None:
        if len(text) > config.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f'text exceeds max_chars ({config.max_chars})')

    def _resolve_lang(text: str, given: Optional[str]) -> str:
        return given if given else detect_lang(text)

    def _resolve_one(text: str, given: Optional[str]) -> str:
        """텍스트 1건 크기 검증 + lang 결정(미지원 포함)."""
        _check_text(text)
        return _resolve_lang(text, given)

    @app.post('/v1/ner',
              summary=_NER_SUMMARY,
              description=_NER_DESCRIPTION,
              response_model=Union[SingleResponse, BatchResponse],
              responses={422: {
                  'model': ErrorResponse,
                  'description': 'JSON 스키마 검증 실패'}},
              dependencies=[Depends(require_key)])
    async def ner(request: Request,
                  req: NERRequest = Body(openapi_examples=_NER_BODY_EXAMPLES)):
        """단일(`text`) 또는 배치(`texts`) NER 추론.

        추론은 전역 guard 안에서 run_in_threadpool 로 실행해 동시 in-flight 를
        max_concurrency 로 묶고 과부하(큐/타임아웃 초과)는 429 로 거절한다.
        검증·언어감지는 guard 밖에서 빠르게 처리하고, 자동감지 `unsupported`
        (ja·vi 신호 부재)는 모델을 호출하지 않고 200 으로 빈 결과를 준다.
        배치는 지원 언어 항목만 언어별 forward 로 묶고(predict_batch) 미지원은
        빈 결과로 둬 입력 순서·lang 1:1 을 보존한다(부분 성공).

        신뢰도 임계값은 모델이 임계값 파일을 로드한 경우 자동 적용된다
        (요청 파라미터 없음 — 서빙은 모델의 운영점을 그대로 낸다).
        """
        if (req.text is None) == (req.texts is None):
            raise HTTPException(
                status_code=400,
                detail="provide exactly one of 'text' or 'texts'")
        if req.lang is not None and req.lang not in SUPPORTED_LANGS:
            raise HTTPException(
                status_code=400, detail=f"unsupported lang '{req.lang}'")

        if req.text is not None:
            lang = _resolve_one(req.text, req.lang)
            if lang == UNSUPPORTED:
                request.state.ner_meta = {
                    'lang': lang, 'batch': 1, 'entities': 0}
                return {'lang': lang, 'entities': []}
            async with guard:
                entities = await run_in_threadpool(
                    registry.predict, req.text, lang)
            request.state.ner_meta = {
                'lang': lang, 'batch': 1, 'entities': len(entities)}
            return {'lang': lang, 'entities': entities}

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
        langs = [_resolve_one(t, req.lang) for t in req.texts]
        results = [{'lang': lang, 'entities': []} for lang in langs]
        # 지원 언어 항목만 언어별 배치 forward, 미지원은 빈 결과로 둔다.
        sup = [(i, req.texts[i], lang) for i, lang in enumerate(langs)
               if lang != UNSUPPORTED]
        if sup:
            sup_texts = [t for _, t, _ in sup]
            sup_langs = [lang for _, _, lang in sup]
            async with guard:
                ents = await run_in_threadpool(
                    registry.predict_batch, sup_texts, sup_langs)
            for (i, _, lang), e in zip(sup, ents):
                results[i] = {'lang': lang, 'entities': e}
        request.state.ner_meta = {
            'lang': _lang_summary(langs),
            'batch': len(req.texts),
            'entities': sum(len(r['entities']) for r in results),
        }
        return {'results': results}

    # 웹 데모 UI 전용 — 외부 소비자용 OpenAPI 명세(=/v1/ner)에 노출하지 않는다
    # (include_in_schema=False). `/v1/ner` 계약과 독립.
    @app.post('/v1/translate',
              response_model=TranslateResponse,
              dependencies=[Depends(require_key)],
              include_in_schema=False)
    async def translate(req: TranslateRequest = Body(...)):
        """마스킹-복원 번역. 추론과 독립이라 NER guard 를 공유하지 않는다.

        translator 미주입(비활성) 시 503. lang 은 ja/vi 만 허용하고, 텍스트
        크기는 NER 과 동일 상한을 재사용한다. 백엔드 호출 실패는 503 으로
        graceful degrade — 호출 측(UI)은 글로스를 생략하고 NER 은 그대로 둔다.
        """
        if translator is None:
            raise HTTPException(
                status_code=503, detail='translation is not enabled')
        if req.lang not in SUPPORTED_LANGS:
            raise HTTPException(
                status_code=400, detail=f"unsupported lang '{req.lang}'")
        _check_text(req.text)
        spans = [s.model_dump() for s in req.spans]
        try:
            result = await run_in_threadpool(
                translator.translate, req.text, req.lang, spans)
        except ValueError:
            # 비정상 span(겹침·범위 밖) — 클라이언트 계약 위반.
            raise HTTPException(status_code=400, detail='invalid spans')
        except TranslationUnavailable:
            raise HTTPException(
                status_code=503, detail='translation backend unavailable')
        return {'lang': req.lang, 'translation': result.translation}

    @app.get('/v1/translate/status', include_in_schema=False)
    async def translate_status():
        """번역 가용성 — 웹 UI 가 페이지 로드 시 1회 조회해 버튼을 켠다.

        `enabled` = 서버 번역 토글. `available` = 토글 ON + 백엔드(vLLM)
        liveness 확인 성공. 폴링용이 아니라 로드 1회용이며, OpenAPI 에는
        노출하지 않는다(웹 UI 전용).
        """
        if translator is None:
            return {'enabled': False, 'available': False}
        available = await run_in_threadpool(translator.available)
        return {'enabled': True, 'available': available}

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    def ui():
        """내부 개발·데모용 NER 추론 웹 페이지(자족적 HTML, 인증 없음).

        동일 출처로 /v1/ner 를 호출하는 vanilla JS 페이지를 그대로 반환한다.
        API 스키마엔 노출하지 않는다(JSON API 가 아니라 브라우저 UI).
        """
        return _UI_HTML

    @app.get('/health', include_in_schema=False)
    def health():
        """언어별 모델 로드 상태·임계값 존재 보고(인증 없음).

        운영·오케스트레이터 헬스체크 전용이라 OpenAPI/Swagger 에는 노출하지
        않는다(엔드포인트 자체는 정상 동작).
        """
        return registry.health()

    return app
