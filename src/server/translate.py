"""웹 데모 UI용 한국어 번역 서비스 — 마스킹-복원 + 온프렘 LLM 백엔드.

`/v1/ner`·`inference.py` 와 독립된 additive 모듈이다. 입력 원문의 PII span 을
sentinel 로 마스킹해 원문 PII 를 번역기에 노출하지 않고, 번역 후 원본값으로
복원한다. 복원 시 sentinel 생존을 검증하며, 소실된 PII 는 훼손·유출 대신
'부재'로 처리한다(graceful degrade — 보존-또는-부재, 절대 훼손 없음). 고유명사
(PER/LOC/ORG/PROD/EVT)는 마스킹하지 않고 프롬프트 지시로 한글 음차한다.

PII 탐지는 이 모듈이 하지 않는다 — 호출자(웹 페이지)가 `/v1/ner` 결과 span 을
그대로 넘기고, 여기서는 그중 PII 라벨만 마스킹 대상으로 고른다.

sentinel 은 `【PII{nonce}_{i}】` 형식이다. lenticular bracket(벤치 생존율
100%)에 **프로세스별 랜덤 nonce** 를 더해, 자연 텍스트의 `【重要】`·`【N】`
표기는 물론 sentinel 형식 자체와의 우연·의도적 충돌까지 사실상 배제한다 —
nonce 는 출력에 남지 않아 클라이언트가 알 수 없으므로, 복원의 전역 문자열
치환이 원문의 sentinel-형식 텍스트를 덮어쓰는 오염을 막는다. 복원·잔여 제거는
이 nonce 형식만 대상으로 한다.

괄호 쌍은 `SentinelFormat` 으로 갈아끼울 수 있다 — 표기가 번역기의 토크나이저에
종속되기 때문이다. nonce 와 번호 부분은 고정이고 바뀌는 것은 감싸는 괄호뿐이다.

이 모듈은 마스킹-복원과 원격 LLM 백엔드(`LLMTranslator`)를 담고, 인프로세스
전용 NMT 백엔드는 `translate_nllb.py` 에 따로 둔다 — 그쪽은 torch·transformers
를 끌고 오므로 `backend=llm` 기동이 그 무게를 지지 않게 임포트를 미룬다.
"""
from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 마스킹(원문 그대로 보존) 대상 = canonical PII 5종. 고유명사는 마스킹하지
# 않고 번역기가 한글 음차한다.
PII_LABELS = frozenset({'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD'})

# sentinel 접두 — 프로세스별 랜덤 nonce. 클라이언트가 예측·재현할 수 없어
# 원문을 만들어 충돌시키는 게 불가능하다.
_SENTINEL_TAG = f'PII{secrets.token_hex(3)}_'
_LANG_NAME = {'ja': '일본어', 'vi': '베트남어'}

_PROMPT_TEMPLATE = (
    '다음 {lang_name} 문장을 한국어로 번역하세요.\n'
    '규칙:\n'
    '1) 문장에 있는 {left}…{right} 형태의 자리표시자(예: {left}PII…{right})는 '
    '위치와 형태를 절대 바꾸지 말고 그대로 두세요.\n'
    '2) 인물·지명·조직·제품명 등 고유명사는 한글로 음차하세요.\n'
    '3) 번역문만 출력하고 설명은 하지 마세요.\n\n'
    '문장: {text}'
)


@dataclass(frozen=True)
class SentinelFormat:
    """sentinel 을 감싸는 괄호 쌍 — 번역기 토크나이저에 종속되는 부분이다.

    기본값 lenticular bracket 은 LLM 경로에서 생존율 100% 다. 반면 전용 NMT
    처럼 SentencePiece 어휘에 그 글자가 없는 모델에서는 양쪽 괄호가 모두
    `<unk>` 로 죽어 sentinel 본체가 멀쩡해도 복원이 전량 실패한다. 그래서
    표기를 번역기 속성으로 두고 엔진이 고르게 한다.
    """

    left: str = '【'
    right: str = '】'

    def token(self, i: int) -> str:
        """i 번째 sentinel 문자열(프로세스 nonce 포함)."""
        return f'{self.left}{_SENTINEL_TAG}{i}{self.right}'

    def pattern(self) -> re.Pattern:
        """이 표기의 sentinel 만 잡는 정규식 — 자연 텍스트 괄호는 비대상."""
        return re.compile(
            rf'{re.escape(self.left)}{re.escape(_SENTINEL_TAG)}'
            rf'(\d+){re.escape(self.right)}')


DEFAULT_SENTINEL = SentinelFormat()


class TranslationUnavailable(Exception):
    """번역 백엔드 미가용 — 비활성·엔드포인트 실패·타임아웃."""


@dataclass
class TranslationResult:
    """번역 결과 + PII 보존 회계.

    dropped_pii > 0 이면 그 PII 는 출력에서 '부재'(훼손·유출 아님)다.
    """

    lang: str
    translation: str
    masked_pii: int
    restored_pii: int
    dropped_pii: int


def _sentinel(i: int, fmt: SentinelFormat = DEFAULT_SENTINEL) -> str:
    """sentinel 토큰 문자열(프로세스 nonce 포함)."""
    return fmt.token(i)


def _mask(text: str, spans: List[dict],
          fmt: SentinelFormat = DEFAULT_SENTINEL) -> tuple[str, Dict[int, str]]:
    """PII 라벨 span 을 sentinel 로 치환한다(오른쪽부터, offset 보존).

    반환: (마스킹된 텍스트, {sentinel_id: 원본 PII 문자열}). PII 가 없으면
    원문과 빈 매핑을 그대로 돌려준다. PII span 은 비중첩·범위 내여야 하며
    (`/v1/ner` 은 비중첩 span 만 생성), 위반 시 `ValueError` 를 올린다 —
    겹치는 span 을 그대로 마스킹하면 원본 슬라이스가 어긋나 PII 조각이
    번역기에 노출될 수 있으므로 방어적으로 거절한다.
    """
    pii = sorted((s for s in spans if s.get('label') in PII_LABELS),
                 key=lambda s: s['start_char'])
    prev_end = 0
    for s in pii:
        start, end = s['start_char'], s['end_char']
        if not (0 <= start < end <= len(text)):
            raise ValueError('pii span out of text bounds')
        if start < prev_end:
            raise ValueError('overlapping pii spans')
        prev_end = end
    id2val: Dict[int, str] = {
        i: text[s['start_char']:s['end_char']] for i, s in enumerate(pii)
    }
    masked = text
    # 오른쪽(뒤) span 부터 치환해야 앞쪽 offset 이 밀리지 않는다.
    for i in reversed(range(len(pii))):
        s = pii[i]
        masked = (masked[:s['start_char']] + fmt.token(i)
                  + masked[s['end_char']:])
    return masked, id2val


def _restore(translated: str, id2val: Dict[int, str],
             fmt: SentinelFormat = DEFAULT_SENTINEL) -> tuple[str, int, int]:
    """sentinel 을 원본 PII 로 복원한다.

    반환: (복원된 텍스트, 복원 성공 수, 소실 수). 번역 중 사라진 sentinel 은
    복원하지 못하며(그 PII 는 부재), 남은 무매핑·환각 sentinel(우리 형식
    `【PII\\d+】`)만 제거한다 — 입력에 원래 있던 `【N】` 자연 표기는 건드리지
    않는다.
    """
    out = translated
    restored = 0
    for i, value in id2val.items():
        token = fmt.token(i)
        if token in out:
            out = out.replace(token, value)
            restored += 1
    dropped = len(id2val) - restored
    # 매핑에 없는 잔여 sentinel(환각·부분 소실)은 화면 노이즈이므로 제거한다.
    # 우리 형식(nonce 포함)만 대상이라 자연 텍스트의 `【N】` 은 보존된다.
    out = fmt.pattern().sub('', out)
    return out, restored, dropped


class LLMTranslator:
    """온프렘 LLM(OpenAI 호환)로 마스킹-복원 번역을 수행한다."""

    def __init__(self, base_url: str, model: str,
                 api_key: Optional[str] = None, timeout_s: float = 30.0,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 sentinel: SentinelFormat = DEFAULT_SENTINEL) -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key or 'none',
                              timeout=timeout_s)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._sentinel = sentinel
        logger.info('LLMTranslator ready: %s @ %s', model, base_url)

    def available(self) -> bool:
        """번역 백엔드(vLLM 등) liveness — 짧은 타임아웃으로 1회 확인한다.

        웹 UI 가 페이지 로드 시 번역 버튼 활성 여부를 정하는 데 쓴다(폴링 아님).
        모델 목록 조회가 성공하면 가용으로 본다.
        """
        try:
            self._client.with_options(timeout=3.0).models.list()
            return True
        except Exception as exc:  # noqa: BLE001 - 미가용을 False 로 흡수
            logger.warning('translation backend not available: %s', exc)
            return False

    def translate(self, text: str, lang: str,
                  spans: List[dict]) -> TranslationResult:
        """원문을 마스킹→번역→복원해 한국어 결과를 낸다.

        LLM 호출 실패·타임아웃은 `TranslationUnavailable` 로 올린다. span 이
        비정상(겹침·범위 밖)이면 `_mask` 가 `ValueError` 를 올린다.
        """
        masked, id2val = _mask(text, spans, self._sentinel)
        prompt = _PROMPT_TEMPLATE.format(
            lang_name=_LANG_NAME.get(lang, lang), text=masked,
            left=self._sentinel.left, right=self._sentinel.right)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self._temperature, max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 백엔드 오류를 한 종류로 흡수
            # 원인(인증·타임아웃·모델명 오류 등)을 로그에 남기고 호출 측엔
            # 일반 503 만 노출한다(내부 정보 비노출).
            logger.warning('translation backend call failed: %s', exc)
            raise TranslationUnavailable(str(exc)) from exc
        raw = (resp.choices[0].message.content or '').strip()
        restored, n_ok, n_drop = _restore(raw, id2val, self._sentinel)
        if n_drop:
            logger.warning('translation dropped %d/%d PII sentinel(s)',
                           n_drop, len(id2val))
        return TranslationResult(
            lang=lang, translation=restored,
            masked_pii=len(id2val), restored_pii=n_ok, dropped_pii=n_drop,
        )


# 백엔드 전용 설정 키 — env 이름과 config 속성을 한 곳에 묶는다. 검증이 이
# 표만 읽으므로 키가 늘어도 규칙이 흩어지지 않는다. 공용 키(BACKEND·MODEL·
# MAX_CONCURRENCY)는 어느 쪽에도 안 들어간다.
_BACKEND_KEYS = {
    'llm': (
        ('NER_SERVER_TRANSLATE_BASE_URL', 'translate_base_url'),
        ('NER_SERVER_TRANSLATE_API_KEY', 'translate_api_key'),
        ('NER_SERVER_TRANSLATE_TIMEOUT_S', 'translate_timeout_s'),
    ),
    'nllb': (
        ('NER_SERVER_TRANSLATE_DEVICE', 'translate_device'),
        ('NER_SERVER_TRANSLATE_NO_REPEAT_NGRAM',
         'translate_no_repeat_ngram'),
    ),
}
BACKENDS = tuple(_BACKEND_KEYS)

# 백엔드 전용 키의 실효 기본값 — config 는 "안 준 것"을 None 으로 남겨야
# 하므로(§config) 기본값은 여기서 넣는다.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_DEVICE = 'cuda:0'
# 반복 억제(`NO_REPEAT_NGRAM`)의 기본값은 여기에 없다 — 켜는 것이 기본이되 그
# 강도의 하한이 sentinel 을 몇 토큰으로 쪼개느냐에 달려 있어, 토크나이저를 쥔
# `translate_nllb.resolve_no_repeat_ngram` 이 정한다.


def _resolve_backend(config) -> str:
    """번역 활성 설정을 검증하고 백엔드 이름을 돌려준다.

    설정만 보고 무엇을 부르는지 알 수 있어야 하므로 기본값으로 때우지 않고
    기동을 실패시킨다 — 미지정·오값, 공용 필수키 누락, 그리고 **고른 백엔드에
    안 맞는 키**(예: `nllb` + `BASE_URL`)가 대상이다. 마지막 것은 값이 조용히
    무시되면 설정 파일이 실제 동작과 어긋난 채 남기 때문이다.
    """
    backend = (getattr(config, 'translate_backend', '') or '').strip().lower()
    if not backend:
        raise ValueError(
            'NER_SERVER_TRANSLATE_BACKEND is required when translation is '
            f'enabled (one of: {", ".join(BACKENDS)})')
    if backend not in _BACKEND_KEYS:
        raise ValueError(
            f'unknown translation backend {backend!r} '
            f'(one of: {", ".join(BACKENDS)})')
    if not config.translate_model:
        raise ValueError(
            'NER_SERVER_TRANSLATE_MODEL is required when translation is '
            'enabled')
    foreign = sorted(
        env for other, keys in _BACKEND_KEYS.items() if other != backend
        for env, attr in keys if getattr(config, attr, None) is not None)
    if foreign:
        raise ValueError(
            f'{", ".join(foreign)} does not apply to translation backend '
            f'{backend!r}')
    if backend == 'llm' and not config.translate_base_url:
        raise ValueError(
            'NER_SERVER_TRANSLATE_BASE_URL is required for translation '
            "backend 'llm'")
    return backend


def build_translator(config):
    """config 로 translator 를 구성한다. 비활성이면 None(엔드포인트는 503).

    활성이면 `translate_backend` 가 구현을 고른다 — `llm` 은 원격 OpenAI 호환
    엔드포인트, `nllb` 는 인프로세스 전용 NMT. 두 구현은 같은 계약
    (`translate`·`available`)을 낸다. 설정 결함은 기동 시점에 올린다.
    """
    if not getattr(config, 'translate_enabled', False):
        return None
    backend = _resolve_backend(config)
    if backend == 'llm':
        timeout_s = config.translate_timeout_s
        return LLMTranslator(
            base_url=config.translate_base_url, model=config.translate_model,
            api_key=config.translate_api_key,
            timeout_s=DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s,
        )
    # torch·transformers 를 여기서 처음 끌어온다 — llm 경로는 안 지나간다.
    from server import translate_nllb
    return translate_nllb.NLLBTranslator(
        model_id=config.translate_model,
        device=config.translate_device or DEFAULT_DEVICE,
        no_repeat_ngram=config.translate_no_repeat_ngram,
    )
