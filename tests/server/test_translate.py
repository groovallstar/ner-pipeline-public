"""번역 서비스 테스트 — 마스킹-복원 PII 보존 + 설정 검증 + 엔드포인트 계약.

핵심 불변식: PII 문자열은 (a) 번역기에 노출되지 않고, (b) 출력에서 원문 그대로
보존되거나(sentinel 생존), (c) 소실 시 훼손 없이 '부재'로 처리된다. 실 LLM 없이
stub translator 로 검증한다.

설정 검증(`build_translator`)은 **설정만 보고 무엇을 부르는지 알 수 있는가**를
본다 — 기본값으로 때우지 않고 기동을 실패시키는지. 동시성은 번역 예산이 NER
예산과 분리돼 서로를 잠식하지 않는지를 양방향으로 본다.
"""

import asyncio
import re
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import server
from server.app import create_app
from server.config import SUPPORTED_LANGS, ServerConfig
from server.translate import (
    LANG_NAME,
    _SENTINEL_TAG,
    TRANSLATABLE_LANGS,
    TranslationResult,
    TranslationUnavailable,
    _mask,
    _restore,
    _sentinel,
    build_translator,
)

# 미지원 언어 픽스처 — 실재하되 지원 계획이 없는 코드. `TRANSLATABLE_LANGS
# <= set(SUPPORTED_LANGS)` 가 아래에서 단언되므로 이 한 줄이 번역 미지원도
# 함께 보장한다.
_UNSUPPORTED_LANG = 'th'
assert _UNSUPPORTED_LANG not in SUPPORTED_LANGS


def _phone_record():
    """전화번호 1건이 든 (text, spans) 를 만든다."""
    prefix = '연락처 '
    phone = '010-1234-5678'
    text = f'{prefix}{phone} 입니다'
    spans = [{
        'label': 'PHONE',
        'start_char': len(prefix),
        'end_char': len(prefix) + len(phone),
        'text': phone,
    }]
    return text, spans, phone


# ---------- 마스킹-복원 단위 ----------

def test_mask_hides_pii_from_translator():
    """마스킹된 텍스트에는 PII 원문이 없고 sentinel 로 대체된다."""
    text, spans, phone = _phone_record()
    masked, id2val = _mask(text, spans)
    assert _sentinel(0) in masked
    assert phone not in masked          # 번역기에 PII 미노출
    assert id2val == {0: phone}


def test_restore_preserves_pii_verbatim():
    """sentinel 이 살아있으면 PII 가 원문 그대로 복원된다."""
    text, spans, phone = _phone_record()
    masked, id2val = _mask(text, spans)
    # 번역기가 sentinel 을 보존한 채 주변만 바꾼 상황을 모사.
    translated = masked.replace('연락처', 'Contact').replace('입니다', 'is')
    restored, n_ok, n_drop = _restore(translated, id2val)
    assert phone in restored            # 원문 그대로 보존
    assert (n_ok, n_drop) == (1, 0)


def test_dropped_sentinel_absent_not_corrupted():
    """sentinel 소실 시 PII 는 훼손이 아니라 '부재'로 처리된다."""
    text, spans, phone = _phone_record()
    _, id2val = _mask(text, spans)
    translated = '연락처가 없습니다'      # sentinel 통째 소실
    restored, n_ok, n_drop = _restore(translated, id2val)
    assert phone not in restored        # 부재(훼손·다른 번호로 바뀜 없음)
    assert '【0】' not in restored        # 잔여 sentinel 노이즈 없음
    assert (n_ok, n_drop) == (0, 1)


def test_non_pii_spans_not_masked():
    """고유명사(PER 등)는 마스킹 대상이 아니다(음차는 번역기 몫)."""
    text = '도쿄 방문'
    spans = [{'label': 'LOC', 'start_char': 0, 'end_char': 2, 'text': '도쿄'}]
    masked, id2val = _mask(text, spans)
    assert masked == text
    assert id2val == {}


def test_multiple_pii_offsets_preserved():
    """여러 PII 도 오른쪽부터 치환해 offset 이 안 밀린다."""
    phone, card = '010-1111-2222', '4111111111111111'
    text = f'전화 {phone} 카드 {card} 끝'
    p0 = text.index(phone)
    p1 = text.index(card)
    spans = [
        {'label': 'PHONE', 'start_char': p0, 'end_char': p0 + len(phone),
         'text': phone},
        {'label': 'CREDIT_CARD', 'start_char': p1, 'end_char': p1 + len(card),
         'text': card},
    ]
    masked, id2val = _mask(text, spans)
    assert phone not in masked and card not in masked
    restored, n_ok, n_drop = _restore(masked, id2val)  # sentinel 그대로
    assert phone in restored and card in restored
    assert (n_ok, n_drop) == (2, 0)


def test_natural_bracket_text_preserved():
    """입력의 자연 `【N】`·`【重要】` 표기는 삭제되지 않는다(sentinel 접두 분리)."""
    phone = '010-1234-5678'
    text = f'【重要】連絡先 {phone} 【9】'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans)
    assert _sentinel(0) in masked and '【重要】' in masked and '【9】' in masked
    assert phone not in masked
    restored, n_ok, n_drop = _restore(masked, id2val)
    assert '【重要】' in restored and '【9】' in restored  # 자연 표기 보존
    assert phone in restored
    assert (n_ok, n_drop) == (1, 0)


def test_natural_bracket_no_pii_misplacement():
    """자연 `【0】` 이 있어도 실제 PII 가 그 위치로 오배치되지 않는다."""
    phone = '010-1234-5678'
    text = f'手順【0】確認後 電話 {phone}'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans)
    restored, n_ok, n_drop = _restore(masked, id2val)
    assert '【0】' in restored                     # 자연 【0】 보존
    assert restored.count(phone) == 1             # PII 는 원위치 1회만
    assert restored.index('【0】') < restored.index(phone)  # 순서 보존
    assert (n_ok, n_drop) == (1, 0)


def test_overlapping_spans_rejected():
    """겹치는 PII span 은 마스킹 전에 거절된다(원본 슬라이스 오염 방지)."""
    text = 'ID 12345678 끝'
    spans = [
        {'label': 'ID_NUM', 'start_char': 3, 'end_char': 11,
         'text': '12345678'},
        {'label': 'CREDIT_CARD', 'start_char': 6, 'end_char': 11,
         'text': '45678'},
    ]
    with pytest.raises(ValueError):
        _mask(text, spans)


def test_out_of_bounds_span_rejected():
    """범위를 벗어난 span 도 거절된다."""
    with pytest.raises(ValueError):
        _mask('짧다', [{'label': 'PHONE', 'start_char': 0, 'end_char': 99,
                       'text': 'x'}])


def test_literal_sentinel_lookalike_in_source_untouched():
    """원문에 sentinel 유사 문자열(`【PII0】`)이 있어도 오염되지 않는다.

    sentinel 은 프로세스 nonce 를 포함하므로 nonce 없는 유사 텍스트와 겹치지
    않는다 — 복원이 그 텍스트를 PII 로 덮어쓰거나 삭제하지 않는다.
    """
    phone = '010-1234-5678'
    text = f'参照コード【PII0】に電話 {phone}'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans)
    restored, n_ok, n_drop = _restore(masked, id2val)
    assert '【PII0】' in restored           # 유사 문자열 보존(오염 없음)
    assert restored.count(phone) == 1      # PII 는 원위치 1회만
    assert (n_ok, n_drop) == (1, 0)


def test_sentinel_uses_lenticular_brackets():
    """sentinel 표기는 프롬프트가 지시하는 `【…】` 그대로다(동작 불변)."""
    assert _sentinel(0).startswith('【') and _sentinel(0).endswith('】')
    assert _SENTINEL_TAG in _sentinel(0)     # nonce 가 붙어 충돌하지 않는다


# ---------- 엔드포인트 계약 ----------

class _Reg:
    """translate 테스트용 최소 registry(엔드포인트가 호출하지 않음)."""

    def health(self):
        return {'status': 'ok', 'langs': {}}

    def predict(self, *a, **k):
        return []

    def predict_batch(self, *a, **k):
        return []


class _EchoTranslator:
    """실 LLM 대신 마스킹 텍스트를 그대로 복원(=완벽한 sentinel 생존)."""

    def __init__(self, available=True):
        self._available = available

    def translate(self, text, lang, spans):
        masked, id2val = _mask(text, spans)
        restored, n_ok, n_drop = _restore(masked, id2val)
        return TranslationResult(lang, restored, len(id2val), n_ok, n_drop)

    def available(self):
        return self._available


class _RaisingTranslator:
    """백엔드 미가용을 모사."""

    def translate(self, text, lang, spans):
        raise TranslationUnavailable('backend down')


def _client(translator=None):
    return TestClient(create_app(_Reg(), ServerConfig(), translator))


def test_translate_disabled_returns_503():
    """translator 미주입(비활성) → 503."""
    text, spans, _ = _phone_record()
    r = _client(translator=None).post(
        '/v1/translate', json={'text': text, 'lang': 'ja', 'spans': spans})
    assert r.status_code == 503
    assert r.json()['error']['status'] == 503


def test_translate_echo_preserves_pii_through_endpoint():
    """엔드포인트 왕복에서도 PII 가 원문 그대로 유지된다."""
    text, spans, phone = _phone_record()
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate', json={'text': text, 'lang': 'vi', 'spans': spans})
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'vi'
    assert phone in body['translation']


def test_translate_unsupported_lang_400():
    """번역 대상 외 lang → 400."""
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate',
        json={'text': 'hi', 'lang': _UNSUPPORTED_LANG, 'spans': []})
    assert r.status_code == 400


def test_translate_accepts_en():
    """lang:'en' 이 200 으로 받아들여지고 PII span 이 보존된다."""
    phone = '010-1234-5678'
    r = _client(translator=_EchoTranslator()).post('/v1/translate', json={
        'text': f'Call Alice at {phone} tomorrow.',
        'lang': 'en',
        'spans': [{'label': 'PHONE', 'start_char': 14,
                   'end_char': 14 + len(phone), 'text': phone}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'en'
    assert phone in body['translation']


def test_translate_rejects_ko_even_though_ner_supports_it():
    """NER 이 받는 ko 를 번역은 400 으로 거절한다.

    한국어로 옮기는 기능이라 ko 원문은 옮길 곳이 없다. `SUPPORTED_LANGS`
    (ja·ko·vi·en)와 `TRANSLATABLE_LANGS`(ja·vi·en)가 갈리는 유일한 지점이고,
    같은 목록을 쓰면 ko 가 조용히 통과해 원문이 그대로 '번역'으로 나온다.
    """
    assert 'ko' in SUPPORTED_LANGS
    assert 'ko' not in TRANSLATABLE_LANGS
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate',
        json={'text': '김민준은 서울에 산다.', 'lang': 'ko', 'spans': []})
    assert r.status_code == 400


def test_translatable_langs_all_have_a_prompt_name():
    """번역 대상 목록과 프롬프트 이름표가 어긋날 수 없다(같은 출처)."""
    assert set(TRANSLATABLE_LANGS) == set(LANG_NAME)
    assert TRANSLATABLE_LANGS <= set(SUPPORTED_LANGS)


def test_web_ui_translatable_list_matches_the_server():
    """웹 UI 의 목록이 서버와 갈라지지 않는지.

    UI 는 vanilla JS 라 서버 상수를 import 할 수 없어 손으로 복사한 목록을
    쥔다. 갈라져도 증상이 조용하다 — 서버가 넓으면 버튼이 없는 언어가
    생기고, UI 가 넓으면 눌렀을 때 400 이 난다. 그래서 한쪽을 고칠 때 다른
    쪽을 잊는 것을 여기서 잡는다.
    """
    ui = (Path(server.__file__).parent / 'static' / 'index.html').read_text(
        encoding='utf-8')
    m = re.search(r'const TRANSLATABLE = new Set\(\[([^\]]*)\]\)', ui)
    assert m, 'web UI has no TRANSLATABLE list'
    declared = set(re.findall(r'"([a-z]{2})"', m.group(1)))
    assert declared == set(TRANSLATABLE_LANGS)


def test_web_ui_gloss_hint_names_every_translatable_lang():
    """번역 안내 문구 두 곳이 번역 대상 언어 이름을 전부 담는지.

    문구는 JS 상수와 정적 HTML 두 벌로 있어 한쪽만 고치기 쉽다. 어긋나도
    에러가 아니라 안내가 거짓이 되므로 조용하다.
    """
    ui = (Path(server.__file__).parent / 'static' / 'index.html').read_text(
        encoding='utf-8')
    const = re.search(r'const GLOSS_HINT =(.*?);', ui, re.S)
    assert const, 'index.html has no GLOSS_HINT constant'
    static = re.search(r'<p[^>]*id="glossStatus"[^>]*>(.*?)</p>', ui, re.S)
    assert static, 'index.html has no glossStatus paragraph'
    for blob in (const.group(1), static.group(1)):
        for name in LANG_NAME.values():
            assert name in blob, name


def test_translate_backend_unavailable_503():
    """백엔드 호출 실패 → 503(graceful degrade)."""
    text, spans, _ = _phone_record()
    r = _client(translator=_RaisingTranslator()).post(
        '/v1/translate', json={'text': text, 'lang': 'ja', 'spans': spans})
    assert r.status_code == 503


def test_translate_empty_spans_ok():
    """PII 가 없어도(빈 spans) 정상 번역 경로를 탄다."""
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate', json={'text': '도쿄 방문', 'lang': 'ja', 'spans': []})
    assert r.status_code == 200
    assert r.json()['translation'] == '도쿄 방문'


def test_translate_overlapping_spans_returns_400():
    """비정상 span(겹침)은 엔드포인트에서 400 으로 거절된다."""
    spans = [
        {'label': 'ID_NUM', 'start_char': 3, 'end_char': 11,
         'text': '12345678'},
        {'label': 'CREDIT_CARD', 'start_char': 6, 'end_char': 11,
         'text': '45678'},
    ]
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate', json={'text': 'ID 12345678 끝', 'lang': 'ja',
                               'spans': spans})
    assert r.status_code == 400


# ---------- 가용성(웹 UI 로드 1회 확인) ----------

def test_status_disabled_when_no_translator():
    """translator 미주입 → enabled/available 모두 False."""
    r = _client(translator=None).get('/v1/translate/status')
    assert r.status_code == 200
    assert r.json() == {'enabled': False, 'available': False}


def test_status_available_when_backend_up():
    """토글 ON + 백엔드 liveness OK → available True."""
    r = _client(translator=_EchoTranslator(available=True)).get(
        '/v1/translate/status')
    assert r.json() == {'enabled': True, 'available': True}


def test_status_unavailable_when_backend_down():
    """토글 ON 이지만 백엔드 미가용 → enabled True, available False."""
    r = _client(translator=_EchoTranslator(available=False)).get(
        '/v1/translate/status')
    assert r.json() == {'enabled': True, 'available': False}


# ---------- 설정 검증(기동 시점) ----------

def _cfg(**kw) -> ServerConfig:
    """번역 활성 config — 나머지 키는 테스트가 채운다."""
    return ServerConfig(translate_enabled=True, **kw)


def test_disabled_builds_nothing():
    """비활성이면 설정을 안 봐도 None(엔드포인트가 503 을 낸다)."""
    assert build_translator(ServerConfig()) is None


def test_model_is_required():
    """MODEL 은 번역 활성 시 필수다."""
    with pytest.raises(ValueError, match='MODEL'):
        build_translator(_cfg(translate_base_url='http://x/v1'))


def test_base_url_is_required():
    """BASE_URL 은 기본값이 없어 명시해야 뜬다.

    기본값을 두면 그 포트에 떠 있는 다른 모델을 조용히 번역기로 쓴다. 원격
    주소 없이 번역만 켜 둔 설정도 여기서 죽는다.
    """
    with pytest.raises(ValueError, match='BASE_URL'):
        build_translator(_cfg(translate_model='m'))


def test_valid_config_builds_translator():
    """필수 둘이 갖춰지면 원격 백엔드를 만든다."""
    translator = build_translator(
        _cfg(translate_model='m',
             translate_base_url='http://localhost:8081/v1'))
    assert type(translator).__name__ == 'LLMTranslator'


# ---------- 동시성(번역 예산은 NER 과 분리된다) ----------

class _BlockingTranslator:
    """슬롯을 붙잡고 있는 번역기 — 동시 점유 수를 관측한다."""

    def __init__(self):
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def translate(self, text, lang, spans):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.release.wait(timeout=5)
        with self._lock:
            self.in_flight -= 1
        return TranslationResult(lang, text, 0, 0, 0)

    def available(self):
        return True


class _BlockingReg(_Reg):
    """NER 슬롯을 붙잡고 있는 registry."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()

    def predict(self, *a, **k):
        self.entered.set()
        self.release.wait(timeout=5)
        return []


def _async_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url='http://test')


async def _wait_until(predicate, timeout_s=5.0):
    """조건이 참이 될 때까지 이벤트 루프를 양보하며 기다린다."""
    for _ in range(int(timeout_s / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def test_translate_concurrency_bounded_and_excess_rejected():
    """동시 in-flight ≤ 상한, 초과는 429 — NER 요청은 그대로 통과한다."""
    payload = {'text': '連絡先', 'lang': 'ja', 'spans': []}

    async def scenario():
        translator = _BlockingTranslator()
        app = create_app(_Reg(), ServerConfig(translate_max_concurrency=2),
                         translator)
        async with _async_client(app) as client:
            holders = [asyncio.create_task(
                client.post('/v1/translate', json=payload)) for _ in range(2)]
            saturated = await _wait_until(
                lambda: translator.in_flight == 2)
            excess = await client.post('/v1/translate', json=payload)
            ner = await client.post('/v1/ner',
                                    json={'text': '東京', 'lang': 'ja'})
            translator.release.set()
            held = await asyncio.gather(*holders)
        return (saturated, translator.max_in_flight, excess.status_code,
                ner.status_code, [r.status_code for r in held])

    saturated, max_in_flight, excess, ner, held = asyncio.run(scenario())
    assert saturated                 # 두 요청이 실제로 동시에 들어갔다
    assert max_in_flight == 2        # 상한을 넘지 않는다
    assert excess == 429             # 초과는 대기 없이 거절
    assert ner == 200                # 번역 포화가 NER 예산을 잠식하지 않는다
    assert held == [200, 200]


def test_ner_saturation_does_not_block_translation():
    """반대 방향 — NER 이 포화여도 번역은 자기 예산으로 돈다."""
    async def scenario():
        registry = _BlockingReg()
        config = ServerConfig(max_concurrency=1, max_queue=0,
                              translate_max_concurrency=2)
        app = create_app(registry, config, _EchoTranslator())
        async with _async_client(app) as client:
            held = asyncio.create_task(
                client.post('/v1/ner', json={'text': '東京', 'lang': 'ja'}))
            saturated = await _wait_until(registry.entered.is_set)
            ner_excess = await client.post('/v1/ner',
                                           json={'text': '大阪', 'lang': 'ja'})
            translated = await client.post(
                '/v1/translate',
                json={'text': '連絡先', 'lang': 'ja', 'spans': []})
            registry.release.set()
            await held
        return saturated, ner_excess.status_code, translated.status_code

    saturated, ner_excess, translated = asyncio.run(scenario())
    assert saturated                 # NER 슬롯이 실제로 점유된 상태에서
    assert ner_excess == 429         # NER 은 자기 상한에서 거절되고
    assert translated == 200         # 번역은 영향을 받지 않는다


@pytest.mark.parametrize('lang,text', [
    ('ja', '電話番号は090-1234-5678です。'),
    ('vi', 'Số điện thoại là 090-1234-5678.'),
])
@pytest.mark.parametrize('enabled', [True, False])
def test_ner_spans_feed_translation_without_exposing_phone(lang, text, enabled):
    """NER 응답을 번역에 연결하고 외부 호출의 마스킹 및 출력 복원을 검증한다."""
    import json

    from openai import OpenAI

    from server.translate import LLMTranslator

    phone = '090-1234-5678'

    class PhoneRegistry(_Reg):
        """가중치 추론만 대체하며 canonical PHONE 응답을 반환한다."""

        def predict(self, text, lang, apply_threshold=True):
            start = text.index(phone)
            return [{'label': 'PHONE', 'start_char': start,
                     'end_char': start + len(phone), 'text': phone, 'score': 1.0}]

    def complete(request):
        # HTTP 경계까지 실제 번역 구현을 실행하여 원문 PII 유출을 검출한다.
        payload = json.loads(request.content)
        prompt = payload['messages'][0]['content']
        assert phone not in prompt
        token = re.search(r'【PII[0-9a-f]+_\d+】', prompt)
        assert token is not None
        return httpx.Response(200, json={
            'id': 'test-completion', 'object': 'chat.completion', 'created': 0,
            'model': 'test-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant',
                            'content': '전화번호는 ' + token.group(0) + '입니다.'}}],
        })

    with httpx.Client(transport=httpx.MockTransport(complete)) as transport:
        translator = LLMTranslator(base_url='http://test.invalid/v1',
                                   model='test-model')
        translator._client.close()
        translator._client = OpenAI(base_url='http://test.invalid/v1',
                                    api_key='test', http_client=transport)
        with TestClient(create_app(PhoneRegistry(), ServerConfig(),
                                   translator if enabled else None)) as client:
            ner = client.post('/v1/ner', json={'text': text, 'lang': lang})
            assert ner.status_code == 200
            spans = ner.json()['entities']
            assert len(spans) == 1
            assert spans[0]['text'] == phone
            response = client.post('/v1/translate', json={
                'text': text, 'lang': ner.json()['lang'], 'spans': spans})
            if not enabled:
                assert response.status_code == 503
                return
            assert response.status_code == 200
            body = response.json()
            assert body['translation'] == '전화번호는 090-1234-5678입니다.'
            assert body['lang'] == lang
