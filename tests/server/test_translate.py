"""번역 서비스 테스트 — 마스킹-복원 PII 보존 + 엔드포인트 계약.

핵심 불변식: PII 문자열은 (a) 번역기에 노출되지 않고, (b) 출력에서 원문 그대로
보존되거나(sentinel 생존), (c) 소실 시 훼손 없이 '부재'로 처리된다. 실 LLM 없이
stub translator 로 검증한다.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import ServerConfig
from server.translate import (
    DEFAULT_SENTINEL,
    SentinelFormat,
    TranslationResult,
    TranslationUnavailable,
    _mask,
    _restore,
    _sentinel,
)


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


def test_alternate_sentinel_format_round_trips():
    """괄호를 갈아끼워도 마스킹-복원 불변식은 그대로다.

    전용 NMT 처럼 lenticular bracket 이 어휘에 없는 번역기를 위해 표기를
    엔진 속성으로 뺐다 — 바뀌는 것은 감싸는 괄호뿐이고 nonce·번호는 같다.
    """
    ascii_fmt = SentinelFormat(left='[', right=']')
    text, spans, phone = _phone_record()
    masked, id2val = _mask(text, spans, ascii_fmt)
    assert _sentinel(0, ascii_fmt) in masked
    assert _sentinel(0) not in masked        # 기본 표기는 섞이지 않는다
    assert phone not in masked               # 번역기에 PII 미노출
    restored, n_ok, n_drop = _restore(masked, id2val, ascii_fmt)
    assert phone in restored and (n_ok, n_drop) == (1, 0)


def test_alternate_sentinel_leaves_default_format_text_alone():
    """ASCII 표기로 돌 때 원문의 `【…】` 자연 표기를 건드리지 않는다."""
    ascii_fmt = SentinelFormat(left='[', right=']')
    phone = '010-1234-5678'
    text = f'【重要】連絡先 {phone}'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans, ascii_fmt)
    restored, n_ok, n_drop = _restore(masked, id2val, ascii_fmt)
    assert '【重要】' in restored
    assert phone in restored and (n_ok, n_drop) == (1, 0)


def test_default_sentinel_unchanged_by_format_parameter():
    """기본 인자로 부르면 종전과 같은 lenticular 표기를 낸다(동작 불변)."""
    assert _sentinel(0).startswith('【') and _sentinel(0).endswith('】')
    assert _sentinel(3) == _sentinel(3, DEFAULT_SENTINEL)


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
    """ja/vi 외 lang → 400."""
    r = _client(translator=_EchoTranslator()).post(
        '/v1/translate', json={'text': 'hi', 'lang': 'en', 'spans': []})
    assert r.status_code == 400


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
