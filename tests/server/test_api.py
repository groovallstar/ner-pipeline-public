"""API 계약·전송 테스트 — stub 추론기로 모델 없이 CI 가능.

실모델을 로드하지 않고 registry stub 을 주입해 엔드포인트 계약(스키마·순서·
lang 에코·감지·에러·인증·헬스)만 검증한다.
"""

import re

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import SUPPORTED_LANGS, ServerConfig
from server.inference import ModelUnavailable

# 미지원 언어 픽스처 — 실재하되 지원 계획이 없는 코드. 리터럴을 파일 곳곳에
# 흩어 두면 언어를 늘릴 때 조용히 뜻을 잃으므로 한 곳에 모으고, 전제를 이
# 파일이 스스로 단언한다.
_UNSUPPORTED_LANG = 'th'
assert _UNSUPPORTED_LANG not in SUPPORTED_LANGS


class StubRegistry:
    """결정적 stub — 첫 단어를 PER span 으로 반환(순서·offset 검증)."""

    def __init__(self, langs=('ja', 'vi', 'ko', 'en'), unavailable=()):
        self._langs = langs
        self._unavailable = set(unavailable)

    def predict(self, text, lang, apply_threshold=True):
        if lang in self._unavailable:
            raise ModelUnavailable(f'model for lang {lang!r} is not loaded')
        first = text.split(' ', 1)[0] if text else ''
        if not first:
            return []
        return [{
            'label': 'PER',
            'start_char': 0,
            'end_char': len(first),
            'text': first,
            'score': 1.0,
        }]

    def predict_batch(self, texts, langs, apply_threshold=True):
        return [self.predict(t, lang, apply_threshold)
                for t, lang in zip(texts, langs)]

    def health(self):
        return {'status': 'ok',
                'langs': {lang: {'loaded': True, 'thresholds': False}
                          for lang in self._langs}}


def _client(registry=None, config=None):
    return TestClient(create_app(registry or StubRegistry(),
                                 config or ServerConfig()))


def test_single_schema():
    """단일 응답 = {lang, entities:[canonical span]}."""
    r = _client().post('/v1/ner', json={'text': 'Alice went home',
                                         'lang': 'vi'})
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'vi'
    assert body['entities'] == [{
        'label': 'PER', 'start_char': 0, 'end_char': 5,
        'text': 'Alice'}]


def test_batch_order_one_to_one():
    """배치 results 는 입력 순서와 1:1."""
    r = _client().post('/v1/ner', json={
        'texts': ['aaa x', 'bbbbb y', 'cc z'], 'lang': 'vi'})
    assert r.status_code == 200
    results = r.json()['results']
    assert [it['entities'][0]['text'] for it in results] == [
        'aaa', 'bbbbb', 'cc']


def test_batch_routes_through_predict_batch():
    """배치는 predict_batch(언어별 묶음)로 가야 한다 — 항목별 predict 회귀 가드.

    predict_batch 만 'B' 마커를 내므로, 배치가 항목별 단건 predict 로 돌면
    (회귀) 첫 단어 'x'/'y' 가 나와 깨진다 — app↔배치 forward 연결을 고정한다.
    """
    class _BatchMarker(StubRegistry):
        def predict_batch(self, texts, langs, apply_threshold=True):
            return [[{'label': 'PER', 'start_char': 0, 'end_char': 1,
                      'text': 'B', 'score': 1.0}] for _ in texts]

    r = _client(registry=_BatchMarker()).post(
        '/v1/ner', json={'texts': ['x', 'y'], 'lang': 'vi'})
    assert [it['entities'][0]['text']
            for it in r.json()['results']] == ['B', 'B']


def test_lang_autodetect_echoed():
    """lang 생략 시 텍스트별 감지 결과를 응답에 에코."""
    r = _client().post('/v1/ner', json={'text': 'これは テスト'})
    assert r.json()['lang'] == 'ja'
    r2 = _client().post('/v1/ner', json={'text': 'Hà Nội là thủ đô'})
    assert r2.json()['lang'] == 'vi'
    r3 = _client().post('/v1/ner', json={'text': '삼성전자는 수원에'})
    assert r3.json()['lang'] == 'ko'


def test_explicit_lang_overrides_detection():
    """명시 lang 은 감지를 우회."""
    r = _client().post('/v1/ner', json={'text': 'これは', 'lang': 'vi'})
    assert r.json()['lang'] == 'vi'


def test_batch_per_text_detection():
    """배치는 텍스트별로 언어가 달라도 각자 감지."""
    r = _client().post('/v1/ner', json={'texts': ['テスト x', 'Hà Nội y']})
    langs = [it['lang'] for it in r.json()['results']]
    assert langs == ['ja', 'vi']


def test_ko_explicit_lang_accepted():
    """명시 `lang:"ko"` 는 400 이 아니라 그대로 추론된다."""
    r = _client().post('/v1/ner', json={'text': '김민준 씨', 'lang': 'ko'})
    assert r.status_code == 200
    assert r.json()['lang'] == 'ko'
    assert r.json()['entities'][0]['text'] == '김민준'


def test_batch_mixes_four_langs_in_order():
    """ja·ko·vi·en 혼합 배치가 언어별로 갈려도 입력 순서·lang 이 1:1."""
    r = _client().post('/v1/ner', json={'texts': [
        'テスト x',      # ja (가나)
        '김민준 y',      # ko (한글)
        'Hà Nội z',      # vi (dot-below)
        'Alice w',       # en (라틴 폴백)
    ]})
    results = r.json()['results']
    assert [it['lang'] for it in results] == ['ja', 'ko', 'vi', 'en']
    assert [it['entities'][0]['text'] for it in results] == [
        'テスト', '김민준', 'Hà', 'Alice']


def test_en_explicit_lang_accepted():
    """명시 lang:'en' 이 200 으로 받아들여지고 lang 을 그대로 에코한다."""
    r = _client().post('/v1/ner', json={'text': 'Alice went to Paris.',
                                        'lang': 'en'})
    assert r.status_code == 200
    body = r.json()
    assert body['lang'] == 'en'
    assert body['entities'][0]['text'] == 'Alice'


def test_unsupported_autodetect_single():
    """자동감지 미지원 → 200 + {lang:'unsupported', entities:[]} (에러 아님)."""
    r = _client().post('/v1/ner', json={'text': '東京都千代田区'})
    assert r.status_code == 200
    assert r.json() == {'lang': 'unsupported', 'entities': []}


def test_unsupported_bypasses_model_predict():
    """미지원 입력은 predict 를 호출하지 않는다(predict 가 터져도 200)."""
    class _Raising(StubRegistry):
        def predict(self, text, lang, apply_threshold=True):
            raise AssertionError('predict called for unsupported input')

    r = _client(registry=_Raising()).post(
        '/v1/ner', json={'text': '東京都千代田区'})
    assert r.status_code == 200
    assert r.json() == {'lang': 'unsupported', 'entities': []}


def test_batch_per_item_partial_unsupported():
    """배치 부분 성공 — 미지원 항목은 빈 결과, 순서·lang 1:1 보존."""
    r = _client().post('/v1/ner', json={'texts': [
        'テスト x',       # ja
        '東京都千代田区',  # unsupported (한자만 — 라틴 글자도 없다)
        'Hà Nội y',       # vi (dot-below ộ)
    ]})
    results = r.json()['results']
    assert [it['lang'] for it in results] == ['ja', 'unsupported', 'vi']
    assert results[1]['entities'] == []
    assert results[0]['entities'][0]['text'] == 'テスト'
    assert results[2]['entities'][0]['text'] == 'Hà'


def test_explicit_unsupported_lang_still_400():
    """명시 lang 이 미지원이면 자동감지와 달리 400(클라이언트 계약)."""
    r = _client().post('/v1/ner',
                       json={'text': 'Hà Nội', 'lang': _UNSUPPORTED_LANG})
    assert r.status_code == 400


def test_neither_text_nor_texts_400():
    r = _client().post('/v1/ner', json={'lang': 'vi'})
    assert r.status_code == 400
    assert 'error' in r.json()


def test_both_text_and_texts_400():
    r = _client().post('/v1/ner', json={'text': 'a', 'texts': ['b']})
    assert r.status_code == 400
    assert 'error' in r.json()


def test_invalid_lang_400():
    r = _client().post('/v1/ner',
                       json={'text': 'a', 'lang': _UNSUPPORTED_LANG})
    assert r.status_code == 400
    assert r.json()['error']['status'] == 400


def test_openapi_ner_description_names_every_supported_lang():
    """Swagger 설명이 실제 지원 언어와 어긋나지 않는지.

    소비자가 읽는 유일한 계약 문구라, 코드에 언어를 늘리고 문구를 안 고치면
    "지원 안 한다"고 적힌 언어가 조용히 200 을 내는 상태가 된다.
    """
    spec = _client().get('/openapi.json').json()
    doc = spec['paths']['/v1/ner']['post']
    blob = doc['summary'] + doc['description']
    for lang in SUPPORTED_LANGS:
        assert f'`{lang}`' in doc['description'], lang
    assert '한국어' in blob


def test_openapi_422_declares_error_envelope():
    """OpenAPI 의 422 선언이 실제 422 응답과 같은 모양이어야 한다.

    선언을 안 덮으면 FastAPI 기본 `HTTPValidationError`(`{"detail":[...]}`)가
    남아, 봉투로 통일된 실제 응답과 기계 계약이 어긋난다 — `/openapi.json` 으로
    클라이언트를 생성·검증하는 소비자가 틀린 계약을 받는 지점이다. 참조 이름만
    보면 모양이 갈려도 통과하므로, 선언한 필드와 실제 응답 필드를 대조한다.
    """
    client = _client()
    spec = client.get('/openapi.json').json()
    schema = (spec['paths']['/v1/ner']['post']['responses']['422']
              ['content']['application/json']['schema'])
    assert schema['$ref'].endswith('/ErrorResponse')

    declared = set(spec['components']['schemas']['ErrorBody']['properties'])
    actual = client.post('/v1/ner', json={'texts': 5})
    assert actual.status_code == 422
    assert set(actual.json()['error']) == declared


def test_max_batch_413():
    """배치 개수 초과 → 413 Payload Too Large."""
    cfg = ServerConfig(max_batch=2)
    r = _client(config=cfg).post('/v1/ner', json={
        'texts': ['a', 'b', 'c'], 'lang': 'vi'})
    assert r.status_code == 413
    assert r.json()['error']['status'] == 413


def test_max_chars_413():
    """텍스트 char 초과 → 413 Payload Too Large."""
    cfg = ServerConfig(max_chars=5)
    r = _client(config=cfg).post('/v1/ner', json={
        'text': 'way too long', 'lang': 'vi'})
    assert r.status_code == 413
    assert r.json()['error']['status'] == 413


def test_max_total_chars_413():
    """개별 텍스트는 한도 내여도 배치 합산이 상한 초과 → 413(작업량 가드)."""
    cfg = ServerConfig(max_chars=100, max_batch=10, max_total_chars=12)
    r = _client(config=cfg).post('/v1/ner', json={
        'texts': ['aaaaa', 'bbbbb', 'ccccc'], 'lang': 'vi'})  # 합 15 > 12
    assert r.status_code == 413
    assert r.json()['error']['status'] == 413


def test_model_unavailable_503():
    reg = StubRegistry(unavailable=('vi',))
    r = _client(registry=reg).post('/v1/ner', json={'text': 'a', 'lang': 'vi'})
    assert r.status_code == 503
    assert r.json()['error']['status'] == 503


def test_api_key_required_when_set():
    cfg = ServerConfig(api_key='secret')
    client = _client(config=cfg)
    no_key = client.post('/v1/ner', json={'text': 'a', 'lang': 'vi'})
    assert no_key.status_code == 401
    ok = client.post('/v1/ner', json={'text': 'a', 'lang': 'vi'},
                     headers={'x-api-key': 'secret'})
    assert ok.status_code == 200


def test_api_key_open_when_unset():
    """env key 미설정이면 인증 없이 오픈."""
    r = _client().post('/v1/ner', json={'text': 'a', 'lang': 'vi'})
    assert r.status_code == 200


def test_health_shape():
    r = _client().get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert set(body['langs']) == set(SUPPORTED_LANGS)


def test_health_no_auth_required():
    """헬스는 API-key 설정돼 있어도 인증 없이 접근 가능."""
    cfg = ServerConfig(api_key='secret')
    r = _client(config=cfg).get('/health')
    assert r.status_code == 200


def test_ui_page_served():
    """루트 UI 는 HTML(200, text/html)을 반환하고 핵심 요소를 담는다."""
    r = _client().get('/')
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('text/html')
    html = r.text
    assert '<textarea' in html
    # 동일 출처 fetch 대상이 페이지에 있어야 한다. 셀렉터 옵션 목록의
    # 정합은 test_ui_selector_options_match_supported_langs 가 본다.
    assert '/v1/ner' in html


def test_ui_selector_options_match_supported_langs():
    """웹 UI 언어 셀렉터가 서버 지원 목록과 어긋나면 실패한다.

    UI 는 서버 목록의 하드코딩 복제본을 쥐고 있고, 어긋남의 증상이 에러가
    아니라 "버튼이 안 뜸" 이라 조용하다. 실제로 ko 서빙을 넣을 때 이 자리가
    낡은 채 남아 있었다.
    """
    html = _client().get('/').text
    m = re.search(r'<select id="lang">(.*?)</select>', html, re.S)
    assert m, 'web UI has no language selector'
    opts = set(re.findall(r'<option value="([a-z]+)"', m.group(1)))
    # 뺄셈이 아니라 합집합으로 비교한다 — `opts - {'auto'}` 는 auto 옵션이
    # 사라져도 참이라, 자동감지를 고르는 유일한 UI 경로가 조용히 없어진다.
    assert opts == set(SUPPORTED_LANGS) | {'auto'}


def test_ui_unsupported_guidance_names_every_supported_lang():
    """미지원 안내 문구가 지원 언어를 전부 이름과 코드로 담는지.

    문구가 낡으면 에러가 아니라 안내가 거짓말이 된다 — 지원하는 언어를
    "지원하지 않는다" 고 읽는 사용자가 생긴다.
    """
    html = _client().get('/').text
    m = re.search(r'const UNSUPPORTED_HINT =(.*?);', html, re.S)
    assert m, 'index.html has no UNSUPPORTED_HINT constant'
    for lang in SUPPORTED_LANGS:
        assert f'({lang})' in m.group(1), lang


def test_ui_page_no_auth_required():
    """UI 페이지는 API-key 설정돼 있어도 인증 없이 접근 가능(헬스와 동일)."""
    cfg = ServerConfig(api_key='secret')
    r = _client(config=cfg).get('/')
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('text/html')
