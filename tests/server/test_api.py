"""API 계약·전송 테스트 — stub 추론기로 모델 없이 CI 가능.

실모델을 로드하지 않고 registry stub 을 주입해 엔드포인트 계약(스키마·순서·
lang 에코·감지·에러·인증·헬스)만 검증한다.
"""

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import ServerConfig
from server.inference import ModelUnavailable


class StubRegistry:
    """결정적 stub — 첫 단어를 PER span 으로 반환(순서·offset 검증)."""

    def __init__(self, langs=('ja', 'vi'), unavailable=()):
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


def test_explicit_lang_overrides_detection():
    """명시 lang 은 감지를 우회."""
    r = _client().post('/v1/ner', json={'text': 'これは', 'lang': 'vi'})
    assert r.json()['lang'] == 'vi'


def test_batch_per_text_detection():
    """배치는 텍스트별로 언어가 달라도 각자 감지."""
    r = _client().post('/v1/ner', json={'texts': ['テスト x', 'Hà Nội y']})
    langs = [it['lang'] for it in r.json()['results']]
    assert langs == ['ja', 'vi']


def test_unsupported_autodetect_single():
    """자동감지 미지원 → 200 + {lang:'unsupported', entities:[]} (에러 아님)."""
    r = _client().post('/v1/ner', json={'text': 'Bonjour à tous'})
    assert r.status_code == 200
    assert r.json() == {'lang': 'unsupported', 'entities': []}


def test_unsupported_bypasses_model_predict():
    """미지원 입력은 predict 를 호출하지 않는다(predict 가 터져도 200)."""
    class _Raising(StubRegistry):
        def predict(self, text, lang, apply_threshold=True):
            raise AssertionError('predict called for unsupported input')

    r = _client(registry=_Raising()).post(
        '/v1/ner', json={'text': 'hello world'})
    assert r.status_code == 200
    assert r.json() == {'lang': 'unsupported', 'entities': []}


def test_batch_per_item_partial_unsupported():
    """배치 부분 성공 — 미지원 항목은 빈 결과, 순서·lang 1:1 보존."""
    r = _client().post('/v1/ner', json={'texts': [
        'テスト x',     # ja
        'hello world',  # unsupported
        'Hà Nội y',     # vi (dot-below ộ)
    ]})
    results = r.json()['results']
    assert [it['lang'] for it in results] == ['ja', 'unsupported', 'vi']
    assert results[1]['entities'] == []
    assert results[0]['entities'][0]['text'] == 'テスト'
    assert results[2]['entities'][0]['text'] == 'Hà'


def test_explicit_unsupported_lang_still_400():
    """명시 lang 이 미지원(ko)이면 자동감지와 달리 400(클라이언트 계약)."""
    r = _client().post('/v1/ner', json={'text': 'Hà Nội', 'lang': 'ko'})
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
    r = _client().post('/v1/ner', json={'text': 'a', 'lang': 'ko'})
    assert r.status_code == 400
    assert r.json()['error']['status'] == 400


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
    assert set(body['langs']) == {'ja', 'vi'}


def test_health_no_auth_required():
    """헬스는 API-key 설정돼 있어도 인증 없이 접근 가능."""
    cfg = ServerConfig(api_key='secret')
    r = _client(config=cfg).get('/health')
    assert r.status_code == 200
