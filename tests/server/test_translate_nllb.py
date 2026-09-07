"""NLLB 백엔드 테스트 — 문장 분할·복원 경로(모델 무의존) + 실모델 PII 보존.

두 층으로 나뉜다.

- **모델 무의존**: 문장 분할과 마스킹→분할→합침→복원 배선. 분할이 새면 NLLB 가
  뒷문장을 통째로 버리고, 표기를 안 옮기면 PII 가 신호 없이 사라진다 — 둘 다
  에러를 내지 않는 실패라 테스트가 없으면 조용히 지나간다.
- **실모델**(`NER_SERVER_TEST_NLLB_MODEL` 지정 시에만): ja·vi·en 인라인 픽스처로
  PII 5종이 verbatim 보존되고 소실·잔여 sentinel 이 0 인지. 가중치가 필요해
  기본은 skip 이다.

    NER_SERVER_TEST_NLLB_MODEL=facebook/nllb-200-distilled-1.3B \\
      uv run pytest tests/server/test_translate_nllb.py
"""

import os
import threading
import time

import pytest

from server.translate import DEFAULT_SENTINEL, _mask, _restore
from server.translate_nllb import (
    ASCII_SENTINEL,
    NLLB_LANG_CODE,
    NLLBTranslator,
    resolve_no_repeat_ngram,
    sentence_batches,
    split_sentences,
)

# 미지원 언어 픽스처 — 이 파일이 검사하는 것이 `NLLB_LANG_CODE` 의 부재이므로
# 전제도 그 dict 에 건다(새 import 없이 정확한 단언).
_UNSUPPORTED_LANG = 'th'
assert _UNSUPPORTED_LANG not in NLLB_LANG_CODE

# ---------- 문장 분할 ----------


def test_ja_splits_on_fullwidth_stop():
    """ja 는 구두점 뒤에 공백이 없어도 문장이 갈린다."""
    assert split_sentences('今日は晴れです。明日は雨だ。', 'ja') == [
        '今日は晴れです。', '明日は雨だ。']


def test_vi_splits_on_period_space():
    """vi 는 마침표 + 공백이 기본 경계다."""
    assert split_sentences('Hôm nay trời đẹp. Ngày mai sẽ mưa.', 'vi') == [
        'Hôm nay trời đẹp.', 'Ngày mai sẽ mưa.']


def test_single_letter_abbreviation_is_not_a_boundary():
    """한 글자 약어 뒤 마침표는 문장 끝이 아니다 — 되붙인다."""
    assert split_sentences('Ông T. Nguyen đã nói vậy.', 'vi') == [
        'Ông T. Nguyen đã nói vậy.']


def test_decimal_and_year_do_not_split():
    """소수점·연도처럼 뒤에 공백이 없는 마침표는 경계가 아니다."""
    assert split_sentences('Giá là 3.14 triệu vào 2019.', 'vi') == [
        'Giá là 3.14 triệu vào 2019.']


def test_empty_and_unsplittable_text_returns_whole():
    """못 나누는 입력은 통짜로 돌려준다(빈 배치로 모델을 부르지 않는다)."""
    assert split_sentences('공백없는한문장', 'ja') == ['공백없는한문장']
    assert split_sentences('mot cau', 'vi') == ['mot cau']


def test_sentinel_survives_splitting():
    """마스킹된 sentinel 이 문장 분할로 쪼개지지 않는다."""
    phone = '0904-123-456'
    text = f'Liên hệ {phone}. Cảm ơn.'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans, ASCII_SENTINEL)
    parts = split_sentences(masked, 'vi')
    assert any(ASCII_SENTINEL.token(0) in part for part in parts)
    restored, n_ok, n_drop = _restore(' '.join(parts), id2val, ASCII_SENTINEL)
    assert phone in restored and (n_ok, n_drop) == (1, 0)


# ---------- 엔진 속성 ----------

def test_ascii_sentinel_differs_from_default_only_in_brackets():
    """NLLB 표기는 괄호만 다르고 nonce·번호 부분은 기본과 같다."""
    assert ASCII_SENTINEL.token(7) == f'[{DEFAULT_SENTINEL.token(7)[1:-1]}]'


# ---------- 반복 억제 강도 ----------

def test_repeat_suppression_on_by_default():
    """미설정이면 켜지되, sentinel 이 복사될 수 있는 하한을 쓴다."""
    assert resolve_no_repeat_ngram(None, 10) == 12


def test_repeat_suppression_can_be_disabled():
    """0 은 명시적 opt-out — 하한으로 끌어올리지 않는다."""
    assert resolve_no_repeat_ngram(0, 10) == 0


def test_too_small_suppression_raised_to_floor():
    """하한보다 작은 값은 올린다 — 그대로 두면 두 번째 PII 가 조용히 소실된다.

    sentinel 은 토큰 열을 공유해서, n 이 그 길이보다 작으면 한 문장의 두 번째
    sentinel 이 반복 금지에 걸려 모델이 글자를 바꿔 내놓는다(실측 n=3 → 5개 중
    2개 소실). 소실은 에러가 아니라 '부재'로 흡수돼 화면에 신호가 없다.
    """
    assert resolve_no_repeat_ngram(3, 10) == 12


def test_larger_suppression_kept():
    """하한 이상이면 설정값을 그대로 쓴다."""
    assert resolve_no_repeat_ngram(20, 10) == 20


# ---------- 배치 상한 · 동시 인코딩 ----------

def test_sentence_batches_bound_generate_size():
    """generate 한 번에 들어가는 문장 수가 상한에 묶이고 순서·개수는 보존된다."""
    sentences = [f's{i}' for i in range(20)]
    batches = sentence_batches(sentences)
    assert [len(b) for b in batches] == [8, 8, 4]
    assert [s for b in batches for s in b] == sentences
    assert sentence_batches([]) == []


class _RecordingTokenizer:
    """`src_lang` 을 인스턴스에 새기고 인코딩 때 읽는 실 토크나이저를 모사."""

    def __init__(self, delay=0.01):
        self.src_lang = None
        self._delay = delay
        self.seen = []

    def __call__(self, sentences, **kwargs):
        entry = self.src_lang
        time.sleep(self._delay)      # 인코딩 구간을 넓혀 경합을 드러낸다
        self.seen.append((entry, self.src_lang, tuple(sentences)))
        return self

    def to(self, device):            # enc.to(device) 자리
        return self


def test_encode_holds_source_language_tag_under_concurrency():
    """동시 요청이 서로의 소스 언어 태그를 덮어쓰지 않는다.

    번역은 전용 guard 안에서 여럿 동시에 돌 수 있는데, NLLB 토크나이저의
    `src_lang` 은 인스턴스에 새겨지는 공유 상태다. 잠그지 않으면 ja 요청이
    vi 태그로 인코딩돼 **에러 없이** 엉뚱한 번역이 나온다.
    """
    translator = object.__new__(NLLBTranslator)
    tok = _RecordingTokenizer()
    translator._tok = tok
    translator._tok_lock = threading.Lock()
    translator._device = 'cpu'

    langs = ['ja', 'vi'] * 5
    threads = [threading.Thread(target=translator._encode,
                                args=([f'{lang}-sentence'], lang))
               for lang in langs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(tok.seen) == len(langs)
    for entry, exit_tag, sentences in tok.seen:
        assert entry == exit_tag                      # 인코딩 중 안 바뀐다
        own_lang = sentences[0].split('-')[0]
        assert entry == NLLB_LANG_CODE[own_lang]      # 자기 태그로 인코딩됐다


class _OverlapProbeTokenizer:
    """토크나이저 안에 동시에 몇이 들어와 있는지 세는 stub."""

    def __init__(self, delay=0.005):
        self.src_lang = None
        self._delay = delay
        self._lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0

    def _enter(self):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)

    def _leave(self):
        with self._lock:
            self.inside -= 1

    def __call__(self, sentences, **kwargs):
        self._enter()
        time.sleep(self._delay)
        self._leave()
        return self

    def batch_decode(self, ids, **kwargs):
        self._enter()
        time.sleep(self._delay)
        self._leave()
        return ['decoded']

    def to(self, device):
        return self


def test_tokenizer_access_is_serialized_across_encode_and_decode():
    """인코딩과 디코딩이 토크나이저 안에서 겹치지 않는다.

    디코딩은 언어 태그를 읽지 않지만 같은 토크나이저 객체를 만진다 — fast
    토크나이저는 내부 상태를 Rust 쪽에서 빌려 쓰므로, 한쪽이 디코드하는 동안
    다른 쪽이 `src_lang` 을 갈면 `Already borrowed` 로 터진다. 그래서 두 구간이
    같은 잠금을 쓴다(생성만 잠금 밖).
    """
    translator = object.__new__(NLLBTranslator)
    tok = _OverlapProbeTokenizer()
    translator._tok = tok
    translator._tok_lock = threading.Lock()
    translator._device = 'cpu'

    def encode_worker():
        for _ in range(5):
            translator._encode(['文'], 'ja')

    def decode_worker():
        for _ in range(5):
            translator._decode([[1, 2, 3]])

    threads = ([threading.Thread(target=encode_worker) for _ in range(3)]
               + [threading.Thread(target=decode_worker) for _ in range(3)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tok.max_inside == 1      # 언제나 한 번에 하나만 들어간다


# ---------- 번역 배선(생성만 대체) ----------

class _StubNLLB(NLLBTranslator):
    """생성 단계만 갈아끼운 번역기 — 그 앞뒤(마스킹·분할·합침·복원)를 본다."""

    def __init__(self):
        self._sentinel = ASCII_SENTINEL
        self.seen = []

    def _generate(self, sentences, lang):
        self.seen.append(list(sentences))
        # 번역기가 sentinel 을 보존한 채 문장별로 결과를 내는 상황을 모사.
        return [f'<{s}>' for s in sentences]


def test_all_sentences_reach_output():
    """문장이 여럿이면 전부 번역되고 뒷문장이 유실되지 않는다."""
    translator = _StubNLLB()
    result = translator.translate('先頭です。二番目です。三番目です。', 'ja', [])
    assert translator.seen == [['先頭です。', '二番目です。', '三番目です。']]
    assert result.translation.count('<') == 3
    assert '三番目' in result.translation


def test_pii_restored_across_sentences():
    """뒷문장의 PII 도 마스킹→복원 왕복을 그대로 지난다."""
    phone, email = '090-1234-5678', 'a@example.jp'
    text = f'電話は{phone}です。連絡は{email}へ。'
    spans = [
        {'label': 'PHONE', 'start_char': text.index(phone),
         'end_char': text.index(phone) + len(phone), 'text': phone},
        {'label': 'EMAIL', 'start_char': text.index(email),
         'end_char': text.index(email) + len(email), 'text': email},
    ]
    translator = _StubNLLB()
    result = translator.translate(text, 'ja', spans)
    # 번역기에 넘어간 문장에는 PII 원문이 없다.
    assert all(phone not in s and email not in s
               for batch in translator.seen for s in batch)
    assert phone in result.translation and email in result.translation
    assert (result.masked_pii, result.restored_pii, result.dropped_pii) == (
        2, 2, 0)


def test_available_is_true_once_loaded():
    """인프로세스 백엔드는 로드된 시점에 가용이다(찔러 볼 원격이 없다)."""
    assert _StubNLLB().available() is True


def test_unsupported_language_rejected():
    """번역 대상 외 언어는 ValueError — 엔드포인트가 400 으로 매핑한다."""
    with pytest.raises(ValueError):
        _StubNLLB().translate('สวัสดี', _UNSUPPORTED_LANG, [])


# ---------- 실모델(가중치 있을 때만) ----------

_MODEL = os.environ.get('NER_SERVER_TEST_NLLB_MODEL')
_DEVICE = os.environ.get('NER_SERVER_TEST_NLLB_DEVICE', 'cuda:0')

requires_model = pytest.mark.skipif(
    not _MODEL, reason='set NER_SERVER_TEST_NLLB_MODEL to run with weights')


def _record(template: str, values):
    """`{}` 자리에 값을 채우고 그 위치로 canonical span 을 만든다."""
    text = template.format(*[v for _, v in values])
    spans, cursor = [], 0
    for label, value in values:
        start = text.index(value, cursor)
        spans.append({'label': label, 'start_char': start,
                      'end_char': start + len(value), 'text': value})
        cursor = start + len(value)
    return text, spans


# PII 5종이 각 1건 이상, 문장 3개에 걸쳐 놓인 인라인 픽스처. 마지막 문장에도
# PII 를 둬 뒷문장 유실이 보존 회계에 드러나게 한다.
_JA_FIXTURE = (
    '連絡先は{}です。電話番号は{}、会員番号は{}です。'
    'カード番号は{}、登録日は{}です。',
    [('EMAIL', 'tanaka@example.co.jp'), ('PHONE', '090-1234-5678'),
     ('ID_NUM', 'JP-4821-9930'), ('CREDIT_CARD', '4111-1111-1111-1111'),
     ('DAT', '2024年3月5日')],
)
_VI_FIXTURE = (
    'Liên hệ {} khi cần. Số điện thoại là {}, mã số là {}. '
    'Thẻ {} được đăng ký ngày {}.',
    [('EMAIL', 'an.nguyen@example.vn'), ('PHONE', '0904-123-456'),
     ('ID_NUM', 'VN-7781-2043'), ('CREDIT_CARD', '5555-4444-3333-2222'),
     ('DAT', '05/03/2024')],
)


# 영어 픽스처. `Inc.` 를 한 번 둬 다중 글자 약어가 문장 중간에서 잘리는
# 기존 한계(vi 와 공유)를 관찰한다 — 잘려도 PII 회계가 통과하면 무해하고,
# 깨지면 그때 별도 이슈로 정규식을 연다.
_EN_FIXTURE = (
    'Contact {} for details. The phone number is {}, member id {}. '
    'Acme Inc. charged card {} on {}.',
    [('EMAIL', 'john.doe@example.com'), ('PHONE', '+1-202-555-0143'),
     ('ID_NUM', 'US-3391-7742'), ('CREDIT_CARD', '4111-2222-3333-4444'),
     ('DAT', 'March 5, 2024')],
)


@pytest.fixture(scope='module')
def loaded_translator():
    """실모델 1회 로드 — 가중치가 없으면 이 모듈의 실모델 테스트는 skip."""
    return NLLBTranslator(model_id=_MODEL, device=_DEVICE)


@requires_model
@pytest.mark.parametrize('lang,fixture', [('ja', _JA_FIXTURE),
                                          ('vi', _VI_FIXTURE),
                                          ('en', _EN_FIXTURE)])
def test_real_model_preserves_pii_verbatim(loaded_translator, lang, fixture):
    """PII 5종이 원문 그대로 남고, 소실·잔여 sentinel 이 0 이다."""
    text, spans = _record(*fixture)
    result = loaded_translator.translate(text, lang, spans)
    missing = [s['text'] for s in spans if s['text'] not in result.translation]
    assert missing == []                                  # verbatim 보존
    assert (result.masked_pii, result.restored_pii, result.dropped_pii) == (
        5, 5, 0)                                          # 소실 0
    assert ASCII_SENTINEL.pattern().search(result.translation) is None
    assert DEFAULT_SENTINEL.pattern().search(result.translation) is None
    assert '[PII' not in result.translation               # 잔여 sentinel 0


@requires_model
def test_real_model_keeps_trailing_sentence(loaded_translator):
    """문장 2개 이상 입력에서 뒷문장이 통째로 유실되지 않는다."""
    text, spans = _record('今日は晴れです。会員番号は{}です。',
                          [('ID_NUM', 'JP-4821-9930')])
    result = loaded_translator.translate(text, 'ja', spans)
    assert 'JP-4821-9930' in result.translation   # 뒷문장이 살아 있다
    assert result.dropped_pii == 0
