"""토크나이저 직렬화 불변식 — 실모델·`/data` 없이 CI 에서 항상 도는 겹.

HF fast 토크나이저는 Rust 객체를 `RefCell` 로 감싸고, 인코딩이 그 상태를
바꾸는 호출(`no_truncation()`)을 지난다. 두 스레드가 같은 인스턴스에 동시에
들어가면 두 번째 borrow 가 `RuntimeError: Already borrowed` 로 터진다.
서버는 추론을 `run_in_threadpool` 로 돌리므로 언어당 하나뿐인 토크나이저에
여러 스레드가 들어온다 — `LangModel._tokenize` 의 잠금이 그걸 막는다.

여기 stub 은 그 `RefCell` 을 모사하되 **겹침을 확률에 맡기지 않는다**.
첫 진입 스레드가 다른 스레드의 진입 시도를 기다려 창을 강제로 열기 때문에,
잠금이 없으면 반드시 겹치고 반드시 터진다. 실모델로 같은 축을 태우는 겹은
`test_inference_integration.py` 에 있다(모델 없으면 skip).
"""

import contextlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ner.classifier.data_utils import build_label_maps
from server.inference import LangModel

_MAX_LENGTH = 32
_THREADS = 8


class _BorrowProbeTokenizer:
    """`RefCell` 을 모사한 fast 토크나이저 — 재진입하면 Already borrowed.

    진입 중에 다른 스레드가 들어오면 실제 Rust 토크나이저와 같은 예외를 낸다.
    첫 진입은 `_opened` 로 창을 열어 두 번째 진입을 기다리므로, 잠금이 없으면
    겹침이 우연이 아니라 필연이다. 잠금이 있으면 아무도 못 들어와 timeout 으로
    빠지고, 그 대기는 인스턴스당 한 번뿐이라 테스트가 느려지지 않는다.
    """

    is_fast = True

    def __init__(self, open_window_s: float = 0.5):
        self._guard = threading.Lock()
        self._inside = 0
        self._window_used = False
        self._arrived = threading.Event()
        self._open_window_s = open_window_s
        self.max_inside = 0
        self.calls = 0

    def __call__(self, text, **kwargs):
        first = self._enter()
        try:
            if first:
                # 다른 스레드가 진입을 시도할 때까지 창을 연다. 잠금이 있으면
                # 아무도 오지 못해 timeout 으로 빠진다.
                self._arrived.wait(self._open_window_s)
            return self._encode(text, kwargs)
        finally:
            self._leave()

    def _enter(self) -> bool:
        with self._guard:
            self._arrived.set()
            if self._inside > 0:
                raise RuntimeError('Already borrowed')
            self._inside += 1
            self.calls += 1
            self.max_inside = max(self.max_inside, self._inside)
            first = not self._window_used
            if first:
                self._window_used = True
                self._arrived.clear()
            return first

    def _leave(self):
        with self._guard:
            self._inside -= 1

    def _encode(self, text, kwargs):
        """공백 토큰화 — offset 계약만 지키면 되고 어휘는 무관하다."""
        spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        if not kwargs.get('return_offsets_mapping'):
            return {'input_ids': list(range(len(spans)))}
        max_length = kwargs['max_length']
        spans = spans[:max_length]
        pad = max_length - len(spans)
        return {
            'input_ids': list(range(len(spans))) + [0] * pad,
            'attention_mask': [1] * len(spans) + [0] * pad,
            'offset_mapping': spans + [(0, 0)] * pad,
        }


def _bare_model(lock):
    """토크나이저 구간만 갖춘 LangModel — 모델·GPU 없이 `_tokenize` 를 태운다."""
    lm = object.__new__(LangModel)
    lm.lang = 'en'
    lm.max_length = _MAX_LENGTH
    lm.tokenizer = _BorrowProbeTokenizer()
    lm.label2id, lm.id2label = build_label_maps()
    lm._tok_lock = lock
    return lm


def _hammer(lm, threads=_THREADS):
    """N 스레드가 동시에 `_tokenize` 를 호출하고 (예외들, 결과수) 를 돌려준다."""
    def one(i):
        return lm._tokenize(f'Alice met Bob in Seoul number {i}')

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(one, i) for i in range(threads)]
        errors, results = [], []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:            # noqa: BLE001 — 수집이 목적
                errors.append(exc)
    return errors, results


def test_lock_serializes_tokenizer_access():
    """잠금이 있으면 동시 진입이 1 을 넘지 않고 예외가 없다."""
    lm = _bare_model(threading.Lock())
    errors, results = _hammer(lm)

    assert errors == []
    assert len(results) == _THREADS
    assert lm.tokenizer.max_inside == 1


def test_without_lock_the_tokenizer_raises_already_borrowed():
    """잠금을 no-op 으로 바꾸면 같은 부하가 `Already borrowed` 로 터진다.

    이 겹이 위 테스트의 판별력을 증명한다 — 없으면 잠금을 지워도 위 단언이
    조용히 통과한다(겹치지 않으면 max_inside 는 1 그대로다).
    """
    lm = _bare_model(contextlib.nullcontext())
    errors, _ = _hammer(lm)

    assert errors, 'no-op 잠금인데 아무도 겹치지 않았다 — 창이 안 열렸다'
    assert all(isinstance(e, RuntimeError) for e in errors)
    assert all('Already borrowed' in str(e) for e in errors)


def test_chunking_and_encoding_share_one_lock_acquisition():
    """분할과 인코딩이 한 번의 잠금 안에서 끝난다.

    `split_for_length` 도 토큰 수를 세느라 같은 토크나이저를 만진다. 둘을 따로
    잠그면 그 사이에 다른 스레드가 끼어 창이 열린다.
    """
    class _CountingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.acquires = 0

        def __enter__(self):
            self.acquires += 1
            return self._lock.__enter__()

        def __exit__(self, *exc):
            return self._lock.__exit__(*exc)

    lock = _CountingLock()
    lm = _bare_model(lock)
    lm._tokenize('Alice met Bob in Seoul')

    assert lock.acquires == 1
    assert lm.tokenizer.calls >= 2      # 토큰 수 세기 + 인코딩


@pytest.mark.parametrize('text', ['', '   ', 'one'])
def test_degenerate_inputs_still_release_the_lock(text):
    """빈·공백·단일 토큰 입력에서도 잠금이 반납된다(다음 호출이 걸리지 않는다)."""
    lm = _bare_model(threading.Lock())
    lm._tokenize(text)
    lm._tokenize('Alice met Bob')       # 반납 안 됐으면 여기서 영영 멈춘다
