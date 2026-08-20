"""Tests for base labeler inheritance, public API, token tracking, and drift guard."""
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ner.labelers.base_vllm_labeler import BaseVllmLabeler
from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler as JV
from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler as KV


class TestInheritance:
    def test_vllm(self):
        assert issubclass(KV, BaseVllmLabeler)
        assert issubclass(JV, BaseVllmLabeler)


class TestPublicAPI:
    @pytest.mark.parametrize("cls", [KV, JV])
    def test_has_label_methods(self, cls):
        for name in ("label", "label_spans", "label_records"):
            fn = getattr(cls, name, None)
            assert callable(fn), f"{cls.__name__}.{name} missing"


class TestDriftGuard:
    """AC13: subclasses override only __init__."""

    DUNDERS = {
        "__init__", "__module__", "__qualname__", "__doc__",
        "__annotations__", "__dict__",
        "__static_attributes__", "__firstlineno__",  # Python 3.12+
    }

    @pytest.mark.parametrize("cls", [KV, JV])
    def test_only_init(self, cls):
        extra = set(cls.__dict__) - self.DUNDERS
        assert extra == set(), f"{cls.__name__} defines extra members: {extra}"


class TestTokenTracking:
    """AC6: BaseVllmLabeler accumulates usage fields from response.usage."""

    def _make_labeler(self):
        return KV(base_url="http://unused", model="dummy", concurrency=1)

    def test_usage_accumulates(self):
        labeler = self._make_labeler()
        assert labeler.total_prompt_tokens == 0
        assert labeler.total_completion_tokens == 0

        mock_response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )
        labeler._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = asyncio.run(labeler._chat("hello"))
        assert result == "[]"
        assert labeler.total_prompt_tokens == 10
        assert labeler.total_completion_tokens == 5

        # Second call accumulates
        asyncio.run(labeler._chat("world"))
        assert labeler.total_prompt_tokens == 20

    def test_usage_none_safe(self):
        labeler = self._make_labeler()
        mock_response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )
        labeler._client.chat.completions.create = AsyncMock(return_value=mock_response)
        asyncio.run(labeler._chat("x"))
        assert labeler.total_prompt_tokens == 0


class TestEventLoopReuse:
    """동기 API 를 여러 번 불러도 클라이언트·세마포어가 같은 루프에 머문다.

    `asyncio.run` 을 호출마다 쓰면 루프가 매번 새로 생기는데, 클라이언트의 연결 풀과
    `asyncio.Semaphore` 는 처음 쓴 루프에 묶인다. 벤치마크는 라벨러 하나로 샘플을 순차
    처리하므로 두 번째 호출부터 어긋나 — 연결 풀은 `Event loop is closed`, 세마포어는
    `bound to a different event loop` — 그 샘플이 통째로 실패했다. 세마포어 쪽은 서버
    없이도 재현되므로 여기서 고정한다: `concurrency=1` 로 대기를 강제하면 옛 구현은
    두 번째 호출에서 깨진다.
    """

    def _labeler(self):
        labeler = KV(base_url="http://unused", model="dummy", concurrency=1)
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )

        async def _slow(*args, **kwargs):
            # 한 번 양보시켜 세마포어에 실제 대기를 만든다. 즉시 반환하면 첫 문장이
            # 끝난 뒤에야 둘째가 시작해 대기가 없고, 그러면 루프 바인딩도 안 일어나
            # 이 테스트가 옛 구현에서도 통과한다.
            await asyncio.sleep(0)
            return response

        labeler._client.chat.completions.create = AsyncMock(side_effect=_slow)
        return labeler

    def test_repeated_sync_calls_share_one_loop(self):
        labeler = self._labeler()
        two_sentences = "서울에서 첫 번째 문장이 끝난다. 부산에서 둘째 문장도 끝난다."

        assert labeler.label_spans(two_sentences) == []
        first_loop = labeler._loop
        assert labeler.label_spans(two_sentences) == []

        assert labeler._loop is first_loop
        assert not first_loop.is_closed()
        labeler.close()

    def test_close_lets_the_next_call_start_a_fresh_loop(self):
        labeler = self._labeler()
        labeler.label_spans("서울에서 한 문장이 끝난다.")
        labeler.close()
        assert labeler._loop is None
        assert labeler.label_spans("서울에서 한 문장이 끝난다.") == []
        labeler.close()


class TestInitSignatureParity:
    """ko and ja subclasses share the same __init__ signature."""

    @pytest.mark.parametrize("pair", [(KV, JV)])
    def test_signature_parity(self, pair):
        ko_sig = inspect.signature(pair[0].__init__)
        ja_sig = inspect.signature(pair[1].__init__)
        assert list(ko_sig.parameters) == list(ja_sig.parameters)
