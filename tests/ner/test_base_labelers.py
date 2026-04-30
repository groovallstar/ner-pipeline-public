"""Tests for base labeler inheritance, public API, token tracking, and drift guard."""
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ner.labelers.base_openai_labeler import BaseOpenAILabeler
from ner.labelers.base_vllm_labeler import BaseVllmLabeler
from ner.labelers.ja.openai_ner_labeler import OpenAINERLabeler as JP
from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler as JV
from ner.labelers.ko.openai_ner_labeler import OpenAINERLabeler as KP
from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler as KV


class TestInheritance:
    def test_vllm(self):
        assert issubclass(KV, BaseVllmLabeler)
        assert issubclass(JV, BaseVllmLabeler)

    def test_openai(self):
        assert issubclass(KP, BaseOpenAILabeler)
        assert issubclass(JP, BaseOpenAILabeler)


class TestPublicAPI:
    @pytest.mark.parametrize("cls", [KV, JV, KP, JP])
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

    @pytest.mark.parametrize("cls", [KV, JV, KP, JP])
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


class TestInitSignatureParity:
    """ko and ja subclasses share the same __init__ signature."""

    @pytest.mark.parametrize("pair", [(KV, JV), (KP, JP)])
    def test_signature_parity(self, pair):
        ko_sig = inspect.signature(pair[0].__init__)
        ja_sig = inspect.signature(pair[1].__init__)
        assert list(ko_sig.parameters) == list(ja_sig.parameters)
