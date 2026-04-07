"""Enhanced NER labeler with self-consistency and two-pass verification.

Wraps any labeler that has a label_spans(text) method.
"""
import json
import logging
import re
from collections import Counter
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class EnhancedLabeler:
    """Wrap a base labeler with self-consistency voting and two-pass verification.

    Args:
        base_labeler: Any labeler with label_spans(text) -> List[dict].
        voting_rounds: Number of inference rounds for self-consistency (1 = disabled).
        two_pass: Enable two-pass verification to filter false positives.
    """

    def __init__(
        self,
        base_labeler: Any,
        voting_rounds: int = 1,
        two_pass: bool = False,
    ) -> None:
        self.base = base_labeler
        self.voting_rounds = voting_rounds
        self.two_pass = two_pass

        # Proxy token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def label_spans(self, text: str) -> List[dict]:
        """Extract entity spans with optional enhancements."""
        if self.voting_rounds > 1:
            spans = self._self_consistency(text)
        else:
            spans = self.base.label_spans(text)

        if self.two_pass:
            spans = self._verify_spans(text, spans)

        self._sync_tokens()
        return spans

    def _self_consistency(self, text: str) -> List[dict]:
        """Run multiple rounds and keep entities that appear in majority."""
        all_rounds: List[List[tuple]] = []

        for _ in range(self.voting_rounds):
            spans = self.base.label_spans(text)
            # Normalize to (text, type) tuples for voting
            round_entities = set()
            for s in spans:
                t = s.get("text", "").strip()
                tp = s.get("type", "").strip()
                if t and tp:
                    round_entities.add((t, tp))
            all_rounds.append(list(round_entities))

        # Count votes
        counter: Counter = Counter()
        for round_entities in all_rounds:
            for entity in round_entities:
                counter[entity] += 1

        # Keep entities with majority votes (> half of rounds)
        threshold = self.voting_rounds / 2
        winners = [
            {"text": text_val, "type": type_val}
            for (text_val, type_val), count in counter.items()
            if count > threshold
        ]
        return winners

    def _verify_spans(self, text: str, spans: List[dict]) -> List[dict]:
        """Two-pass: ask the LLM to verify extracted entities."""
        if not spans:
            return spans

        entities_str = ", ".join(
            f'"{s["text"]}"({s["type"]})' for s in spans
        )

        verify_prompt = (
            f"以下のテキストから抽出された固有表現が正しいか検証してください。\n"
            f"正しいものだけをJSON配列で返してください。\n\n"
            f"テキスト: {text}\n"
            f"抽出された固有表現: {entities_str}\n\n"
            f"正しい固有表現のみ出力（JSON配列）:"
        )

        try:
            # Use the base labeler's internal chat method
            if hasattr(self.base, '_chat'):
                import asyncio
                raw = asyncio.run(self.base._chat(verify_prompt))
            elif hasattr(self.base, '_call_llm'):
                # Ollama labeler
                raw = self.base._llm.invoke(
                    [__import__('langchain_core.messages', fromlist=['HumanMessage']).HumanMessage(content=verify_prompt)]
                )
                raw = str(raw.content).strip()
            else:
                return spans

            verified = self._parse_verified(raw, spans)
            return verified if verified else spans  # fallback to original if parse fails
        except Exception as e:
            logger.warning("Two-pass verification failed: %s", e)
            return spans

    @staticmethod
    def _parse_verified(raw: str, original_spans: List[dict]) -> Optional[List[dict]]:
        """Parse verification response. Only accepts list of dicts with text+type."""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        def _valid_span_list(data) -> Optional[List[dict]]:
            if not isinstance(data, list):
                return None
            result = []
            for item in data:
                if isinstance(item, dict) and "text" in item and "type" in item:
                    result.append(item)
            return result if result else None

        try:
            data = json.loads(raw)
            valid = _valid_span_list(data)
            if valid is not None:
                return valid
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                valid = _valid_span_list(json.loads(match.group()))
                if valid is not None:
                    return valid
            except json.JSONDecodeError:
                pass
        return None

    def _sync_tokens(self):
        """Sync token counts from base labeler."""
        self.total_prompt_tokens = getattr(self.base, "total_prompt_tokens", 0)
        self.total_completion_tokens = getattr(self.base, "total_completion_tokens", 0)
