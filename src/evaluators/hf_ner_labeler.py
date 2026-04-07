"""HuggingFace token-classification NER labeler for benchmark baseline."""

import logging
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from evaluators.tag_aligner import normalize_tag

logger = logging.getLogger(__name__)


class HFNERLabeler:
    """Wraps a HuggingFace token-classification model for NER inference."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "token-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=device,
            aggregation_strategy="none",
        )

        # Build label normalization map from model config
        self._id2label = self._model.config.id2label
        logger.info("Loaded HF model %s with labels: %s", model_name, list(self._id2label.values()))

    def label(self, text: str) -> List[dict]:
        """Label text, returning NERRecords compatible with benchmark runner."""
        tokens = text.split()
        if not tokens:
            return []
        bio_tags = self.label_sentence(tokens)
        return [{"tokens": tokens, "ner_tags": bio_tags, "id": "0"}]

    def label_sentence(self, tokens: List[str]) -> List[str]:
        """Label pre-tokenized tokens, returning BIO tags aligned to input tokens.

        Uses the HF pipeline on reconstructed text, then maps subword predictions
        back to the original token boundaries via character offsets.
        """
        text = " ".join(tokens)
        return self._predict_and_align(text, tokens)

    def label_syllables(self, text: str, syllable_tokens: List[str]) -> List[str]:
        """Label text and align predictions to KLUE-style syllable tokens.

        Args:
            text: Original sentence text (word-segmented, no entity annotations).
            syllable_tokens: KLUE syllable tokens (including space tokens).

        Returns:
            BIO tags aligned to syllable_tokens.
        """
        return self._predict_and_align(text, syllable_tokens)

    def _predict_and_align(self, text: str, tokens: List[str]) -> List[str]:
        """Run HF pipeline and align predictions to arbitrary token grid via char offsets."""
        if not text.strip():
            return ["O"] * len(tokens)

        raw_preds = self._pipe(text)

        # Build char→token index map from the original text
        # For syllable tokens, we need to map against the original text character positions
        char_to_tok = self._build_char_map(text, tokens)

        # Map subword predictions to token-level tags
        token_tags = ["O"] * len(tokens)
        token_scores = [0.0] * len(tokens)

        for pred in raw_preds:
            entity_label = pred.get("entity", "")
            score = pred.get("score", 0.0)
            start = pred.get("start", 0)
            end = pred.get("end", 0)

            if entity_label in ("O", "PAD") or not entity_label:
                continue

            # Map each character in the prediction span to token indices
            for char_pos in range(start, end):
                tok_idx = char_to_tok.get(char_pos)
                if tok_idx is not None and score > token_scores[tok_idx]:
                    token_scores[tok_idx] = score
                    token_tags[tok_idx] = normalize_tag(entity_label)

        # Fix B/I continuity: consecutive same-entity tokens should be I- after first B-
        prev_entity = None
        for i in range(len(token_tags)):
            tag = token_tags[i]
            if tag == "O":
                prev_entity = None
                continue
            parts = tag.split("-", 1)
            if len(parts) != 2:
                prev_entity = None
                continue
            prefix, entity = parts
            if entity == prev_entity and prefix == "B":
                token_tags[i] = f"I-{entity}"
            prev_entity = entity if prefix in ("B", "I") else None

        return token_tags

    @staticmethod
    def _build_char_map(text: str, tokens: List[str]) -> dict:
        """Build character-position to token-index map.

        Handles both word-level tokens (split by space) and KLUE syllable tokens
        by greedily matching each token against the text.
        """
        char_to_tok = {}
        text_pos = 0

        for tok_idx, tok in enumerate(tokens):
            if tok.strip() == "":
                # Space token: skip one space character in text
                if text_pos < len(text) and text[text_pos] == " ":
                    text_pos += 1
                continue

            # Find this token in text starting from current position
            found = text.find(tok, text_pos)
            if found == -1:
                # Try single character match for syllable tokens
                if len(tok) == 1 and text_pos < len(text):
                    if text[text_pos] == tok:
                        char_to_tok[text_pos] = tok_idx
                        text_pos += 1
                    elif text_pos + 1 < len(text) and text[text_pos] == " ":
                        text_pos += 1
                        if text_pos < len(text) and text[text_pos] == tok:
                            char_to_tok[text_pos] = tok_idx
                            text_pos += 1
                continue

            for i in range(found, found + len(tok)):
                char_to_tok[i] = tok_idx
            text_pos = found + len(tok)

        return char_to_tok
