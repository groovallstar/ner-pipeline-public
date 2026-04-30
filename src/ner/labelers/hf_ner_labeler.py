"""벤치마크 베이스라인용 HuggingFace 토큰 분류 NER 라벨러."""

import logging
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from ner.labelers.tag_aligner import normalize_tag

logger = logging.getLogger(__name__)


class HFNERLabeler:
    """NER 추론을 위해 HuggingFace 토큰 분류 모델을 래핑한다."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        lang: str = "ko",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.lang = lang

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

        # 모델 설정에서 레이블 정규화 맵을 구성한다
        self._id2label = self._model.config.id2label
        logger.info("Loaded HF model %s with labels: %s", model_name, list(self._id2label.values()))

    def label(self, text: str) -> List[dict]:
        """텍스트를 라벨링하여 벤치마크 러너와 호환되는 NERRecord를 반환한다."""
        tokens = text.split()
        if not tokens:
            return []
        bio_tags = self.label_sentence(tokens)
        return [{"tokens": tokens, "ner_tags": bio_tags, "id": "0"}]

    def label_sentence(self, tokens: List[str]) -> List[str]:
        """사전 토큰화된 토큰을 라벨링하여 입력 토큰에 정렬된 BIO 태그를 반환한다.

        재구성된 텍스트에 HF 파이프라인을 적용한 후, 문자 오프셋을 통해
        서브워드 예측을 원래 토큰 경계에 매핑한다.
        """
        text = " ".join(tokens)
        return self._predict_and_align(text, tokens)

    def label_syllables(self, text: str, syllable_tokens: List[str]) -> List[str]:
        """텍스트를 라벨링하여 KLUE 방식 음절 토큰에 예측을 정렬한다.

        Args:
            text: 원문 문장 텍스트 (단어 분절, 엔티티 어노테이션 없음).
            syllable_tokens: KLUE 음절 토큰 (공백 토큰 포함).

        Returns:
            syllable_tokens에 정렬된 BIO 태그.
        """
        return self._predict_and_align(text, syllable_tokens)

    def _predict_and_align(self, text: str, tokens: List[str]) -> List[str]:
        """HF 파이프라인을 실행하고 문자 오프셋을 통해 임의 토큰 그리드에 예측을 정렬한다."""
        if not text.strip():
            return ["O"] * len(tokens)

        raw_preds = self._pipe(text)

        # 원문 텍스트에서 문자 → 토큰 인덱스 맵을 구성한다
        # 음절 토큰의 경우 원문 문자 위치에 대해 매핑해야 한다
        char_to_tok = self._build_char_map(text, tokens)

        # 서브워드 예측을 토큰 수준 태그로 매핑한다
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
                    token_tags[tok_idx] = normalize_tag(entity_label, lang=self.lang)

        # B/I 연속성 보정: 연속된 동일 엔티티 토큰은 첫 B- 이후 I-여야 한다
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
        """문자 위치에서 토큰 인덱스로의 맵을 구성한다.

        각 토큰을 텍스트에 탐욕적으로 매핑하여 단어 수준 토큰과
        KLUE 음절 토큰을 모두 처리한다.
        """
        char_to_tok = {}
        text_pos = 0

        for tok_idx, tok in enumerate(tokens):
            if tok.strip() == "":
                # 공백 토큰: 텍스트에서 공백 문자 하나를 건너뛴다
                if text_pos < len(text) and text[text_pos] == " ":
                    text_pos += 1
                continue

            # 현재 위치부터 텍스트에서 이 토큰을 찾는다
            found = text.find(tok, text_pos)
            if found == -1:
                # 음절 토큰을 위해 단일 문자 매칭을 시도한다
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
