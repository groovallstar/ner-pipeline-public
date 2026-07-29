"""전용 NMT(NLLB) 번역 엔진 — 벤치 하네스용 in-process 백엔드.

`LLMTranslator` 와 같은 계약(`available`·`translate`)을 내주므로 `run_bench` 가
LLM 후보와 같은 자리에 끼워 돌린다. 마스킹-복원은 프로덕션(`server.translate`)
것을 그대로 쓰고, 엔진 속성 두 가지만 다르다.

- **sentinel 은 ASCII 괄호다.** NLLB 의 SentencePiece 어휘에 lenticular
  bracket 이 없어 양쪽이 `<unk>` 로 죽는다 — sentinel 본체는 통과하는데 복원이
  괄호째 매칭하니 전량 실패로 집계된다. 괄호만 옮기면 nonce 를 포함해 복원된다.
- **번역 전에 문장으로 쪼갠다.** NLLB 는 문장 단위 모델이라 통짜 입력을 주면
  뒷문장을 통째로 버린다. 경계 규칙은 "마침표류 + 공백"을 기본으로 두고 한 글자
  약어만 예외로 뺐다 — 반대로 경계인 경우를 열거하면 연도·자리표시자·닫는
  따옴표에서 샌다.

프롬프트가 없는 모델이라 음차를 지시할 수단이 없고, LLM 경로의 프롬프트 규칙은
여기에 적용되지 않는다.
"""
from __future__ import annotations

import re
from typing import List, Optional

from server.translate import (
    DEFAULT_SENTINEL,
    SentinelFormat,
    TranslationResult,
    _mask,
    _restore,
)

# FLORES-200 언어 태그.
NLLB_LANG_CODE = {'ja': 'jpn_Jpan', 'vi': 'vie_Latn', 'ko': 'kor_Hang'}

# NLLB 가 살려내는 표기 — 괄호만 ASCII 로 옮기고 nonce·번호는 그대로 둔다.
ASCII_SENTINEL = SentinelFormat(left='[', right=']')

# 문장 경계. ja 는 구두점 뒤 공백이 없어 별도로 본다.
_JA_BOUNDARY = re.compile(r'(?<=[。！？])')
_LATIN_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
# 한 글자 약어(`T. Nguyen`·`D. C.`) 뒤는 문장 끝이 아니다.
_ABBREV_TAIL = re.compile(r'(?:^|[\s(\[])[A-Za-zÀ-ỹ]\.$')


def split_sentences(text: str, lang: str) -> List[str]:
    """번역 단위로 문장을 나눈다 — 빈 조각은 버리고, 못 나누면 통짜로 준다."""
    if lang == 'ja':
        parts = [p for p in _JA_BOUNDARY.split(text) if p.strip()]
        return parts or [text]
    merged: List[str] = []
    for part in _LATIN_BOUNDARY.split(text):
        # 직전 조각이 한 글자 약어로 끝났으면 경계가 아니라 되붙인다.
        if merged and _ABBREV_TAIL.search(merged[-1]):
            merged[-1] = f'{merged[-1]} {part}'
        else:
            merged.append(part)
    return [p for p in merged if p.strip()] or [text]


def max_token_repeat(text: str) -> int:
    """같은 어절이 연속으로 반복된 최대 횟수 — 생성 붕괴 탐지용.

    beam search 는 반복 억제 없이 돌면 한 어절을 `max_new_tokens` 까지 되풀이해
    출력을 통째로 버리는 수가 있다. 평균 품질 지표는 이 붕괴를 흡수해버리므로
    건수로 따로 센다.
    """
    tokens = text.split()
    if not tokens:
        return 0
    best = run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


class NLLBTranslator:
    """in-process transformers 로 NLLB 를 태운다(fp16, beam search).

    HTTP 백엔드가 아니라 기동 즉시 가용이므로 `available` 은 로드 성공 여부다.
    """

    def __init__(self, model_id: str, device: str = 'cuda:0',
                 beams: int = 4, max_new_tokens: int = 256,
                 no_repeat_ngram: int = 0,
                 sentinel: SentinelFormat = ASCII_SENTINEL) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._device = torch.device(device)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, dtype=torch.float16).to(self._device).eval()
        self._beams = beams
        self._max_new_tokens = max_new_tokens
        self._no_repeat_ngram = no_repeat_ngram
        self._sentinel = sentinel
        self._ko_id = self._tok.convert_tokens_to_ids(NLLB_LANG_CODE['ko'])
        if self._ko_id is None or self._ko_id == self._tok.unk_token_id:
            raise RuntimeError(
                f'target language token {NLLB_LANG_CODE["ko"]!r} not in the '
                f'tokenizer vocabulary of {model_id}')

    def available(self) -> bool:
        """로드된 시점에 가용이다(원격 백엔드가 없어 확인할 대상이 없다)."""
        return True

    def _generate(self, sentences: List[str], lang: str) -> List[str]:
        """문장 묶음을 한 배치로 번역한다."""
        self._tok.src_lang = NLLB_LANG_CODE[lang]
        enc = self._tok(sentences, return_tensors='pt', padding=True,
                        truncation=True, max_length=512).to(self._device)
        kwargs = {}
        if self._no_repeat_ngram:
            kwargs['no_repeat_ngram_size'] = self._no_repeat_ngram
        with self._torch.inference_mode():
            out = self._model.generate(
                **enc, forced_bos_token_id=self._ko_id,
                num_beams=self._beams, max_new_tokens=self._max_new_tokens,
                **kwargs)
        return self._tok.batch_decode(out, skip_special_tokens=True)

    def translate(self, text: str, lang: str,
                  spans: List[dict]) -> TranslationResult:
        """원문을 마스킹→문장별 번역→복원해 한국어 결과를 낸다."""
        if lang not in NLLB_LANG_CODE:
            raise ValueError(f'unsupported source language: {lang}')
        masked, id2val = _mask(text, spans, self._sentinel)
        decoded = self._generate(split_sentences(masked, lang), lang)
        # 원문 언어와 무관하게 출력은 한국어라 문장 사이는 공백으로 잇는다.
        raw = ' '.join(d.strip() for d in decoded if d.strip())
        restored, n_ok, n_drop = _restore(raw, id2val, self._sentinel)
        return TranslationResult(
            lang=lang, translation=restored, masked_pii=len(id2val),
            restored_pii=n_ok, dropped_pii=n_drop)

    def peak_vram_mib(self) -> Optional[int]:
        """이 프로세스가 device 에 잡은 최대 할당량(MiB)."""
        if self._device.type != 'cuda':
            return None
        return round(
            self._torch.cuda.max_memory_allocated(self._device) / 1024 / 1024)


def sentinel_for(engine_kind: str) -> SentinelFormat:
    """엔진 종류별 sentinel 표기 — 토크나이저 종속성을 한 곳에 모은다."""
    return ASCII_SENTINEL if engine_kind == 'nllb' else DEFAULT_SENTINEL
