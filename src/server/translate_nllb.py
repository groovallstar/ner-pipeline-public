"""전용 NMT(NLLB) 번역 백엔드 — 서버 프로세스 안에서 직접 돌린다.

`LLMTranslator` 와 같은 계약(`available`·`translate`)을 내주므로 `/v1/translate`
는 어느 쪽이 꽂혔는지 몰라도 된다. 마스킹-복원은 `server.translate` 것을 그대로
쓰고, 엔진 속성 셋만 다르다.

- **sentinel 은 ASCII 괄호다.** NLLB 의 SentencePiece 어휘에 lenticular
  bracket 이 없어 양쪽이 `<unk>` 로 죽는다 — sentinel 본체는 통과하는데 복원이
  괄호째 매칭하니 전량 실패로 집계된다(근거: `certified/translate_bench/
  nllb-1.3b-sentinel-ascii-2080ti/summary.json`, 현행 표기 0/186 → ASCII
  186/186). 마스킹-복원은 소실을 '부재'로 처리해 에러를 내지 않으므로, 표기를
  안 옮기면 화면에서 PII 가 신호 없이 사라진다.
- **번역 전에 문장으로 쪼갠다.** NLLB 는 문장 단위 모델이라 통짜 입력을 주면
  뒷문장을 통째로 버린다. 경계 규칙은 "마침표류 + 공백"을 기본으로 두고 한 글자
  약어만 예외로 뺐다 — 반대로 경계인 경우를 열거하면 연도·자리표시자·닫는
  따옴표에서 샌다.
- **반복을 억제하되, sentinel 이 복사될 만큼만 억제한다.** 억제 없이 beam search
  를 돌리면 한 어절을 `max_new_tokens` 까지 되풀이해 출력을 통째로 버리는
  레코드가 나온다. 그런데 `no_repeat_ngram_size` 를 작게 잡으면 **sentinel 자체가
  반복 금지에 걸린다** — 한 문장에 PII 가 둘이면 두 번째 sentinel 은 앞엣것과
  토큰 열을 공유해 그대로 쓸 수 없고, 모델이 글자를 바꿔 내보내 복원이 실패한다
  (실측: n=3·9 에서 5개 중 2개 소실, n=12 에서 0개). 그래서 억제 강도의 하한을
  sentinel 토큰 길이에서 계산한다(`resolve_no_repeat_ngram`).

프롬프트가 없는 모델이라 음차를 지시할 수단이 없고, LLM 경로의 프롬프트 규칙은
여기에 적용되지 않는다.

원격 HTTP 가 아니라 **NER 과 같은 GPU 를 쓴다** — 그래서 번역은 전용 동시성
상한 안에서만 돌고(`app.py`), 한 요청이 만드는 문장 배치도 아래 상한으로 묶어
입력 길이가 그대로 VRAM 피크가 되지 않게 한다. 동시에 도는 만큼 **공유 상태도
생긴다** — 소스 언어 태그가 토크나이저 인스턴스에 새겨지므로 토크나이저를 만지는
구간(인코딩·디코딩)을 직렬화한다(`_encode`·`_decode`).
"""
from __future__ import annotations

import logging
import re
import threading
from typing import List

from server.translate import (
    SentinelFormat,
    TranslationResult,
    _mask,
    _restore,
)

logger = logging.getLogger(__name__)

# FLORES-200 언어 태그.
NLLB_LANG_CODE = {'ja': 'jpn_Jpan', 'vi': 'vie_Latn', 'ko': 'kor_Hang',
                  'en': 'eng_Latn'}

# NLLB 가 살려내는 표기 — 괄호만 ASCII 로 옮기고 nonce·번호는 그대로 둔다.
ASCII_SENTINEL = SentinelFormat(left='[', right=']')

# 한 번의 generate 에 넣는 문장 수 상한. 긴 입력(max_chars 2만 자)이 수백
# 문장으로 갈릴 수 있는데, 그걸 한 배치로 태우면 요청 하나가 VRAM 피크를
# 입력 길이에 비례해 밀어 올린다. 나눠 태우면 피크가 이 상한에 묶인다.
_MAX_SENTENCES_PER_BATCH = 8

# 인코더 입력 토큰 상한 — 문장 단위라 넉넉하다(초과분은 잘린다).
_MAX_INPUT_TOKENS = 512

# 반복 억제 하한의 여유분 — sentinel 토큰 길이에 이만큼 얹는다. 하한을 정확히
# sentinel 길이에 맞추면 앞뒤 문맥이 한 토큰만 겹쳐도 다시 걸린다(실측: 길이
# 10 인 sentinel 이 n=10 에서 하나 소실, n=12 에서 전부 보존).
_NGRAM_MARGIN = 2


def sentence_batches(sentences: List[str]) -> List[List[str]]:
    """문장을 generate 한 번 분량으로 끊는다 — VRAM 피크를 입력 길이에서 뗀다."""
    return [sentences[i:i + _MAX_SENTENCES_PER_BATCH]
            for i in range(0, len(sentences), _MAX_SENTENCES_PER_BATCH)]


def resolve_no_repeat_ngram(configured, sentinel_tokens: int) -> int:
    """설정값과 sentinel 길이로 실제 `no_repeat_ngram_size` 를 정한다.

    - `None`(미설정): sentinel 이 복사될 수 있는 하한을 쓴다 — 억제는 켜지되
      PII 보존을 깨지 않는 가장 강한 값이다.
    - `0`: 끈다(명시적 opt-out — 생성 붕괴를 감수한다는 뜻).
    - 하한보다 작은 양수: 하한으로 올리고 경고를 남긴다. 그대로 따르면 한 문장에
      PII 가 둘 이상일 때 두 번째부터 조용히 소실되는데, 그 실패는 에러가 아니라
      '부재'로 흡수돼 화면에서 보이지 않기 때문이다.
    """
    floor = sentinel_tokens + _NGRAM_MARGIN
    if configured is None:
        return floor
    if configured <= 0:
        return 0
    if configured < floor:
        logger.warning(
            'no_repeat_ngram_size %d raised to %d so PII sentinels stay '
            'copyable', configured, floor)
        return floor
    return configured

# 문장 경계. ja 는 구두점 뒤에 공백이 없어 별도로 본다.
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


class NLLBTranslator:
    """in-process transformers 로 NLLB 를 태운다(beam search).

    dtype 은 device 를 따른다 — CUDA 는 fp16(VRAM 절감), 그 외는 fp32(CPU 는
    fp16 커널이 없거나 느리다). 원격 백엔드가 아니라 로드 성공이 곧 가용이다.
    """

    def __init__(self, model_id: str, device: str = 'cuda:0',
                 beams: int = 4, max_new_tokens: int = 256,
                 no_repeat_ngram=None,
                 sentinel: SentinelFormat = ASCII_SENTINEL) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._device = torch.device(device)
        dtype = (torch.float16 if self._device.type == 'cuda'
                 else torch.float32)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, dtype=dtype).to(self._device).eval()
        self._beams = beams
        self._max_new_tokens = max_new_tokens
        self._sentinel = sentinel
        # 토크나이저 접근을 직렬화하는 잠금 — 사유는 `_encode`·`_decode`.
        self._tok_lock = threading.Lock()
        # 하한은 이 토크나이저가 sentinel 을 몇 토큰으로 쪼개느냐에 달렸다
        # (nonce 가 프로세스마다 달라 길이도 조금씩 다르다) — 그래서 상수가
        # 아니라 실제 표기를 재서 정한다.
        sentinel_tokens = len(self._tok(sentinel.token(0),
                                        add_special_tokens=False)['input_ids'])
        self._no_repeat_ngram = resolve_no_repeat_ngram(
            no_repeat_ngram, sentinel_tokens)
        self._ko_id = self._tok.convert_tokens_to_ids(NLLB_LANG_CODE['ko'])
        if self._ko_id is None or self._ko_id == self._tok.unk_token_id:
            raise RuntimeError(
                f'target language token {NLLB_LANG_CODE["ko"]!r} not in the '
                f'tokenizer vocabulary of {model_id}')
        logger.info('NLLBTranslator ready: %s @ %s (no_repeat_ngram=%d)',
                    model_id, self._device, self._no_repeat_ngram)

    def available(self) -> bool:
        """로드된 시점에 가용이다(원격 백엔드가 없어 확인할 대상이 없다)."""
        return True

    def _encode(self, sentences: List[str], lang: str):
        """소스 언어 태그를 걸고 토큰화한다 — 토크나이저 접근은 직렬화한다.

        NLLB 토크나이저는 `src_lang` 을 인스턴스에 새겨 두고 인코딩할 때 그것을
        읽는다. 번역은 동시에 여럿 돌 수 있으므로(번역 전용 guard, 기본 2)
        ja·vi·en 요청이 겹치면 한쪽이 **상대의 언어 태그로 인코딩**돼 조용히
        엉뚱한 번역이 나온다 — 에러가 아니라 잘못된 출력이라 화면에 신호가 없다.
        """
        with self._tok_lock:
            self._tok.src_lang = NLLB_LANG_CODE[lang]
            enc = self._tok(sentences, return_tensors='pt', padding=True,
                            truncation=True, max_length=_MAX_INPUT_TOKENS)
        return enc.to(self._device)

    def _decode(self, ids) -> List[str]:
        """생성 결과를 문자열로 되돌린다 — 인코딩과 같은 잠금을 쓴다.

        디코드는 태그를 읽지 않지만 같은 토크나이저 객체를 만진다. fast 토크
        나이저는 내부 상태를 Rust 쪽에서 빌려 쓰므로, 한쪽이 디코드하는 동안
        다른 쪽이 `src_lang` 을 갈면 `Already borrowed` 로 터진다. 8문장
        디코드는 밀리초 단위라 직렬화 비용이 무시할 만하다(오래 걸리는 생성만
        잠금 밖이다).
        """
        with self._tok_lock:
            return self._tok.batch_decode(ids, skip_special_tokens=True)

    def _generate(self, sentences: List[str], lang: str) -> List[str]:
        """문장 묶음을 번역한다 — VRAM 피크를 묶으려 나눠 태운다."""
        out: List[str] = []
        for chunk in sentence_batches(sentences):
            enc = self._encode(chunk, lang)
            kwargs = {}
            if self._no_repeat_ngram:
                kwargs['no_repeat_ngram_size'] = self._no_repeat_ngram
            with self._torch.inference_mode():
                ids = self._model.generate(
                    **enc, forced_bos_token_id=self._ko_id,
                    num_beams=self._beams,
                    max_new_tokens=self._max_new_tokens, **kwargs)
            out.extend(self._decode(ids))
        return out

    def translate(self, text: str, lang: str,
                  spans: List[dict]) -> TranslationResult:
        """원문을 마스킹→문장별 번역→복원해 한국어 결과를 낸다.

        span 이 비정상(겹침·범위 밖)이면 `_mask` 가 `ValueError` 를 올린다
        (엔드포인트가 400 으로 매핑). 미지원 언어도 `ValueError` 다.
        """
        if lang not in NLLB_LANG_CODE:
            raise ValueError(f'unsupported source language: {lang}')
        masked, id2val = _mask(text, spans, self._sentinel)
        decoded = self._generate(split_sentences(masked, lang), lang)
        # 원문 언어와 무관하게 출력은 한국어라 문장 사이는 공백으로 잇는다.
        raw = ' '.join(d.strip() for d in decoded if d.strip())
        restored, n_ok, n_drop = _restore(raw, id2val, self._sentinel)
        if n_drop:
            logger.warning('translation dropped %d/%d PII sentinel(s)',
                           n_drop, len(id2val))
        return TranslationResult(
            lang=lang, translation=restored, masked_pii=len(id2val),
            restored_pii=n_ok, dropped_pii=n_drop)
