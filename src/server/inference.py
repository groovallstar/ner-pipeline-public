"""추론 코어 — 언어별 모델을 1회 로드·재사용해 텍스트 → canonical span.

`ner.classifier` 의 data_utils(build_label_maps / encode_row /
decode_bio_to_spans)와 confidence_threshold(임계값 fit·apply)에 의존한다.
학습·평가 모듈(train_eval 등)은 import 하지 않고, 추론 루프(softmax→argmax→
BIO decode)를 서버 안에서 직접 돈다.

출력 span 은 프로젝트 canonical 형식 `{label, start_char, end_char, text,
score}` — `.jsonl` 데이터 관례와 일치해 API 결과를 파이프라인에 되먹일 수
있다.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from ner.classifier.confidence_threshold import (
    apply_thresholds,
    load_thresholds,
)
from ner.classifier.data_utils import (
    build_label_maps,
    decode_bio_to_spans,
    encode_row,
)
from server.chunking import split_for_length
from server.config import ServerConfig

logger = logging.getLogger(__name__)

# 정밀도 → autocast dtype. fp32 는 autocast off, bf16 은 배치(B>1) forward 만
# autocast 로 묶어 matmul 을 bf16 으로 돌린다(softmax 등은 fp32 승격). 단건
# (B=1)은 작은 [1,L] 행렬이라 autocast 오버헤드가 tensor-core 이득을 넘어
# 오히려 느려져 fp32 로 둔다(분기는 _infer_chunks).
_AUTOCAST_DTYPE: Dict[str, Optional[torch.dtype]] = {
    'fp32': None,
    'bf16': torch.bfloat16,
}


class ModelUnavailable(Exception):
    """요청한 언어 모델이 로드되지 않았을 때 — API 는 503 으로 매핑."""


def _load_tokenizer(model_dir: str, lang: str):
    """언어별 토크나이저 로드. ja 는 slow(MeCab), vi 는 fast 우선·실패 시 slow.

    fast/slow 분기는 data_utils.encode_row 가 토크나이저 capability 로 다시
    판단하므로, 여기서는 로드 가능한 형태만 확보하면 된다.
    """
    if lang == 'ja':
        return AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except (TypeError, ValueError, OSError):
        return AutoTokenizer.from_pretrained(model_dir, use_fast=False)


def _to_canonical(span: dict, text: str) -> dict:
    """내부 {type,start,end,score?} → canonical {label,start_char,...}."""
    out = {
        'label': span['type'],
        'start_char': span['start'],
        'end_char': span['end'],
        'text': text[span['start']:span['end']],
    }
    if 'score' in span:
        out['score'] = span['score']
    return out


class LangModel:
    """한 언어의 모델·토크나이저·임계값을 보유하고 추론을 수행한다.

    상태는 로드 시점에 고정되고 요청 간 변하지 않는다(무상태 핸들러 보장).
    """

    def __init__(self, lang: str, model_dir: str, thresholds_path: str,
                 max_length: int = 256, precision: str = 'bf16',
                 device: Optional[torch.device] = None):
        self.lang = lang
        self.max_length = max_length
        self.precision = precision
        self.autocast_dtype = _AUTOCAST_DTYPE[precision]
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = _load_tokenizer(model_dir, lang)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir, dtype=torch.float32).to(self.device).eval()
        self.label2id, self.id2label = build_label_maps()
        # graceful: 파일 있으면 적용, 없으면 raw(빈 dict). confidence_threshold.
        # load_thresholds 는 부재 파일을 가드하지 않으므로 여기서 존재 확인.
        self.thresholds = (load_thresholds(thresholds_path)
                           if os.path.exists(thresholds_path) else {})
        self.has_thresholds = bool(self.thresholds)

    def _encode_chunks(self, chunks: List[Tuple[str, int]]):
        """chunks → (feats, offs_list, bases) — 토큰 인코딩(CPU)."""
        feats, offs_list, bases = [], [], []
        for sub, base in chunks:
            feat, offs = encode_row(
                {'text': sub, 'entities': []}, self.tokenizer,
                self.label2id, self.lang, self.max_length)
            feats.append(feat)
            offs_list.append(offs)
            bases.append(base)
        return feats, offs_list, bases

    def _forward_feats(self, feats: List[dict]):
        """feats(list) → (pred_ids, confs) numpy [N, max_length].

        각 feat 는 encode_row 에서 max_length 패딩되므로 [N, max_length] 한
        배치로 쌓아 1 forward 한다. bf16 은 배치(B>1)에서만 이득이라 단건
        (B=1)은 fp32 로 둔다(autocast off).
        """
        input_ids = torch.tensor(
            [f['input_ids'] for f in feats], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            [f['attention_mask'] for f in feats],
            dtype=torch.long).to(self.device)
        use_autocast = (self.autocast_dtype is not None
                        and input_ids.shape[0] > 1)
        with torch.no_grad(), torch.autocast(
                self.device.type, dtype=self.autocast_dtype,
                enabled=use_autocast):
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            conf_t, pred_t = probs.max(dim=-1)
        return pred_t.cpu().numpy(), conf_t.cpu().numpy()

    def _decode_chunk(self, pred_row, conf_row, offs, base) -> List[dict]:
        """한 chunk 의 (pred,conf) row → 글로벌 offset 내부 span 리스트."""
        n = len(offs)
        pred_ids = [int(x) for x in pred_row[:n]]
        confs = [float(x) for x in conf_row[:n]]
        spans = []
        for sp in decode_bio_to_spans(
                pred_ids, offs, self.id2label, confs=confs):
            shifted = dict(sp)
            shifted['start'] = sp['start'] + base
            shifted['end'] = sp['end'] + base
            spans.append(shifted)
        return spans

    def _infer_chunks(self, chunks: List[Tuple[str, int]]) -> List[dict]:
        """여러 (substring, base_offset) chunk 를 한 forward 로 묶어 추론하고
        글로벌 offset 내부 span({type,start,end,score})으로 병합한다.

        chunk 1개면 배치 차원 1 로 단건 추론과 동일한 결과를 내고
        (behavior-invariant), 장문에서만 GPU 가 chunk 들을 병렬 처리한다.
        """
        feats, offs_list, bases = self._encode_chunks(chunks)
        pred_np, conf_np = self._forward_feats(feats)
        spans: List[dict] = []
        for i, (offs, base) in enumerate(zip(offs_list, bases)):
            spans.extend(
                self._decode_chunk(pred_np[i], conf_np[i], offs, base))
        return spans

    def predict_many(self, texts: List[str],
                     abstain: bool = True) -> List[List[dict]]:
        """여러 텍스트를 언어 공통 배치 forward 로 추론(입력 순서 보존).

        각 텍스트를 chunk 분할한 뒤 전 텍스트의 chunk 를 [TotalChunks,
        max_length] 한 배치로 묶어 1 forward 한다(B>1 → bf16). chunk 를 원
        텍스트로 되돌려 글로벌 offset·임계값을 적용하고 텍스트별 canonical
        span 리스트를 돌려준다 — 텍스트별 단건 predict 순차 호출과 span 이
        동일하다(배치 등가).
        """
        feats, offs_list, bases, owners = [], [], [], []
        for ti, text in enumerate(texts):
            chunks = split_for_length(text, self.tokenizer, self.max_length)
            f, o, b = self._encode_chunks(chunks)
            feats.extend(f)
            offs_list.extend(o)
            bases.extend(b)
            owners.extend([ti] * len(f))
        per_text: List[List[dict]] = [[] for _ in texts]
        if feats:
            pred_np, conf_np = self._forward_feats(feats)
            for i, (offs, base, ti) in enumerate(
                    zip(offs_list, bases, owners)):
                per_text[ti].extend(
                    self._decode_chunk(pred_np[i], conf_np[i], offs, base))
        out: List[List[dict]] = []
        for ti, text in enumerate(texts):
            spans = per_text[ti]
            if abstain and self.thresholds:
                spans = apply_thresholds([spans], self.thresholds)[0]
            out.append([_to_canonical(s, text) for s in spans])
        return out

    def predict(self, text: str, abstain: bool = True) -> List[dict]:
        """텍스트 → canonical span 리스트.

        긴 입력은 chunk 분할 후 모든 chunk 를 한 배치 forward 로 추론하고 각
        span 을 원문 글로벌 offset 으로 병합한다. abstain=True 면 로드된
        임계값을 적용(없으면 raw).
        """
        chunks = split_for_length(text, self.tokenizer, self.max_length)
        spans = self._infer_chunks(chunks)
        # 임계값은 canonical 변환 전 내부 span({type,...})에 적용한다 —
        # confidence_threshold.apply_thresholds 가 type 필드로 필터하며,
        # 이는 학습-시점 eval 경로와 동일하다(parity 보장). decode 가 동일
        # (type,start,end)를 이미 병합해 중복이 없다.
        if abstain and self.thresholds:
            spans = apply_thresholds([spans], self.thresholds)[0]
        return [_to_canonical(s, text) for s in spans]


class ModelRegistry:
    """언어별 LangModel 보관소. 로드 실패한 언어는 503 으로 응답된다."""

    def __init__(self, models: Dict[str, LangModel],
                 requested: Optional[List[str]] = None):
        self._models = models
        # health 가 미로드 언어도 loaded=false 로 보고하도록 요청 목록 보존
        self._requested = list(requested) if requested else list(models)

    @classmethod
    def load(cls, config: ServerConfig) -> 'ModelRegistry':
        """설정의 각 언어 모델을 로드한다(실패는 로깅 후 건너뜀 — graceful)."""
        models: Dict[str, LangModel] = {}
        for lang in config.langs:
            try:
                models[lang] = LangModel(
                    lang=lang,
                    model_dir=config.model_dir(lang),
                    thresholds_path=config.thresholds_path(lang),
                    max_length=config.max_length,
                    precision=config.precision,
                )
                logger.info('Loaded %s model (thresholds=%s)',
                            lang, models[lang].has_thresholds)
            except Exception as exc:  # noqa: BLE001 — 부팅은 계속(503 처리)
                logger.warning('Failed to load %s model: %s', lang, exc)
        return cls(models, requested=list(config.langs))

    def predict(self, text: str, lang: str,
                abstain: bool = True) -> List[dict]:
        """언어 모델로 추론. 미로드 언어면 ModelUnavailable."""
        model = self._models.get(lang)
        if model is None:
            raise ModelUnavailable(f'model for lang {lang!r} is not loaded')
        return model.predict(text, abstain=abstain)

    def predict_batch(self, texts: List[str], langs: List[str],
                      abstain: bool = True) -> List[List[dict]]:
        """(text, lang) 배치를 언어별로 묶어 추론하고 입력 순서로 복원한다.

        같은 언어의 텍스트들을 한 forward 배치(predict_many)로 묶어 GPU
        활용도를 높인다. 미로드 언어가 하나라도 있으면 ModelUnavailable.
        """
        groups: Dict[str, List[int]] = {}
        for idx, lang in enumerate(langs):
            groups.setdefault(lang, []).append(idx)
        results: List[Optional[List[dict]]] = [None] * len(texts)
        for lang, idxs in groups.items():
            model = self._models.get(lang)
            if model is None:
                raise ModelUnavailable(
                    f'model for lang {lang!r} is not loaded')
            sub_out = model.predict_many(
                [texts[i] for i in idxs], abstain=abstain)
            for i, spans in zip(idxs, sub_out):
                results[i] = spans
        return results

    def health(self) -> dict:
        """언어별 로드 상태·임계값 존재 여부 보고(미로드 언어 포함)."""
        langs = {}
        for lang in self._requested:
            model = self._models.get(lang)
            if model is None:
                langs[lang] = {'loaded': False, 'thresholds': False}
            else:
                langs[lang] = {'loaded': True,
                               'thresholds': model.has_thresholds}
        all_loaded = all(v['loaded'] for v in langs.values()) if langs \
            else False
        return {'status': 'ok' if all_loaded else 'degraded',
                'langs': langs}
