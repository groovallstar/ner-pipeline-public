"""추론 코어 — 언어별 모델을 1회 로드·재사용해 텍스트 → canonical span.

안정적 `ner.classifier.data_utils`(build_label_maps / encode_row /
decode_bio_to_spans)만 의존한다. 학습·평가 모듈(train_eval 등)은 import 하지
않고, 추론 루프(softmax→argmax→BIO decode)를 서버 안에서 직접 돈다.

출력 span 은 프로젝트 canonical 형식 `{label, start_char, end_char, text,
score}` — `.jsonl` 데이터 관례와 일치해 API 결과를 파이프라인에 되먹일 수
있다.
"""

import logging
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from ner.classifier.data_utils import (
    build_label_maps,
    decode_bio_to_spans,
    encode_row,
)
from ner.server.chunking import split_for_length
from ner.server.config import ServerConfig
from ner.server.thresholds import apply_thresholds, load_thresholds

logger = logging.getLogger(__name__)


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
                 max_length: int = 256,
                 device: Optional[torch.device] = None):
        self.lang = lang
        self.max_length = max_length
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = _load_tokenizer(model_dir, lang)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir, dtype=torch.float32).to(self.device).eval()
        self.label2id, self.id2label = build_label_maps()
        self.thresholds = load_thresholds(thresholds_path)
        self.has_thresholds = bool(self.thresholds)

    def _infer_spans(self, text: str) -> List[dict]:
        """길이 한도 내 텍스트 → 내부 {type,start,end,score} span."""
        row = {'text': text, 'entities': []}
        feat, offs = encode_row(
            row, self.tokenizer, self.label2id, self.lang, self.max_length)
        input_ids = torch.tensor(
            [feat['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            [feat['attention_mask']], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)
        conf_t, pred_t = probs.max(dim=-1)
        n = len(offs)
        pred_ids = [int(x) for x in pred_t[0].cpu().numpy()[:n]]
        confs = [float(x) for x in conf_t[0].cpu().numpy()[:n]]
        return decode_bio_to_spans(pred_ids, offs, self.id2label, confs=confs)

    def predict(self, text: str, abstain: bool = True) -> List[dict]:
        """텍스트 → canonical span 리스트.

        긴 입력은 chunk 후 각 span 을 원문 글로벌 offset 으로 병합한다.
        abstain=True 면 로드된 임계값을 적용(없으면 raw).
        """
        spans: List[dict] = []
        for sub, base in split_for_length(
                text, self.tokenizer, self.max_length):
            for sp in self._infer_spans(sub):
                shifted = dict(sp)
                shifted['start'] = sp['start'] + base
                shifted['end'] = sp['end'] + base
                spans.append(shifted)
        # decode_bio_to_spans 가 동일 (type,start,end) 를 이미 단일 span 으로
        # 병합하므로, 임계값은 중복 없는 span 집합에 적용된다(dedup→threshold
        # 순서 모호성 없음).
        canonical = [_to_canonical(s, text) for s in spans]
        if abstain and self.thresholds:
            canonical = apply_thresholds(canonical, self.thresholds)
        return canonical


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
