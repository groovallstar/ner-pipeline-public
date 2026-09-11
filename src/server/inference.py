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
import threading
import unicodedata
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


class ModelUnavailable(Exception):
    """요청한 언어 모델이 로드되지 않았을 때 — API 는 503 으로 매핑."""


def _load_tokenizer(model_dir: str, lang: str):
    """언어별 토크나이저 로드 — ja 만 slow 를 명시하고 나머지는 fast 를 요청한다.

    ja 를 이름으로 특례 두는 것은 BertJapaneseTokenizer 가 fast 판을 갖지
    않기 때문이다. 나머지는 use_fast=True 로 요청하지만 **요청이 곧 결과는
    아니다** — fast 판이 없는 모델이면 transformers 가 예외 없이 slow 를
    돌려주므로 except 절도 안 탄다. 실제로 fast 로 열리는 것은 ko(WordPiece)
    와 en(ByteLevel BPE)이고, PhoBERT 로 서빙하는 vi 는 slow 로 열린다.
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
        # graceful: 파일 있으면 적용, 없으면 raw(빈 dict). confidence_threshold.
        # load_thresholds 는 부재 파일을 가드하지 않으므로 여기서 존재 확인.
        self.thresholds = (load_thresholds(thresholds_path)
                           if os.path.exists(thresholds_path) else {})
        self.has_thresholds = bool(self.thresholds)
        # 토크나이저 접근을 직렬화하는 잠금 — 사유는 `_tokenize`. 인스턴스마다
        # 따로라 언어끼리는 경합하지 않는다.
        self._tok_lock = threading.Lock()

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

    def _tokenize(self, text: str):
        """text → (feats, offs_list, bases) — 토크나이저 접근은 직렬화한다.

        HF fast 토크나이저는 Rust 객체를 `RefCell` 로 감싸고 있고, 인코딩은
        `no_truncation()` 처럼 그 객체의 상태를 **바꾸는** 호출을 지난다. 서버는
        추론을 `run_in_threadpool` 로 돌리므로 최대 `max_concurrency` 스레드가
        언어당 하나뿐인 이 토크나이저에 동시에 들어오는데, 그러면 두 번째
        borrow 가 `RuntimeError: Already borrowed` 로 터져 요청이 500 이 된다
        (실측: 동시 2 요청만으로 ko·en 의 40~50% 가 실패).

        분할(`split_for_length`)도 토큰 수를 세느라 같은 객체를 만지므로 인코딩과
        한 잠금 안에 둔다. slow 토크나이저(ja)는 이 경로가 아니지만 함께 잠근다 —
        MeCab 은 스레드 안전이 검증된 바 없고, `is_fast` 로 갈라 두면 잠기는
        언어와 아닌 언어가 조용히 어긋난다. 잠금이 인스턴스마다 따로라 ja 를
        잠가도 다른 언어의 처리량에는 닿지 않는다.

        forward 는 잠금 밖이다 — GPU 구간이 길어 여기까지 직렬화하면 동시성
        상한이 무의미해진다.
        """
        with self._tok_lock:
            chunks = split_for_length(text, self.tokenizer, self.max_length)
            return self._encode_chunks(chunks)

    def _forward_feats(self, feats: List[dict]):
        """feats(list) → (pred_ids, confs) numpy [N, max_length].

        각 feat 는 encode_row 에서 max_length 패딩되므로 [N, max_length] 한
        배치로 쌓아 1 forward 한다. 추론은 fp32 로만 돈다 — 단건·배치가 같은
        커널을 타 배치화가 결과를 바꾸지 않는다(결정적, 운영점 정합).
        """
        input_ids = torch.tensor(
            [f['input_ids'] for f in feats], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            [f['attention_mask'] for f in feats],
            dtype=torch.long).to(self.device)
        with torch.no_grad():
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

    def _infer_encoded(self, feats: List[dict], offs_list, bases) -> List[dict]:
        """인코딩된 chunk 들을 한 forward 로 묶어 추론하고 글로벌 offset 내부
        span({type,start,end,score})으로 병합한다 — 토크나이저를 만지지 않는다.

        chunk 1개면 배치 차원 1 로 단건 추론과 동일한 결과를 내고
        (behavior-invariant), 장문에서만 GPU 가 chunk 들을 병렬 처리한다.
        """
        if not feats:
            return []
        pred_np, conf_np = self._forward_feats(feats)
        spans: List[dict] = []
        for i, (offs, base) in enumerate(zip(offs_list, bases)):
            spans.extend(
                self._decode_chunk(pred_np[i], conf_np[i], offs, base))
        return spans

    def predict_many(self, texts: List[str],
                     apply_threshold: bool = True) -> List[List[dict]]:
        """여러 텍스트를 언어 공통 배치 forward 로 추론(입력 순서 보존).

        각 텍스트를 chunk 분할한 뒤 전 텍스트의 chunk 를 [TotalChunks,
        max_length] 한 배치로 묶어 1 forward 한다. chunk 를 원 텍스트로
        되돌려 글로벌 offset·임계값을 적용하고 텍스트별 canonical span
        리스트를 돌려준다 — 추론이 fp32 전용이라(모델 로드 시 고정) 배치
        크기가 커널을 바꾸지 않으므로, 텍스트별 단건 predict 순차 호출과
        span 이 동일하다(배치 등가).
        """
        texts = [unicodedata.normalize('NFC', t) for t in texts]
        feats, offs_list, bases, owners = [], [], [], []
        for ti, text in enumerate(texts):
            f, o, b = self._tokenize(text)
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
            if apply_threshold and self.thresholds:
                spans = apply_thresholds([spans], self.thresholds)[0]
            out.append([_to_canonical(s, text) for s in spans])
        return out

    def predict(self, text: str, apply_threshold: bool = True) -> List[dict]:
        """텍스트 → canonical span 리스트.

        긴 입력은 chunk 분할 후 모든 chunk 를 한 배치 forward 로 추론하고 각
        span 을 원문 글로벌 offset 으로 병합한다. apply_threshold=True 면 로드된
        임계값을 적용(없으면 raw). 입력은 NFC 로 정규화한다 — 모델은 NFC 로
        학습됐고, NFD(분해형) 입력은 결합부호가 별도 토큰이 돼 span 이 깨진다
        (offset 은 정규화된 텍스트 기준).
        """
        text = unicodedata.normalize('NFC', text)
        spans = self._infer_encoded(*self._tokenize(text))
        # 임계값은 canonical 변환 전 내부 span({type,...})에 적용한다 —
        # confidence_threshold.apply_thresholds 가 type 필드로 필터하며,
        # 이는 학습-시점 eval 경로와 동일하다(parity 보장). 반복 offset의
        # 라벨 충돌은 공용 디코더에서 처리한다.
        if apply_threshold and self.thresholds:
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
                )
                logger.info('Loaded %s model (thresholds=%s)',
                            lang, models[lang].has_thresholds)
            except Exception as exc:  # noqa: BLE001 — 부팅은 계속(503 처리)
                logger.warning('Failed to load %s model: %s', lang, exc)
        return cls(models, requested=list(config.langs))

    def predict(self, text: str, lang: str,
                apply_threshold: bool = True) -> List[dict]:
        """언어 모델로 추론. 미로드 언어면 ModelUnavailable."""
        model = self._models.get(lang)
        if model is None:
            raise ModelUnavailable(f'model for lang {lang!r} is not loaded')
        return model.predict(text, apply_threshold=apply_threshold)

    def predict_batch(self, texts: List[str], langs: List[str],
                      apply_threshold: bool = True) -> List[List[dict]]:
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
                [texts[i] for i in idxs], apply_threshold=apply_threshold)
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
