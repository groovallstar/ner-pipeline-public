"""classifier 학습/평가 코어.

HuggingFace Trainer 래퍼 + char-offset span F1 (src/ner/metrics 공용).
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner.classifier.data_utils import NER_TYPES, decode_bio_to_spans
from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)


class BertCRFForTokenClassification(nn.Module):
    """BERT token-classification + linear-chain CRF head.

    학습: -log_likelihood (Viterbi forward).
    평가: Viterbi decode 로 BIO 일관성 강제.
    save/load 는 base HF 모델과 CRF transition matrix 를 별개로 저장.

    HuggingFace Trainer 가 PreTrainedModel attribute 를 일부 가정하므로
    호환용 더미 속성을 노출한다.
    """

    # Trainer._issue_warnings_after_load 가 PreTrainedModel 가정 attribute
    _keys_to_ignore_on_save: Optional[list] = None

    def __init__(self,
                 base_model_name_or_path: str,
                 num_labels: int,
                 id2label: Optional[Dict[int, str]] = None,
                 label2id: Optional[Dict[str, int]] = None) -> None:
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_model_name_or_path,
            num_labels=num_labels,
            id2label=id2label or {},
            label2id=label2id or {},
            ignore_mismatched_sizes=True,
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.base(
            input_ids=input_ids, attention_mask=attention_mask
        )
        emissions = outputs.logits

        if labels is None:
            return {'logits': emissions}

        # CRF mask: -100 (ignore_index) 위치 제외 + attention mask 적용.
        # CRF 는 첫 step 이 항상 valid 여야 함 (CLS 토큰).
        mask = (labels != -100) & (attention_mask == 1)
        mask[:, 0] = True
        safe_labels = labels.masked_fill(labels == -100, 0)
        loss = -self.crf(
            emissions, safe_labels, mask=mask, reduction='mean'
        )
        return {'loss': loss, 'logits': emissions}

    def decode(self, input_ids, attention_mask) -> List[List[int]]:
        """Viterbi decode. attention_mask 가 1 인 토큰만 디코드."""
        with torch.no_grad():
            emissions = self.base(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            return self.crf.decode(emissions, mask=attention_mask.bool())

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.base.save_pretrained(output_dir)
        torch.save(
            self.crf.state_dict(), os.path.join(output_dir, 'crf.pt')
        )

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        num_labels: int,
                        id2label: Optional[Dict[int, str]] = None,
                        label2id: Optional[Dict[str, int]] = None
                        ) -> 'BertCRFForTokenClassification':
        instance = cls(model_path, num_labels, id2label, label2id)
        crf_path = os.path.join(model_path, 'crf.pt')
        if os.path.exists(crf_path):
            instance.crf.load_state_dict(
                torch.load(crf_path, weights_only=True, map_location='cpu')
            )
        return instance


class WeightedTrainer(Trainer):
    """class-weighted cross-entropy 를 적용하는 Trainer 변형.

    None 이면 표준 CE 와 동일.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def _save(self, output_dir: Optional[str] = None,
              state_dict=None) -> None:
        # BertCRFForTokenClassification 은 nn.Module 직속이라 safetensors
        # 의 contiguous 검사에 걸린다. CRF wrapper 의 save_pretrained 로
        # base 모델 + crf.pt 분리 저장.
        if isinstance(self.model, BertCRFForTokenClassification):
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            torch.save(
                self.args, os.path.join(output_dir, 'training_args.bin')
            )
            return
        super()._save(output_dir, state_dict)

    def _load_best_model(self) -> None:
        # CRF wrapper 는 base + crf.pt 분리 저장이라 기본 _load_best_model
        # (model.safetensors 만) 으로는 CRF transition matrix 가 누락된다.
        if isinstance(self.model, BertCRFForTokenClassification):
            best = self.state.best_model_checkpoint
            if not best:
                return
            device = next(self.model.parameters()).device
            new_base = AutoModelForTokenClassification.from_pretrained(
                best,
                num_labels=self.model.num_labels,
                ignore_mismatched_sizes=True,
            ).to(device)
            self.model.base = new_base
            crf_path = os.path.join(best, 'crf.pt')
            if os.path.exists(crf_path):
                self.model.crf.load_state_dict(
                    torch.load(crf_path, weights_only=True,
                               map_location=device)
                )
            return
        super()._load_best_model()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        # CRF 모델은 forward 에서 -log_likelihood 를 직접 계산하므로
        # labels 를 전달하고 outputs['loss'] 를 그대로 사용한다.
        if isinstance(model, BertCRFForTokenClassification):
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (self.class_weights.to(logits.device)
                  if self.class_weights is not None else None)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def _build_compute_metrics(id2label: Dict[int, str]):
    """Trainer 의 epoch eval 단계에서 overall_f1 / ner_f1 / pii_f1 을 산출.

    metric_for_best_model='ner_f1' 로 지정하면 NER 5종 best epoch 가 선택된다.
    span 일관성을 위해 char-offset 가 아니라 BIO 시퀀스 기준 token-level micro F1 을 사용
    (best-model 선택 신호용 — 본 평가는 끝난 뒤 char-offset span F1 로 다시 측정).
    """
    from seqeval.metrics import f1_score

    def _filter_to_types(seq, types):
        out = []
        for tag in seq:
            if tag == 'O':
                out.append('O')
            elif tag[2:] in types:
                out.append(tag)
            else:
                out.append('O')
        return out

    def compute(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        true_labels: List[List[str]] = []
        true_preds: List[List[str]] = []
        for pred_seq, label_seq in zip(preds, labels):
            tp, tl = [], []
            for p, lab in zip(pred_seq, label_seq):
                if lab == -100:
                    continue
                tp.append(id2label.get(int(p), 'O'))
                tl.append(id2label.get(int(lab), 'O'))
            true_preds.append(tp)
            true_labels.append(tl)

        overall_f1 = f1_score(true_labels, true_preds, average='micro')

        ner_labels = [_filter_to_types(s, NER_TYPES) for s in true_labels]
        ner_preds = [_filter_to_types(s, NER_TYPES) for s in true_preds]
        any_ner = any(any(t != 'O' for t in s) for s in ner_labels)
        ner_f1 = f1_score(ner_labels, ner_preds, average='micro') if any_ner else 0.0

        return {'overall_f1': overall_f1, 'ner_f1': ner_f1}

    return compute


class NERDataset(Dataset):
    """encode_dataset 가 반환한 features 리스트를 tensor 로 묶어주는 dataset."""

    def __init__(self, features: List[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self.features[idx]
        return {
            'input_ids': torch.tensor(f['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(f['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(f['labels'], dtype=torch.long),
        }


def fine_tune(*, model_name: str,
              train_features: List[dict],
              eval_features: List[dict],
              label2id: Dict[str, int],
              id2label: Dict[int, str],
              output_dir: str,
              epochs: int = 5,
              batch_size: int = 16,
              lr: float = 5e-5,
              class_weights: Optional[torch.Tensor] = None,
              metric_for_best: str = 'eval_loss',
              init_model_path: Optional[str] = None,
              precision: str = 'fp16',
              use_crf: bool = False) -> Tuple[float, str]:
    """HF Trainer 로 fine-tune. best 모델을 output_dir/best 에 저장.

    Args:
        class_weights: BIO 21 라벨용 weight tensor (None = 표준 CE)
        metric_for_best: 'eval_loss' (낮을수록 좋음) 또는 'ner_f1' / 'overall_f1' (높을수록 좋음)
        init_model_path: None 또는 기존 HF 모델 경로 (curriculum stage 2 용)
        use_crf: True 면 BertCRFForTokenClassification 으로 학습.
            CRF loss 가 -log_likelihood 라 class_weights 는 무시된다.

    Returns:
        (학습 시간 sec, best 모델 디렉토리 경로)
    """
    init_path = init_model_path if init_model_path is not None else model_name
    if use_crf:
        model = BertCRFForTokenClassification(
            init_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            init_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    greater_is_better = (metric_for_best != 'eval_loss')
    needs_compute_metrics = (metric_for_best in ('ner_f1', 'overall_f1'))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        save_total_limit=1,
        report_to=[],
        logging_steps=100,
        fp16=(torch.cuda.is_available() and precision == 'fp16'),
        bf16=(torch.cuda.is_available() and precision == 'bf16'),
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=NERDataset(train_features),
        eval_dataset=NERDataset(eval_features),
        class_weights=class_weights,
        compute_metrics=_build_compute_metrics(id2label) if needs_compute_metrics else None,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    best_dir = os.path.join(output_dir, 'best')
    trainer.save_model(best_dir)
    return elapsed, best_dir


def evaluate_model(*, model_path: str,
                   eval_features: List[dict],
                   eval_offsets: List[List[Tuple[int, int]]],
                   eval_rows: List[dict],
                   id2label: Dict[int, str],
                   batch_size: int = 32,
                   return_spans: bool = False,
                   capture_scores: bool = False) -> dict:
    """best 모델 로드 → predict → BIO decode → strict + relaxed span F1 계산.

    eval_rows 는 augmenters JSONL 형식 그대로 (label/start_char/end_char).
    내부에서 metrics 모듈이 요구하는 (type/start/end) 형식으로 변환.

    Args:
        return_spans: True 면 반환 dict 에 'gold_spans_list',
            'pred_spans_list' 키를 추가한다 (kfold pooled 평가용).
        capture_scores: True 면 토큰 softmax 신뢰도를 포착해 각 pred span 에
            'score'(conf_mean)를 부착한다 (abstention fit·apply 용). non-CRF
            경로만 지원 — CRF(Viterbi)는 score 미부착. score 는 metrics 에
            영향 없음(여전히 strict/relaxed 는 임계값 미적용 raw 기준).

    Returns:
        {"strict": {overall, per_entity}, "relaxed": {overall, per_entity}}
        strict = (start, end, type) 정확 일치 (게이트 측정 default).
        relaxed = SemEval'13 Partial — type 일치 + char-offset overlap 시 0.5점.
        return_spans=True 면 'gold_spans_list', 'pred_spans_list' 추가.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CRF 모델 감지: best 디렉토리에 crf.pt 가 있으면 CRF wrapper 로 로드.
    is_crf = os.path.exists(os.path.join(model_path, 'crf.pt'))
    if is_crf:
        label2id = {v: k for k, v in id2label.items()}
        model = BertCRFForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        ).to(device).eval()
    else:
        # 평가는 항상 float32 로 (DeBERTa-v3 family 의 fp16 NaN underflow 회피)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, dtype=torch.float32
        ).to(device).eval()

    pred_spans_list: List[List[dict]] = []
    gold_spans_list: List[List[dict]] = []

    n = len(eval_features)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = eval_features[i:i + batch_size]
            input_ids = torch.tensor(
                [f['input_ids'] for f in batch], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask'] for f in batch], dtype=torch.long
            ).to(device)
            if is_crf:
                # Viterbi decode — variable-length tag list per sentence
                pred_lists = model.decode(input_ids, attention_mask)
                for j, pred_ids in enumerate(pred_lists):
                    offs = eval_offsets[i + j]
                    pred_ids_list = [int(x) for x in pred_ids[:len(offs)]]
                    spans = decode_bio_to_spans(pred_ids_list, offs, id2label)
                    pred_spans_list.append(spans)
                    gold = [
                        {'type': e['label'],
                         'start': e['start_char'],
                         'end': e['end_char']}
                        for e in eval_rows[i + j]['entities']
                    ]
                    gold_spans_list.append(gold)
            else:
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                if capture_scores:
                    probs = torch.softmax(logits, dim=-1)
                    conf_t, pred_t = probs.max(dim=-1)
                    confs_np = conf_t.cpu().numpy()
                    preds = pred_t.cpu().numpy()
                else:
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    confs_np = None
                for j, pred_ids in enumerate(preds):
                    offs = eval_offsets[i + j]
                    pred_ids_list = [int(x) for x in pred_ids[:len(offs)]]
                    cf = ([float(x) for x in confs_np[j][:len(offs)]]
                          if confs_np is not None else None)
                    spans = decode_bio_to_spans(
                        pred_ids_list, offs, id2label, confs=cf)
                    pred_spans_list.append(spans)
                    gold = [
                        {'type': e['label'],
                         'start': e['start_char'],
                         'end': e['end_char']}
                        for e in eval_rows[i + j]['entities']
                    ]
                    gold_spans_list.append(gold)

    result = {
        'strict': compute_offset_span_f1(gold_spans_list, pred_spans_list),
        'relaxed': compute_offset_span_f1_relaxed(
            gold_spans_list, pred_spans_list
        ),
    }
    if return_spans:
        result['gold_spans_list'] = gold_spans_list
        result['pred_spans_list'] = pred_spans_list
    return result
