"""classifier 학습/평가 코어.

HuggingFace Trainer 래퍼 + char-offset span F1 (src/ner/metrics 공용).
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ner.classifier.data_utils import NER_TYPES, decode_bio_to_spans
from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)


class WeightedTrainer(Trainer):
    """class-weighted cross-entropy 를 적용하는 Trainer 변형.

    None 이면 표준 CE 와 동일.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
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
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                tp.append(id2label.get(int(p), 'O'))
                tl.append(id2label.get(int(l), 'O'))
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
              precision: str = 'fp16') -> Tuple[float, str]:
    """HF Trainer 로 fine-tune. best 모델을 output_dir/best 에 저장.

    Args:
        class_weights: BIO 21 라벨용 weight tensor (None = 표준 CE)
        metric_for_best: 'eval_loss' (낮을수록 좋음) 또는 'ner_f1' / 'overall_f1' (높을수록 좋음)
        init_model_path: None 또는 기존 HF 모델 경로 (curriculum stage 2 용)

    Returns:
        (학습 시간 sec, best 모델 디렉토리 경로)
    """
    init_path = init_model_path if init_model_path is not None else model_name
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
                   batch_size: int = 32) -> dict:
    """best 모델 로드 → predict → BIO decode → strict + relaxed span F1 계산.

    eval_rows 는 augmenters JSONL 형식 그대로 (label/start_char/end_char).
    내부에서 metrics 모듈이 요구하는 (type/start/end) 형식으로 변환.

    Returns:
        {"strict": {overall, per_entity}, "relaxed": {overall, per_entity}}
        strict = (start, end, type) 정확 일치 (게이트 측정 default).
        relaxed = SemEval'13 Partial — type 일치 + char-offset overlap 시 0.5점.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 평가는 항상 float32 로 (DeBERTa-v3 family 의 fp16 NaN underflow 회피)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, torch_dtype=torch.float32
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1).cpu().numpy()

            for j, pred_ids in enumerate(preds):
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

    return {
        'strict': compute_offset_span_f1(gold_spans_list, pred_spans_list),
        'relaxed': compute_offset_span_f1_relaxed(
            gold_spans_list, pred_spans_list
        ),
    }
