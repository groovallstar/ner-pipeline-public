"""classifier 학습/평가 코어.

HuggingFace Trainer 래퍼 + char-offset span F1 (src/ner/metrics 공용).
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ner.classifier.data_utils import (
    decode_bio_to_spans,
    merge_email_fragments,
)
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
              precision: str = 'fp16',
              train_seed: Optional[int] = None) -> Tuple[float, str]:
    """HF Trainer 로 fine-tune. best 모델을 output_dir/best 에 저장.

    best 모델 선택은 valid eval_loss 기준 (낮을수록 좋음).

    Args:
        class_weights: BIO 21 라벨용 weight tensor (None = 표준 CE)
        train_seed: 학습 seed — 데이터 split seed 와 분리. None(기본)이면
            모델 생성 전 시드를 건너뛰어 기존 비시드 헤드 init 동작을
            유지한다(BC). 값을 주면 그 seed 로 헤드 init·dropout·셔플을
            고정해 재현 가능한 run 을 만든다.

    Returns:
        (학습 시간 sec, best 모델 디렉토리 경로)
    """
    # 학습 seed 를 모델 생성 *전에* 고정한다 — from_pretrained 의 새 분류
    # 헤드 random init 은 Trainer 내부 set_seed(학습 루프 직전)보다 먼저
    # 일어나, 여기서 시드해야 train_seed 가 헤드 init 까지 제어한다.
    # train_seed=None(기본)이면 시드를 건너뛰어 기존 비시드 동작을 유지한다(BC).
    if train_seed is not None:
        set_seed(train_seed)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=1,
        report_to=[],
        logging_steps=100,
        fp16=(torch.cuda.is_available() and precision == 'fp16'),
        bf16=(torch.cuda.is_available() and precision == 'bf16'),
        seed=train_seed if train_seed is not None else 42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=NERDataset(train_features),
        eval_dataset=NERDataset(eval_features),
        class_weights=class_weights,
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
                   capture_scores: bool = False,
                   capture_timing: bool = False) -> dict:
    """best 모델 로드 → predict → BIO decode → strict + relaxed span F1 계산.

    eval_rows 는 augmenters JSONL 형식 그대로 (label/start_char/end_char).
    내부에서 metrics 모듈이 요구하는 (type/start/end) 형식으로 변환.

    Args:
        return_spans: True 면 반환 dict 에 'gold_spans_list',
            'pred_spans_list' 키를 추가한다 (kfold pooled 평가용).
        capture_scores: True 면 토큰 softmax 신뢰도를 포착해 각 pred span 에
            'score'(conf_mean)를 부착한다 (신뢰도 임계값 fit·apply 용). score 는
            metrics 에 영향 없음(여전히 strict/relaxed 는 임계값 미적용 raw 기준).
        capture_timing: True 면 반환 dict 에 'load_seconds'(모델 로드),
            'infer_seconds'(추론 루프 전체) 키를 초 단위로 추가한다.

    Returns:
        {"strict": {overall, per_entity}, "relaxed": {overall, per_entity}}
        strict = (start, end, type) 정확 일치 (게이트 측정 default).
        relaxed = SemEval'13 Partial — type 일치 + char-offset overlap 시 0.5점.
        return_spans=True 면 'gold_spans_list', 'pred_spans_list' 추가.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 평가는 항상 float32 로 (DeBERTa-v3 family 의 fp16 NaN underflow 회피)
    t_load = time.perf_counter()
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, dtype=torch.float32
    ).to(device).eval()
    load_seconds = time.perf_counter() - t_load

    pred_spans_list: List[List[dict]] = []
    gold_spans_list: List[List[dict]] = []

    n = len(eval_features)
    t_infer = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = eval_features[i:i + batch_size]
            input_ids = torch.tensor(
                [f['input_ids'] for f in batch], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask'] for f in batch], dtype=torch.long
            ).to(device)
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
                spans = merge_email_fragments(
                    decode_bio_to_spans(
                        pred_ids_list, offs, id2label, confs=cf),
                    eval_rows[i + j]['text'])
                pred_spans_list.append(spans)
                gold = [
                    {'type': e['label'],
                     'start': e['start_char'],
                     'end': e['end_char']}
                    for e in eval_rows[i + j]['entities']
                ]
                gold_spans_list.append(gold)
    infer_seconds = time.perf_counter() - t_infer

    result = {
        'strict': compute_offset_span_f1(gold_spans_list, pred_spans_list),
        'relaxed': compute_offset_span_f1_relaxed(
            gold_spans_list, pred_spans_list
        ),
    }
    if return_spans:
        result['gold_spans_list'] = gold_spans_list
        result['pred_spans_list'] = pred_spans_list
    if capture_timing:
        result['load_seconds'] = load_seconds
        result['infer_seconds'] = infer_seconds
    return result
