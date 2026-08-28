"""토큰 char-offset 정렬 감사 — 라벨 경계 위반과 gold 왕복 복원율을 잰다.

두 가지를 재며, 둘 다 학습된 모델 없이 결정적으로 돌아간다.

1. **경계 위반 라벨** — 엔티티 라벨(`B-`/`I-`)을 받은 토큰이 엔티티 밖 char 를
   포함하거나(`s < es` 또는 `e > ee`), 엔티티 종료 경계에 놓인 길이 0 조각
   (`s == e` 이고 `s == ee`)인 건수. 엔티티 *내부*의 길이 0(`es < s < ee`)은
   라벨이 옳으므로 위반이 아니다 — 토크나이저가 raw 로 길이 0 offset 을 주는
   자리(연속 공백 등)가 여기 해당한다.
2. **왕복 복원율** — gold 라벨을 그대로 `decode_bio_to_spans` 에 넣어 gold
   엔티티가 strict(start·end·type 완전일치)로 되살아나는 비율. 복원 실패는
   *모델이 완벽해도* 그 엔티티를 strict 로 맞힐 수 없다는 뜻이라, 채점의
   구조적 상한이 된다.

`--model-dir` 을 주면 셋째로 **고정 체크포인트 재평가**를 한다. 학습이 없어
결정적이며, 토큰별 예측 label id 시퀀스의 해시를 함께 남긴다 — offset 정렬을
바꾼 전후로 이 해시가 같아야 점수 변화가 정렬 변경에서만 왔다고 말할 수 있다
(input_ids 는 offset 과 무관하므로 같아야 정상이고, 다르면 정렬 외의 무언가가
움직인 것이다).

사용:
    python src/ner/scripts/audit_offset_alignment.py \\
        --data /abs/origin.jsonl --model-name roberta-base --out /abs/out.json

    python src/ner/scripts/audit_offset_alignment.py \\
        --data /abs/test.jsonl --model-dir /abs/model --out /abs/out.json
"""
import argparse
import hashlib
import json
import os
from collections import Counter

from transformers import AutoTokenizer

from ner.classifier.data_utils import (
    build_label_maps,
    decode_bio_to_spans,
    encode_row,
    load_jsonl,
)
from ner.metrics.span_metrics import compute_offset_span_f1


def require_abs(path, label):
    """데이터·모델 경로는 전체(절대) 경로만 허용 — 상대 경로면 종료."""
    if not os.path.isabs(path):
        raise SystemExit(
            f'Error: {label} must be an absolute path (got: {path})')


def classify_violation(start, end, entities, etype):
    """엔티티 라벨을 받은 토큰 하나를 위반 종류로 분류. 정상이면 None.

    host 는 이 토큰을 담을 수 있는 같은 타입 엔티티다. 라벨을 붙인 매처가
    `es <= s and e <= ee` 를 요구하므로 정상 경로에서 `outside_char` 는 나올 수
    없다 — 그래서 이 가지는 매칭 규칙이 바뀌면 울리는 불변식 감시로 남긴다.
    """
    hosts = [e for e in entities
             if e['label'] == etype
             and e['start_char'] <= start <= e['end_char']]
    if not hosts:
        return 'no_host'
    host = hosts[0]
    if start < host['start_char'] or end > host['end_char']:
        return 'outside_char'
    if start == end and start == host['end_char']:
        return 'zero_at_end'
    return None


def audit(rows, tokenizer, label2id, id2label, lang, max_length):
    """경계 위반 건수와 gold 왕복 복원율을 함께 센다."""
    violations = Counter()
    by_type = Counter()
    recovered = Counter()
    missed = Counter()
    entity_labeled = 0

    for row in rows:
        feat, offsets = encode_row(
            row, tokenizer, label2id, lang, max_length)
        entities = row['entities']

        for (start, end), label_id in zip(offsets, feat['labels']):
            if label_id == -100:
                continue
            label = id2label[label_id]
            if label == 'O':
                continue
            entity_labeled += 1
            kind = classify_violation(start, end, entities, label[2:])
            if kind is not None:
                violations[kind] += 1
                by_type[f'{kind}/{label[2:]}'] += 1

        decoded = {
            (s['start'], s['end'], s['type'])
            for s in decode_bio_to_spans(feat['labels'], offsets, id2label)
        }
        for entity in entities:
            key = (entity['start_char'], entity['end_char'], entity['label'])
            bucket = recovered if key in decoded else missed
            bucket[entity['label']] += 1

    total_ok = sum(recovered.values())
    total_miss = sum(missed.values())
    total = total_ok + total_miss
    per_type = {}
    for etype in sorted(set(recovered) | set(missed)):
        n = recovered[etype] + missed[etype]
        per_type[etype] = {
            'recovered': recovered[etype],
            'missed': missed[etype],
            'ceiling': recovered[etype] / n if n else 0.0,
        }
    return {
        'rows': len(rows),
        'entity_labeled_tokens': entity_labeled,
        'boundary_violations': {
            'total': sum(violations.values()),
            'by_kind': dict(sorted(violations.items())),
            'by_kind_type': dict(sorted(by_type.items())),
        },
        'roundtrip': {
            'entities': total,
            'recovered': total_ok,
            'missed': total_miss,
            'ceiling': total_ok / total if total else 0.0,
            'per_type': per_type,
        },
    }


def evaluate_checkpoint(rows, tokenizer, label2id, id2label, lang,
                        max_length, model_dir, batch_size):
    """고정 체크포인트로 test 를 재평가하고 예측 label id 해시를 남긴다.

    train_eval.evaluate_model 을 쓰지 않고 추론 루프를 직접 도는 것은 토큰별
    argmax 를 그대로 손에 쥐어 해시해야 하기 때문이다 — 그 해시가 정렬 변경
    전후로 같아야 점수 차이를 정렬 탓으로 돌릴 수 있다.
    """
    import torch
    from transformers import AutoModelForTokenClassification

    features = []
    offsets_list = []
    for row in rows:
        feat, offsets = encode_row(
            row, tokenizer, label2id, lang, max_length)
        features.append(feat)
        offsets_list.append(offsets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, dtype=torch.float32
    ).to(device).eval()

    digest = hashlib.sha256()
    input_digest = hashlib.sha256()
    pred_spans_list = []
    gold_spans_list = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            input_ids = torch.tensor(
                [f['input_ids'] for f in batch], dtype=torch.long).to(device)
            attention_mask = torch.tensor(
                [f['attention_mask'] for f in batch],
                dtype=torch.long).to(device)
            preds = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.argmax(dim=-1).cpu().numpy()
            for j, pred_ids in enumerate(preds):
                offsets = offsets_list[i + j]
                ids = [int(x) for x in pred_ids[:len(offsets)]]
                digest.update((','.join(map(str, ids)) + '\n').encode())
                input_digest.update(
                    (','.join(map(str, batch[j]['input_ids'])) + '\n').encode())
                pred_spans_list.append(
                    decode_bio_to_spans(ids, offsets, id2label))
                gold_spans_list.append([
                    {'type': e['label'],
                     'start': e['start_char'],
                     'end': e['end_char']}
                    for e in rows[i + j]['entities']
                ])

    return {
        'model_dir': model_dir,
        'rows': len(rows),
        'pred_label_ids_sha256': digest.hexdigest(),
        'input_ids_sha256': input_digest.hexdigest(),
        'strict': compute_offset_span_f1(gold_spans_list, pred_spans_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Audit token char-offset alignment: count entity-label '
                    'boundary violations and measure the gold round-trip '
                    'ceiling. With --model-dir, also re-evaluate a fixed '
                    'checkpoint and hash its token-level predictions. '
                    'Paths must be absolute.')
    parser.add_argument('--data', required=True,
                        help='Absolute corpus JSONL path')
    parser.add_argument('--lang', default='en',
                        choices=['ja', 'vi', 'ko', 'en'],
                        help='Language context for the encoder branch')
    parser.add_argument('--model-name', default=None,
                        help='Tokenizer name or path for the audit '
                             '(defaults to --model-dir when given)')
    parser.add_argument('--model-dir', default=None,
                        help='Absolute checkpoint dir to re-evaluate')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Tokenizer max length')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Inference batch size for --model-dir')
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only the first N rows (0 = all)')
    parser.add_argument('--out', default=None,
                        help='Absolute path to write the result JSON')
    args = parser.parse_args()

    require_abs(args.data, 'data path')
    if args.model_dir:
        require_abs(args.model_dir, 'model dir')
    if args.out:
        require_abs(args.out, 'output path')

    tokenizer_src = args.model_name or args.model_dir
    if not tokenizer_src:
        raise SystemExit('Error: give --model-name or --model-dir')

    label2id, id2label = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[:args.limit]

    result = {
        'data': args.data,
        'lang': args.lang,
        'tokenizer': tokenizer_src,
        'max_length': args.max_length,
        'audit': audit(rows, tokenizer, label2id, id2label,
                       args.lang, args.max_length),
    }
    if args.model_dir:
        result['checkpoint'] = evaluate_checkpoint(
            rows, tokenizer, label2id, id2label, args.lang,
            args.max_length, args.model_dir, args.batch_size)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
