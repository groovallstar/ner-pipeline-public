"""classifier 학습/평가용 데이터 유틸.

augmenters 가 produce 한 JSONL 파일을 소비한다. 결합도는 얕게 — 직접 import
결합 없이 파일 형식만 contract 로 사용한다.

JSONL 입력 형식 (augmenters/pii, augmenters/wikiann_vi 출력):
    {"text": "...",
     "entities": [{"label": str, "start_char": int, "end_char": int, "text": str}],
     "id": "..."}

라벨 셋: canonical 10종 평면 (NER 5 + PII 5)
- NER 5: PER, LOC, ORG, PROD, EVT
- PII 5: DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD
- BIO 변환: O + 10*B + 10*I = 21 labels

토크나이저 분기:
- JA (BertJapaneseTokenizer, slow): tokenize() 후 text.find() 로 subword char span 추적
- VI (XLMRobertaTokenizer, fast): return_offsets_mapping=True 사용
"""

import json
import random
from typing import Dict, List, Tuple

# canonical 10종 평면 (단일 출처: docs/manual/data/canonical-entity-schema.md)
CANONICAL_LABELS = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
    'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD',
]


def build_label_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    """canonical 10종 → BIO 21 labels (O + 10*B + 10*I)."""
    labels = ['O']
    for t in CANONICAL_LABELS:
        labels.append(f'B-{t}')
        labels.append(f'I-{t}')
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def load_jsonl(path: str) -> List[dict]:
    """JSONL 파일 로드 + entities 의 label 이 canonical 10종 안에 있는지 검증."""
    rows = []
    valid = set(CANONICAL_LABELS)
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            r = json.loads(line)
            for e in r['entities']:
                if e['label'] not in valid:
                    raise ValueError(
                        f"unknown label '{e['label']}' at line {ln} (id={r.get('id')})"
                    )
            rows.append(r)
    return rows


def split_train_test(rows: List[dict], test_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """행 단위 셔플 후 train/test 분할."""
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    return shuffled[n_test:], shuffled[:n_test]


def split_train_valid_test(rows: List[dict],
                           valid_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """행 단위 셔플 후 train/valid/test 3-way 분할.

    분할 순서: shuffle → 앞부분 test, 그 다음 valid, 나머지 train.
    같은 seed 면 결정적이며, test 셋은 split_train_test 와 동일한 prefix.
    """
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    test = shuffled[:n_test]
    valid = shuffled[n_test:n_test + n_valid]
    train = shuffled[n_test + n_valid:]
    return train, valid, test


def split_kfold_stratified(rows, n_folds=5, fold_index=0, seed=42,
                           strat_labels=('PROD', 'EVT')):
    """층화 K-fold 분할. (train, valid, test) 3-way 반환.

    층화 기준: 각 row 의 entities 에 등장하는 strat_labels 부분집합
    (예: 없음/PROD만/EVT만/둘다 = 최대 4개 층). stratum key 는 정렬된 tuple.

    각 층 내에서 row 인덱스를 random.Random(seed) 로 셔플한 뒤 라운드로빈
    (shuffled_position % n_folds) 으로 fold 에 배정한다. 층은 stratum key
    정렬 순서로 결정적으로 순회하므로, 같은 seed 면 fold_index 와 무관하게
    fold 배정이 항상 동일하다.

    test = fold_index 에 배정된 rows, valid = (fold_index+1) % n_folds 에
    배정된 rows, train = 나머지.

    핵심 보장: fold_index 를 0..n_folds-1 로 바꿔가며 호출하면 모든 row 가
    정확히 한 번씩 test 에 등장한다 (valid 도 동일).

    층 크기가 n_folds 로 나누어 떨어지지 않으면 나머지 row 는 낮은 번호의
    fold 에 먼저 배정된다 (라운드로빈 잔여분).

    Args:
        rows: augmenters JSONL 형식 row 리스트
        n_folds: fold 개수 (>= 3 — test/valid 외에 train 이 최소 1 fold 필요)
        fold_index: test 로 쓸 fold 번호 (0 <= fold_index < n_folds)
        seed: 셔플 시드
        strat_labels: 층화 기준 라벨 튜플

    Returns:
        (train_rows, valid_rows, test_rows)
    """
    if n_folds < 3:
        raise ValueError(
            f'n_folds must be >= 3 for 3-way split, got {n_folds}'
        )
    if n_folds > len(rows):
        raise ValueError(
            f'n_folds ({n_folds}) must not exceed number of rows '
            f'({len(rows)})'
        )
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(
            f'fold_index must be in [0, {n_folds}), got {fold_index}'
        )

    strat_set = set(strat_labels)
    # 층화 기준: row 의 entities 에 등장하는 strat_labels 부분집합 (정렬 tuple)
    strata: Dict[tuple, List[int]] = {}
    for i, row in enumerate(rows):
        present = {
            e['label'] for e in row['entities']
            if e['label'] in strat_set
        }
        key = tuple(sorted(present))
        strata.setdefault(key, []).append(i)

    # fold 별 row 인덱스 버킷
    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for key in sorted(strata.keys()):
        bucket = strata[key][:]
        rng.shuffle(bucket)
        for pos, row_idx in enumerate(bucket):
            fold_indices[pos % n_folds].append(row_idx)

    test_idx = fold_indices[fold_index]
    valid_idx = fold_indices[(fold_index + 1) % n_folds]
    test_set = set(test_idx)
    valid_set = set(valid_idx)
    train_rows = [
        rows[i] for i in range(len(rows))
        if i not in test_set and i not in valid_set
    ]
    valid_rows = [rows[i] for i in valid_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, valid_rows, test_rows


def _encode_ja(text: str, tokenizer, max_length: int):
    """JA: slow tokenizer 의 tokenize() 결과를 text 에 greedy match 로 정렬.

    BertJapaneseTokenizer 는 fast 가 아니므로 return_offsets_mapping 미지원.
    각 subword surface 를 text.find(surface, pos) 로 찾아 char span 부여.
    '##' 접두 subword 는 strip 후 매칭한다.
    """
    tokens = tokenizer.tokenize(text)
    input_ids = [tokenizer.cls_token_id]
    char_offsets: List[Tuple[int, int]] = [(0, 0)]

    pos = 0
    for tok in tokens:
        if len(input_ids) >= max_length - 1:
            break
        surface = tok[2:] if tok.startswith('##') else tok
        idx = text.find(surface, pos)
        if idx < 0:
            # text 와 일치 실패 (UNK 등) — pos 유지, span 0-length
            start, end = pos, pos
        else:
            start, end = idx, idx + len(surface)
            pos = end
        sid = tokenizer.convert_tokens_to_ids(tok)
        input_ids.append(sid)
        char_offsets.append((start, end))

    input_ids.append(tokenizer.sep_token_id)
    char_offsets.append((0, 0))

    attention_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        char_offsets.append((0, 0))

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, char_offsets


def _encode_vi(text: str, tokenizer, max_length: int):
    """VI: fast tokenizer 의 offset_mapping 을 그대로 사용."""
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True,
    )
    return (
        {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']},
        list(enc['offset_mapping']),
    )


def _bio_labels_from_offsets(char_offsets: List[Tuple[int, int]],
                             entities: List[dict],
                             label2id: Dict[str, int]) -> List[int]:
    """char offset 리스트 + gold entities → BIO label id 리스트.

    매칭 규칙: 토큰 span [s,e) 가 entity span [es,ee) 안에 완전히 포함되면
    해당 entity 라벨 부여. 직전 토큰이 동일 entity 였으면 I-, 아니면 B-.
    (s,e)=(0,0) 는 special token 으로 -100 (loss ignore).
    """
    o_id = label2id['O']
    labels: List[int] = []
    last_entity = None  # (start_char, end_char, label)

    for start, end in char_offsets:
        if start == 0 and end == 0:
            labels.append(-100)
            last_entity = None
            continue

        matched = None
        for e in entities:
            if e['start_char'] <= start and end <= e['end_char']:
                matched = e
                break

        if matched is None:
            labels.append(o_id)
            last_entity = None
        else:
            key = (matched['start_char'], matched['end_char'], matched['label'])
            tag = 'I' if last_entity == key else 'B'
            labels.append(label2id[f"{tag}-{matched['label']}"])
            last_entity = key

    return labels


def encode_row(row: dict, tokenizer, label2id, lang: str, max_length: int = 256):
    """단일 row → (features, char_offsets).

    토크나이저 capability 로 분기:
    - fast tokenizer (offset_mapping 지원): _encode_vi 경로 (lang 무관)
    - slow tokenizer: _encode_ja 경로 (BertJapaneseTokenizer 같은 MeCab 기반)
    lang 인자는 모델 선택의 컨텍스트로만 유지.
    """
    text = row['text']
    if getattr(tokenizer, 'is_fast', False):
        enc, offs = _encode_vi(text, tokenizer, max_length)
    else:
        enc, offs = _encode_ja(text, tokenizer, max_length)

    labels = _bio_labels_from_offsets(offs, row['entities'], label2id)
    return (
        {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels,
        },
        offs,
    )


def encode_dataset(rows: List[dict], tokenizer, label2id, lang: str, max_length: int = 256):
    """전체 dataset 인코딩. (features_list, offsets_list) 반환.

    features_list 는 학습/평가 모델 입력, offsets_list 는 평가 시 BIO → span 디코드용.
    """
    features = []
    offsets_list = []
    for row in rows:
        feat, offs = encode_row(row, tokenizer, label2id, lang, max_length)
        features.append(feat)
        offsets_list.append(offs)
    return features, offsets_list


NER_TYPES = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')
PII_TYPES = ('DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD')


def class_weights_tensor(label2id: Dict[str, int],
                         w_ner: float = 1.0,
                         w_pii: float = 1.0,
                         w_o: float = 1.0):
    """label2id 기준 BIO 21 라벨에 대한 weight tensor (torch).

    NER 5종 BIO → w_ner, PII 5종 BIO → w_pii, O → w_o.
    """
    import torch
    weights = [w_o] * len(label2id)
    for label, idx in label2id.items():
        if label == 'O':
            weights[idx] = w_o
        elif label[2:] in NER_TYPES:
            weights[idx] = w_ner
        elif label[2:] in PII_TYPES:
            weights[idx] = w_pii
    return torch.tensor(weights, dtype=torch.float32)


def boundary_weights_tensor(label2id: Dict[str, int],
                            w_b: float = 1.0,
                            w_i: float = 1.0,
                            w_o: float = 1.0):
    """B-/I-/O 토큰별 weight tensor — entity 경계 학습 강조용.

    `B-XXX` (entity 시작) 에 w_b, `I-XXX` (내부) 에 w_i, `O` 에 w_o.
    `class_weights_tensor` 와 elementwise 곱으로 결합 가능.
    """
    import torch
    weights = [w_o] * len(label2id)
    for label, idx in label2id.items():
        if label == 'O':
            weights[idx] = w_o
        elif label.startswith('B-'):
            weights[idx] = w_b
        elif label.startswith('I-'):
            weights[idx] = w_i
    return torch.tensor(weights, dtype=torch.float32)


def mask_pii_in_features(features: List[dict], label2id: Dict[str, int]) -> List[dict]:
    """features 의 labels 에서 PII BIO 라벨을 모두 O 로 치환한 새 features 리스트.

    NER warmup curriculum 의 1단계 학습용. -100(special token)은 보존한다.
    원본 features 는 변경하지 않음.
    """
    o_id = label2id['O']
    pii_ids = set()
    for label, idx in label2id.items():
        if label != 'O' and label[2:] in PII_TYPES:
            pii_ids.add(idx)

    out = []
    for f in features:
        new_labels = [
            o_id if (l in pii_ids) else l
            for l in f['labels']
        ]
        out.append({
            'input_ids': f['input_ids'],
            'attention_mask': f['attention_mask'],
            'labels': new_labels,
        })
    return out


def decode_bio_to_spans(label_ids: List[int],
                        char_offsets: List[Tuple[int, int]],
                        id2label: Dict[int, str]) -> List[dict]:
    """BIO label id 시퀀스 + char offsets → list of {type, start, end} spans.

    동일 단어에 같은 char span 이 반복되어도 max(end) 로 병합되어 단일 span 으로 수렴.
    """
    spans: List[dict] = []
    current = None
    for lid, (start, end) in zip(label_ids, char_offsets):
        if start == 0 and end == 0:
            if current:
                spans.append(current)
                current = None
            continue
        label = id2label.get(int(lid), 'O')
        if label == 'O':
            if current:
                spans.append(current)
                current = None
        elif label.startswith('B-'):
            if current:
                spans.append(current)
            current = {'type': label[2:], 'start': start, 'end': end}
        elif label.startswith('I-'):
            etype = label[2:]
            if current and current['type'] == etype:
                current['end'] = max(current['end'], end)
            else:
                # B- 누락된 I- 는 새 span 시작으로 관용 처리
                current = {'type': etype, 'start': start, 'end': end}
    if current:
        spans.append(current)
    return spans
