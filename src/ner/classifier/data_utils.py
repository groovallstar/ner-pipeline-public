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
- fast (XLM-R·DeBERTa-V3 등): return_offsets_mapping=True + 공백·후행부호 trim
- PhoBERT (slow, 단어분절 전제): pyvi 분절 후 단어별 BPE, 단어 char span 부여
- JA (BertJapaneseTokenizer, slow): tokenize() 후 text.find() 로 subword char span 추적
"""

import json
import random
from typing import Dict, List, Optional, Tuple

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
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
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


def split_train_valid_test(rows: List[dict],
                           valid_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """행 단위 셔플 후 train/valid/test 3-way 분할.

    분할 순서: shuffle → 앞부분 test, 그 다음 valid, 나머지 train.
    같은 seed 면 결정적이다.
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


def _context_word_count(row: dict) -> int:
    """엔티티 span 밖(문맥) 실단어 수 — '엔티티만 있는 단편' 판별용.

    구두점·괄호 등 알파넘 문자가 없는 토큰은 단어로 세지 않는다(위키
    동음이의 '( 회사 )'·리다이렉트 단편이 문맥어로 잘못 잡히는 것 방지).
    """
    text = row['text']
    spans = [(e['start_char'], e['end_char']) for e in row['entities']]
    n = 0
    pos = 0
    for tok in text.split():
        j = text.find(tok, pos)
        pos = j + len(tok)
        if any(s <= j < e for s, e in spans):
            continue
        if any(ch.isalnum() for ch in tok):
            n += 1
    return n


def split_holdout_deploy(rows: List[dict],
                         n_test: int = 100,
                         valid_ratio: float = 0.2,
                         seed: int = 42,
                         group_key: Optional[str] = None,
                         test_require_types: Optional[set] = None,
                         test_min_context_words: int = 0
                         ) -> Tuple[List[dict], List[dict], List[dict]]:
    """배포용 단일 분할: test n_test 행을 홀드아웃 후 나머지를 train/valid.

    배포 모델 학습용 — K-fold(교차검증 추정)와 달리 한 모델을 만들고 고정
    test 로 최종 점검한다. 남은 데이터는 valid_ratio 만큼 검증에 배분(나머지
    train).

    group_key 가 주어지면 같은 row[group_key] 값을 공유하는 행을 한 unit 으로
    묶어 통째로 한 split 에만 둔다 → test 원문이 train/valid 로 새는 cross-split
    누출이 구조적으로 불가능해진다(WikiANN 류 원문 중복 코퍼스에서 배포 점검의
    정직성 보장). group_key=None 이면 행 단위 분할.

    분할: unit 을 seed 로 셔플 → 앞에서부터 누적 행 수가 n_test 이상이 될
    때까지 test(unit 통째라 정확히 n_test 가 아닐 수 있음) → 남은 unit 에서
    누적 행 수가 (남은 행)*valid_ratio 이상이 될 때까지 valid → 나머지 train.
    같은 seed 면 결정적이다.

    Args:
        rows: augmenters JSONL 형식 row 리스트
        n_test: 홀드아웃할 test 최소 행 수
        valid_ratio: 남은(비-test) 행 중 valid 비율 [0, 1)
        seed: 셔플 시드
        group_key: 같은 split 으로 묶을 그룹 키 필드명(예: 'orig').
        test_require_types: 주어지면 test 는 unit 의 모든 row 가 이 라벨을
            ≥1개 가질 때만 뽑는다(예: canonical 전체 → 빈 샘플 제외). 미적격
            unit 은 train/valid 로 간다. None 이면 필터 없음.
        test_min_context_words: test 는 unit 의 모든 row 가 엔티티 밖 문맥어를
            이 수 이상 가질 때만(예: 3 → 엔티티+문맥 1~2단어짜리 단편 제외).
            0 이면 필터 없음. test_require_types 와 AND 로 적용.

    Returns:
        (train_rows, valid_rows, test_rows)
    """
    if n_test <= 0:
        raise ValueError(f'n_test must be positive, got {n_test}')
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(
            f'valid_ratio must be in [0, 1), got {valid_ratio}'
        )

    # 분할 단위 구성 (group_key=None → 행 1개가 unit)
    if group_key is None:
        units: List[List[int]] = [[i] for i in range(len(rows))]
    else:
        groups: Dict[object, List[int]] = {}
        order: List[object] = []
        for i, row in enumerate(rows):
            k = row[group_key]
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(i)
        units = [groups[k] for k in order]

    rng = random.Random(seed)
    rng.shuffle(units)

    # test 적격: unit 의 모든 row 가 (test_require_types 라벨 ≥1개) ∧
    # (엔티티 밖 문맥어 ≥ test_min_context_words) 를 만족해야 test 후보다.
    # 한 row 라도 미달이면 unit 은 train/valid 로 (test 에 빈·단편 샘플 차단).
    req = set(test_require_types) if test_require_types else None

    def _row_ok(r: dict) -> bool:
        if req is not None and not any(
                e['label'] in req for e in r['entities']):
            return False
        if (test_min_context_words > 0
                and _context_word_count(r) < test_min_context_words):
            return False
        return True

    def _eligible(unit: List[int]) -> bool:
        return all(_row_ok(rows[ri]) for ri in unit)

    # 셔플 순서로 적격 unit 을 누적 행수 n_test 이상까지 test, 나머지 remaining
    test_idx: List[int] = []
    remaining: List[List[int]] = []
    for unit in units:
        if len(test_idx) < n_test and _eligible(unit):
            test_idx.extend(unit)
        else:
            remaining.append(unit)
    if len(test_idx) < n_test:
        raise ValueError(
            f'not enough eligible rows for n_test={n_test} '
            f'(got {len(test_idx)}; require_types={test_require_types})'
        )

    # 남은 unit → valid_ratio 만큼 valid, 나머지 train
    n_valid_target = int(sum(len(g) for g in remaining) * valid_ratio)
    valid_idx: List[int] = []
    v = 0
    while v < len(remaining) and len(valid_idx) < n_valid_target:
        valid_idx.extend(remaining[v])
        v += 1
    train_idx: List[int] = [i for g in remaining[v:] for i in g]

    train = [rows[i] for i in train_idx]
    valid = [rows[i] for i in valid_idx]
    test = [rows[i] for i in test_idx]
    return train, valid, test


def split_kfold_stratified(rows, n_folds=5, fold_index=0, seed=42,
                           strat_labels=('PROD', 'EVT'), group_key=None):
    """층화 K-fold 분할. (train, valid, test) 3-way 반환.

    분할 단위(unit): group_key 가 None 이면 row 1개가 unit 1개(행 단위).
    group_key 가 주어지면 같은 row[group_key] 값을 공유하는 row 들을 한
    unit 으로 묶고, unit 을 통째로 한 fold 에 배정한다. 이렇게 하면 같은
    원문에서 파생된 여러 행(예: 같은 원문에 서로 다른 PII 를 주입한 행들)이
    train·test 로 갈리는 cross-fold 누출이 구조적으로 불가능해진다.

    층화 기준: 각 unit 안 row 들의 entities 에 등장하는 strat_labels 합집합
    (예: 없음/PROD만/EVT만/둘다 = 최대 4개 층). stratum key 는 정렬된 tuple.

    각 층 내에서 unit 인덱스를 random.Random(seed) 로 셔플한 뒤 라운드로빈
    (shuffled_position % n_folds) 으로 fold 에 배정한다. 층은 stratum key
    정렬 순서로 결정적으로 순회하므로, 같은 seed 면 fold_index 와 무관하게
    fold 배정이 항상 동일하다.

    test = fold_index 에 배정된 unit 들의 rows, valid = (fold_index+1) %
    n_folds 에 배정된 unit 들의 rows, train = 나머지.

    핵심 보장: fold_index 를 0..n_folds-1 로 바꿔가며 호출하면 모든 row 가
    정확히 한 번씩 test 에 등장한다 (valid 도 동일). group_key 가 None 이면
    기존 행 단위 분할과 완전히 동일하다(같은 seed → 동일 결과).

    층 크기가 n_folds 로 나누어 떨어지지 않으면 나머지 unit 은 낮은 번호의
    fold 에 먼저 배정된다 (라운드로빈 잔여분).

    Args:
        rows: augmenters JSONL 형식 row 리스트
        n_folds: fold 개수 (>= 3 — test/valid 외에 train 이 최소 1 fold 필요)
        fold_index: test 로 쓸 fold 번호 (0 <= fold_index < n_folds)
        seed: 셔플 시드
        strat_labels: 층화 기준 라벨 튜플
        group_key: 같은 fold 로 묶을 그룹 키 필드명(예: 'orig'). None 이면
            행 단위 분할.

    Returns:
        (train_rows, valid_rows, test_rows)
    """
    if n_folds < 3:
        raise ValueError(
            f'n_folds must be >= 3 for 3-way split, got {n_folds}'
        )
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(
            f'fold_index must be in [0, {n_folds}), got {fold_index}'
        )

    # 분할 단위 구성. group_key=None 이면 unit=행 1개(기존과 동일한 순서).
    if group_key is None:
        units: List[List[int]] = [[i] for i in range(len(rows))]
    else:
        groups: Dict[object, List[int]] = {}
        order: List[object] = []
        for i, row in enumerate(rows):
            k = row[group_key]
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(i)
        units = [groups[k] for k in order]

    if n_folds > len(units):
        raise ValueError(
            f'n_folds ({n_folds}) must not exceed number of split units '
            f'({len(units)}; group_key={group_key!r})'
        )

    strat_set = set(strat_labels)
    # 층화 기준: unit 안 row 들의 strat_labels 합집합 (정렬 tuple)
    strata: Dict[tuple, List[int]] = {}
    for ui, unit in enumerate(units):
        present: set = set()
        for ri in unit:
            present |= {
                e['label'] for e in rows[ri]['entities']
                if e['label'] in strat_set
            }
        key = tuple(sorted(present))
        strata.setdefault(key, []).append(ui)

    # fold 별 unit 인덱스 버킷
    fold_units: List[List[int]] = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for key in sorted(strata.keys()):
        bucket = strata[key][:]
        rng.shuffle(bucket)
        for pos, ui in enumerate(bucket):
            fold_units[pos % n_folds].append(ui)

    def _row_indices(fold_unit_idx: List[int]) -> List[int]:
        out: List[int] = []
        for ui in fold_unit_idx:
            out.extend(units[ui])
        return out

    test_idx = _row_indices(fold_units[fold_index])
    valid_idx = _row_indices(fold_units[(fold_index + 1) % n_folds])
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


# 후행에서 떼어낼 문장부호 — SentencePiece 계열이 entity 끝 토큰에 흡착시키는
# 마침표·쉼표만. ')'·':' 등은 entity 에 정당히 포함될 수 있어 제외한다.
_TRAIL_PUNCT = '.,'


def _trim_offset(text: str, start: int, end: int) -> Tuple[int, int]:
    """fast tokenizer offset 에서 선행 공백 + 후행 공백·문장부호를 제외해
    char span 을 토큰의 entity 관련 내용에 맞춘다.

    두 가지 토크나이저 입도 차이를 흡수한다:
    1. DeBERTa-V3(SentencePiece)는 `▁` 토큰 offset 에 선행 공백을 포함시켜
       (예: '▁Võ' → ' Võ') entity 첫 토큰 start 가 경계보다 1 작아진다.
    2. 다국어 SentencePiece 는 숫자형 entity 끝과 문장부호를 한 토큰으로
       병합한다(예: '568.' → 끝이 entity 경계를 넘어감).
    공백·후행 `.`/`,` 를 trim 하면 토크나이저 무관하게 정렬되고, 이미 분리해
    내는 XLM-R 계열에는 사실상 no-op 이다. (start,end)=(0,0) 특수토큰·
    zero-length 는 그대로 둔다.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _TRAIL_PUNCT):
        end -= 1
    return (start, end)


def _encode_vi(text: str, tokenizer, max_length: int, trim: bool = True):
    """VI: fast tokenizer 의 offset_mapping 을 (기본) 공백·후행부호 trim 후 사용.

    trim=False 는 정렬 수정 이전 동작을 재현하는 진단용 — SentencePiece 계열의
    offset 어긋남으로 라벨이 붕괴(F1 ≈ 0)하는 as-is 벤치마크 재현에만 쓴다.
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True,
    )
    raw = enc['offset_mapping']
    offsets = [_trim_offset(text, s, e) for s, e in raw] if trim else list(raw)
    return (
        {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']},
        offsets,
    )


def _is_phobert(tokenizer) -> bool:
    """PhoBERT 계열(단어분절 필요 + slow BPE) 토크나이저 판별."""
    name = type(tokenizer).__name__.lower()
    path = getattr(tokenizer, 'name_or_path', '').lower()
    return 'phobert' in name or 'phobert' in path


def _word_spans_vi(text: str) -> List[Tuple[str, int, int]]:
    """pyvi 단어분절 → [(분절표면, start_char, end_char)] 원문 char 정렬.

    pyvi 는 다음절 단어를 '_' 로 잇고 문장부호 주위에 공백을 넣어 원문과
    char 정렬이 깨진다. 분절 표면을 음절('_' 분리)로 쪼개 원문에서 cursor
    순차 탐색해 각 단어의 원문 char span 을 복원한다.
    """
    from pyvi import ViTokenizer
    seg = ViTokenizer.tokenize(text)
    out: List[Tuple[str, int, int]] = []
    pos = 0
    for token in seg.split(' '):
        if not token:
            continue
        start = end = None
        for syll in token.split('_'):
            idx = text.find(syll, pos)
            if idx < 0:
                continue
            if start is None:
                start = idx
            end = idx + len(syll)
            pos = end
        if start is not None:
            out.append((token, start, end))
    return out


def _encode_phobert(text: str, tokenizer, max_length: int):
    """PhoBERT: pyvi 단어분절 후 단어별 BPE, 각 subword 에 단어의 원문 char
    span 을 부여한다.

    PhoBERT 는 fast tokenizer·offset_mapping 미지원이고 입력이 단어분절을
    전제하므로, 분절 단어 단위로 char span 을 정렬한다. entity 가 단어 경계에
    정렬되는 한 단어 단위 span 으로 BIO 라벨·디코드가 정확히 복원된다.
    """
    word_spans = _word_spans_vi(text)
    input_ids = [tokenizer.cls_token_id]
    char_offsets: List[Tuple[int, int]] = [(0, 0)]
    for surface, start, end in word_spans:
        for sub in tokenizer.tokenize(surface):
            if len(input_ids) >= max_length - 1:
                break
            input_ids.append(tokenizer.convert_tokens_to_ids(sub))
            char_offsets.append((start, end))

    input_ids.append(tokenizer.sep_token_id)
    char_offsets.append((0, 0))

    attention_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        char_offsets.append((0, 0))

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, char_offsets


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


def encode_row(row: dict, tokenizer, label2id, lang: str, max_length: int = 256,
               trim_offsets: bool = True):
    """단일 row → (features, char_offsets).

    토크나이저 capability 로 분기:
    - fast tokenizer (offset_mapping 지원): _encode_vi 경로 (lang 무관)
    - slow tokenizer: _encode_ja 경로 (BertJapaneseTokenizer 같은 MeCab 기반)
    lang 인자는 모델 선택의 컨텍스트로만 유지. trim_offsets=False 는 fast 경로
    의 offset trim 을 끄는 진단용(as-is 붕괴 재현).
    """
    text = row['text']
    if getattr(tokenizer, 'is_fast', False):
        enc, offs = _encode_vi(text, tokenizer, max_length, trim=trim_offsets)
    elif _is_phobert(tokenizer):
        enc, offs = _encode_phobert(text, tokenizer, max_length)
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


def encode_dataset(rows: List[dict], tokenizer, label2id, lang: str,
                   max_length: int = 256, trim_offsets: bool = True):
    """전체 dataset 인코딩. (features_list, offsets_list) 반환.

    features_list 는 학습/평가 모델 입력, offsets_list 는 평가 시 BIO → span 디코드용.
    trim_offsets=False 는 fast 경로 offset trim 을 끄는 진단용(as-is 붕괴 재현).
    """
    features = []
    offsets_list = []
    for row in rows:
        feat, offs = encode_row(
            row, tokenizer, label2id, lang, max_length, trim_offsets
        )
        features.append(feat)
        offsets_list.append(offs)
    return features, offsets_list


NER_TYPES = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')
PII_TYPES = ('DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD')


def boundary_weights_tensor(label2id: Dict[str, int],
                            w_b: float = 1.0,
                            w_i: float = 1.0,
                            w_o: float = 1.0):
    """B-/I-/O 토큰별 weight tensor — entity 경계 학습 강조용.

    `B-XXX` (entity 시작) 에 w_b, `I-XXX` (내부) 에 w_i, `O` 에 w_o.
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
            o_id if (lid in pii_ids) else lid
            for lid in f['labels']
        ]
        out.append({
            'input_ids': f['input_ids'],
            'attention_mask': f['attention_mask'],
            'labels': new_labels,
        })
    return out


def decode_bio_to_spans(label_ids: List[int],
                        char_offsets: List[Tuple[int, int]],
                        id2label: Dict[int, str],
                        confs: Optional[List[float]] = None) -> List[dict]:
    """BIO label id 시퀀스 + char offsets → list of {type, start, end} spans.

    동일 단어에 같은 char span 이 반복되어도 max(end) 로 병합되어 단일 span 으로 수렴.

    confs 가 주어지면(토큰별 신뢰도, label_ids 와 동일 길이) 각 span 에
    `score` = span 구성 토큰 신뢰도의 평균(conf_mean)을 부착한다. confs 미입력
    시에는 score 키를 달지 않아 기존 동작과 완전히 동일하다(BC).
    """
    spans: List[dict] = []
    current = None
    cur_confs: Optional[List[float]] = None

    def _push() -> None:
        # confs 입력 시에만 score 부착 — 그 외엔 기존 출력과 동일
        if confs is not None and cur_confs:
            current['score'] = float(sum(cur_confs) / len(cur_confs))
        spans.append(current)

    for idx, (lid, (start, end)) in enumerate(zip(label_ids, char_offsets)):
        cf = confs[idx] if confs is not None else None
        if start == 0 and end == 0:
            if current:
                _push()
                current = None
                cur_confs = None
            continue
        label = id2label.get(int(lid), 'O')
        if label == 'O':
            if current:
                _push()
                current = None
                cur_confs = None
        elif label.startswith('B-'):
            if current:
                _push()
            current = {'type': label[2:], 'start': start, 'end': end}
            cur_confs = [cf] if cf is not None else None
        elif label.startswith('I-'):
            etype = label[2:]
            if current and current['type'] == etype:
                current['end'] = max(current['end'], end)
                if cur_confs is not None:
                    cur_confs.append(cf)
            else:
                # B- 누락된 I- 는 새 span 시작으로 관용 처리
                current = {'type': etype, 'start': start, 'end': end}
                cur_confs = [cf] if cf is not None else None
    if current:
        _push()
    return spans
