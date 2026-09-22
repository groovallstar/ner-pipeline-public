"""classifier 학습/평가용 데이터 유틸.

augmenters 가 produce 한 JSONL 파일을 소비한다. 결합도는 얕게 — 직접 import
결합 없이 파일 형식만 contract 로 사용한다.

JSONL 입력 형식 (augmenters 출력):
    {"text": "...",
     "entities": [{"label": str, "start_char": int, "end_char": int, "text": str}],
     "id": "..."}

라벨 셋: canonical 10종 평면 (NER 5 + PII 5)
- NER 5: PER, LOC, ORG, PROD, EVT
- PII 5: DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD
- BIO 변환: O + 10*B + 10*I = 21 labels

토크나이저 분기:
- fast (XLM-R·DeBERTa-V3 등): return_offsets_mapping=True + 공백·후행부호 trim
  - 그중 바이트 BPE(RoBERTa): 앞 공백(`load_tokenizer`) + 공백류 한 칸 정규화
- PhoBERT (slow, 단어분절 전제): pyvi 분절 후 단어별 BPE, 단어 char span 부여
- JA (BertJapaneseTokenizer, slow): tokenize() 후 text.find() 로 subword char span 추적
"""

import hashlib
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


def _build_units(rows: List[dict],
                 group_key: Optional[str]) -> List[List[int]]:
    """분할 단위(unit)를 구성한다.

    group_key 가 None 이면 행 1개가 unit 1개(행 단위 분할). 그렇지 않으면
    같은 row[group_key] 값을 공유하는 행들을 한 unit 으로 묶는다. unit 은
    첫 등장 순서를 보존하므로, 모든 그룹의 크기가 1인 경우 행 단위 분할과
    완전히 동일한 순서가 된다.

    필드가 없는 행이 있으면 KeyError 로 크게 실패한다 — 조용히 넘기면
    누출 보호가 무력화된 채 통과한다.
    """
    if group_key is None:
        return [[i] for i in range(len(rows))]
    groups: Dict[object, List[int]] = {}
    order: List[object] = []
    for i, row in enumerate(rows):
        k = row[group_key]
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    return [groups[k] for k in order]


def group_stats(rows: List[dict],
                group_key: Optional[str]) -> Tuple[int, int]:
    """(행 수, 그룹 수)를 반환한다. group_key=None 이면 그룹 수 = 행 수."""
    if group_key is None:
        return len(rows), len(rows)
    return len(rows), len({row[group_key] for row in rows})


def dataset_fingerprint(rows: List[dict]) -> str:
    """데이터 내용의 순서 민감 지문(sha256 앞 16자리)을 계산한다.

    각 행을 (text, 정렬된 (label,start_char,end_char) 튜플)로 정규화해 파일
    순서대로 해시한다. 이 지문이 실험의 '자'(test gold) 를 대표한다 —
    같은 지문이면 두 실험은 같은 문장·같은 정답을 재고 있다.

    순서에 민감한 이유: 분할이 seed shuffle 이라 행 순서가 fold 멤버십을 바꾼다.
    행을 재정렬만 해도 test gold 의 fold 분해가 달라지므로 다른 자로 본다.
    """
    h = hashlib.sha256()
    for row in rows:
        spans = sorted(
            (e['label'], e['start_char'], e['end_char'])
            for e in row['entities']
        )
        h.update(repr((row['text'], spans)).encode('utf-8'))
        h.update(b'\x00')  # 행 경계 — 연접 모호성 방지
    return h.hexdigest()[:16]


def _coarsens(rows: List[dict], group_key: Optional[str], field: str) -> bool:
    """field 의 그룹이 group_key 의 그룹을 쪼개지 않고 통째로 포함하는가.

    후보 필드가 선언한 키의 그룹 경계를 가로지르면(같은 group_key 값의 행이
    서로 다른 field 값을 가지면) 그것은 형제 표시가 아니라 그냥 다른 축의
    범주다 — 예: 문서 도메인·라벨 유무. 그런 필드는 그룹 수가 적더라도 형제를
    묶는 키가 아니므로 후보에서 뺀다.

    group_key=None(행 단위)이면 모든 그룹이 싱글턴이라 어떤 필드든 포함한다.
    """
    if group_key is None:
        return True
    seen: Dict[object, object] = {}
    for row in rows:
        key, value = row[group_key], row[field]
        if key in seen:
            if seen[key] != value:
                return False
        else:
            seen[key] = value
    return True


def stronger_group_keys(rows: List[dict],
                        group_key: Optional[str]) -> Dict[str, int]:
    """선언한 group_key 보다 행을 더 강하게 묶는 후보 필드를 찾는다.

    더 강하다 = (1) 그룹 수가 더 적고, (2) 선언한 키의 그룹을 쪼개지 않는다.
    (2)가 없으면 도메인·카테고리 같은 범주 필드가 그룹 수만 적다는 이유로
    올바른 형제 키를 밀어낸다.

    이 검사가 필요한 이유: 고유값 필드(예: 행 일련번호)를 group_key 로 주면
    모든 unit 이 싱글턴이 되어 그룹 보호가 no-op 이 되는데, 같은 필드로 누출을
    세면 중복이 0 이라 "누출 없음" 으로 잘못 읽힌다.

    후보에서 제외: 선언한 키 자신, 값이 hashable 하지 않은 필드(entities 등),
    일부 행에만 있는 필드, 그룹 수가 1 인 필드(전체를 한 덩어리로 묶어 분할
    자체가 불가능하므로 후보가 아니다).

    한계: group_key=None 이면 (2)가 항상 참이라 범주 필드도 후보로 잡힐 수
    있다. 이때는 후보가 여럿 보고되므로 사람이 형제 표시를 고른다.

    Returns:
        {필드명: 그룹 수} — 비어 있으면 선언한 키가 가장 강하다.
    """
    if not rows:
        return {}
    base = group_stats(rows, group_key)[1]
    common = set(rows[0])
    for row in rows[1:]:
        common &= set(row)
    out: Dict[str, int] = {}
    for field in sorted(common):
        if field == group_key:
            continue
        try:
            n_groups = len({row[field] for row in rows})
        except TypeError:  # list/dict 값 — 그룹 키가 될 수 없다
            continue
        if 1 < n_groups < base and _coarsens(rows, group_key, field):
            out[field] = n_groups
    return out


def validate_group_key(rows: List[dict], group_key: Optional[str]) -> None:
    """선언한 group_key 가 형제 행을 실제로 묶는지 검증한다 (fail-loud).

    세 가지를 본다: 필드 존재, 값의 유효성, 더 강한 후보의 부재. 하나라도
    위반하면 ValueError — 조용한 통과는 누출을 "이상 없음" 으로 만든다.
    """
    if not rows:
        return
    if group_key is not None:
        missing = sum(1 for row in rows if group_key not in row)
        if missing:
            raise ValueError(
                f'group key {group_key!r} missing in {missing}/{len(rows)} '
                f'rows; cannot verify leak-free split'
            )
        empty = sum(1 for row in rows if row[group_key] is None)
        if empty:
            raise ValueError(
                f'group key {group_key!r} is null in {empty}/{len(rows)} '
                f'rows; cannot verify leak-free split'
            )
    stronger = stronger_group_keys(rows, group_key)
    if stronger:
        declared = 'none (row-level)' if group_key is None else repr(group_key)
        found = ', '.join(f'{f}={n}' for f, n in sorted(stronger.items()))
        raise ValueError(
            f'group key {declared} leaves {group_stats(rows, group_key)[1]} '
            f'groups over {len(rows)} rows, but a stronger key exists '
            f'({found}). Sibling rows would be split across train/test. '
            f'Pass the stronger field to --group-key.'
        )


def split_train_valid_test(rows: List[dict],
                           valid_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42,
                           group_key: Optional[str] = None
                           ) -> Tuple[List[dict], List[dict], List[dict]]:
    """unit 단위 셔플 후 train/valid/test 3-way 분할.

    group_key 가 주어지면 같은 row[group_key] 값을 공유하는 행을 한 unit 으로
    묶어 통째로 한 split 에만 둔다 — 같은 원문에서 파생된 형제 행이 train·test
    로 갈리는 누출이 구조적으로 불가능해진다. group_key=None 이면 행 단위
    분할이며 기존 동작과 완전히 동일하다(같은 seed → 동일 결과).

    분할 순서: unit 셔플 → 앞에서부터 누적 행 수가 목표에 이를 때까지 test,
    다음 valid, 나머지 train. unit 을 통째로 배정하므로 group_key 가 있으면
    실제 비율이 목표를 약간 넘을 수 있다.
    """
    units = _build_units(rows, group_key)
    rng = random.Random(seed)
    rng.shuffle(units)
    n = len(rows)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    test_idx: List[int] = []
    valid_idx: List[int] = []
    train_idx: List[int] = []
    for unit in units:
        if len(test_idx) < n_test:
            test_idx.extend(unit)
        elif len(valid_idx) < n_valid:
            valid_idx.extend(unit)
        else:
            train_idx.extend(unit)
    return ([rows[i] for i in train_idx],
            [rows[i] for i in valid_idx],
            [rows[i] for i in test_idx])


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
    units = _build_units(rows, group_key)

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
    units = _build_units(rows, group_key)

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
    애초에 zero-length 인 offset 은 그대로 둔다.

    다만 토큰이 공백·문장부호만으로 이뤄져 trim 결과가 비면 원래 offset 을
    돌려준다. `_bio_labels_from_offsets` 의 포함 검사가 양끝을 포함하므로
    (`es <= s and e <= ee`), 길이 0 조각은 엔티티가 *끝나는* 지점에서 그 조건을
    통과해 엔티티 밖 문장부호가 엔티티 안으로 라벨된다. 원래 offset 을 유지하면
    그 조각이 엔티티 밖 char 를 들고 있어 포함 검사에서 떨어진다. 엔티티가
    문장부호로 끝나는 경우(`Inc.`)에도 마지막 char 가 살아 있어야 gold 라벨을
    디코드했을 때 원래 span 이 복원된다.
    """
    orig = (start, end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _TRAIL_PUNCT):
        end -= 1
    if start == end and orig[0] != orig[1]:
        return orig
    return (start, end)


def is_byte_level(tokenizer) -> bool:
    """바이트 BPE(GPT-2·RoBERTa 계열) fast 토크나이저인지 판별한다.

    이 계열만 공백의 유무·종류·개수를 토큰 모양(`Ġ`·`Ċ`·`ĉ`)에 새긴다.
    `add_prefix_space` 속성은 SentencePiece 계열에도 있어 판별에 쓰지 않는다.
    """
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        return False
    from tokenizers import pre_tokenizers
    return isinstance(backend.pre_tokenizer, pre_tokenizers.ByteLevel)


def load_tokenizer(name_or_path: str):
    """학습·포장·분석이 함께 쓰는 토크나이저 로더.

    fast 를 먼저 열고, 없으면(BertJapaneseTokenizer 같은 MeCab 기반) slow 로
    내려간다. 바이트 BPE 는 `add_prefix_space=True` 로 다시 연다. 그러지 않으면
    문자열 첫 단어만 `Ġ` 없는 토큰이 되어, 학습에서 공백 뒤 모양으로만 본 단어가
    첫 자리에서는 낯선 조각으로 쪼개진다(주소 `alice@…` 의 `al`+`ice`).

    포장이 이 로더로 연 토크나이저를 `model/` 에 저장하므로 설정이 모델과 함께
    배포되고, 서버는 저장된 파일을 그대로 연다. 학습과 포장이 다른 로더를 쓰면
    배포 모델이 학습 때와 다른 토큰을 받는다.
    """
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except (TypeError, ValueError, OSError):
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)
    if is_byte_level(tokenizer) and not tokenizer.add_prefix_space:
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, use_fast=True, add_prefix_space=True)
    return tokenizer


def canonical_whitespace(text: str) -> Tuple[str, List[int]]:
    """앞뒤 공백을 떼고 공백류의 연속을 공백 한 칸으로 접는다.

    돌려주는 목록은 접힌 텍스트의 각 글자가 원문 몇 번째 글자였는지다.
    바이트 BPE 는 개행·탭(`Ċ`·`ĉ`)과 연속 공백의 여분(단독 `Ġ`)을 따로 된
    토큰으로 쓰는데, en 코퍼스에는 개행·탭이 없고 연속 공백은 카드번호 표기
    안에만 있다. 그런 토큰이 주소 앞에 오면 모델이 주소를 깨뜨리므로, 모델이
    학습에서 본 공백 한 칸 모양으로 맞춘다.
    """
    chars: List[str] = []
    index: List[int] = []
    pending = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending = bool(chars)
            continue
        if pending:
            chars.append(' ')
            index.append(i - 1)
            pending = False
        chars.append(ch)
        index.append(i)
    return ''.join(chars), index


def _unfold_offset(index: List[int], start: int, end: int,
                   n_chars: int) -> Tuple[int, int]:
    """접힌 텍스트의 offset 을 원문 위치로 되돌린다.

    (0,0) 은 특수 토큰 표지라 그대로 둔다. 실제 토큰은 첫 글자와 마지막 글자를
    옮긴다 — 접힌 공백은 토큰 가운데에만 올 수 있으므로 양끝 위치면 충분하다.
    """
    if start == end == 0:
        return (0, 0)
    if end > start:
        return (index[start], index[end - 1] + 1)
    pos = index[start] if start < len(index) else n_chars
    return (pos, pos)


def _encode_vi(text: str, tokenizer, max_length: int):
    """VI: fast tokenizer 의 offset_mapping 을 공백·후행부호 trim 후 사용.

    바이트 BPE 는 `canonical_whitespace` 로 접은 텍스트를 인코딩하고 offset 을
    원문 위치로 되돌린다. 다른 fast 토크나이저는 원문을 그대로 넣는다.
    """
    text_for_tok, index = text, None
    if is_byte_level(tokenizer):
        folded, fold_index = canonical_whitespace(text)
        if folded != text:
            text_for_tok, index = folded, fold_index
    enc = tokenizer(
        text_for_tok,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True,
    )
    raw = enc['offset_mapping']
    if index is not None:
        raw = [_unfold_offset(index, s, e, len(text)) for s, e in raw]
    offsets = [_trim_offset(text, s, e) for s, e in raw]
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
                   max_length: int = 256):
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


def decode_bio_to_spans(label_ids: List[int],
                        char_offsets: List[Tuple[int, int]],
                        id2label: Dict[int, str],
                        confs: Optional[List[float]] = None) -> List[dict]:
    """BIO label id 시퀀스 + char offsets → list of {type, start, end} spans.

    연속한 동일 offset은 첫 서브워드 라벨을 따른다. 같은 라벨의 후속 B/I는
    이어 붙이고, 충돌하는 라벨은 무시하여 같은 문자 구간을 중복 출력하지 않는다.

    confs 가 주어지면(토큰별 신뢰도, label_ids 와 동일 길이) 각 span 에
    `score` = span 구성 토큰 신뢰도의 평균(conf_mean)을 부착한다. confs 미입력
    시에는 score 키를 달지 않아 기존 동작과 완전히 동일하다(BC).
    """
    spans: List[dict] = []
    current = None
    cur_confs: Optional[List[float]] = None
    previous_offset = None
    first_label = 'O'

    def _push() -> None:
        # confs 입력 시에만 score 부착 — 그 외엔 기존 출력과 동일
        if confs is not None and cur_confs:
            current['score'] = float(sum(cur_confs) / len(cur_confs))
        spans.append(current)

    for idx, (lid, (start, end)) in enumerate(zip(label_ids, char_offsets)):
        cf = confs[idx] if confs is not None else None
        if start == 0 and end == 0:
            previous_offset = None
            if current:
                _push()
                current = None
                cur_confs = None
            continue
        label = id2label.get(int(lid), 'O')
        if (start, end) == previous_offset:
            if first_label == 'O' or label[2:] != first_label[2:]:
                continue
            label = 'I-' + first_label[2:]
        else:
            previous_offset = (start, end)
            first_label = label
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
                if current:
                    _push()
                current = {'type': etype, 'start': start, 'end': end}
                cur_confs = [cf] if cf is not None else None
    if current:
        _push()
    return spans


EMAIL_TYPE = 'EMAIL'


def merge_email_fragments(spans: List[dict], text: str) -> List[dict]:
    """공백 없이 이어진 EMAIL 조각을 한 span 으로 합친다.

    서브워드마다 따로 라벨을 내는 토크나이저에서는 주소의 한 조각만 O 로
    떨어져도, 혹은 같은 주소 안에서 B 가 다시 나와도 디코더가 주소를 여러
    span 으로 낸다. 이메일에는 공백이 없으므로 사이에 공백이 없는 두 EMAIL
    조각은 같은 주소다. 사이의 O 글자까지 덮어 한 span 으로 만든다.

    합치지 않는 경우가 셋이다. 사이에 공백이 있으면 다른 주소일 수 있고,
    합친 결과에 `@` 가 둘 이상이면 쉼표로 붙여 쓴 주소 목록이며, 두 조각
    사이에 다른 엔티티가 있으면 그것을 삼키게 된다. score 가 있으면 조각 중
    가장 낮은 값을 준다 — 평균을 내면 확신 없는 조각이 임계값을 통과한다.
    """
    out: List[dict] = []
    for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
        prev = out[-1] if out else None
        if (prev is not None
                and prev['type'] == EMAIL_TYPE
                and span['type'] == EMAIL_TYPE
                and not any(c.isspace()
                            for c in text[prev['end']:span['start']])
                and text[prev['start']:max(prev['end'], span['end'])]
                .count('@') <= 1):
            merged = dict(prev)
            merged['end'] = max(prev['end'], span['end'])
            if 'score' in prev and 'score' in span:
                merged['score'] = min(prev['score'], span['score'])
            out[-1] = merged
            continue
        out.append(dict(span))
    return out
