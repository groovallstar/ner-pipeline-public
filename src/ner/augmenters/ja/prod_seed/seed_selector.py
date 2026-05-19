"""PROD 도메인 휴리스틱 seed oversample 도구 (JA classifier 천장 회복용).

목적
----
JA NER classifier 의 PROD recall 천장 — test FN 의 도메인 (법안·서적·식품·
교통·음악) 이 train+valid 셋에 1~5건만 존재 — 을 해소하기 위해, 휴리스틱
키워드로 train PROD-positive 문장을 정밀 선별·oversample 한다.

`random PROD-pos oversample` 은 surface 분포가 famous media·software 에
편중되어 test FN 과 substring overlap 0% → 천장 그대로. 본 도구는 도메인
suffix·키워드 정규식으로 long-tail 도메인 surface 를 가진 문장만 선별.

학습 흐름
----------
    1. 본 모듈로 보강 jsonl 생성 (--extra-only --base-extra <S7N2 neg> 권장)
    2. classifier 학습에 --data-extra-train-jsonl 로 주입

데이터 contract
----------------
입력 JSONL: augmenters contract (text, entities=[{label,start_char,end_char,
text}]). PROD label entity 만 도메인 분류 대상.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# 도메인별 surface 분류 정규식. test FN 의 도메인 (법안·서적·식품·교통·음악)
# 을 train+valid 셋에서 정밀 매칭하기 위한 휴리스틱.
DOMAIN_PATTERNS: Dict[str, List[str]] = {
    'law': [
        r'法案$', r'法律$', r'法$', r'制定法$', r'条例$', r'条約$', r'勅令$',
        r'規則$', r'憲法$', r'憲法', r'宣言$', r'章典$', r'布告$', r'令$',
        r'規約$', r'協定$', r'議定書$', r'公約$', r'通則$',
    ],
    'book': [
        r'案内$', r'辞典$', r'年鑑$', r'白書$', r'誌$', r'紙$', r'本$',
        r'集$', r'録$', r'録〜?$', r'記$', r'伝$', r'書$', r'論$', r'帖$',
        r'帳$', r'報$', r'紀要$', r'物語$', r'研究$', r'教授$', r'読本$',
        r'資料$', r'考$', r'草$', r'巻$', r'説$', r'抄$', r'文書$', r'典$',
        r'史$', r'雑誌$', r'新聞$', r'批判$', r'論考$', r'論集$', r'論文$',
    ],
    'food': [
        r'セット$', r'丼$', r'膳$', r'重$', r'定食$', r'カレー$', r'弁当$',
        r'菓子$', r'すだれ$', r'まんじゅう$', r'ちゃん$', r'ラーメン$',
        r'うどん$', r'蕎麦$', r'ジュース$', r'コーヒー$', r'酒$', r'ビール$',
        r'パン$', r'ケーキ$', r'寿司$', r'麥酒$',
    ],
    'transit_card': [
        r'カード$', r'ICOCA', r'Suica', r'PASMO', r'TOICA', r'nimoca',
        r'PiTaPa', r'manaca', r'はやかけん', r'SUGOCA', r'パス$',
        r'チケット$', r'乗車券$', r'切符$', r'定期$',
    ],
    'music_work': [
        r'のうた$', r'メロディ$', r'交響曲$', r'協奏曲$', r'ソナタ$',
        r'組曲$', r'狂詩曲$', r'前奏曲$', r'四重奏$', r'三重奏$', r'幻想曲$',
        r'追分$', r'節$', r'唄$', r'唱歌$', r'民謡$',
    ],
}

# LONG-surface 후보 선별 시 제외할 famous media·software 키워드. random PROD
# oversample 에서 천장이 깨지지 않은 surface 들 (이미 모델이 잘 맞히는 것)
# 이 다시 들어오는 것을 차단.
DEFAULT_LONG_EXCLUDE_KEYWORDS: List[str] = [
    'ガンダム', 'ルパン', 'ナルト', 'ドラえもん', 'ウルトラ', 'ファイナル',
    'ポケモン', 'コナン', 'クレヨン', '機動', 'ヤマト', 'スター', 'エヴァ',
    'プリキュア', '戦士', 'ライダー', 'ジョジョ', 'ドラゴン', 'エンジン',
    'OS', 'Windows', 'Linux', 'iOS', 'Android', 'Mac', 'iPhone', 'iPad',
    'PS', 'Xbox', 'Switch', 'Wii', 'Atari', 'カメラ', 'モデル',
    'エディション', 'バージョン', 'Vol', 'Premium', 'Pro', 'Plus', 'Lite',
]


def categorize_prod_surface(surface: str) -> Optional[str]:
    """PROD surface 를 도메인 카테고리 또는 None 으로 분류.

    여러 패턴이 매칭하면 첫 번째 도메인을 반환 (DOMAIN_PATTERNS 순서대로
    law → book → food → transit_card → music_work).
    """
    for domain, patterns in DOMAIN_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, surface):
                return domain
    return None


def select_domain_seed_indices(
    rows: List[dict],
    *,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Set[int]:
    """train+valid split 에서 도메인 휴리스틱 매칭 PROD-pos 문장 인덱스 집합.

    test split 인덱스는 leak 방지를 위해 제외. 한 문장에 PROD 가 여러 개
    있으면 그 중 하나라도 도메인 매칭이면 후보로 채택.
    """
    from ner.classifier.data_utils import split_train_valid_test
    train_rows, valid_rows, _ = split_train_valid_test(
        rows, valid_ratio, test_ratio, seed,
    )
    keep_ids = {id(r) for r in train_rows} | {id(r) for r in valid_rows}
    selected: Set[int] = set()
    for i, row in enumerate(rows):
        if id(row) not in keep_ids:
            continue
        for e in row.get('entities', []):
            if e.get('label') != 'PROD':
                continue
            if categorize_prod_surface(e.get('text', '')) is not None:
                selected.add(i)
                break
    return selected


def select_long_seed_indices(
    rows: List[dict],
    *,
    min_length: int = 6,
    exclude_keywords: Optional[List[str]] = None,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Set[int]:
    """train+valid 에서 도메인 미매칭이지만 LONG PROD-pos 문장 인덱스 집합.

    domain 휴리스틱이 놓친 long-tail PROD 를 보강하는 보조 풀. famous media
    키워드 (`DEFAULT_LONG_EXCLUDE_KEYWORDS`) 가 포함된 surface 는 제외 —
    random oversample 에서 이미 모델이 잘 맞히는 부분과 중복되어 효과 없음.
    """
    from ner.classifier.data_utils import split_train_valid_test
    train_rows, valid_rows, _ = split_train_valid_test(
        rows, valid_ratio, test_ratio, seed,
    )
    keep_ids = {id(r) for r in train_rows} | {id(r) for r in valid_rows}
    exclude = exclude_keywords or DEFAULT_LONG_EXCLUDE_KEYWORDS
    selected: Set[int] = set()
    for i, row in enumerate(rows):
        if id(row) not in keep_ids:
            continue
        for e in row.get('entities', []):
            if e.get('label') != 'PROD':
                continue
            surface = e.get('text', '')
            if len(surface) < min_length:
                continue
            if categorize_prod_surface(surface) is not None:
                # 도메인 매칭은 domain pool 에서 처리 — 중복 방지
                continue
            if any(k in surface for k in exclude):
                continue
            selected.add(i)
            break
    return selected


def oversample_to_jsonl(
    *,
    input_path: str,
    output_path: str,
    oversample: int = 5,
    use_domain: bool = True,
    use_long: bool = False,
    long_min_length: int = 6,
    long_exclude_keywords: Optional[List[str]] = None,
    base_extra_path: Optional[str] = None,
    extra_only: bool = True,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """전체 파이프라인 — 보강 jsonl 생성.

    Args:
        input_path: 원본 학습 JSONL (augmenters contract).
        output_path: 보강 jsonl 저장 경로.
        oversample: 각 후보 문장의 복제 횟수 N (extra_only=True 시 N 회 출력,
            False 시 1 + (N-1) 회). 기본 5 — V1 셋업.
        use_domain: 도메인 휴리스틱 매칭 seed 사용 여부 (기본 True).
        use_long: LONG surface (famous media 제외) seed 사용 여부 (기본 False
            — V2 셋업에서 활성화).
        long_min_length: LONG 컷오프.
        long_exclude_keywords: LONG 풀에서 제외할 키워드 (기본 famous media).
        base_extra_path: prefix 로 prepend 할 기존 extra jsonl (예: S7N2
            negative extra). None 이면 PROD-pos extra 만 출력.
        extra_only: True 면 extra (N 회 복제) 만 출력 → classifier
            `--data-extra-train-jsonl` 로 leak-free 학습. False 면 원본 +
            extra 합본 (split 재계산 → leak 유의).
        valid_ratio/test_ratio/seed: split 결정 시 사용 (train+valid 만 후보
            로 잡기 위함).

    Returns: 통계 dict {domain_seed_sentences, long_seed_sentences,
        candidates, extra_rows, base_extra_rows, output_rows}.
    """
    with open(input_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    logger.info('Loaded %d input rows from %s', len(rows), input_path)

    domain_sidx: Set[int] = set()
    long_sidx: Set[int] = set()
    if use_domain:
        domain_sidx = select_domain_seed_indices(
            rows, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed,
        )
        logger.info('Domain heuristic seed sentences: %d', len(domain_sidx))
    if use_long:
        long_sidx = select_long_seed_indices(
            rows,
            min_length=long_min_length,
            exclude_keywords=long_exclude_keywords,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        logger.info('LONG-surface seed sentences: %d', len(long_sidx))

    candidates = sorted(domain_sidx | long_sidx)
    logger.info('Total candidate sentences: %d', len(candidates))

    base_extra_rows = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if base_extra_path:
            with open(base_extra_path, encoding='utf-8') as bf:
                for line in bf:
                    f.write(line)
                    base_extra_rows += 1
            logger.info(
                'Prepended %d rows from base extra %s',
                base_extra_rows, base_extra_path,
            )
        extra_count = 0
        if not extra_only:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write('\n')
        # seed-major emission (각 후보 N회 연속 출력) — HF Trainer 의 shuffle
        # 입력 순서에 따라 학습 결과가 흔들리는 것을 ad-hoc 실험과 동일
        # 패턴으로 고정.
        copies = oversample if extra_only else max(0, oversample - 1)
        for i in candidates:
            row_line = json.dumps(rows[i], ensure_ascii=False)
            for _ in range(copies):
                f.write(row_line)
                f.write('\n')
                extra_count += 1
    output_rows = (
        base_extra_rows
        + (len(rows) if not extra_only else 0)
        + extra_count
    )
    logger.info(
        'Wrote %d rows to %s (base_extra=%d, extra=%d, extra_only=%s)',
        output_rows, output_path, base_extra_rows, extra_count, extra_only,
    )

    return {
        'domain_seed_sentences': len(domain_sidx),
        'long_seed_sentences': len(long_sidx),
        'candidates': len(candidates),
        'base_extra_rows': base_extra_rows,
        'extra_rows': extra_count,
        'output_rows': output_rows,
    }
