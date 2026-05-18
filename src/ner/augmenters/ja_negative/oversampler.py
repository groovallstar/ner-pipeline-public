"""환각 부정 예시 oversample 도구 (JA classifier S7 promote).

목적
----
JA NER classifier 의 환각 (FP HALLUCINATION) 감소를 위해, 학습 셋에서
환각 surface 가 등장하지만 그 위치가 entity 라벨로 표시되지 않은 문장을
N 배 oversample 한 보강 데이터셋을 생성한다.

학습 흐름
----------
    1. classifier 의 error_analysis 로 production 모델 진단
       (`results/.../v3_boundary/error_analysis.json` 생성)
    2. 본 모듈로 보강 jsonl 생성 (--extra-only 권장 — leak-free 학습용)
    3. classifier 학습에 --data-extra-train-jsonl 로 주입

이슈 #58 검증 결과 (v3 / N=2 / ambiguous 제외)
    overall strict F1: 0.9368 → 0.9624 (+2.56pp vs production)
    환각 (FP HALL): 60 → 30 (-50%)
    게이트 통과 (P AND R ≥ 0.95): 5/10 → 7/10
"""

import json
import logging
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def extract_seeds(
    diagnosis_path: str,
    allowed_pred_types: Optional[Set[str]] = None,
) -> Set[str]:
    """error_analysis.json 의 HALLUCINATION FP entries 에서 surface 추출.

    Args:
        diagnosis_path: classifier `error_analysis.py --with-diagnosis` 출력 JSON
        allowed_pred_types: 주어지면 그 pred type 으로 잡힌 환각 surface 만
            추출 (예: PER/LOC/ORG/PROD/EVT 로 NER-only 분리)

    Returns: surface 집합 (중복 제거).
    """
    with open(diagnosis_path, encoding='utf-8') as f:
        dump = json.load(f)
    seeds: Set[str] = set()
    for sr in dump.get('sentences', []):
        for entry in sr.get('fp', []):
            if entry.get('error_class') == 'HALLUCINATION':
                pred = entry.get('pred', {})
                if (allowed_pred_types is not None
                        and pred.get('type') not in allowed_pred_types):
                    continue
                surface = pred.get('text', '').strip()
                if surface:
                    seeds.add(surface)
    return seeds


def filter_ambiguous_seeds(
    seeds: Set[str], train_rows: List[dict],
) -> Tuple[Set[str], Set[str]]:
    """학습 셋에서 entity surface 로 등장한 seed 를 제외한다.

    가설 검증 (이슈 #58):
        ambiguous surface (학습 셋에 entity 라벨로 등장하면서 동시에 다른
        위치에서 환각으로 잡힌 것) 를 부정 예시로 강화하면 모델이 그
        surface 자체를 entity 회피 → recall 회귀. 자동 제외로 회귀 차단.

    Returns: (filtered, excluded)
    """
    entity_surfaces: Set[str] = set()
    for r in train_rows:
        text = r['text']
        for e in r.get('entities', []):
            sf = text[e['start_char']:e['end_char']]
            entity_surfaces.add(sf)
    filtered = {s for s in seeds if s not in entity_surfaces}
    excluded = seeds - filtered
    return filtered, excluded


def find_candidate_indices(rows: List[dict], seeds: Set[str]) -> List[int]:
    """seed surface 가 row text 에 등장하면서 그 위치가 entity 라벨 밖에
    있는 row 의 인덱스 (한 row 당 1번만 카운트)."""
    candidates: List[int] = []
    for i, row in enumerate(rows):
        text = row['text']
        entities = row.get('entities', [])
        if _row_has_unlabeled_seed(text, entities, seeds):
            candidates.append(i)
    return candidates


def _row_has_unlabeled_seed(
    text: str, entities: List[dict], seeds: Set[str],
) -> bool:
    """row 안에 seed surface 가 등장하면서 그 위치가 entity 에 포함되지
    않은 경우가 하나라도 있으면 True."""
    for seed in seeds:
        start = 0
        while True:
            pos = text.find(seed, start)
            if pos < 0:
                break
            end = pos + len(seed)
            covered = any(
                e['start_char'] <= pos and end <= e['end_char']
                for e in entities
            )
            if not covered:
                return True
            start = pos + 1
    return False


def oversample_to_jsonl(
    *,
    input_path: str,
    diagnosis_path: str,
    output_path: str,
    oversample: int = 2,
    auto_exclude_ambiguous: bool = True,
    seed_pred_types: Optional[Set[str]] = None,
    extra_only: bool = False,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """전체 파이프라인 — 보강 jsonl 생성.

    Args:
        input_path: 원본 학습 JSONL (augmenters JSONL contract)
        diagnosis_path: error_analysis.json (HALLUCINATION 추출용)
        output_path: 보강 jsonl 저장 경로
        oversample: 각 후보 문장의 총 등장 횟수 N (extra_only=False 시
            1 + (N-1) 추가 = N 번, extra_only=True 시 (N-1) 번만 출력)
        auto_exclude_ambiguous: True 면 학습 셋에 entity 로 등장한 seed
            를 자동 제외 (recall 회귀 차단, 권장 default)
        seed_pred_types: 환각 pred type 필터 (예: {'PER','LOC','ORG','PROD',
            'EVT'} 로 NER-only). None = 전체
        extra_only: True 면 extra (oversample N-1 회 복제) 만 출력 →
            classifier `--data-extra-train-jsonl` 로 leak-free 학습 가능.
            False = 원본 + extra 합본 출력 (학습 시 split 재계산 → leak 유의)
        valid_ratio/test_ratio/seed: train split 결정 시 사용 (auto_exclude_
            ambiguous 가 train 셋만 참조하기 위함)

    Returns: 통계 dict {seeds_initial, seeds_excluded, seeds_final,
        candidates, extra_rows, output_rows}
    """
    with open(input_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    logger.info('Loaded %d input rows from %s', len(rows), input_path)

    seeds = extract_seeds(diagnosis_path, allowed_pred_types=seed_pred_types)
    seeds_initial = len(seeds)
    logger.info(
        'Extracted %d hallucination seeds (pred_types=%s)',
        seeds_initial,
        sorted(seed_pred_types) if seed_pred_types else 'ALL',
    )

    excluded: Set[str] = set()
    if auto_exclude_ambiguous:
        # train split 만 ambiguous 판단 기준으로 사용 (valid/test leak 방지)
        from ner.classifier.data_utils import split_train_valid_test
        train_rows, _, _ = split_train_valid_test(
            rows, valid_ratio, test_ratio, seed,
        )
        seeds, excluded = filter_ambiguous_seeds(seeds, train_rows)
        logger.info(
            'Excluded %d ambiguous seeds (left %d): %s',
            len(excluded), len(seeds), sorted(excluded),
        )

    candidates = find_candidate_indices(rows, seeds)
    logger.info('Found %d candidate sentences', len(candidates))

    with open(output_path, 'w', encoding='utf-8') as f:
        extra_count = 0
        if not extra_only:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write('\n')
        for _ in range(oversample - 1):
            for i in candidates:
                f.write(json.dumps(rows[i], ensure_ascii=False))
                f.write('\n')
                extra_count += 1
    output_rows = (len(rows) if not extra_only else 0) + extra_count
    logger.info(
        'Wrote %d rows to %s (extra=%d, extra_only=%s)',
        output_rows, output_path, extra_count, extra_only,
    )

    return {
        'seeds_initial': seeds_initial,
        'seeds_excluded': len(excluded),
        'seeds_final': len(seeds),
        'candidates': len(candidates),
        'extra_rows': extra_count,
        'output_rows': output_rows,
    }
