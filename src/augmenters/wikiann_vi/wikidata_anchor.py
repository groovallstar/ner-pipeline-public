"""Wikipedia 인터링크 + Wikidata P31을 이용한 8종 앵커 검증 유틸.

흐름:
1. 재라벨 JSONL의 엔티티 표면형을 unique set으로 수집
2. vi.wikipedia.org/w/api.php로 페이지 → Q-ID 조회 (batch ≤ 50)
3. www.wikidata.org/w/api.php로 Q-ID → P31(instance of) claim 조회 (batch ≤ 50)
4. P31 Q-ID를 canonical 8종으로 매핑 (`WIKIDATA_TO_CANONICAL` 테이블)
5. 재라벨 타입 vs Wikidata 추론 타입 일치율 집계

네트워크·캐시 제약:
- 호출 간 0.2s sleep (Wikimedia 관례 준수)
- 결과를 JSON 파일로 캐시해 재실행 시 재호출 방지
- 네트워크 실패는 개별 엔티티 단위로 소프트 폴백 (reason='network')

상세 기준: `docs/manual/data/canonical-entity-schema.md`,
`docs/manual/data/vietnamese-ner-8types.md` §5.2 두 번째 검증 레이어.
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

_VI_WIKI_API = 'https://vi.wikipedia.org/w/api.php'
_WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
_UA = 'ner_pipeline/issue-13 (research; https://github.com/groovallstar/ner_pipeline)'
_BATCH = 50
_SLEEP = 0.2


# Wikidata Q-ID → canonical 8종 (수작업 curated, 주요 케이스).
# 본 테이블에 없는 Q-ID는 '미매핑(unmapped)'으로 집계되며, 리포트에 상위
# unmapped Q-ID가 기록돼 후속 확장의 판단 근거가 된다.
WIKIDATA_TO_CANONICAL: Dict[str, str] = {
    # PER
    'Q5': 'PER',             # human
    'Q215627': 'PER',        # person
    'Q95074': 'PER',         # fictional character
    'Q15632617': 'PER',      # fictional human

    # LOC (자연·행정)
    'Q6256': 'LOC',          # country
    'Q3024240': 'LOC',       # historical country
    'Q515': 'LOC',           # city
    'Q15284': 'LOC',         # municipality
    'Q3957': 'LOC',          # town
    'Q532': 'LOC',           # village
    'Q5119': 'LOC',          # capital city
    'Q82794': 'LOC',         # geographic region
    'Q1549591': 'LOC',       # big city
    'Q2074737': 'LOC',       # state of Vietnam (tỉnh)
    'Q17366755': 'LOC',      # administrative territorial entity (Việt Nam)
    'Q4022': 'LOC',          # river
    'Q8502': 'LOC',          # mountain
    'Q46831': 'LOC',         # mountain range
    'Q165': 'LOC',           # sea
    'Q23397': 'LOC',         # lake
    'Q23442': 'LOC',         # island
    'Q33837': 'LOC',         # archipelago
    'Q39816': 'LOC',         # valley
    'Q39594': 'LOC',         # bay (vịnh)
    'Q124734': 'LOC',        # strait
    'Q34876': 'LOC',         # cape
    'Q3250511': 'LOC',       # gulf
    'Q486972': 'LOC',        # human settlement (generic)
    'Q7376585': 'LOC',       # rural commune (Việt Nam xã)
    'Q7275': 'LOC',          # state
    'Q35657': 'LOC',         # U.S. state
    'Q484170': 'LOC',        # commune of France
    'Q1615742': 'LOC',       # province of China
    'Q2824648': 'LOC',       # province of Vietnam
    'Q1289426': 'LOC',       # county of China
    'Q24764': 'LOC',         # municipality of the Philippines

    # FAC (물리적 개별 건축물·교통시설)
    'Q41176': 'FAC',         # building
    'Q811979': 'FAC',        # architectural structure
    'Q16917': 'FAC',         # hospital
    'Q1248784': 'FAC',       # airport
    'Q55488': 'FAC',         # railway station
    'Q124757': 'FAC',        # bus station
    'Q3914': 'FAC',          # school (초·중·고)
    'Q159334': 'FAC',        # secondary school
    'Q9842': 'FAC',          # primary school
    'Q44613': 'FAC',         # monastery
    'Q24398318': 'FAC',      # religious building
    'Q16970': 'FAC',         # church building
    'Q210272': 'FAC',        # temple / pagoda
    'Q33506': 'FAC',         # museum
    'Q7075': 'FAC',          # library
    'Q22806': 'FAC',         # national library
    'Q24354': 'FAC',         # theatre
    'Q483110': 'FAC',        # stadium
    'Q12876': 'FAC',         # tunnel
    'Q12280': 'FAC',         # bridge
    'Q57821': 'FAC',         # fortification
    'Q23413': 'FAC',         # castle
    'Q105731': 'FAC',        # tower

    # CORP (영리 법인·기업·방송·운송 회사·대학 법인 본체)
    'Q4830453': 'CORP',      # business
    'Q783794': 'CORP',       # company
    'Q891723': 'CORP',       # public company
    'Q219577': 'CORP',       # holding company
    'Q18388277': 'CORP',     # technology company
    'Q1002697': 'CORP',      # periodical
    'Q11032': 'CORP',        # newspaper
    'Q1616075': 'CORP',      # television station
    'Q14350': 'CORP',        # radio station
    'Q2085381': 'CORP',      # publisher
    'Q270791': 'CORP',       # state-owned enterprise
    'Q43229': 'CORP',        # organization (약 fallback; 더 구체 없는 경우)
    'Q46970': 'CORP',        # airline
    'Q249556': 'CORP',       # railway company
    'Q11229656': 'CORP',     # bank
    # 대학 법인 본체 — Stockmark 실측 `〜大学` 115건 전수 100% CORP
    'Q3918': 'CORP',         # university
    'Q38723': 'CORP',        # higher education institution
    'Q875538': 'CORP',       # public university

    # PROD (물건·작품·소프트웨어)
    'Q2424752': 'PROD',      # product
    'Q7397': 'PROD',         # software
    'Q9143': 'PROD',         # programming language
    'Q17155032': 'PROD',     # phone
    'Q11424': 'PROD',        # film
    'Q7889': 'PROD',         # video game
    'Q571': 'PROD',          # book
    'Q8261': 'PROD',         # novel
    'Q482994': 'PROD',       # album
    'Q134556': 'PROD',       # single (music)
    'Q7725634': 'PROD',      # literary work
    'Q15416': 'PROD',        # television program
    'Q5398426': 'PROD',      # television series
    'Q3405677': 'PROD',      # automobile model
    'Q105543609': 'PROD',    # musical work / composition
    'Q21198342': 'PROD',     # manga series

    # EVT (일회성 사건·전쟁·조약)
    'Q1190554': 'EVT',       # occurrence (event)
    'Q178561': 'EVT',        # battle
    'Q198': 'EVT',           # war
    'Q625994': 'EVT',        # conference
    'Q1775415': 'EVT',       # festival
    'Q175331': 'EVT',        # demonstration
    'Q350604': 'EVT',        # armed conflict
    'Q13418847': 'EVT',      # historical event
    'Q2761147': 'EVT',       # military operation
    'Q131569': 'EVT',        # treaty
    'Q1407217': 'EVT',       # national sports competition (일회 대회)
    'Q27020041': 'EVT',      # sports season

    # POL (정당·정부·군·국제기구)
    'Q7278': 'POL',          # political party
    'Q327333': 'POL',        # government agency
    'Q7210356': 'POL',       # political organization
    'Q183061': 'POL',        # cabinet
    'Q8719': 'POL',          # military
    'Q749622': 'POL',        # armed forces
    'Q610311': 'POL',        # military unit
    'Q484652': 'POL',        # international organization
    'Q1463313': 'POL',       # intergovernmental organization
    'Q41487': 'POL',         # national assembly
    'Q35798': 'POL',         # court
    'Q28083049': 'POL',      # national intelligence agency
    'Q61883': 'POL',         # air force
    'Q4508': 'POL',          # navy
    'Q772547': 'POL',        # armed forces
    'Q15925165': 'POL',      # specific intl organization (e.g. IOM)

    # ORG (스포츠·협회·대학 부속 조직)
    # 주의: 대학 법인 본체(university/higher education/public university)는
    # Stockmark 실측 100% CORP이므로 CORP 블록으로 이동. Q2385804(educational
    # institution)은 초중고·사설 학원까지 포함하는 상위 클래스라 ORG fallback
    # 유지 (하위 Q3918 등이 매칭되면 그 쪽이 우선됨).
    'Q2385804': 'ORG',       # educational institution (초중고 포함 fallback)
    'Q748019': 'ORG',        # scientific society
    'Q955824': 'ORG',        # learned society
    'Q4438121': 'ORG',       # sports organization
    'Q847017': 'ORG',        # sports club
    'Q476028': 'ORG',        # association football club
    'Q13406463': 'ORG',      # sports league
    'Q15991290': 'ORG',      # sports season / league
    'Q215380': 'ORG',        # musical group / band
    'Q1478443': 'ORG',       # football federation
    'Q135408445': 'ORG',     # men's national football team
    'Q15991303': 'ORG',      # association football league
    'Q2178147': 'ORG',       # trade association
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': _UA})
    return s


def fetch_qids(
    titles: List[str], session: Optional[requests.Session] = None,
) -> Dict[str, Optional[str]]:
    """vi.wikipedia 페이지 제목 → Wikidata Q-ID. 없는 경우 None."""
    if not titles:
        return {}
    s = session or _session()
    result: Dict[str, Optional[str]] = {t: None for t in titles}
    unique = list({t for t in titles if t})

    for i in range(0, len(unique), _BATCH):
        batch = unique[i:i + _BATCH]
        params = {
            'action': 'query',
            'prop': 'pageprops',
            'ppprop': 'wikibase_item',
            'redirects': 1,
            'titles': '|'.join(batch),
            'format': 'json',
            'formatversion': 2,
        }
        try:
            r = s.get(_VI_WIKI_API, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning('vi.wiki fetch failed (batch %d): %s', i, exc)
            time.sleep(_SLEEP)
            continue

        pages = data.get('query', {}).get('pages', [])
        # 정규화 매핑 (redirect·normalized title)
        norm = {
            n['from']: n['to']
            for n in data.get('query', {}).get('normalized', [])
        }
        redir = {
            r['from']: r['to']
            for r in data.get('query', {}).get('redirects', [])
        }

        for page in pages:
            if 'missing' in page:
                continue
            qid = (page.get('pageprops', {}) or {}).get('wikibase_item')
            if not qid:
                continue
            title = page.get('title')
            # 원래 입력 제목을 역매핑
            originals = [
                orig for orig in batch
                if _resolve_title(orig, norm, redir) == title
                or orig == title
            ]
            for orig in originals:
                result[orig] = qid
        time.sleep(_SLEEP)
    return result


def _resolve_title(
    orig: str, norm: Dict[str, str], redir: Dict[str, str],
) -> str:
    """MediaWiki 정규화·리다이렉트 체인을 따라 최종 제목을 반환한다."""
    seen: Set[str] = set()
    cur = orig
    while cur in norm and cur not in seen:
        seen.add(cur)
        cur = norm[cur]
    while cur in redir and cur not in seen:
        seen.add(cur)
        cur = redir[cur]
    return cur


def fetch_p31(
    qids: List[str], session: Optional[requests.Session] = None,
) -> Dict[str, List[str]]:
    """Q-ID → P31 (instance of) Q-ID 리스트."""
    if not qids:
        return {}
    s = session or _session()
    result: Dict[str, List[str]] = {q: [] for q in qids}
    unique = list(set(qids))

    for i in range(0, len(unique), _BATCH):
        batch = unique[i:i + _BATCH]
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(batch),
            'props': 'claims',
            'format': 'json',
        }
        try:
            r = s.get(_WIKIDATA_API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning('Wikidata fetch failed (batch %d): %s', i, exc)
            time.sleep(_SLEEP)
            continue

        entities = data.get('entities', {}) or {}
        for qid, ent in entities.items():
            claims = (ent.get('claims') or {}).get('P31', [])
            p31_ids: List[str] = []
            for c in claims:
                val = (
                    ((c.get('mainsnak') or {}).get('datavalue') or {})
                    .get('value') or {}
                )
                tgt = val.get('id')
                if tgt:
                    p31_ids.append(tgt)
            result[qid] = p31_ids
        time.sleep(_SLEEP)
    return result


def anchor_type(p31_qids: List[str]) -> Optional[str]:
    """P31 Q-ID 리스트에서 canonical 8종을 결정한다. 매핑 0건이면 None."""
    for qid in p31_qids:
        tgt = WIKIDATA_TO_CANONICAL.get(qid)
        if tgt:
            return tgt
    return None


def _iter_entities(records: List[dict], span_key: str):
    """(surface_form, predicted_type, record_id) 튜플 생성기."""
    for r in records:
        for s in r.get(span_key, []):
            t = (s.get('text') or '').strip()
            y = (s.get('type') or '').strip()
            if t and y:
                yield t, y, r.get('id')


def run_anchor(
    records: List[dict],
    span_key: str = 'gold_spans_8type',
    cache_path: Optional[Path] = None,
) -> Dict:
    """재라벨 JSONL에 대한 Wikidata 앵커 검증 전체 파이프라인.

    Returns:
        {
          "total_entities": N, "unique_surfaces": U,
          "with_qid": X, "with_p31": Y, "with_mapped_type": Z,
          "matches": M, "mismatches": K,
          "per_type_agreement": {type: {agreed, total, ratio}},
          "unmapped_qids": {qid: count},
          "samples": [...],  # 예시 pairs
        }
    """
    cache: Dict = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding='utf-8'))

    # 1. 표면형 수집
    entities = list(_iter_entities(records, span_key))
    unique_surfaces = sorted({t for t, _, _ in entities})
    print(
        f'Total entities: {len(entities)}, '
        f'unique surfaces: {len(unique_surfaces)}'
    )

    # 2. 표면형 → Q-ID
    qid_cache: Dict[str, Optional[str]] = cache.get('qid', {})
    missing = [t for t in unique_surfaces if t not in qid_cache]
    print(f'Fetching Q-IDs for {len(missing)} surfaces...')
    fetched = fetch_qids(missing)
    qid_cache.update(fetched)

    # 3. Q-ID → P31
    all_qids = sorted({q for q in qid_cache.values() if q})
    p31_cache: Dict[str, List[str]] = cache.get('p31', {})
    missing_p31 = [q for q in all_qids if q not in p31_cache]
    print(f'Fetching P31 for {len(missing_p31)} Q-IDs...')
    fetched_p31 = fetch_p31(missing_p31)
    p31_cache.update(fetched_p31)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {'qid': qid_cache, 'p31': p31_cache},
                ensure_ascii=False, indent=2,
            ),
            encoding='utf-8',
        )

    # 4. 집계
    matches = 0
    mismatches = 0
    with_qid = 0
    with_p31 = 0
    with_mapped = 0
    per_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'agreed': 0, 'total': 0}
    )
    unmapped_counter: Counter = Counter()
    samples: List[dict] = []

    for surface, pred_type, rid in entities:
        qid = qid_cache.get(surface)
        if not qid:
            continue
        with_qid += 1
        p31 = p31_cache.get(qid) or []
        if not p31:
            continue
        with_p31 += 1
        anchor = anchor_type(p31)
        if not anchor:
            unmapped_counter[p31[0]] += 1
            continue
        with_mapped += 1
        per_type[pred_type]['total'] += 1
        if anchor == pred_type:
            matches += 1
            per_type[pred_type]['agreed'] += 1
        else:
            mismatches += 1
            if len(samples) < 30:
                samples.append({
                    'surface': surface,
                    'id': rid,
                    'predicted': pred_type,
                    'anchor': anchor,
                    'qid': qid,
                    'p31_head': p31[:3],
                })

    total = with_mapped
    return {
        'total_entities': len(entities),
        'unique_surfaces': len(unique_surfaces),
        'with_qid': with_qid,
        'with_p31': with_p31,
        'with_mapped_type': with_mapped,
        'matches': matches,
        'mismatches': mismatches,
        'agreement': (matches / total) if total else float('nan'),
        'per_type_agreement': {
            t: {
                'agreed': v['agreed'],
                'total': v['total'],
                'ratio': (
                    v['agreed'] / v['total'] if v['total'] else 0.0
                ),
            }
            for t, v in per_type.items()
        },
        'unmapped_qids': dict(unmapped_counter.most_common(20)),
        'mismatch_samples': samples,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m augmenters.wikiann_vi.wikidata_anchor',
        description='Wikipedia/Wikidata anchor verification',
    )
    p.add_argument('--input', required=True, help='Relabel JSONL path')
    p.add_argument(
        '--span-key', default='gold_spans_8type',
        help='Field holding 8-type spans',
    )
    p.add_argument(
        '--cache', default='data/wikiann_vi/wikidata_cache.json',
        help='JSON cache file for API responses',
    )
    p.add_argument(
        '--json-out', default=None,
        help='Full result JSON output path',
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    records = [
        json.loads(line) for line in Path(args.input).read_text(
            encoding='utf-8'
        ).splitlines() if line.strip()
    ]
    result = run_anchor(
        records,
        span_key=args.span_key,
        cache_path=Path(args.cache) if args.cache else None,
    )
    print()
    print('=== Wikidata Anchor Result ===')
    print(f"Entities           : {result['total_entities']}")
    print(f"Unique surfaces    : {result['unique_surfaces']}")
    print(f"With Wikidata Q-ID : {result['with_qid']}")
    print(f"With P31 claims    : {result['with_p31']}")
    print(f"Mapped to 8-type   : {result['with_mapped_type']}")
    print(
        f"Agreement          : {result['matches']} / "
        f"{result['with_mapped_type']} = {result['agreement']:.4f}"
    )
    print('Per-type agreement:')
    for t, d in sorted(
        result['per_type_agreement'].items(),
        key=lambda kv: -kv[1]['total'],
    ):
        print(
            f"  {t}: {d['agreed']}/{d['total']} = {d['ratio']:.4f}"
        )
    print('Top unmapped P31 Q-IDs (to consider for table expansion):')
    for qid, cnt in list(result['unmapped_qids'].items())[:10]:
        print(f'  {qid}: {cnt}')

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f"Wrote result -> {args.json_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
