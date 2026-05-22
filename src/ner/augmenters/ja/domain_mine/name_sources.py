"""B2: 4도메인 명칭 수집 (e-Gov 법령 API + Wikidata SPARQL + curated).

- law: e-Gov 법령 API v1 (XML) 法令名一覧 (政府標準利用規約2.0 / CC BY 호환)
- book: Wikidata SPARQL — 일본어 literary work 라벨
- music_work: Wikidata SPARQL — 일본 musical work 라벨
- transit_card: 폐쇄적 소집합이라 curated 상수

HTTP·캐시 관례는 wikiann_vi.wikidata_anchor 를 차용 (requests.Session +
User-Agent, soft-fail, JSON 캐시). 파싱은 순수 함수로 분리해 단위 테스트
한다 (parse_law_list / parse_sparql_names / normalize_names).
"""

import argparse
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ner.augmenters.ja.domain_mine.schema import DOMAINS

logger = logging.getLogger(__name__)

_UA = 'ner_pipeline (research; https://github.com/groovallstar/ner_pipeline)'
_EGOV_LAWLIST = 'https://laws.e-gov.go.jp/api/1/lawlists/{category}'
_WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
_QID_RE = re.compile(r'^Q\d+$')

# e-Gov 법령 종류 코드 (1=全法令, 2=憲法・法律, 3=政令・勅令, 4=府省令).
DEFAULT_LAW_CATEGORY = 2

# Wikidata SPARQL — rdfs:label + LANG 필터로 라벨 서비스의 Q-ID 폴백 회피.
SPARQL_QUERIES: Dict[str, str] = {
    'book': (
        'SELECT DISTINCT ?itemLabel WHERE {{ '
        '?item wdt:P31/wdt:P279* wd:Q7725634 . '   # literary work
        '?item wdt:P407 wd:Q5287 . '               # language = Japanese
        '?item rdfs:label ?itemLabel . '
        'FILTER(LANG(?itemLabel) = "ja") }} LIMIT {limit}'
    ),
    'music_work': (
        'SELECT DISTINCT ?itemLabel WHERE {{ '
        '?item wdt:P31/wdt:P279* wd:Q2188189 . '   # musical work
        '?item wdt:P495 wd:Q17 . '                 # country of origin = JP
        '?item rdfs:label ?itemLabel . '
        'FILTER(LANG(?itemLabel) = "ja") }} LIMIT {limit}'
    ),
}

# 交通系ICカード — 전국 상호이용 + 주요 지역 카드 (폐쇄적 소집합, curated).
TRANSIT_CARDS: List[str] = [
    'Suica', 'PASMO', 'ICOCA', 'PiTaPa', 'TOICA', 'manaca', 'Kitaca',
    'SUGOCA', 'nimoca', 'はやかけん', 'PASPY', 'SAPICA', 'りゅーと',
    'icsca', 'IruCa', 'ナイスパス', 'LuLuCa', 'めぐりん',
]


def parse_law_list(xml_text: str) -> List[str]:
    """e-Gov lawlists XML 에서 LawName 텍스트를 전부 추출한다 (순수)."""
    root = ET.fromstring(xml_text)
    names = []
    for info in root.iter('LawNameListInfo'):
        el = info.find('LawName')
        if el is not None and el.text:
            names.append(el.text.strip())
    return names


def parse_sparql_names(data: Dict, var: str = 'itemLabel') -> List[str]:
    """SPARQL JSON 결과에서 라벨 값을 추출한다 (Q-ID 폴백 제외, 순수)."""
    out = []
    for b in data.get('results', {}).get('bindings', []):
        v = (b.get(var) or {}).get('value')
        if v and not _QID_RE.match(v):
            out.append(v.strip())
    return out


def normalize_names(names: List[str], min_len: int = 2,
                    max_len: int = 40) -> List[str]:
    """공백 제거 + 길이 필터 + 순서 보존 dedup (순수)."""
    seen = set()
    out = []
    for n in names:
        n = (n or '').strip()
        if not n or len(n) < min_len or len(n) > max_len:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': _UA})
    return s


def fetch_law_names(category: int = DEFAULT_LAW_CATEGORY,
                    session: Optional[requests.Session] = None,
                    timeout: int = 30) -> List[str]:
    """e-Gov 법령 API 에서 法令名一覧을 가져온다. 실패 시 빈 리스트."""
    s = session or _session()
    url = _EGOV_LAWLIST.format(category=category)
    try:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        return parse_law_list(r.text)
    except Exception as exc:
        logger.warning('e-Gov fetch failed (category %s): %s', category, exc)
        return []


def fetch_wikidata_names(query: str,
                         session: Optional[requests.Session] = None,
                         timeout: int = 60) -> List[str]:
    """Wikidata SPARQL 쿼리로 일본어 라벨 리스트를 가져온다."""
    s = session or _session()
    try:
        r = s.get(_WIKIDATA_SPARQL,
                  params={'query': query, 'format': 'json'},
                  timeout=timeout)
        r.raise_for_status()
        return parse_sparql_names(r.json())
    except Exception as exc:
        logger.warning('Wikidata SPARQL fetch failed: %s', exc)
        return []


def collect_domain_names(domain: str, limit: int = 3000,
                         law_category: int = DEFAULT_LAW_CATEGORY,
                         session: Optional[requests.Session] = None
                         ) -> List[str]:
    """도메인별 명칭 수집을 디스패치한다 (정규화까지 적용)."""
    if domain == 'law':
        raw = fetch_law_names(law_category, session)
    elif domain == 'transit_card':
        raw = list(TRANSIT_CARDS)
    elif domain in SPARQL_QUERIES:
        raw = fetch_wikidata_names(
            SPARQL_QUERIES[domain].format(limit=limit), session)
    else:
        raise ValueError(f'unknown domain: {domain}')
    return normalize_names(raw)


def load_or_fetch(domain: str, cache_dir: Path, limit: int = 3000,
                  law_category: int = DEFAULT_LAW_CATEGORY,
                  session: Optional[requests.Session] = None,
                  refresh: bool = False) -> List[str]:
    """도메인 명칭을 캐시에서 읽거나 없으면 수집 후 캐시에 쓴다."""
    cache_path = cache_dir / f'{domain}.json'
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
        return data['names']
    names = collect_domain_names(domain, limit, law_category, session)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({'domain': domain, 'count': len(names), 'names': names},
                   ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return names


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ja.domain_mine.name_sources',
        description='Collect domain name lists (law/book/music/transit)',
    )
    p.add_argument('--domains', nargs='+', default=list(DOMAINS),
                   help='Domains to collect (default: all 4)')
    p.add_argument('--cache-dir', default='data/domain_mine/names',
                   help='Directory for per-domain name cache JSON')
    p.add_argument('--limit', type=int, default=3000,
                   help='SPARQL result LIMIT per domain')
    p.add_argument('--law-category', type=int, default=DEFAULT_LAW_CATEGORY,
                   help='e-Gov law category (2=Constitution/Acts)')
    p.add_argument('--refresh', action='store_true',
                   help='Ignore cache and re-fetch')
    p.add_argument('--log-level', default='INFO')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    cache_dir = Path(args.cache_dir)
    session = _session()
    for domain in args.domains:
        names = load_or_fetch(domain, cache_dir, args.limit,
                              args.law_category, session, args.refresh)
        out_path = cache_dir / f'{domain}.json'
        print(f'{domain}: {len(names)} names -> {out_path}')
        for n in names[:5]:
            print(f'  - {n}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
