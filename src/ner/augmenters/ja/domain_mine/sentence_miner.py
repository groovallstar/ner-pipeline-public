"""B3: 일본어 위키백과 문장 마이닝 (명칭 포함 문장 + offset 추출).

B2 가 모은 도메인 명칭마다, MediaWiki API (ja.wikipedia) 전문검색 →
intro extract → 문장 분리 → 명칭이 실제 등장하는 문장만 PROD anchor
offset 과 함께 추출한다. 나머지 엔티티는 B4 에서 교차검증 재라벨한다.

위키백과는 현 학습 데이터 (Stockmark NER) 와 같은 출처·라이선스
(CC-BY-SA) 라 분포 정합성이 높다. 문장 분리·offset 은 순수 함수로
분리해 단위 테스트한다. HTTP·캐시 관례는 wikidata_anchor 차용.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ner.augmenters.ja.domain_mine.schema import DOMAINS, find_occurrences

logger = logging.getLogger(__name__)

_JA_WIKI_API = 'https://ja.wikipedia.org/w/api.php'
_UA = 'ner_pipeline (research; https://github.com/groovallstar/ner_pipeline)'
_BATCH = 20
_SLEEP = 0.2
_SENT_END = re.compile(r'(?<=[。！？])')


def split_sentences(text: str) -> List[str]:
    """일본어 문장 분리 (。！？ 기준, 개행 분리 + strip). 순수."""
    out = []
    for chunk in _SENT_END.split(text):
        for seg in chunk.split('\n'):
            seg = seg.strip()
            if seg:
                out.append(seg)
    return out


def extract_sentences_with_name(text: str, name: str, min_len: int = 8,
                                max_len: int = 200) -> List[Dict]:
    """text 를 문장 분리해, name 이 등장하는 문장 + anchor offset 반환.

    각 결과: {'text': 문장, 'anchors': [(start, end), ...]}. offset 은
    문장 문자열 기준이라 그대로 entity span 으로 쓸 수 있다 (순수).
    """
    results = []
    for sent in split_sentences(text):
        if len(sent) < min_len or len(sent) > max_len:
            continue
        anchors = find_occurrences(sent, name)
        if anchors:
            results.append({'text': sent, 'anchors': anchors})
    return results


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': _UA})
    return s


def search_titles(name: str, session: Optional[requests.Session] = None,
                  limit: int = 5, timeout: int = 20) -> List[str]:
    """ja.wikipedia 전문검색으로 name 관련 문서 제목을 반환한다."""
    s = session or _session()
    params = {
        'action': 'query', 'list': 'search', 'srsearch': name,
        'srlimit': limit, 'format': 'json', 'formatversion': 2,
    }
    try:
        r = s.get(_JA_WIKI_API, params=params, timeout=timeout)
        r.raise_for_status()
        hits = r.json().get('query', {}).get('search', [])
        return [h['title'] for h in hits if h.get('title')]
    except Exception as exc:
        logger.warning('ja.wiki search failed (%s): %s', name, exc)
        return []


def fetch_extracts(titles: List[str],
                   session: Optional[requests.Session] = None,
                   timeout: int = 30) -> Dict[str, str]:
    """문서 제목 → intro plaintext extract (batch ≤ 20)."""
    if not titles:
        return {}
    s = session or _session()
    out: Dict[str, str] = {}
    unique = list(dict.fromkeys(titles))
    for i in range(0, len(unique), _BATCH):
        batch = unique[i:i + _BATCH]
        params = {
            'action': 'query', 'prop': 'extracts', 'explaintext': 1,
            'exintro': 1, 'redirects': 1, 'titles': '|'.join(batch),
            'format': 'json', 'formatversion': 2,
        }
        try:
            r = s.get(_JA_WIKI_API, params=params, timeout=timeout)
            r.raise_for_status()
            for page in r.json().get('query', {}).get('pages', []):
                ext = page.get('extract')
                if ext:
                    out[page.get('title', '')] = ext
        except Exception as exc:
            logger.warning('ja.wiki extract failed (batch %d): %s', i, exc)
        time.sleep(_SLEEP)
    return out


def mine_name(name: str, domain: str,
              session: Optional[requests.Session] = None,
              max_titles: int = 5, max_sentences: int = 3,
              min_len: int = 8, max_len: int = 200) -> List[Dict]:
    """단일 명칭의 마이닝 레코드 리스트 (anchor offset 포함)."""
    s = session or _session()
    titles = list(dict.fromkeys([name] + search_titles(name, s, max_titles)))
    extracts = fetch_extracts(titles, s)
    records: List[Dict] = []
    seen_text = set()
    for title, text in extracts.items():
        for hit in extract_sentences_with_name(text, name, min_len, max_len):
            if hit['text'] in seen_text:
                continue
            seen_text.add(hit['text'])
            records.append({
                'text': hit['text'], 'domain': domain, 'name': name,
                'source_title': title, 'anchors': hit['anchors'],
            })
            if len(records) >= max_sentences:
                return records
    return records


def mine_names(names: List[str], domain: str,
               session: Optional[requests.Session] = None,
               max_titles: int = 5, max_sentences: int = 3,
               min_len: int = 8, max_len: int = 200) -> List[Dict]:
    """명칭 리스트 전체를 마이닝한다 (호출 간 sleep, 명칭 단위 soft-fail)."""
    s = session or _session()
    out: List[Dict] = []
    for i, name in enumerate(names):
        out.extend(mine_name(name, domain, s, max_titles, max_sentences,
                             min_len, max_len))
        if (i + 1) % 50 == 0:
            logger.info('%s: mined %d/%d names -> %d records',
                        domain, i + 1, len(names), len(out))
        time.sleep(_SLEEP)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.augmenters.ja.domain_mine.sentence_miner',
        description='Mine ja.wikipedia sentences containing domain names',
    )
    p.add_argument('--domains', nargs='+', default=list(DOMAINS))
    p.add_argument('--names-dir', default='data/domain_mine/names',
                   help='Dir of B2 per-domain name cache JSON')
    p.add_argument('--out-dir', default='data/domain_mine/mined',
                   help='Output dir for per-domain mined JSONL')
    p.add_argument('--max-names', type=int, default=0,
                   help='Cap names per domain (0 = all)')
    p.add_argument('--max-titles', type=int, default=5)
    p.add_argument('--max-sentences', type=int, default=3)
    p.add_argument('--log-level', default='INFO')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    names_dir = Path(args.names_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    for domain in args.domains:
        cache = json.loads(
            (names_dir / f'{domain}.json').read_text(encoding='utf-8'))
        names = cache['names']
        if args.max_names:
            names = names[:args.max_names]
        records = mine_names(names, domain, session,
                             args.max_titles, args.max_sentences)
        out_path = out_dir / f'{domain}.jsonl'
        with out_path.open('w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'{domain}: {len(records)} records -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
