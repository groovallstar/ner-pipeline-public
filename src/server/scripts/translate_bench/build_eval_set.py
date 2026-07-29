"""번역 벤치마크 평가셋 빌더.

NTREX-128(뉴스, 긴 문장 포함, CC BY-SA 4.0) JA/VI 원문 + KO 레퍼런스를
길이 층화로 표본하고, 절반에 canonical PII 5종을 suffix 주입한다
(`ner.augmenters.pii` 재사용, gold span 코드 보장). 나머지 절반은 clean.

출력: `data/eval_set.jsonl`
  {id, lang, text, base_text, ko_ref, pii_spans, base_len, has_pii}
  - text      : PII 주입 후 문장(has_pii=False 면 base_text 와 동일)
  - base_text : 주입 전 원문(음차·뜻전달 채점 기준)
  - ko_ref    : NTREX KO 레퍼런스(LLM-judge 참고)
  - pii_spans : {label, start, end, text} — verbatim 보존 대상

NTREX 원문은 결정적 파생물(eval_set.jsonl)만 커밋한다. raw 는 scratch 로
내려받아 재현에만 쓴다.
"""
from __future__ import annotations

import json
import random
import unicodedata
import urllib.request
from pathlib import Path

from ner.augmenters.pii.config import InjectionConfig
from ner.augmenters.pii.injector import PIIInjector
from ner.augmenters.pii.schema import Record

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
REPO_ROOT = HERE.parents[3]
RAW = REPO_ROOT / 'results' / 'translate_bench' / 'ntrex_raw'
OUT = DATA / 'eval_set.jsonl'

BASE = ('https://raw.githubusercontent.com/MicrosoftTranslator/'
        'NTREX/main/NTREX-128')
FILES = {
    'ja': 'newstest2019-ref.jpn.txt',
    'vi': 'newstest2019-ref.vie.txt',
    'ko': 'newstest2019-ref.kor.txt',
}
# verbatim 보존 대상 PII 5종만 주입(고유명사는 음차 대상이라 제외).
PII_LABELS = ['PHONE', 'DAT', 'ID_NUM', 'EMAIL', 'CREDIT_CARD']
# 주입 문장당 PII 개수 분포(>=1 보장).
INJECT_DENSITY = {1: 0.4, 2: 0.4, 3: 0.2}
PER_LANG = 14  # 언어별 표본 수(총 28)
# 고정 seed — 표본·주입 결정성. 날짜·번호가 아닌 임의 상수.
SEED = 1234


def fetch(name: str) -> list[str]:
    """NTREX 파일을 scratch 캐시로 내려받고 NFC 정규화 라인 리스트를 반환한다."""
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / name
    if not dst.exists():
        url = f'{BASE}/{name}'
        print(f'downloading {url}')
        urllib.request.urlretrieve(url, dst)
    lines = dst.read_text(encoding='utf-8').splitlines()
    return [unicodedata.normalize('NFC', ln.strip()) for ln in lines]


def stratified_indices(lengths: list[int], rng: random.Random) -> list[int]:
    """길이 층화 표본: long(top 25%) 6 + medium(mid 50%) 4 + short(bottom 25%) 4."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    n = len(order)
    q1, q3 = n // 4, (3 * n) // 4
    short_pool, mid_pool, long_pool = order[:q1], order[q1:q3], order[q3:]
    picks: list[int] = []
    picks += rng.sample(long_pool, 6)   # 긴 문장 강조(문중 PII 스트레스)
    picks += rng.sample(mid_pool, 4)
    picks += rng.sample(short_pool, 4)
    return picks


def build_lang(lang: str, src: list[str], ko: list[str],
               rng: random.Random) -> list[dict]:
    """한 언어의 표본 레코드(절반 PII 주입, 절반 clean 교대)를 만든다."""
    lengths = [len(s) for s in src]
    idxs = stratified_indices(lengths, rng)
    inj_cfg = InjectionConfig(
        lang=lang, density=dict(INJECT_DENSITY),
        pii_labels=list(PII_LABELS), seed=SEED,
    )
    injector = PIIInjector(inj_cfg)
    records: list[dict] = []
    for rank, i in enumerate(idxs):
        base_text = src[i]
        ko_ref = ko[i] if i < len(ko) else ''
        has_pii = (rank % 2 == 0)   # 결정적 교대
        if has_pii:
            rec = injector.inject(Record(text=base_text, entities=[]))
            text = rec.text
            pii_spans = [
                {'label': e.label, 'start': e.start_char,
                 'end': e.end_char, 'text': e.text}
                for e in rec.entities
            ]
        else:
            text, pii_spans = base_text, []
        records.append({
            'id': f'ntrex-{lang}-{i:04d}',
            'lang': lang,
            'text': text,
            'base_text': base_text,
            'ko_ref': ko_ref,
            'pii_spans': pii_spans,
            'base_len': len(base_text),
            'has_pii': has_pii,
        })
    return records


def main() -> None:
    rng = random.Random(SEED)
    ja_src = fetch(FILES['ja'])
    vi_src = fetch(FILES['vi'])
    ko_ref = fetch(FILES['ko'])
    print(f'lines: ja={len(ja_src)} vi={len(vi_src)} ko={len(ko_ref)}')
    all_recs: list[dict] = []
    all_recs += build_lang('ja', ja_src, ko_ref, rng)
    all_recs += build_lang('vi', vi_src, ko_ref, rng)
    DATA.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_pii = sum(1 for r in all_recs if r['has_pii'])
    lens = [r['base_len'] for r in all_recs]
    pii_labels: dict[str, int] = {}
    for r in all_recs:
        for s in r['pii_spans']:
            pii_labels[s['label']] = pii_labels.get(s['label'], 0) + 1
    print(f'\nwrote {len(all_recs)} records -> {OUT}')
    print(f'  with PII: {n_pii} | clean: {len(all_recs) - n_pii}')
    print(f'  base_len: min={min(lens)} '
          f'median={sorted(lens)[len(lens) // 2]} max={max(lens)}')
    print(f'  PII label counts: {pii_labels}')


if __name__ == '__main__':
    main()
