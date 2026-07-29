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

**표본은 앞에서만 자란다.** `--per-lang` 을 키우면 처음 14개(언어별)는 그대로
두고 뒤에 덧붙는다 — 기존 표본이 새 표본의 접두 부분집합이라, 예전에 측정한
수치를 그 부분집합에서 그대로 재현할 수 있다. 표본을 갈아엎으면 확장 전후
점수를 나란히 놓을 근거가 사라지므로 이 불변식을 유지한다. 확장분은 본
난수열을 건드리지 않도록 별도 seed 를 쓴다.
"""
from __future__ import annotations

import argparse
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
FROZEN_PER_LANG = 14  # 최초 동결 표본 수(언어별) — 확장해도 이 앞부분은 불변
# 동결분의 층별 배분(long/mid/short). 확장분도 같은 비율을 따른다.
FROZEN_STRATA = (6, 4, 4)
# 고정 seed — 표본·주입 결정성. 날짜·번호가 아닌 임의 상수.
SEED = 1234
# 확장분 전용 seed — 본 난수열과 분리해야 동결분 추첨이 안 흔들린다.
EXTRA_SEED = 5678


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


def _strata(lengths: list[int]) -> tuple[list[int], list[int], list[int]]:
    """길이 오름차순 인덱스를 short(하위 25%)·mid(중간 50%)·long(상위 25%)로."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    n = len(order)
    q1, q3 = n // 4, (3 * n) // 4
    return order[:q1], order[q1:q3], order[q3:]


def stratified_indices(lengths: list[int], rng: random.Random,
                       per_lang: int, extra_rng: random.Random) -> list[int]:
    """길이 층화 표본 — 동결분 14개(long 6·mid 4·short 4) + 확장분.

    동결분은 최초 버전과 완전히 같은 난수 호출 순서를 타므로 `per_lang` 을
    키워도 앞 14개는 변하지 않는다. 확장분은 이미 뽑힌 인덱스를 제외한
    나머지에서 같은 층 비율로 `extra_rng` 가 뽑는다.
    """
    short_pool, mid_pool, long_pool = _strata(lengths)
    picks: list[int] = []
    picks += rng.sample(long_pool, FROZEN_STRATA[0])  # 긴 문장 강조(문중 PII 스트레스)
    picks += rng.sample(mid_pool, FROZEN_STRATA[1])
    picks += rng.sample(short_pool, FROZEN_STRATA[2])

    extra = per_lang - FROZEN_PER_LANG
    if extra <= 0:
        return picks

    taken = set(picks)
    # 동결분과 같은 층 비율(6:4:4). 반올림 잔여는 short 가 흡수한다.
    n_long = round(extra * FROZEN_STRATA[0] / FROZEN_PER_LANG)
    n_mid = round(extra * FROZEN_STRATA[1] / FROZEN_PER_LANG)
    n_short = extra - n_long - n_mid
    for pool, k in ((long_pool, n_long), (mid_pool, n_mid),
                    (short_pool, n_short)):
        avail = [i for i in pool if i not in taken]
        if k > len(avail):
            raise ValueError(f'per-lang {per_lang} exceeds stratum capacity '
                             f'({k} requested, {len(avail)} available)')
        drawn = extra_rng.sample(avail, k)
        picks += drawn
        taken.update(drawn)
    return picks


def build_lang(lang: str, src: list[str], ko: list[str],
               rng: random.Random, per_lang: int,
               extra_rng: random.Random) -> list[dict]:
    """한 언어의 표본 레코드(절반 PII 주입, 절반 clean 교대)를 만든다."""
    lengths = [len(s) for s in src]
    idxs = stratified_indices(lengths, rng, per_lang, extra_rng)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-lang', type=int, default=FROZEN_PER_LANG,
                    help='samples per language (frozen first 14 are preserved)')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    if args.per_lang < FROZEN_PER_LANG:
        ap.error(f'--per-lang must be >= {FROZEN_PER_LANG} '
                 '(the frozen sample is never shrunk)')

    rng = random.Random(SEED)
    ja_src = fetch(FILES['ja'])
    vi_src = fetch(FILES['vi'])
    ko_ref = fetch(FILES['ko'])
    print(f'lines: ja={len(ja_src)} vi={len(vi_src)} ko={len(ko_ref)}')
    all_recs: list[dict] = []
    # 확장분 난수열은 언어별로 분리 — ja 확장 개수가 vi 추첨을 흔들지 않게.
    for idx, (lang, src) in enumerate((('ja', ja_src), ('vi', vi_src))):
        all_recs += build_lang(lang, src, ko_ref, rng, args.per_lang,
                               random.Random(EXTRA_SEED + idx))
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_pii = sum(1 for r in all_recs if r['has_pii'])
    lens = [r['base_len'] for r in all_recs]
    pii_labels: dict[str, int] = {}
    for r in all_recs:
        for s in r['pii_spans']:
            pii_labels[s['label']] = pii_labels.get(s['label'], 0) + 1
    print(f'\nwrote {len(all_recs)} records -> {out_path}')
    print(f'  with PII: {n_pii} | clean: {len(all_recs) - n_pii}')
    print(f'  base_len: min={min(lens)} '
          f'median={sorted(lens)[len(lens) // 2]} max={max(lens)}')
    print(f'  PII label counts: {pii_labels}')


if __name__ == '__main__':
    main()
