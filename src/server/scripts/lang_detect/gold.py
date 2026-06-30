"""gold 다국어 언어감지 평가셋 빌더 — FLORES-200 dev + 파생 4종.

자연어 10종(ja/vi/ko/en/zh/fr/pt/de/es/tr)은 FLORES-200 dev 에서 고정 seed
로 N 문장씩 샘플하고, 파생 4종(한자only-ja·무부호-vi·romaji-ja·vi+ja 코드
스위칭)은 그 샘플에서 결정적으로 만든다.

각 gold 행은 {text, bucket, true_lang, expected, report_only, source}.
expected 는 제품 결정에 따른 *감지기 목표 출력*이라 true_lang 과 다를 수
있다 — 무부호-vi 는 true_lang=vi 이나 expected=unsupported(수용된 한계),
romaji-ja 도 expected=unsupported(가나 부재). 가나를 가진 vi+ja 는 가나
override 로 expected=ja.

재생성(romaji 파생에 pykakasi 필요):
    uv run --with pykakasi python -m server.scripts.lang_detect.gold \\
        --flores-dir <dir>/flores200_dataset/dev --n 100
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
GOLD_PATH = _PKG_DIR / 'gold.jsonl'
MANIFEST_PATH = _PKG_DIR / 'gold_manifest.json'

# FLORES-200 dev 언어 코드 → 파일 스템
FLORES_FILES = {
    'ja': 'jpn_Jpan',
    'vi': 'vie_Latn',
    'ko': 'kor_Hang',
    'en': 'eng_Latn',
    'zh': 'zho_Hans',
    'fr': 'fra_Latn',
    'pt': 'por_Latn',
    'de': 'deu_Latn',
    'es': 'spa_Latn',
    'tr': 'tur_Latn',
}

# 버킷 → (true_lang, expected, report_only).
# expected = 제품 결정에 따른 감지기 목표 출력(true_lang 과 다를 수 있음).
# report_only = 한계 정량용(헤드라인 목표 제외, 수치만 보고).
BUCKET_SPEC = {
    'ja_kana':       ('ja',    'ja',          False),
    'ja_kanji_only': ('ja',    'ja',          True),
    'vi_diacritics': ('vi',    'vi',          False),
    'vi_khong_dau':  ('vi',    'unsupported', False),
    'romaji_ja':     ('ja',    'unsupported', False),
    'vi_ja_switch':  ('vi+ja', 'ja',          False),
    'ko': ('ko', 'unsupported', False),
    'en': ('en', 'unsupported', False),
    'zh': ('zh', 'unsupported', False),
    'fr': ('fr', 'unsupported', False),
    'pt': ('pt', 'unsupported', False),
    'de': ('de', 'unsupported', False),
    'es': ('es', 'unsupported', False),
    'tr': ('tr', 'unsupported', False),
}

_KANA_RANGES = (
    (0x3040, 0x309F), (0x30A0, 0x30FF),
    (0x31F0, 0x31FF), (0xFF66, 0xFF9D),
)
# CJK 통합 한자(기본·확장 A) — 한자 run 추출용
_KANJI_RE = re.compile(r'[㐀-䶿一-鿿]{2,}')


def _has_kana(text):
    """가나 보유 여부(파생 필터용)."""
    return any(lo <= ord(c) <= hi
               for c in text for lo, hi in _KANA_RANGES)


def _read_flores(flores_dir, stem):
    """FLORES dev 파일에서 비어있지 않은 문장 리스트를 읽는다."""
    path = Path(flores_dir) / f'{stem}.dev'
    with open(path, encoding='utf-8') as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _sample(lines, n, rng):
    """고정 rng 로 n 개 인덱스를 비복원 추출해 원순서로 반환."""
    k = min(n, len(lines))
    idx = sorted(rng.sample(range(len(lines)), k))
    return [lines[i] for i in idx]


def _strip_diacritics(text):
    """vi 성조·부호를 모두 제거해 không-dấu(무부호) 문자열로 만든다."""
    nfd = unicodedata.normalize('NFD', text)
    out = []
    for ch in nfd:
        if unicodedata.combining(ch):
            continue  # 결합부호 제거
        out.append({'đ': 'd', 'Đ': 'D'}.get(ch, ch))
    return unicodedata.normalize('NFC', ''.join(out))


def _longest_kanji_run(text):
    """문장에서 가장 긴 한자(≥2) 연속 구간을 반환(없으면 None)."""
    runs = _KANJI_RE.findall(text)
    return max(runs, key=len) if runs else None


def _romanize_ja(sentences):
    """pykakasi 로 ja 문장들을 Hepburn romaji 로 변환(가나·한자 제거).

    pykakasi 가 변환하지 못한 잔여 가나(예: 중점 ・ U+30FB)는 라틴 romaji
    가 아니므로 제거해 '가나 없는 romaji' 라는 버킷 불변식을 보장한다 —
    그래야 romaji→unsupported 경로만 시험한다(중점이 남으면 가나 override
    가 발동해 ja 로 잡힌다).
    """
    try:
        import pykakasi
    except ImportError as exc:
        raise SystemExit(
            'romaji bucket needs pykakasi; rerun with '
            '`uv run --with pykakasi ...`') from exc
    kks = pykakasi.kakasi()
    out = []
    for s in sentences:
        toks = [it['hepburn'] for it in kks.convert(s)]
        romaji = ' '.join(t for t in toks if t)
        # 잔여 가나(중점·장음 등 미변환 문자) 제거 후 공백 정규화
        clean = ''.join(' ' if _has_kana(ch) else ch for ch in romaji)
        out.append(' '.join(clean.split()))
    return out


def _row(text, bucket, source):
    """gold 행 dict 생성 — BUCKET_SPEC 에서 라벨을 파생(단일 출처)."""
    true_lang, expected, report_only = BUCKET_SPEC[bucket]
    return {'text': text, 'bucket': bucket, 'true_lang': true_lang,
            'expected': expected, 'report_only': report_only,
            'source': source}


def build(flores_dir, n, seed):
    """gold 행 리스트 + 매니페스트를 만든다."""
    rng = random.Random(seed)
    samples = {}
    # 자연어 10종 — 고정 정렬순으로 샘플(seed 결정성)
    for lang in sorted(FLORES_FILES):
        lines = _read_flores(flores_dir, FLORES_FILES[lang])
        samples[lang] = _sample(lines, n, rng)
    ja, vi = samples['ja'], samples['vi']
    rows = []
    # ja_kana — 가나 보유 ja(FLORES-ja 는 전수 보유)
    for s in ja:
        if _has_kana(s):
            rows.append(_row(s, 'ja_kana', 'flores:jpn_Jpan'))
    # ja_kanji_only — 최장 한자 run(≥2) 추출(zh 와 스크립트 동일 스트레스)
    for s in ja:
        run = _longest_kanji_run(s)
        if run:
            rows.append(
                _row(run, 'ja_kanji_only', 'derived:kanji_run(jpn_Jpan)'))
    # romaji_ja — Hepburn romaji(가나·한자 제거)
    for s in _romanize_ja(ja):
        if s:
            rows.append(_row(s, 'romaji_ja', 'derived:romaji(jpn_Jpan)'))
    # vi_diacritics — 원문 그대로(술어로 필터 금지: 누출 방지)
    for s in vi:
        rows.append(_row(s, 'vi_diacritics', 'flores:vie_Latn'))
    # vi_khong_dau — 부호 제거(수용된 한계 → unsupported)
    for s in vi:
        rows.append(
            _row(_strip_diacritics(s), 'vi_khong_dau',
                 'derived:strip_diacritics(vie_Latn)'))
    # vi_ja_switch — vi 문장 + ja 문장 연결(가나 혼입 → override 검증)
    for v, j in zip(vi, ja):
        rows.append(
            _row(f'{v} {j}', 'vi_ja_switch',
                 'derived:concat(vie_Latn+jpn_Jpan)'))
    # 미지원 자연어 8종 — 원문 그대로
    for lang in ('ko', 'en', 'zh', 'fr', 'pt', 'de', 'es', 'tr'):
        stem = FLORES_FILES[lang]
        for s in samples[lang]:
            rows.append(_row(s, lang, f'flores:{stem}'))
    return rows, _manifest(rows, flores_dir, n, seed)


def _manifest(rows, flores_dir, n, seed):
    """버킷별 건수·출처·expected 를 집계한 매니페스트."""
    counts = Counter(r['bucket'] for r in rows)
    buckets = {}
    for b in BUCKET_SPEC:
        true_lang, expected, report_only = BUCKET_SPEC[b]
        src = next((r['source'] for r in rows if r['bucket'] == b), None)
        buckets[b] = {'count': counts.get(b, 0), 'true_lang': true_lang,
                      'expected': expected, 'report_only': report_only,
                      'source': src}
    return {
        'dataset': 'FLORES-200 dev + derived',
        'flores_dir': str(flores_dir),
        'sample_n': n,
        'seed': seed,
        'total': len(rows),
        'natural_langs': sorted(FLORES_FILES),
        'buckets': buckets,
        'notes': [
            'leak-free: hand-rules untrained; FLORES dev not used to fit '
            'any rule.',
            'representative: professionally translated full sentences '
            '(FLORES) + explicitly labelled derived stress buckets.',
            'non-gameable: vi_diacritics is not filtered by the predicate '
            '(all sampled vi sentences kept).',
            'derived 4 buckets are deterministic functions of the same '
            'fixed FLORES sample.',
        ],
    }


def write(rows, manifest,
          gold_path=GOLD_PATH, manifest_path=MANIFEST_PATH):
    """gold.jsonl + manifest.json 을 기록한다."""
    with open(gold_path, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description='Build gold multilingual lang-detect eval set')
    ap.add_argument('--flores-dir', required=True,
                    help='FLORES-200 dev directory (extracted)')
    ap.add_argument('--n', type=int, default=100,
                    help='sentences sampled per natural language')
    ap.add_argument('--seed', type=int, default=42, help='sampling seed')
    args = ap.parse_args()
    rows, manifest = build(args.flores_dir, args.n, args.seed)
    write(rows, manifest)
    print(f'wrote {len(rows)} rows to {GOLD_PATH}')
    print(f'wrote manifest to {MANIFEST_PATH}')


if __name__ == '__main__':
    main()
