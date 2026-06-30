"""언어감지 후보 벤치마크 CLI — 손규칙 vs fastText-LID.

같은 gold 평가셋에 두 후보를 돌려 혼동행렬·클래스 P/R·버킷 recall 과
falsifiable 목표(ja-kana recall=100%, pan-Latin vi false-accept=0, 미지원
버킷 recall 등)를 산출해 콘솔 표 + results/lang_detect_bench.json 으로
저장한다. lid-model 생략 시 손규칙만 평가한다.

    uv run --with fasttext python -m server.scripts.lang_detect.bench \\
        --lid-model /path/to/lid.176.ftz
"""

import argparse
import json
from pathlib import Path

from server.scripts.lang_detect import GOLD_PATH
from server.scripts.lang_detect.candidates import (
    hand_rules, make_fasttext_lid)
from server.scripts.lang_detect.confusion import evaluate, load_gold

# pan-Latin 미지원 언어(vi false-accept 하드 0 대상)
_PAN_LATIN = ('fr', 'pt', 'de', 'es', 'tr')
# 미지원으로 떨어져야 하는 버킷(가나·vi 신호 부재)
_UNSUPPORTED_BUCKETS = ('ko', 'en', 'zh', 'vi_khong_dau', 'romaji_ja')


def _targets(result):
    """평가 결과에서 falsifiable 목표 지표를 추출한다."""
    pb = result['per_bucket']
    vi_fa = {b: pb[b]['pred_dist'].get('vi', 0) for b in _PAN_LATIN}
    return {
        'ja_kana_recall': pb['ja_kana']['recall'],
        'vi_diacritics_recall': pb['vi_diacritics']['recall'],
        'vi_false_accept_pan_latin': vi_fa,
        'vi_false_accept_total': sum(vi_fa.values()),
        'unsupported_recall': {
            b: pb[b]['recall'] for b in _UNSUPPORTED_BUCKETS},
        'vi_ja_switch_recall': pb['vi_ja_switch']['recall'],
        'kanji_only_ja_recall_reported': pb['ja_kanji_only']['recall'],
    }


def _fmt(x):
    """비율을 백분율 문자열로(None 은 '-')."""
    return '-' if x is None else f'{x * 100:.2f}%'


def run(gold_path, lid_model):
    """후보별 평가 결과 dict 를 만든다(lid_model 없으면 손규칙만)."""
    rows = load_gold(gold_path)
    out = {'gold': str(gold_path), 'n_rows': len(rows), 'candidates': {}}
    cand = {'hand_rules': hand_rules}
    if lid_model:
        cand['fasttext_lid'] = make_fasttext_lid(lid_model)
    for name, det in cand.items():
        res = evaluate(det, rows)
        res['targets'] = _targets(res)
        out['candidates'][name] = res
    return out


def _print_summary(out):
    """후보별 핵심 목표를 콘솔 표로 출력한다."""
    for name, res in out['candidates'].items():
        t = res['targets']
        print(f'\n=== {name} ===')
        print(f"  ja-kana recall       : {_fmt(t['ja_kana_recall'])}"
              '  (target 100%)')
        print(f"  vi-diacritics recall : "
              f"{_fmt(t['vi_diacritics_recall'])}  (report)")
        print(f"  vi false-accept total: {t['vi_false_accept_total']}"
              '  (target 0)')
        for b, v in t['vi_false_accept_pan_latin'].items():
            print(f'      {b}: {v}')
        print('  unsupported recall:')
        for b, v in t['unsupported_recall'].items():
            print(f'      {b}: {_fmt(v)}')
        print(f"  vi+ja switch recall  : "
              f"{_fmt(t['vi_ja_switch_recall'])}  (kana override -> ja)")
        print(f"  kanji-only-ja recall : "
              f"{_fmt(t['kanji_only_ja_recall_reported'])}"
              '  (report-only limitation)')


def main():
    ap = argparse.ArgumentParser(
        description='Benchmark lang-detect candidates on the gold set')
    ap.add_argument('--gold', default=str(GOLD_PATH),
                    help='gold jsonl path')
    ap.add_argument('--lid-model', default=None,
                    help='fastText lid.176.ftz path (omit=hand-rules only)')
    ap.add_argument('--out', default='results/lang_detect_bench.json',
                    help='results json output path')
    args = ap.parse_args()
    out = run(args.gold, args.lid_model)
    _print_summary(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f'\nwrote results to {args.out}')


if __name__ == '__main__':
    main()
