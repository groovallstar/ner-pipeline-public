"""번역 엔진 벤치마크 러너 — 프로덕션 마스킹-복원 경로로 후보를 스윕한다.

각 후보(OpenAI 호환 vLLM 엔드포인트)를 `server.translate.LLMTranslator` 로
그대로 태워(실 nonce-sentinel 마스킹-복원) 자동지표를 재고, 선택적으로
중립 LLM-judge 로 음차·뜻전달을 채점한다. 품질은 하드웨어 독립이라 임의
GPU 에서 재고, 2080Ti 정합을 위해 후보는 4-bit 양자화 체크포인트로 서빙한다.

자동지표(객관):
  - PII 보존   : 원본 PII 가 출력에 verbatim 생존한 수 / 마스킹한 수
  - 지연        : 문장당 왕복 wall-clock(마스킹·복원 포함, LLM 호출이 지배)
LLM-judge(참고, 편향 주의 — 판정자가 후보와 계열이 겹치면 유리 편향 가능):
  - adequacy   : 원문 뜻 전달 정확·완전성(1~5)
  - translit   : 고유명사 한글 음차 자연스러움(1~5; 고유명사 없으면 5)

사용:
  uv run python -m server.scripts.translate_bench.run_bench \
    --engine gemma31b cyankiwi/gemma-4-31B-it-AWQ-8bit http://localhost:8081/v1 \
    --engine qwen25-7b Qwen/Qwen2.5-7B-Instruct-AWQ http://localhost:8090/v1 \
    --judge-url http://localhost:8082/v1 --judge-model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import List, Optional

from server.translate import LLMTranslator, TranslationUnavailable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_EVAL = HERE / 'data' / 'eval_set.jsonl'
DEFAULT_OUT = REPO_ROOT / 'results' / 'translate_bench'

# 판정자 응답에서 첫 JSON 객체 추출용.
_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)

_JUDGE_PROMPT = (
    '당신은 기계번역 품질 평가자다. 아래 {lang_name}→한국어 번역을 채점하라.\n'
    '문장 끝에 붙은 전화번호·카드번호·이메일 등 숫자/연락처 조각은 원문 보존용'
    ' 주입물이니 채점에서 무시하라.\n\n'
    '원문({lang_name}): {source}\n'
    '참조 번역(KO): {ko_ref}\n'
    '평가 대상(KO): {candidate}\n\n'
    '두 축을 1~5 정수로 채점하고 JSON 만 출력하라(설명 금지):\n'
    '- adequacy: 원문 뜻을 정확하고 완전하게 전달하는가 (1 매우 나쁨 ~ 5 완벽)\n'
    '- translit: 인명·지명·조직 등 고유명사를 자연스러운 한글로 음차했는가 '
    '(1~5, 고유명사가 없으면 5)\n'
    '형식: {{"adequacy": <1-5>, "translit": <1-5>, "note": "<한 줄 근거>"}}'
)

_LANG_NAME = {'ja': '일본어', 'vi': '베트남어'}


def load_eval(path: Path) -> List[dict]:
    """평가셋 jsonl 로드."""
    return [json.loads(ln) for ln in
            path.read_text(encoding='utf-8').splitlines() if ln.strip()]


def _to_prod_spans(pii_spans: List[dict]) -> List[dict]:
    """평가셋 span(start/end) → 프로덕션 span(start_char/end_char) 변환."""
    return [{'label': s['label'], 'start_char': s['start'],
             'end_char': s['end'], 'text': s['text']} for s in pii_spans]


def run_engine(name: str, translator: LLMTranslator,
               records: List[dict]) -> List[dict]:
    """한 엔진으로 전 레코드를 번역하고 행별 지표를 수집한다."""
    rows: List[dict] = []
    for r in records:
        spans = _to_prod_spans(r['pii_spans'])
        pii_values = [s['text'] for s in r['pii_spans']]
        t0 = time.perf_counter()
        err = None
        try:
            res = translator.translate(r['text'], r['lang'], spans)
            translated = res.translation
            masked, restored, dropped = (
                res.masked_pii, res.restored_pii, res.dropped_pii)
        except (TranslationUnavailable, ValueError) as exc:
            translated, masked, restored, dropped = '', len(spans), 0, len(spans)
            err = str(exc)
        dt = time.perf_counter() - t0
        # verbatim 재확인(복원 회계와 교차검증).
        verbatim = sum(1 for v in pii_values if v and v in translated)
        rows.append({
            'engine': name, 'id': r['id'], 'lang': r['lang'],
            'has_pii': r['has_pii'], 'base_text': r['base_text'],
            'ko_ref': r['ko_ref'], 'translated': translated,
            'n_pii': masked, 'pii_restored': restored, 'pii_dropped': dropped,
            'pii_verbatim': verbatim, 'latency_s': round(dt, 3), 'error': err,
        })
    return rows


def judge_rows(rows: List[dict], judge: LLMTranslator, judge_model: str) -> None:
    """LLM-judge 로 각 행에 adequacy/translit 점수를 in-place 부착한다."""
    client = judge._client  # OpenAI 호환 클라이언트 재사용
    for row in rows:
        if row['error'] or not row['translated']:
            continue
        prompt = _JUDGE_PROMPT.format(
            lang_name=_LANG_NAME.get(row['lang'], row['lang']),
            source=row['base_text'], ko_ref=row['ko_ref'],
            candidate=row['translated'])
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0, max_tokens=256)
            raw = (resp.choices[0].message.content or '').strip()
            m = _JSON_RE.search(raw)
            data = json.loads(m.group(0)) if m else {}
            row['adequacy'] = int(data.get('adequacy')) \
                if data.get('adequacy') is not None else None
            row['translit'] = int(data.get('translit')) \
                if data.get('translit') is not None else None
            row['judge_note'] = data.get('note', '')
        except Exception as exc:  # noqa: BLE001 - 판정 실패는 None 으로 흡수
            row['adequacy'] = row['translit'] = None
            row['judge_note'] = f'judge error: {exc}'


def summarize(rows: List[dict]) -> dict:
    """엔진×언어 집계(보존·지연·judge 평균)를 dict 로 반환한다."""
    from collections import defaultdict
    agg: dict = defaultdict(lambda: {
        'n': 0, 'lat': [], 'pii_ok': 0, 'pii_tot': 0,
        'adq': [], 'trl': [], 'errors': 0})
    for r in rows:
        a = agg[(r['engine'], r['lang'])]
        a['n'] += 1
        a['lat'].append(r['latency_s'])
        a['pii_ok'] += r['pii_restored']
        a['pii_tot'] += r['n_pii']
        if r.get('error'):
            a['errors'] += 1
        if r.get('adequacy') is not None:
            a['adq'].append(r['adequacy'])
        if r.get('translit') is not None:
            a['trl'].append(r['translit'])
    out: dict = {}
    for (eng, lang), a in sorted(agg.items()):
        out[f'{eng}|{lang}'] = {
            'engine': eng, 'lang': lang, 'n': a['n'], 'errors': a['errors'],
            'pii_preserved': a['pii_ok'], 'pii_total': a['pii_tot'],
            'pii_preserve_pct': round(100 * a['pii_ok'] / a['pii_tot'], 1)
            if a['pii_tot'] else None,
            'latency_mean_s': round(statistics.mean(a['lat']), 2) if a['lat'] else None,
            'latency_p50_s': round(statistics.median(a['lat']), 2) if a['lat'] else None,
            'latency_max_s': round(max(a['lat']), 2) if a['lat'] else None,
            'adequacy_mean': round(statistics.mean(a['adq']), 2) if a['adq'] else None,
            'translit_mean': round(statistics.mean(a['trl']), 2) if a['trl'] else None,
        }
    return out


def print_summary(summary: dict) -> None:
    """요약 표를 stdout 에 출력한다."""
    print('\n=== SUMMARY (engine x lang) ===')
    hdr = (f'{"engine":22} {"lang":4} {"n":>3} {"PII보존":>10} '
           f'{"lat_mean":>9} {"lat_max":>8} {"adequacy":>9} {"translit":>9}')
    print(hdr)
    for row in summary.values():
        pii = (f'{row["pii_preserved"]}/{row["pii_total"]}'
               if row['pii_total'] else '-')
        adq = f'{row["adequacy_mean"]}' if row['adequacy_mean'] is not None else '-'
        trl = f'{row["translit_mean"]}' if row['translit_mean'] is not None else '-'
        lat_m = f'{row["latency_mean_s"]}s' if row['latency_mean_s'] is not None else '-'
        lat_x = f'{row["latency_max_s"]}s' if row['latency_max_s'] is not None else '-'
        print(f'{row["engine"]:22} {row["lang"]:4} {row["n"]:>3} {pii:>10} '
              f'{lat_m:>9} {lat_x:>8} {adq:>9} {trl:>9}')


def write_dump(rows: List[dict], path: Path) -> None:
    """원문/출력/참조 나란히 markdown 덤프(사람 눈 확인용)."""
    lines: List[str] = ['# 번역 벤치 덤프 (원문 / 후보 출력 / 참조)\n']
    by_id: dict = {}
    for r in rows:
        by_id.setdefault(r['id'], []).append(r)
    for rid, group in by_id.items():
        g0 = group[0]
        lines.append(f'\n## {rid} ({g0["lang"]}, has_pii={g0["has_pii"]})')
        lines.append(f'- **원문**: {g0["base_text"]}')
        lines.append(f'- **참조(KO)**: {g0["ko_ref"]}')
        for r in group:
            adq = r.get('adequacy', '-')
            trl = r.get('translit', '-')
            tag = f'[adq={adq} trl={trl} PII={r["pii_restored"]}/{r["n_pii"]}]'
            lines.append(f'- **{r["engine"]}** {tag}: {r["translated"]}')
    path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Translation engine benchmark sweep')
    p.add_argument('--engine', nargs=3, action='append',
                   metavar=('NAME', 'MODEL', 'BASE_URL'),
                   help='Add an engine (repeatable): NAME MODEL BASE_URL')
    p.add_argument('--eval', type=Path, default=DEFAULT_EVAL,
                   help='Eval set jsonl path')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT,
                   help='Output directory (scratch)')
    p.add_argument('--timeout', type=float, default=60.0,
                   help='Per-call timeout seconds')
    p.add_argument('--max-tokens', type=int, default=2048,
                   help='Max output tokens per translation')
    p.add_argument('--judge', dest='judge', action='store_true', default=True,
                   help='Run LLM-judge (default on)')
    p.add_argument('--no-judge', dest='judge', action='store_false',
                   help='Skip LLM-judge')
    p.add_argument('--judge-url', default='http://localhost:8082/v1',
                   help='Judge backend base_url (neutral, not a candidate)')
    p.add_argument('--judge-model', default='cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit',
                   help='Judge model name')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engines = args.engine or [[
        'gemma31b', 'cyankiwi/gemma-4-31B-it-AWQ-8bit',
        'http://localhost:8081/v1']]
    records = load_eval(args.eval)
    print(f'loaded {len(records)} eval records from {args.eval}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for name, model, base_url in engines:
        tr = LLMTranslator(base_url=base_url, model=model,
                           timeout_s=args.timeout, max_tokens=args.max_tokens)
        if not tr.available():
            print(f'[skip] engine {name}: backend not available at {base_url}')
            continue
        print(f'running engine {name} ({model}) ...')
        rows = run_engine(name, tr, records)
        all_rows += rows
        n_err = sum(1 for r in rows if r['error'])
        print(f'  done: {len(rows)} rows, {n_err} errors')

    judge: Optional[LLMTranslator] = None
    if args.judge and all_rows:
        judge = LLMTranslator(base_url=args.judge_url, model=args.judge_model,
                              timeout_s=args.timeout)
        if judge.available():
            print(f'judging with {args.judge_model} @ {args.judge_url} ...')
            judge_rows(all_rows, judge, args.judge_model)
        else:
            print(f'[skip] judge backend not available at {args.judge_url}')

    out_rows = args.out_dir / 'bench_results.jsonl'
    with out_rows.open('w', encoding='utf-8') as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = summarize(all_rows)
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_dump(all_rows, args.out_dir / 'dump.md')
    print(f'\nwrote {len(all_rows)} rows -> {out_rows}')
    print(f'wrote summary -> {args.out_dir / "summary.json"}')
    print(f'wrote dump    -> {args.out_dir / "dump.md"}')
    print_summary(summary)


if __name__ == '__main__':
    main()
