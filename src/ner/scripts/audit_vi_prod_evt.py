"""VI PROD/EVT 천장 진단 — silver↔model 불일치 추출 + §3 기준 판정 준비.

리포트 §천장의 PROD(~0.71)/EVT 원인을 (a) silver 노이즈 / (b) 모델 한계 /
(c) schema 미규정 중 무엇인지 가리기 위한 감사 도구. classifier가 저장한
test_predictions.json(gold_spans=silver, pred_spans=model)에서 PROD/EVT가
얽힌 불일치 span을 전수 추출하고, canonical §3 기준 판정을 위한 케이스
JSONL을 만든다.

서브커맨드:
  extract   — 불일치 + 일치 대조군 추출 → cases JSONL
  (adjudicate 는 vLLM 판정 단계에서 추가)

사용:
    python src/ner/scripts/audit_vi_prod_evt.py extract \
        --pred results/classifier/vi/canonical5fold/phobert-base-v2/fold0/test_predictions.json \
        --out results/classifier/vi/audit/prod_evt_cases.jsonl
"""
import argparse
import asyncio
import json
import os
import random
import re
from typing import Dict, List, Optional

TARGET = ('PROD', 'EVT')

# canonical §3.1/§3.2/§3.3 회색지대 규칙 condensation (판정 루브릭).
# VI gold 가 아직 안 담은 부분(서비스 제외·제조 artifact·named 프로젝트 등)을
# 명시해 silver/model 어느 쪽이 §3 에 맞는지 판정시킨다.
RUBRIC = """\
You adjudicate one Vietnamese NER span against canonical schema §3
(PROD/EVT/ORG boundary). Decide the SINGLE correct label for the span.

PROD = tangible products; creative works (music/film/book/manga/game/TV
  program); packaged software/OS; format/model-named manufactured goods
  (vehicles, weapons, ships, aircraft, spacecraft, locomotives WITH a
  type/class/model name, e.g. "IV号戦車", "Galaxy S24").
  NOT PROD (→ non-entity O, unless it is an organization):
  services / SaaS / telecom / online media / financial products /
  memberships / courses; online-operated games (MMO); technical
  standards/specs/formats/protocols/licenses; awards / medals / honors /
  measurement standards. Model number ALONE ("RV522") is NOT PROD.
EVT = one-time events; wars; treaties; tournaments/cups/championships;
  the SPECIFIC-year edition of a periodic competition ("World Cup 2022");
  natural disasters & major accidents; named economic/political crises;
  periodic compound event nouns (elections, referendum, census); dated
  bounded uprisings/movements/massacres; exhibitions/tours/premieres;
  named projects/plans/strategies/programs.
  NOT EVT: regular/ongoing leagues, teams, clubs (→ ORG); abstract
  topics/debates ("~problem", "~controversy") → O; century-scale eras
  ("~era/period") → O; multi-year processes/states (cold war, religious
  reformation, industrial revolution) → O.
ORG = companies; record labels; regular leagues/teams/orchestras;
  infrastructure operators; facilities (stations, airports, hospitals,
  museums, stadiums, schools, universities).
LAW / statute / regulation / bill → non-entity (O).
"""


def _exact_key(s: dict) -> tuple:
    return (s['start'], s['end'], s['type'])


def _overlap(a: dict, b: dict) -> bool:
    """두 span 의 char 구간이 겹치면 True."""
    return a['start'] < b['end'] and b['start'] < a['end']


def _counterpart_label(span: dict, others: List[dict]) -> Optional[str]:
    """다른 쪽(gold/pred)에서 span 과 겹치는 첫 span 의 type. 없으면 None.

    정확 일치(같은 start/end)를 우선, 없으면 최대 overlap span 을 택한다.
    """
    exact = [o for o in others
             if o['start'] == span['start'] and o['end'] == span['end']]
    if exact:
        return exact[0]['type']
    overlaps = [o for o in others if _overlap(span, o)]
    if not overlaps:
        return None
    best = max(overlaps, key=lambda o: min(o['end'], span['end'])
               - max(o['start'], span['start']))
    return best['type']


def extract_cases(rows: List[dict]) -> Dict[str, List[dict]]:
    """PROD/EVT 불일치(FN/FP) + 일치 대조군 추출.

    FN: silver(gold)가 PROD/EVT 인데 model 이 정확 일치 못 함 (gold 기준 anchor).
    FP: model(pred)이 PROD/EVT 인데 gold 가 정확 일치 못 함, 단 이미 FN 으로
        잡힌 영역과 겹치면 중복 제거(gold anchor 우선).
    대조군: silver=model 둘 다 같은 PROD/EVT span (정확 일치) — 오라벨 점검용.
    """
    fn_cases: List[dict] = []
    fp_cases: List[dict] = []
    control: List[dict] = []

    for row in rows:
        text = row['text']
        sid = row['id']
        gold = row['gold_spans']
        pred = row['pred_spans']
        pred_exact = {_exact_key(p) for p in pred}
        gold_exact = {_exact_key(g) for g in gold}
        fn_regions: List[dict] = []

        # FN + 대조군 (gold anchor)
        for g in gold:
            if g['type'] not in TARGET:
                continue
            if _exact_key(g) in pred_exact:
                control.append(_case(sid, text, g, g['type'], g['type'],
                                     'MATCH'))
                continue
            model_label = _counterpart_label(g, pred) or 'O'
            fn_cases.append(_case(sid, text, g, g['type'], model_label, 'FN'))
            fn_regions.append(g)

        # FP (pred anchor), FN 영역과 겹치면 skip
        for p in pred:
            if p['type'] not in TARGET:
                continue
            if _exact_key(p) in gold_exact:
                continue
            if any(_overlap(p, r) for r in fn_regions):
                continue
            silver_label = _counterpart_label(p, gold) or 'O'
            fp_cases.append(_case(sid, text, p, silver_label, p['type'], 'FP'))

    return {'FN': fn_cases, 'FP': fp_cases, 'MATCH': control}


def _case(sid: str, text: str, span: dict, silver: str, model: str,
          direction: str) -> dict:
    """판정용 케이스 레코드 1건."""
    return {
        'sentence_id': sid,
        'text': text,
        'start': span['start'],
        'end': span['end'],
        'surface': text[span['start']:span['end']],
        'silver_label': silver,
        'model_label': model,
        'direction': direction,
    }


def _build_prompt(case: dict) -> str:
    """단일 케이스 판정 프롬프트 — 루브릭 + 문장 + span + JSON 출력 지시."""
    return (
        f'{RUBRIC}\n'
        f'SENTENCE: {case["text"]}\n'
        f'SPAN: "{case["surface"]}" (chars {case["start"]}..{case["end"]})\n'
        f'(For reference, silver label = {case["silver_label"]}, '
        f'model prediction = {case["model_label"]}; judge independently.)\n\n'
        'Return ONLY JSON: {"correct": "<PROD|EVT|ORG|PER|LOC|DAT|O>", '
        '"rule": "<short rule id>", "ambiguous": <true|false>, '
        '"reason": "<one short sentence>"}\n'
        'Set ambiguous=true ONLY if §3 genuinely does not determine the '
        'label (the class boundary is underspecified for this span).'
    )


def _parse_json(content: str) -> Optional[dict]:
    """`<think>` 제거 후 JSON 파싱. 실패 시 첫 `{...}` 블록 재시도."""
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None


async def _adjudicate(cases: List[dict], base_url: str, model: str,
                      concurrency: int, temperature: float) -> List[dict]:
    """vLLM 으로 각 케이스의 §3 정답 라벨 판정. cases 에 'adj' 키 부착."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key='none', timeout=120.0)
    sem = asyncio.Semaphore(concurrency)
    done = [0]

    async def one(case: dict) -> dict:
        async with sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': _build_prompt(case)}],
                temperature=temperature,
            )
            adj = _parse_json(resp.choices[0].message.content)
            done[0] += 1
            if done[0] % 25 == 0:
                print(f'  adjudicated {done[0]}/{len(cases)}')
            return {**case, 'adj': adj}

    return await asyncio.gather(*[one(c) for c in cases])


def _base_counts(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    """원본 pred 에서 type 별 exact-match TP/FP/FN 집계 (PROD/EVT)."""
    out = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in TARGET}
    for row in rows:
        gset = {_exact_key(g) for g in row['gold_spans']}
        pset = {_exact_key(p) for p in row['pred_spans']}
        for t in TARGET:
            g_t = {k for k in gset if k[2] == t}
            p_t = {k for k in pset if k[2] == t}
            out[t]['tp'] += len(g_t & p_t)
            out[t]['fn'] += len(g_t - p_t)
            out[t]['fp'] += len(p_t - g_t)
    return out


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {'precision': round(pr, 4), 'recall': round(rc, 4),
            'f1': round(f1, 4)}


def analyze(adjudicated: List[dict], rows: List[dict]) -> dict:
    """판정 결과 → silver오류/모델오류/둘다/schema-갭 3분류 + 보정 F1.

    분류: ambiguous → schema_gap. 아니면 정답(correct) 대비
    silver/model 일치로 silver_error / model_error / both_error / agree.
    보정 F1: FP 중 silver_error 는 TP 로(silver 누락), FN 중 silver_error 는
    gold 에서 제거(허위 gold). schema_gap 은 보정에서 제외(별도 보고).
    """
    base = _base_counts(rows)
    cls = {t: {'silver_error': 0, 'model_error': 0, 'both_error': 0,
               'schema_gap': 0, 'agree': 0, 'unparsed': 0} for t in TARGET}
    # 보정 델타: FP→TP, gold 제거(FN 제거 + support 감소)
    delta = {t: {'fp_to_tp': 0, 'fn_drop': 0, 'fp_gap': 0, 'fn_gap': 0}
             for t in TARGET}

    for c in adjudicated:
        adj = c.get('adj')
        t = c['silver_label'] if c['silver_label'] in TARGET \
            else c['model_label']
        if t not in TARGET:
            continue
        bucket = cls[t]
        if not adj or 'correct' not in adj:
            bucket['unparsed'] += 1
            continue
        if adj.get('ambiguous'):
            bucket['schema_gap'] += 1
            if c['direction'] == 'FP':
                delta[t]['fp_gap'] += 1
            elif c['direction'] == 'FN':
                delta[t]['fn_gap'] += 1
            continue
        correct = adj['correct']
        sok = (correct == c['silver_label'])
        mok = (correct == c['model_label'])
        if c['direction'] == 'MATCH':
            bucket['agree' if sok else 'both_error'] += 1
            continue
        if sok and not mok:
            bucket['model_error'] += 1
        elif mok and not sok:
            bucket['silver_error'] += 1
            if c['direction'] == 'FP':
                delta[t]['fp_to_tp'] += 1
            elif c['direction'] == 'FN':
                delta[t]['fn_drop'] += 1
        else:
            bucket['both_error'] += 1

    report = {'per_type': {}}
    for t in TARGET:
        b = base[t]
        orig = _prf(b['tp'], b['fp'], b['fn'])
        # schema_gap 케이스는 분모에서 제외(보정 불가)
        tp2 = b['tp'] + delta[t]['fp_to_tp']
        fp2 = b['fp'] - delta[t]['fp_to_tp'] - delta[t]['fp_gap']
        fn2 = b['fn'] - delta[t]['fn_drop'] - delta[t]['fn_gap']
        corrected = _prf(tp2, fp2, fn2)
        report['per_type'][t] = {
            'base_counts': b, 'orig_f1': orig,
            'corrected_f1': corrected, 'classification': cls[t],
            'corrected_counts': {'tp': tp2, 'fp': fp2, 'fn': fn2},
        }
    return report


def _load_pred(paths: List[str]) -> List[dict]:
    """하나 이상의 test_predictions(.json/.jsonl) 를 로드·연결.

    grouped K-fold 전수 감사는 fold별 예측 파일을 모두 합쳐 누출-free
    전체 코퍼스를 한 번에 본다(각 행은 정확히 한 fold 의 test 라 중복 없음).
    단일 경로도 허용(백워드 호환).
    """
    rows: List[dict] = []
    for path in paths:
        if path.endswith('.jsonl'):
            rows.extend(json.loads(ln)
                        for ln in open(path, encoding='utf-8'))
        else:
            rows.extend(json.load(open(path)))
    return rows


def main():
    p = argparse.ArgumentParser(
        description='Extract VI PROD/EVT silver-vs-model disagreements '
                    'for canonical §3 adjudication.')
    sub = p.add_subparsers(dest='cmd', required=True)
    pe = sub.add_parser('extract', help='Extract disagreement cases')
    pe.add_argument('--pred', required=True, nargs='+',
                    help='One or more test_predictions.json paths '
                         '(pass all grouped folds for full leak-free audit)')
    pe.add_argument('--out', required=True, help='Output cases JSONL')
    pe.add_argument('--control-size', type=int, default=40,
                    help='Number of agreement control cases to sample')
    pe.add_argument('--seed', type=int, default=42)

    pa = sub.add_parser('adjudicate', help='vLLM §3 adjudication of cases')
    pa.add_argument('--cases', required=True, help='cases JSONL from extract')
    pa.add_argument('--out', required=True, help='adjudicated JSONL')
    pa.add_argument('--base-url', default='http://localhost:8081/v1')
    pa.add_argument('--model', default='cyankiwi/gemma-4-31B-it-AWQ-8bit')
    pa.add_argument('--concurrency', type=int, default=8)
    pa.add_argument('--temperature', type=float, default=0.0)

    pn = sub.add_parser('analyze', help='Classify + corrected F1')
    pn.add_argument('--adjudicated', required=True)
    pn.add_argument('--pred', required=True, nargs='+',
                    help='Original predictions (one or more fold files)')
    pn.add_argument('--out', default=None)

    args = p.parse_args()

    if args.cmd == 'extract':
        rows = _load_pred(args.pred)
        cases = extract_cases(rows)

        rng = random.Random(args.seed)
        control = cases['MATCH']
        if len(control) > args.control_size:
            control = rng.sample(control, args.control_size)

        ordered = cases['FN'] + cases['FP'] + control
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for c in ordered:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')

        def _by_type(items, d):
            prod = sum(1 for c in items
                       if 'PROD' in (c['silver_label'], c['model_label']))
            evt = sum(1 for c in items
                      if 'EVT' in (c['silver_label'], c['model_label']))
            print(f'  {d}: {len(items)} (PROD={prod} EVT={evt})')

        print(f'rows={len(rows)} → cases written to {args.out}')
        _by_type(cases['FN'], 'FN (silver PROD/EVT, model differs)')
        _by_type(cases['FP'], 'FP (model PROD/EVT, silver differs)')
        print(f'  MATCH control: {len(control)} '
              f'(of {len(cases["MATCH"])} total)')
        print(f'  TOTAL cases: {len(ordered)}')

    elif args.cmd == 'adjudicate':
        cases = [json.loads(ln)
                 for ln in open(args.cases, encoding='utf-8')]
        result = asyncio.run(_adjudicate(
            cases, args.base_url, args.model, args.concurrency,
            args.temperature))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for c in result:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        unparsed = sum(1 for c in result if not c.get('adj'))
        print(f'adjudicated {len(result)} → {args.out} '
              f'(unparsed={unparsed})')

    elif args.cmd == 'analyze':
        adj = [json.loads(ln)
               for ln in open(args.adjudicated, encoding='utf-8')]
        rows = _load_pred(args.pred)
        report = analyze(adj, rows)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
