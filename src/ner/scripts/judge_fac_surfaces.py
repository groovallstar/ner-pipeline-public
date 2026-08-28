"""FAC 표면을 canonical §3 규칙으로 판정한다. 두 모델을 각각 돌린다."""
import asyncio
import json
import re
import sys
import pathlib
from openai import AsyncOpenAI

BACKENDS = [('a', 'http://localhost:8081/v1'), ('b', 'http://localhost:8082/v1')]

PROMPT = """You label an English named entity against a fixed schema.

Work in this order.

STEP 1 - What does the name DENOTE? Read the contexts and decide what real
thing the name stands for. The surface form alone is not enough and often
misleads: a name shaped like two place names joined by a hyphen may denote a
railway line, or it may denote a shopping center named after the intersection
it sits on. A bare number may denote a highway. The contexts decide, not the
shape of the name. Look at what people DO with it in the sentences - travel
along it, or visit and build on it.

STEP 2 - Label that thing.
  * A single man-made structure standing in ONE place -> ORG.
    Stations, airports, bridges, tunnels, buildings, towers, stadiums, arenas,
    parks, hotels, hospitals, museums, libraries, churches, temples, shopping
    centers, plazas, harbors, ports, dams, power plants, military bases, forts,
    prisons, campuses, restaurants, clubs, theaters, cemeteries.
  * A route that CONNECTS multiple points -> LOC.
    Roads, streets, avenues, boulevards, highways, freeways, expressways,
    numbered routes, railway and subway LINES, metro/rail networks, canals,
    waterways, trails, paths.

STEP 3 - Metonymy. Do not let a borrowed sense change the answer you reached in
STEP 1. If the name denotes ONE real thing and another sense merely borrows
that name, keep the label of the thing. "Wall Street" denotes a street -> LOC,
even when the sentence means the financial industry. Only when a SEPARATE
entity genuinely bears the same name does the context choose between them.

STEP 4 - If the span is not a named entity at all under this schema -> DROP.
Common phrases, sentence fragments, bare verbs, mis-tagged tokens. Do NOT use
DROP merely because you are unsure of ORG vs LOC.

Answer with one JSON object and nothing else:
{{"verdict": "ORG" or "LOC" or "DROP", "reason": "<10 words max>"}}

SURFACE: {surface}

CONTEXTS:
{contexts}"""


def parse(raw):
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', cleaned, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    v = str(d.get('verdict', '')).upper().strip()
    return {'verdict': v, 'reason': str(d.get('reason', ''))[:120]} if v in ('ORG', 'LOC', 'DROP') else None


async def one(client, model, entry, sem):
    ctx = '\n'.join(f'{i+1}. {c[:400]}' for i, c in enumerate(entry['contexts']))
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=200,
                    messages=[{'role': 'user',
                               'content': PROMPT.format(surface=entry['surface'], contexts=ctx)}])
                got = parse(r.choices[0].message.content or '')
                if got:
                    return got
            except Exception as e:
                if attempt == 2:
                    return {'verdict': 'ERROR', 'reason': str(e)[:120]}
                await asyncio.sleep(2)
        return {'verdict': 'UNPARSED', 'reason': ''}


async def run(tag, url, entries):
    client = AsyncOpenAI(base_url=url, api_key='x')
    model = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(16)
    res = await asyncio.gather(*(one(client, model, e, sem) for e in entries))
    return tag, res


async def main():
    entries = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = pathlib.Path(sys.argv[2])
    results = await asyncio.gather(*(run(t, u, entries) for t, u in BACKENDS))
    by_tag = dict(results)
    merged = []
    for i, e in enumerate(entries):
        merged.append({
            'surface': e['surface'],
            'occurrences': e['occurrences'],
            'source_types': e['source_types'],
            'a': by_tag['a'][i], 'b': by_tag['b'][i],
            'agreement': 'high' if by_tag['a'][i]['verdict'] == by_tag['b'][i]['verdict'] else 'conflict',
        })
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding='utf-8')
    agree = sum(1 for m in merged if m['agreement'] == 'high')
    print(f'판정 {len(merged)} · 일치 {agree} · 갈림 {len(merged)-agree}')

asyncio.run(main())
