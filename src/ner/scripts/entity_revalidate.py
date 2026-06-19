"""silver gold 의 한 라벨 span 을 LLM 으로 재검증(keep/drop/retype).

PROD 같은 LLM-relabel silver 라벨에 섞인 junk(비-제품)·정의혼동(브랜드↔회사,
작품↔날짜/사건)을 canonical rubric 기준으로 정리한다. 대상 라벨 span 만
판정하고 나머지 라벨(사람 KLUE gold 인 PER/LOC/DAT, 합성 PII)은 불변.

판정: KEEP(라벨 유지) / DROP(개체명 아님 → 제거) / 다른 canonical 타입(재배정).
경계(offset)는 건드리지 않는다 — 존재/타입만 본다(boundary 버킷은 별도).

사용:
    python -m ner.scripts.entity_revalidate --input data/klue/pii_all.jsonl \
        --label PROD --output data/klue/pii_all.prodclean.jsonl \
        --url http://localhost:8081/v1 --model cyankiwi/gemma-4-31B-it-AWQ-8bit
"""
import argparse
import asyncio
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

CANON = {'PER', 'LOC', 'ORG', 'PROD', 'EVT', 'DAT'}

# 라벨별 판정 rubric (ner_prompts.py 의 PROD 정의와 정합)
GUIDE = {
    'PROD': (
        'PROD(제품·작품)=시판 물품(전자기기·식품·약품·차량·무기·함정·항공기)·'
        '창작 작품(영화·드라마·노래·음반·서적·만화·게임·방송 프로그램)·'
        '패키지 SW/OS·브랜드+모델명. '
        '제외(DROP): 일반 구절·감상평("예뻤다 끝!","마지막","해피엔딩")·'
        '지시어+장르("이 영화")·법령·무형서비스/웹사이트/플랫폼·named 프로젝트·'
        '기술표준·상/훈장·모델번호 단독. '
        '재배정: 회사/브랜드 주체는 ORG, 사건·대회·전쟁은 EVT, 인물/배역명으로만 '
        '쓰였으면 PER, 지명은 LOC, 날짜표현은 DAT.'
    ),
}

PROMPT = """{guide}

문장: {sentence}

위 문장에서 아래 각 span 이 {label} 로 올바른지 canonical 기준으로 판정하라.
- 올바르면 "{label}"
- 개체명이 아니면(일반구절·감상평·지시어 등) "DROP"
- 다른 타입이면 해당 타입(PER/LOC/ORG/EVT/DAT)
span 목록: {spans}
JSON 배열로만 출력(설명 금지): [{{"text": "<span>", "verdict": "<{label}|DROP|PER|LOC|ORG|EVT|DAT>"}}]"""


def _parse(text: str) -> list:
    """LLM 응답에서 JSON 배열 추출."""
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


async def _judge(client, model, rec, label, sem, cache_fp, retries=5):
    """한 레코드의 대상 라벨 span 을 LLM 으로 판정 → {id, verdicts} 캐시 기록.

    전송 실패 시 backoff 재시도, 끝까지 실패하면 캐시에 안 써 다음 resume
    이 재시도하게 둔다(서버 다운 내성). 성공 시 verdict 를 즉시 flush.
    """
    spans = [e['text'] for e in rec['entities'] if e['label'] == label]
    prompt = PROMPT.format(
        guide=GUIDE[label], sentence=rec['text'], label=label,
        spans=json.dumps(spans, ensure_ascii=False),
    )
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model, temperature=0.0,
                    messages=[{'role': 'user', 'content': prompt}],
                )
            out = {}
            for item in _parse(resp.choices[0].message.content):
                t, v = item.get('text'), item.get('verdict')
                if t is not None and v:
                    out[t] = v
            cache_fp.write(json.dumps(
                {'id': rec['id'], 'verdicts': out}, ensure_ascii=False) + '\n')
            cache_fp.flush()
            return True
        except Exception as exc:  # noqa: BLE001 — 네트워크/파싱 모두 재시도
            if attempt == retries - 1:
                logger.warning('judge failed id=%s: %s', rec['id'], exc)
                return False
            await asyncio.sleep(2 ** attempt)


def _load_cache(path):
    """캐시 jsonl → {rec_id: verdicts}. 없으면 빈 dict."""
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done[r['id']] = r['verdicts']
    return done


async def _run(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.url, api_key='none')
    sem = asyncio.Semaphore(args.concurrency)
    rows = [json.loads(line) for line in open(args.input)]
    targets = [r for r in rows if any(
        e['label'] == args.label for e in r['entities'])]

    cache_path = args.output + '.cache.jsonl'
    done = _load_cache(cache_path)
    todo = [r for r in targets if r['id'] not in done]
    logger.info('records=%d %s-records=%d cached=%d todo=%d',
                len(rows), args.label, len(targets), len(done), len(todo))

    with open(cache_path, 'a', encoding='utf-8') as cache_fp:
        results = await asyncio.gather(*[
            _judge(client, args.model, r, args.label, sem, cache_fp)
            for r in todo])
    failed = results.count(False)
    if failed:
        logger.warning('%d records failed (re-run to resume)', failed)

    done = _load_cache(cache_path)  # 방금 쓴 것 포함 재로드
    kept = dropped = 0
    retyped = {}
    with open(args.output, 'w', encoding='utf-8') as fout:
        for rec in rows:
            v = done.get(rec['id'], {})
            ents = []
            for e in rec['entities']:
                if e['label'] != args.label:
                    ents.append(e)
                    continue
                verdict = v.get(e['text'], args.label)  # 무응답 시 보수적 keep
                if verdict == 'DROP':
                    dropped += 1
                    continue
                if verdict in CANON and verdict != args.label:
                    e = {**e, 'label': verdict}
                    retyped[verdict] = retyped.get(verdict, 0) + 1
                else:
                    kept += 1
                ents.append(e)
            rec['entities'] = ents
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
    logger.info('kept=%d dropped=%d retyped=%s failed=%d',
                kept, dropped, retyped, failed)


def main():
    p = argparse.ArgumentParser(
        description='LLM re-validation of one silver label (keep/drop/retype).')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--label', required=True, choices=sorted(GUIDE))
    p.add_argument('--url', default='http://localhost:8081/v1')
    p.add_argument('--model', default='cyankiwi/gemma-4-31B-it-AWQ-8bit')
    p.add_argument('--concurrency', type=int, default=6)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(_run(args))


if __name__ == '__main__':
    main()
