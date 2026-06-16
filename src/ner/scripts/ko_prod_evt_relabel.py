"""KLUE seed gold에 PROD/EVT를 LLM 재라벨로 증분 — canonical NER 5종 완성.

ko 라벨러(PROD/EVT 포함 6종)로 gold 문장을 재라벨해 PROD/EVT span만
수확하고, 기존 4종(PER/LOC/ORG/DAT)을 보존한 채 비-overlap 증분 병합한다.
4종은 KLUE 유래 gold가 권위이므로 재라벨에서는 PROD/EVT만 채택하며,
기존 span과 겹치는 PROD/EVT는 버리고 'reclassify 후보'로 별도 집계한다
(canonical상 KLUE OG는 대부분 진짜 조직 → 후보는 거의 없을 것으로 기대).

모드:
  relabel  gold 문장 → 6종 재라벨 → PROD/EVT char-span JSONL  [vLLM, GPU]
  merge    gold + relabel → 비-overlap PROD/EVT 증분 → 5종 gold JSONL

사용:
    python -m ner.scripts.ko_prod_evt_relabel relabel \
        --gold data/klue/origin.jsonl \
        --out results/classifier/ko/prod_evt_relabel.jsonl \
        --base-url http://localhost:8081/v1 \
        --model cyankiwi/gemma-4-31B-it-AWQ-8bit [--limit 500]
    python -m ner.scripts.ko_prod_evt_relabel merge \
        --gold data/klue/origin.jsonl \
        --relabel results/classifier/ko/prod_evt_relabel.jsonl \
        --out data/klue/origin.jsonl  # in-place 5종 승격 (gold 선로드 후 기록)

merge는 gold를 메모리에 모두 읽은 뒤 기록하므로 --out == --gold(in-place)가
안전하다. 기본 policy=contains-replace: outermost(완전 포함) PROD/EVT가 KLUE
조각을 교체하고, flat(overlap 0) gold를 보장한다(flat BIO 분류기 계약).
"""
import argparse
import asyncio
import collections
import json
import logging
import os
from typing import List

from ner.labelers.ja.span_matcher import match_spans
from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler

logger = logging.getLogger(__name__)

PROD_EVT = ("PROD", "EVT")


def _load_jsonl(path: str) -> List[dict]:
    """JSONL 파일을 레코드 리스트로 읽는다."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_ids(path: str) -> set:
    """기존 출력에서 완료된 레코드 id를 모아 resume를 지원한다."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def _is_noise(label: str, text: str) -> bool:
    """체계적 노이즈 PROD/EVT 필터 — 단일 글자, 단독 숫자(PROD).

    '무'(단일 글자)·'320'(모델번호 단독) 류를 버린다. EVT의 '6.25'는
    isdigit()=False라 보존된다.
    """
    t = text.strip()
    if len(t) <= 1:
        return True
    if label == "PROD" and t.isdigit():
        return True
    return False


def _classify_overlap(p: dict, ents: List[dict]) -> tuple:
    """재라벨 PROD/EVT span p와 기존 gold span들의 관계를 분류한다.

    flat BIO 분류기는 overlap span을 표현 불가 → 충돌 시 하나만 남겨야 한다.
    canonical·표준(outermost) 기준으로 *완전 포함*만 replace한다:
      add      겹치는 gold 없음 → p 추가
      replace  겹치는 gold가 모두 p에 완전 포함 + p가 엄격히 더 김
               → 그 조각들을 p로 교체 (예: '쓰촨 대지진'EVT ⊃ '쓰촨'LOC)
      skip     동일 span(타입 불일치)·p가 gold의 부분집합·부분 overlap
               → 보수적으로 gold 유지, p 폐기
    returns (action, overlapped_gold_spans).
    """
    ps, pe = p["start_char"], p["end_char"]
    over = [
        e for e in ents
        if ps < e["end_char"] and e["start_char"] < pe
    ]
    if not over:
        return "add", []
    contained = all(
        ps <= e["start_char"] and e["end_char"] <= pe for e in over
    )
    longer = (pe - ps) > max(e["end_char"] - e["start_char"] for e in over)
    if contained and longer:
        return "replace", over
    return "skip", over


# ── relabel ───────────────────────────────────────────────────────────


async def _relabel_one(labeler: VllmNERLabeler, rec: dict) -> dict:
    """레코드 1건을 6종 재라벨해 PROD/EVT char-span만 반환한다.

    전체 span을 match_spans로 정렬(consumed-range로 위치 확정)한 뒤
    PROD/EVT만 골라낸다 — 부분 문자열 충돌을 방지하기 위함.
    """
    text = rec["text"]
    raw = await labeler.alabel_spans(text, split=False)
    offset_spans = match_spans(text, raw)
    pe = [
        {
            "label": s["type"],
            "start_char": s["start"],
            "end_char": s["end"],
            "text": s["text"],
        }
        for s in offset_spans
        if s["type"] in PROD_EVT and not _is_noise(s["type"], s["text"])
    ]
    return {"id": rec["id"], "text": text, "spans": pe}


async def _relabel_async(
    records: List[dict], out_path: str, labeler: VllmNERLabeler
) -> None:
    """남은 레코드를 동시 재라벨하고 완료되는 대로 증분 기록한다."""
    tasks = [asyncio.ensure_future(_relabel_one(labeler, r)) for r in records]
    n_done = 0
    n_pe = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            n_pe += len(res["spans"])
            if n_done % 200 == 0:
                logger.info(
                    "relabeled %d/%d (PROD+EVT spans so far: %d)",
                    n_done, len(records), n_pe,
                )
    logger.info(
        "relabel done: %d records, %d PROD/EVT spans, "
        "tokens prompt=%d completion=%d",
        n_done, n_pe, labeler.total_prompt_tokens,
        labeler.total_completion_tokens,
    )


def run_relabel(args: argparse.Namespace) -> None:
    """relabel 모드 엔트리 — gold 문장을 PROD/EVT char-span으로 재라벨."""
    gold = _load_jsonl(args.gold)
    if args.limit:
        gold = gold[: args.limit]
    done = _done_ids(args.out)
    todo = [r for r in gold if r["id"] not in done]
    logger.info(
        "gold=%d, already_done=%d, todo=%d -> %s",
        len(gold), len(done), len(todo), args.out,
    )
    if not todo:
        logger.info("nothing to do")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    labeler = VllmNERLabeler(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    asyncio.run(_relabel_async(todo, args.out, labeler))


# ── merge ───────────────────────────────────────────────────────────────


def run_merge(args: argparse.Namespace) -> None:
    """merge 모드 — PROD/EVT를 기존 gold에 병합해 5종 gold를 만든다.

    policy=contains-replace: outermost(완전 포함) PROD/EVT가 KLUE 조각을
    교체. policy=additive: overlap PROD/EVT를 전부 폐기(보수).
    """
    gold = _load_jsonl(args.gold)
    relabel = {r["id"]: r["spans"] for r in _load_jsonl(args.relabel)}

    added = collections.Counter()
    overridden = collections.Counter()  # replace로 교체된 KLUE span 타입
    base_dist = collections.Counter()
    replaced_log = []  # outermost 교체 기록
    skipped_log = []   # exact/subset/partial → gold 유지
    out_rows = []

    for rec in gold:
        ents = list(rec["entities"])
        for e in ents:
            base_dist[e["label"]] += 1
        # 긴 span부터 처리 — 중첩된 재라벨 span의 결정 안정화
        for sp in sorted(
            relabel.get(rec["id"], []),
            key=lambda s: -(s["end_char"] - s["start_char"]),
        ):
            action, over = _classify_overlap(sp, ents)
            if action == "skip" or (
                action == "replace" and args.policy == "additive"
            ):
                clash = over[0]
                skipped_log.append({
                    "id": rec["id"], "text": sp["text"], "new": sp["label"],
                    "existing": clash["label"], "existing_text": clash["text"],
                })
                continue
            if action == "replace":
                for e in over:
                    if e in ents:
                        ents.remove(e)
                        overridden[e["label"]] += 1
                replaced_log.append({
                    "id": rec["id"], "text": sp["text"], "new": sp["label"],
                    "overridden": [
                        {"label": e["label"], "text": e["text"]} for e in over
                    ],
                })
            ents.append(sp)
            added[sp["label"]] += 1
        ents.sort(key=lambda e: (e["start_char"], e["end_char"]))
        out_rows.append(
            {"id": rec["id"], "text": rec["text"], "entities": ents}
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_dist = collections.Counter()
    for r in out_rows:
        for e in r["entities"]:
            final_dist[e["label"]] += 1
    logger.info(
        "merged gold (policy=%s) -> %s (%d records)",
        args.policy, args.out, len(out_rows),
    )
    logger.info("base dist:  %s", dict(base_dist))
    logger.info(
        "added PROD=%d EVT=%d | replaced=%d (overridden KLUE: %s) | "
        "skipped=%d",
        added.get("PROD", 0), added.get("EVT", 0), len(replaced_log),
        dict(overridden), len(skipped_log),
    )
    logger.info("final dist: %s", dict(final_dist))
    for c in skipped_log[:15]:
        logger.info(
            "  skip [%s] '%s' relabel=%s vs gold=%s('%s')",
            c["id"], c["text"], c["new"], c["existing"], c["existing_text"],
        )
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            for c in replaced_log:
                f.write(json.dumps(
                    {"action": "replace", **c}, ensure_ascii=False) + "\n")
            for c in skipped_log:
                f.write(json.dumps(
                    {"action": "skip", **c}, ensure_ascii=False) + "\n")
        logger.info(
            "replace/skip report (%d+%d) -> %s",
            len(replaced_log), len(skipped_log), args.report_out,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="KLUE gold PROD/EVT relabel + non-overlap merge"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_re = sub.add_parser("relabel", help="LLM relabel PROD/EVT (vLLM)")
    p_re.add_argument("--gold", required=True, help="seed gold JSONL")
    p_re.add_argument("--out", required=True, help="relabel output JSONL")
    p_re.add_argument(
        "--base-url", default="http://localhost:8081/v1",
        help="vLLM OpenAI-compatible base URL"
    )
    p_re.add_argument(
        "--model", default="cyankiwi/gemma-4-31B-it-AWQ-8bit",
        help="served model name"
    )
    p_re.add_argument("--limit", type=int, default=0, help="pilot: first N")
    p_re.add_argument("--concurrency", type=int, default=16)
    p_re.add_argument("--max-tokens", type=int, default=2048)
    p_re.set_defaults(func=run_relabel)

    p_mg = sub.add_parser("merge", help="merge PROD/EVT into 5-type gold")
    p_mg.add_argument("--gold", required=True, help="seed gold JSONL")
    p_mg.add_argument("--relabel", required=True, help="relabel JSONL")
    p_mg.add_argument("--out", required=True, help="merged 5-type gold JSONL")
    p_mg.add_argument(
        "--policy", choices=["contains-replace", "additive"],
        default="contains-replace",
        help="overlap policy: outermost replace (default) or additive"
    )
    p_mg.add_argument(
        "--report-out", default="",
        help="optional JSONL of replace/skip decisions"
    )
    p_mg.set_defaults(func=run_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
