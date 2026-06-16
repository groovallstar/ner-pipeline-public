"""KLUE NER(음절 BIO) → canonical char-span gold JSONL 변환.

KLUE 6종 중 canonical 대응 4종만 남겨 재매핑한다:
PS→PER, LC→LOC, OG→ORG, DT→DAT. TI·QT는 canonical 대응이 없어 드롭.
출력은 분류기(`ner.classifier.data_utils.load_jsonl`) 계약과 동일한
{id, text, entities:[{label, start_char, end_char, text}]} JSONL이며,
한국어 canonical 10종 gold의 첫 증분(NER 3종 + DAT). PROD/EVT·PII 6종은
후속 증분에서 같은 gold에 누적한다.
"""

import argparse
import json
import logging
from pathlib import Path

from ner.labelers.dataset_loader import HFTokenDatasetLoader

logger = logging.getLogger(__name__)

# KLUE 약어 → canonical. 미등록(TI/QT)은 드롭한다.
_KLUE_TO_CANONICAL = {"PS": "PER", "LC": "LOC", "OG": "ORG", "DT": "DAT"}

# 출력 기준 디렉토리: 프로젝트 루트 data/ko_klue/ (src/ner/scripts/ → parents[3])
_OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "ko_klue"


def bio_to_char_spans(tokens, tags):
    """음절 BIO 토큰/태그 → (재구성 텍스트, canonical char-span 엔티티 리스트).

    비공백 음절은 그대로, 공백 토큰('' 또는 ' ')은 공백 1칸으로 재구성하고
    char offset을 그 텍스트 기준으로 매긴다. 공백 토큰은 자체 BIO 태그를
    따른다(KLUE는 다어절 엔티티 내부 공백을 I-로 태깅 → 엔티티에 포함;
    O면 경계). TI/QT 등 canonical 미대응 스팬은 드롭한다.
    """
    parts = []
    pos = 0
    spans = []  # (label, start_char, end_char)
    cur = None  # [label, start_char, end_char]

    for tok, tag in zip(tokens, tags):
        is_space = tok.strip() == ""
        piece = " " if is_space else tok
        start = pos
        parts.append(piece)
        pos += len(piece)

        raw = tag[2:] if (tag.startswith("B-") or tag.startswith("I-")) else ""
        label = _KLUE_TO_CANONICAL.get(raw)

        if tag.startswith("B-"):
            if cur:
                spans.append(tuple(cur))
                cur = None
            if label:
                cur = [label, start, pos]
        elif tag.startswith("I-") and cur and label == cur[0]:
            cur[2] = pos
        else:
            if cur:
                spans.append(tuple(cur))
                cur = None

    if cur:
        spans.append(tuple(cur))

    text = "".join(parts)
    ents = [
        {"label": lab, "start_char": s, "end_char": e, "text": text[s:e]}
        for (lab, s, e) in spans
    ]
    return text, ents


def convert(splits):
    """지정 split들을 로드·변환해 단일 레코드 리스트로 합친다."""
    loader = HFTokenDatasetLoader()
    rows = []
    for split in splits:
        recs = loader.load("klue", config="ner", split=split)
        logger.info("Loaded %d KLUE records [%s]", len(recs), split)
        for rec in recs:
            text, ents = bio_to_char_spans(rec["tokens"], rec["ner_tags"])
            rows.append(
                {"id": str(len(rows)), "text": text, "entities": ents}
            )
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert KLUE NER to canonical char-span gold JSONL"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"],
        help="KLUE splits to merge (default: train validation)"
    )
    parser.add_argument(
        "--output", default=str(_OUT_DIR / "klue_canonical.jsonl"),
        help="Output JSONL path"
    )
    args = parser.parse_args()

    rows = convert(args.splits)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ent = sum(len(r["entities"]) for r in rows)
    logger.info(
        "Wrote %d records, %d entities -> %s", len(rows), n_ent, out_path
    )


if __name__ == "__main__":
    main()
