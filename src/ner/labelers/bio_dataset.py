"""BIO 토큰 시퀀스 NER 데이터셋 스펙 레지스트리, 로더, span 추출기.

docs/ko/bio-span-process.md §2~§4에 문서화된
토큰화 단위 × BIO 변형 × span 추출 처리 과정을 구현한다.

공개 API:
    TokenUnit, DatasetSpec, REGISTRY,
    DatasetNotFoundError,
    normalize_bio_variant, extract_spans, load
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from datasets import ClassLabel, load_dataset
from datasets.exceptions import DatasetNotFoundError as HFDatasetNotFoundError


class DatasetNotFoundError(Exception):
    """데이터셋 스펙이 REGISTRY에 없거나 HF 로드에 실패할 때 발생한다."""


class TokenUnit(str, Enum):
    """NER 데이터셋의 토큰화 단위."""
    SYLLABLE = "syllable"   # KLUE: 명시적 공백 토큰을 포함한 음절 수준
    WORD = "word"           # WikiANN 방식: 공백 분리 단어
    MORPHEME = "morpheme"   # KMOU/Stockmark: 형태소 분석 단위


@dataclass(frozen=True)
class DatasetSpec:
    """BIO 토큰 시퀀스 NER 데이터셋의 불변 메타데이터."""
    name: str               # HuggingFace 데이터셋 이름
    lang: str               # 언어 코드 (예: "ko")
    unit: TokenUnit
    bio_variant: str        # "iob2" | "kmou_i_alone"
    joiner: str             # span 텍스트 재구성용 토큰 결합자
    config: Optional[str] = None  # HF datasets 설정 이름


# ---------------------------------------------------------------------------
# 레지스트리 — 초기 항목: 한국어 전용
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetSpec] = {
    "klue": DatasetSpec(
        name="klue",
        lang="ko",
        unit=TokenUnit.SYLLABLE,
        bio_variant="iob2",
        joiner="",
        config="ner",
    ),
    # nlp-kmu/kor_ner는 github.com/kmounlp/NER의 HF 캐시 버전이다.
    # 형태소 수준 토큰(어간에서 조사 분리).
    # ClassLabel 이름: ["I", "O", "B_OG", "B_TI", "B_LC", "B_DT", "B_PS"]
    "nlp-kmu/kor_ner": DatasetSpec(
        name="nlp-kmu/kor_ner",
        lang="ko",
        unit=TokenUnit.MORPHEME,
        bio_variant="kmou_i_alone",
        joiner=" ",
        config=None,
    ),
}


# ---------------------------------------------------------------------------
# BIO 변형 정규화
# ---------------------------------------------------------------------------

def normalize_bio_variant(tags: List[str], variant: str) -> List[str]:
    """데이터셋별 BIO 태그를 표준 IOB2로 변환한다.

    지원 변형:
        "iob2"          — 항등 변환 (이미 표준)
        "kmou_i_alone"  — B_TYPE → B-TYPE, 단독 I → I-{prev_type}
    """
    if variant == "iob2":
        return list(tags)

    if variant == "kmou_i_alone":
        out: List[str] = []
        current_type: Optional[str] = None
        for tag in tags:
            if tag == "O":
                out.append("O")
                current_type = None
            elif tag.startswith("B_"):
                entity_type = tag[2:]
                out.append(f"B-{entity_type}")
                current_type = entity_type
            elif tag == "I":
                if current_type is not None:
                    out.append(f"I-{current_type}")
                else:
                    out.append("O")
            elif tag.startswith("B-"):
                out.append(tag)
                current_type = tag[2:]
            elif tag.startswith("I-"):
                out.append(tag)
            else:
                out.append("O")
                current_type = None
        return out

    raise ValueError(f"unknown bio_variant: {variant!r}")


# ---------------------------------------------------------------------------
# IOB2 태그에서 span 추출
# ---------------------------------------------------------------------------

def extract_spans(
    tokens: List[str],
    bio_tags: List[str],
    joiner: str,
) -> List[dict]:
    """IOB2 태그가 붙은 토큰 시퀀스에서 엔티티 span을 추출한다.

    Args:
        tokens: 토큰 리스트 (음절 수준의 경우 공백만 있는 토큰 포함 가능).
        bio_tags: 토큰과 1:1로 정렬된 IOB2 태그.
        joiner: span 내 토큰 결합 문자열 (음절은 "", 단어/형태소는 " ").

    Returns:
        {"text": str, "type": str} 딕셔너리의 리스트.
    """
    spans: List[dict] = []
    current_chars: List[str] = []
    current_type: str = ""

    for tok, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            # 진행 중인 span을 플러시한다
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            # 새 span 시작
            current_type = tag[2:]
            current_chars = [tok] if tok.strip() else []
        elif tag.startswith("I-") and current_type and tag[2:] == current_type:
            # 현재 span을 이어간다
            if tok.strip():
                current_chars.append(tok)
        else:
            # O 또는 타입 불일치 I-: 플러시 후 초기화
            if current_chars and current_type:
                spans.append({"text": joiner.join(current_chars), "type": current_type})
            current_chars = []
            current_type = ""

    # 마지막 span 플러시
    if current_chars and current_type:
        spans.append({"text": joiner.join(current_chars), "type": current_type})

    return spans


# ---------------------------------------------------------------------------
# 엔드투엔드 로더
# ---------------------------------------------------------------------------

def load(
    spec_name: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """BIO NER 데이터셋을 엔드투엔드로 로드한다: HF 로드 → IOB2 정규화 → span 추출.

    Args:
        spec_name: REGISTRY의 키 (예: "klue", "nlp-kmu/kor_ner").
        split: 데이터셋 분할 (예: "train", "validation", "test").
        max_samples: 반환할 레코드 수 제한.

    Returns:
        레코드 리스트: {"id": str, "tokens": list[str], "bio_tags": list[str],
                        "spans": list[{"text": str, "type": str}]}

    Raises:
        DatasetNotFoundError: spec_name이 REGISTRY에 없거나 HF 로드에 실패한 경우.
    """
    try:
        spec = REGISTRY[spec_name]
    except KeyError:
        raise DatasetNotFoundError(f"unknown spec: {spec_name!r}")

    # JSONL 폴백을 먼저 시도한다 (Python 3.13 + datasets 비호환성 회피)
    jsonl_path = Path("/data/ner") / spec.name.replace("/", "_") / f"{split}.jsonl"
    if jsonl_path.exists():
        return _load_from_jsonl(jsonl_path, spec, max_samples)

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/work/.huggingface/datasets")

    try:
        hf_dataset = load_dataset(
            spec.name,
            spec.config,
            split=split,
            cache_dir=cache_dir,
        )
    # ValueError 는 존재하지 않는 config·split 을 가리키므로 부재로 묶는다.
    # 반면 TypeError 같은 호출 오류는 데이터 부재가 아니라 코드 결함이라
    # 삼키지 않고 그대로 전파한다.
    except (HFDatasetNotFoundError, FileNotFoundError, ValueError) as exc:
        raise DatasetNotFoundError(f"HF load failed for {spec.name!r}: {exc}") from exc

    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    # 정수 → 문자열 변환을 위해 ClassLabel을 감지한다
    features = hf_dataset.features
    tag_feature = features.get("ner_tags")
    label_feature = None
    if hasattr(tag_feature, "feature"):
        label_feature = tag_feature.feature
    elif isinstance(tag_feature, ClassLabel):
        label_feature = tag_feature

    records: List[dict] = []
    for i, row in enumerate(hf_dataset):
        tokens = list(row["tokens"])
        raw_tags = row["ner_tags"]

        # ClassLabel이 있을 때 정수 태그를 문자열로 디코딩한다
        if label_feature is not None and isinstance(label_feature, ClassLabel):
            raw_tags_str = [label_feature.int2str(t) for t in raw_tags]
        else:
            raw_tags_str = [str(t) for t in raw_tags]

        bio_tags = normalize_bio_variant(raw_tags_str, spec.bio_variant)
        spans = extract_spans(tokens, bio_tags, spec.joiner)

        records.append({
            "id": str(i),
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": spans,
            "sentence": spec.joiner.join(tokens),
        })

    return records


def _load_from_jsonl(
    path: Path,
    spec: DatasetSpec,
    max_samples: Optional[int],
) -> List[dict]:
    """사전 내보낸 JSONL 레코드를 로드하고 BIO 정규화 + span 추출을 적용한다."""
    raw_records: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))
            if max_samples is not None and len(raw_records) >= max_samples:
                break

    records: List[dict] = []
    for i, row in enumerate(raw_records):
        tokens = row.get("tokens", [])
        raw_tags_str = row.get("ner_tags", [])
        bio_tags = normalize_bio_variant(raw_tags_str, spec.bio_variant)
        spans = extract_spans(tokens, bio_tags, spec.joiner)
        records.append({
            "id": row.get("id", str(i)),
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": spans,
            "sentence": spec.joiner.join(tokens),
        })

    return records
