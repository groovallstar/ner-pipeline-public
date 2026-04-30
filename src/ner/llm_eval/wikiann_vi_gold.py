"""WikiANN-vi 인간 주석 BIO gold → offset span 변환 모듈.

본 모듈은 silver(LLM 라벨러 산출) 와의 정량 비교를 위해 WikiANN-vi 의
PER · LOC · ORG 3종 gold 를 silver 와 동일한 offset span 형식으로 노출한다.

silver 측 BIO→span 변환과 동일한 함수 (``labelers.vi.dataset_loader.
bio_to_offset_spans``) 를 재사용해 토큰 결합·offset 정렬 정합성을 보장한다.
"""
from __future__ import annotations

import os
from typing import Dict, List

from datasets import ClassLabel, load_dataset

from labelers.vi.dataset_loader import bio_to_offset_spans

GOLD_TYPES = ('PER', 'LOC', 'ORG')


def load_wikiann_vi_gold(
    split: str,
    *,
    hf_name: str = 'unimelb-nlp/wikiann',
    hf_config: str = 'vi',
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> List[dict]:
    """WikiANN-vi 한 split 을 (id, text, gold_spans) 리스트로 반환.

    silver 의 ``id`` 필드 (str(i)) 와 동일한 키 규칙을 사용해 silver-gold
    매칭에 그대로 쓸 수 있다.
    """
    cache = cache_dir or os.environ.get(
        'HF_DATASETS_CACHE', '/work/.huggingface/datasets',
    )
    ds = load_dataset(
        hf_name, hf_config, split=split,
        cache_dir=cache, trust_remote_code=False,
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    feature = ds.features['ner_tags'].feature
    is_class_label = isinstance(feature, ClassLabel)

    records: List[dict] = []
    for i, row in enumerate(ds):
        tokens = list(row['tokens'])
        raw_tags = row['ner_tags']
        tags = (
            [feature.int2str(t) for t in raw_tags]
            if is_class_label
            else [str(t) for t in raw_tags]
        )
        text, spans = bio_to_offset_spans(tokens, tags)
        # WikiANN 기본 라벨은 이미 PER/LOC/ORG 3종. 안전망으로 필터.
        spans = [s for s in spans if s['type'] in GOLD_TYPES]
        records.append({
            'id': str(i),
            'text': text,
            'gold_spans': spans,
        })
    return records


def load_all_splits(
    splits: tuple = ('train', 'validation', 'test'),
    *,
    max_samples: int | None = None,
) -> Dict[str, List[dict]]:
    """주어진 split 들을 한 번에 로드해 dict 로 반환."""
    return {
        s: load_wikiann_vi_gold(s, max_samples=max_samples)
        for s in splits
    }
