"""WikiANN Vietnamese NER 데이터셋 로더.

데이터셋: unimelb-nlp/wikiann (config='vi')
형식: 토큰 리스트 + BIO 태그 리스트 (3 엔티티: PER, LOC, ORG)

JA loader(Stockmark)와 동일한 `gold_spans` 스키마로 변환한다:
    {"id": str, "text": str,
     "gold_spans": [{"text", "type", "start", "end"}]}

텍스트는 tokens를 단일 공백으로 join하여 재구성하고, 각 span의 start/end는
재구성 텍스트 상의 문자 오프셋을 가리킨다.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import ClassLabel, load_dataset


class VietnameseDatasetLoader:
    """WikiANN-vi를 offset span 형식으로 로드한다."""

    DATASET_NAME = 'unimelb-nlp/wikiann'
    CONFIG = 'vi'

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or os.environ.get(
            'HF_DATASETS_CACHE', '/work/.huggingface/datasets'
        )

    def load(
        self,
        name: Optional[str] = None,
        config: Optional[str] = None,
        split: str = 'test',
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        """WikiANN-vi를 offset span 레코드로 로드한다.

        Args:
            name: 데이터셋 이름 (기본값: unimelb-nlp/wikiann)
            config: 언어 config (기본값: vi)
            split: 'train' | 'validation' | 'test'
            max_samples: 반환 레코드 수 제한
        """
        dataset_name = name or self.DATASET_NAME
        dataset_config = config or self.CONFIG
        hf_dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        if max_samples is not None:
            hf_dataset = hf_dataset.select(
                range(min(max_samples, len(hf_dataset)))
            )

        label_feature = hf_dataset.features['ner_tags'].feature
        is_class_label = isinstance(label_feature, ClassLabel)

        records: List[dict] = []
        for i, row in enumerate(hf_dataset):
            tokens = list(row['tokens'])
            raw_tags = row['ner_tags']
            tags = (
                [label_feature.int2str(t) for t in raw_tags]
                if is_class_label
                else [str(t) for t in raw_tags]
            )
            text, gold_spans = bio_to_offset_spans(tokens, tags)
            records.append({
                'id': str(i),
                'text': text,
                'gold_spans': gold_spans,
            })
        return records

    @staticmethod
    def load_local(
        path: str, max_samples: Optional[int] = None
    ) -> List[dict]:
        """로컬 JSONL(PII 주입 결과 등)을 load() 동일 스키마로 읽는다.

        입력: {text, entities:[{label|type, start_char|start, end_char|end, text}]}
        출력: {id, text, gold_spans:[{text, type, start, end}]}
        """
        records: List[dict] = []
        with open(Path(path), encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                gold_spans = []
                for ent in data.get('entities', []):
                    gold_spans.append({
                        'text': ent.get('text', ''),
                        'type': ent.get(
                            'label', ent.get('type', '')
                        ),
                        'start': int(ent.get(
                            'start_char', ent.get('start', 0)
                        )),
                        'end': int(ent.get(
                            'end_char', ent.get('end', 0)
                        )),
                    })
                records.append({
                    'id': str(data.get('id', i)),
                    'text': data['text'],
                    'gold_spans': gold_spans,
                })
                if (max_samples is not None
                        and len(records) >= max_samples):
                    break
        return records


def bio_to_offset_spans(
    tokens: List[str], tags: List[str]
) -> Tuple[str, List[dict]]:
    """토큰·BIO 태그를 (text, gold_spans)로 변환한다.

    text는 tokens를 단일 공백으로 join하고, 각 span의 start/end는 재구성
    텍스트 상의 문자 오프셋을 가리킨다.
    """
    if len(tokens) != len(tags):
        raise ValueError(
            f'tokens/tags length mismatch: '
            f'{len(tokens)} vs {len(tags)}'
        )

    offsets: List[int] = []
    cursor = 0
    for idx, tok in enumerate(tokens):
        if idx > 0:
            cursor += 1
        offsets.append(cursor)
        cursor += len(tok)

    text = ' '.join(tokens)

    spans: List[dict] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith('B-'):
            entity_type = tag[2:]
            j = i + 1
            while j < n and tags[j] == f'I-{entity_type}':
                j += 1
            start = offsets[i]
            end = offsets[j - 1] + len(tokens[j - 1])
            spans.append({
                'text': text[start:end],
                'type': entity_type,
                'start': start,
                'end': end,
            })
            i = j
        else:
            i += 1
    return text, spans


def offset_spans_to_bio(
    tokens: List[str], spans: List[dict]
) -> List[str]:
    """tokens와 offset spans로부터 BIO 태그를 복원한다 (왕복 검증용).

    공백 join 텍스트에 대해 spans를 토큰 경계에 맞춰 BIO로 역변환한다.
    경계 불일치 span은 건너뛴다.
    """
    tags = ['O'] * len(tokens)
    offsets: List[int] = []
    cursor = 0
    for idx, tok in enumerate(tokens):
        if idx > 0:
            cursor += 1
        offsets.append(cursor)
        cursor += len(tok)

    for span in spans:
        start = int(span['start'])
        end = int(span['end'])
        entity_type = span['type']
        start_tok: Optional[int] = None
        end_tok: Optional[int] = None
        for idx, off in enumerate(offsets):
            tok_end = off + len(tokens[idx])
            if off == start:
                start_tok = idx
            if tok_end == end:
                end_tok = idx
        if (start_tok is None or end_tok is None
                or start_tok > end_tok):
            continue
        tags[start_tok] = f'B-{entity_type}'
        for k in range(start_tok + 1, end_tok + 1):
            tags[k] = f'I-{entity_type}'
    return tags
