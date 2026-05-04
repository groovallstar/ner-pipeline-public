"""WikiANN-vi canonical 덤프 로더.

canonical 5종으로 축소된 JSONL 덤프
(`data/wikiann_vi/vi_wikiann_recall_{split}.jsonl`, recall-merge
결과)를 그대로 읽는다. HF 원본(WikiANN 3종) 로딩은 본 모듈의 책임이
아니다 — 재라벨 파이프라인은 augmenters/wikiann_vi 쪽에서 직접 HF를
읽는다. 폴백 없음.

출력 스키마: `{id, text, gold_spans:[{text, type, start, end}]}`
— 기본 span_key는 `gold_spans_8type_merged` (recall 정책 병합 결과의
필드명. 데이터 호환성을 위해 필드명 그대로 유지).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union


class VietnameseDatasetLoader:
    """canonical WikiANN-vi 덤프를 gold_spans 레코드로 반환한다."""

    DEFAULT_PATH: dict = {
        'train': Path(
            'data/wikiann_vi/vi_wikiann_recall_train.jsonl'
        ),
        'validation': Path(
            'data/wikiann_vi/vi_wikiann_recall_validation.jsonl'
        ),
        'test': Path(
            'data/wikiann_vi/vi_wikiann_recall_test.jsonl'
        ),
    }
    SPAN_KEY = 'gold_spans_8type_merged'

    def load(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        span_key: Optional[str] = None,
    ) -> List[dict]:
        """canonical WikiANN-vi 덤프를 로드한다.

        Args:
            split: 'train' | 'validation' | 'test'. `path` 지정 시 무시.
            max_samples: 반환할 레코드 수 상한.
            path: 임의 JSONL 경로. split 대신 사용.
            span_key: gold로 사용할 span 필드. 기본값은
                `gold_spans_8type_merged` (recall 정책 병합, canonical 5종).
                WikiANN 원본 3종으로 평가하려면 `'gold_spans'` 지정.

        파일이 없으면 `FileNotFoundError`. 폴백 없음.
        """
        if path is not None:
            target = Path(path)
        elif split in self.DEFAULT_PATH:
            target = self.DEFAULT_PATH[split]
        else:
            raise ValueError(
                f'unknown split {split!r}; expected one of '
                f'{list(self.DEFAULT_PATH)}'
            )

        if not target.exists():
            raise FileNotFoundError(
                f'WikiANN-vi canonical dump not found: {target}. '
                'Generate it via ner.augmenters.wikiann_vi + '
                'ner.augmenters.wikiann_vi.merge_confidence.'
            )
        return _read_records(
            target, max_samples, span_key or self.SPAN_KEY,
        )

    @staticmethod
    def load_local(
        path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        """임의 JSONL을 labelers/vi 스키마로 읽는다.

        입력이 재라벨 덤프(`gold_spans_8type_merged` 계열)가 아니라 PII
        증강 산출물(`entities` 스키마)인 경우를 처리한다.
        """
        records: List[dict] = []
        with open(Path(path), encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                gold_spans = [
                    {
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
                    }
                    for ent in data.get('entities', [])
                ]
                records.append({
                    'id': str(data.get('id', i)),
                    'text': data['text'],
                    'gold_spans': gold_spans,
                })
                if (max_samples is not None
                        and len(records) >= max_samples):
                    break
        return records


def _read_records(
    path: Path, max_samples: Optional[int], span_key: str,
) -> List[dict]:
    """WikiANN-vi 덤프 JSONL을 gold_spans 레코드로 반환한다.

    재라벨 산출물은 레코드 최상위에 `gold_spans`(WikiANN 3종),
    `gold_spans_8type` 또는 `gold_spans_8type_merged`(canonical 5종)를
    담고 있다. `span_key`로 어느 필드를 gold로 삼을지 선택한다.
    """
    records: List[dict] = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            raw_spans = data.get(span_key, [])
            gold_spans = [
                {
                    'text': s.get('text', ''),
                    'type': s.get('type', ''),
                    'start': int(s.get('start', 0)),
                    'end': int(s.get('end', 0)),
                }
                for s in raw_spans
            ]
            records.append({
                'id': str(data.get('id', i)),
                'text': data['text'],
                'gold_spans': gold_spans,
            })
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


# ── BIO ↔ offset span 유틸 ──────────────────────────────────────────
# HF 원본(WikiANN) 재라벨 파이프라인(augmenters/wikiann_vi)이 BIO→span
# 변환을 위해 사용한다. 본 모듈의 로딩 경로와는 분리된 유틸이다.


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
