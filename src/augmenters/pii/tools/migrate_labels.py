"""#17 Phase 1 — PII JSONL 라벨 마이그레이션 CLI.

기존 PII 주입 산출물의 `ADDRESS` 라벨을 `LOC`로, `DOB` 라벨을 `DAT`로
치환한다. JSONL 본체와 부수 파일(`.stats.json`·`.verify.json`)의 라벨 키도
동일하게 치환하며, 이미 치환된 대상은 no-op(idempotent)이다.

사용 예:
    python -m augmenters.pii.tools.migrate_labels data/stockmark/pii_test.jsonl
    python -m augmenters.pii.tools.migrate_labels data/stockmark/ --dry-run

span 오프셋은 건드리지 않으며 라벨 이외 필드는 원형을 유지한다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, str] = {
    'ADDRESS': 'LOC',
    'DOB': 'DAT',
}

# 부수 파일(.stats.json, .verify.json)의 per_label 섹션을 병합할 때,
# 새 키가 이미 존재하면 하위 카운트를 합친다.


def _merge_numeric_dict(
    target: dict, incoming: dict,
) -> None:
    """숫자 카운트 기반 dict를 in-place로 누적 합산한다."""
    for k, v in incoming.items():
        if isinstance(v, (int, float)):
            target[k] = target.get(k, 0) + v
        elif isinstance(v, dict):
            sub = target.setdefault(k, {})
            if isinstance(sub, dict):
                _merge_numeric_dict(sub, v)
        else:
            target.setdefault(k, v)


def migrate_entity(entity: dict) -> tuple[dict, bool]:
    """단일 엔티티 dict의 `label` 필드를 치환한다.

    Returns:
        (migrated entity, changed flag)
    """
    label = entity.get('label')
    if label in LABEL_MAP:
        new_entity = dict(entity)
        new_entity['label'] = LABEL_MAP[label]
        return new_entity, True
    return entity, False


def migrate_jsonl(path: str, dry_run: bool = False) -> Counter:
    """JSONL 파일의 모든 엔티티 라벨을 치환하고 변경 카운트를 반환한다."""
    changes: Counter = Counter()
    out_lines: list[str] = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            raw = raw.rstrip('\n')
            if not raw:
                out_lines.append(raw)
                continue
            record = json.loads(raw)
            new_entities = []
            for ent in record.get('entities', []):
                migrated, changed = migrate_entity(ent)
                if changed:
                    changes[ent['label']] += 1
                new_entities.append(migrated)
            record['entities'] = new_entities
            out_lines.append(json.dumps(record, ensure_ascii=False))

    if not dry_run and changes:
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for line in out_lines:
                f.write(line + '\n')
        os.replace(tmp_path, path)
    return changes


def migrate_per_label_dict(per_label: dict) -> tuple[dict, Counter]:
    """`per_label` dict(stats/verify용)의 키를 치환·병합한다."""
    changes: Counter = Counter()
    new_dict: dict = {}
    for key, value in per_label.items():
        new_key = LABEL_MAP.get(key, key)
        if new_key != key:
            changes[key] += 1
        if new_key in new_dict and isinstance(value, dict):
            _merge_numeric_dict(new_dict[new_key], value)
        elif new_key in new_dict and isinstance(value, (int, float)):
            new_dict[new_key] += value
        else:
            new_dict[new_key] = value
    return new_dict, changes


def migrate_json(path: str, dry_run: bool = False) -> Counter:
    """stats.json / verify.json 의 `per_label` 섹션을 치환한다."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    changes: Counter = Counter()
    per_label = data.get('per_label')
    if isinstance(per_label, dict):
        new_per_label, delta = migrate_per_label_dict(per_label)
        data['per_label'] = new_per_label
        changes.update(delta)
    if not dry_run and changes:
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    return changes


def _collect_targets(path: str) -> list[str]:
    """파일 또는 디렉토리 경로에서 대상 파일 목록을 모은다."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    targets: list[str] = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and (
            name.endswith('.jsonl')
            or name.endswith('.stats.json')
            or name.endswith('.verify.json')
        ):
            targets.append(full)
    return targets


def migrate_path(path: str, dry_run: bool = False) -> dict[str, Counter]:
    """파일/디렉토리 단위 치환 결과를 파일별 Counter로 반환한다."""
    results: dict[str, Counter] = {}
    for target in _collect_targets(path):
        if target.endswith('.jsonl'):
            results[target] = migrate_jsonl(target, dry_run=dry_run)
        elif target.endswith('.json'):
            results[target] = migrate_json(target, dry_run=dry_run)
    return results


def _format_report(results: dict[str, Counter], dry_run: bool) -> str:
    lines = []
    prefix = '[dry-run] ' if dry_run else ''
    total: Counter = Counter()
    for path, delta in results.items():
        if not delta:
            lines.append(f'{prefix}{path}: no-op (already migrated)')
            continue
        summary = ', '.join(
            f'{k}→{LABEL_MAP[k]}×{v}' for k, v in sorted(delta.items())
        )
        lines.append(f'{prefix}{path}: {summary}')
        total.update(delta)
    if total:
        lines.append('')
        lines.append(
            f'{prefix}Total: ' + ', '.join(
                f'{k}→{LABEL_MAP[k]}×{v}' for k, v in sorted(total.items())
            )
        )
    return '\n'.join(lines) if lines else f'{prefix}No files found.'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Migrate ADDRESS→LOC and DOB→DAT in PII JSONL bundles.',
    )
    parser.add_argument(
        'path',
        help='JSONL file, stats/verify JSON file, or directory.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report changes without writing files.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(message)s',
    )
    results = migrate_path(args.path, dry_run=args.dry_run)
    print(_format_report(results, args.dry_run))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
