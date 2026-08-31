"""EN `FAC` 디스크 마이그레이션 원장 — 손댄 span 을 지문에 묶는다.

`data/**` 는 gitignore 이고 주입 텍스트는 LLM 산출이라 다시 만들 수 없다.
그래서 "어떤 span 이 왜 손대졌는가" 가 저장소에 남는 자리는 이 원장뿐이고,
원장이 실물과 어긋나지 않는지를 여기서 본다.

**지문이 없으면 역적용은 정의상 참이다.** 원장을 되돌려 나온 코퍼스를 원장이
선언한 개수로 다시 재면, 역적용이 원장대로 뒤집었으니 차이가 언제나 원장 행
수가 된다. 그래서 역적용의 대조 상대는 원장이 아니라 **마이그레이션 전 파일의
SHA256** 이다 — 원장에 없는 변경이 섞였거나 있는 변경이 빠졌으면 그 해시가
안 나온다. KO 원장이 `gold_sha256` 을 갖는 이유와 같다
(`tests/ner/labelers/test_ko_locorg_ledger.py`).

후 지문은 반대편을 막는다 — gitignore 된 코퍼스를 나중에 조용히 편집하면
원장은 그대로인데 실물만 달라지고, 그 어긋남이 여기서 붉어진다.

변환 산출물(`{split}.jsonl`)은 원장이 손대지 않는다. 원천 + 코드에서 결정적
으로 다시 만들어지므로 지문만 남겨 **재현성 앵커**로 쓴다.
"""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.__main__ import SPLIT_FILES, convert_one_split
from ner.augmenters.ontonotes_en.convert import load_id2label
from ner.scripts import migrate_en_fac_disk_corpus as migration

_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _ROOT / 'data' / 'ontonotes_en'
RAW_DIR = DATA_DIR / 'raw'
PII_DIR = DATA_DIR / 'pii'
LEDGER_PATH = (
    _ROOT / 'src' / 'ner' / 'augmenters' / 'ontonotes_en'
    / 'data' / 'fac_disk_migration_ledger.json'
)
SCRIPT_PATH = Path(migration.__file__)

# 손댄 자리의 수 — 원장을 고치는 사람이 이 숫자도 손으로 고쳐야 한다.
GOLDEN_MOVED = 274
GOLDEN_REMOVED = 15

# **모호해서 안 건드린 자리.** 비어 있어도 "비어 있음" 을 세어 대조한다 —
# 조용히 채우면 어느 엔티티를 뒤집을지가 임의 선택이 되는데, 개수 등식은
# 어느 쪽을 뒤집든 1 이동이라 안 걸리고 후 지문은 그 결과를 봉인한다.
GOLDEN_AMBIGUOUS = 0

# 판정 표는 손대라는데 주입이 이미 떨궈 대상이 없던 자리. 변환 쪽 판정 수와
# 실제 적용 수의 차가 여기 있다 — 등식으로 쓰지 않고 값으로만 박는다.
GOLDEN_ABSENT = 5

# 판정 표가 `ORG` 아닌 답을 낸 원본 `FAC` span 수. 위 넷의 합과 같아야 한다.
GOLDEN_TABLE_VERDICTS = {'LOC': 277, 'DROP': 17}

# 주입 모듈 — 마이그레이션이 이 경로를 지나면 재생 대조가 항진명제가 된다.
REPLAY_MODULE_PREFIX = 'ner.augmenters.pii'


def _ledger():
    return json.loads(LEDGER_PATH.read_text(encoding='utf-8'))


def _entries(ledger):
    """원장이 다루는 자리 전량 — 적용한 것과 안 한 것을 합친다."""
    return (ledger['move'] + ledger['remove']
            + ledger['not_applied']['ambiguous']
            + ledger['not_applied']['absent_from_pii'])


def _span_key(entry):
    """변환 산출물 기준 span 신원. 주입 offset 은 문장이 다시 쓰여 못 쓴다."""
    return (entry['split'], entry['id'],
            entry['start_char'], entry['end_char'], entry['verdict'])


def _corpus_files(ledger):
    return {
        **{f'pii/{split}.jsonl': PII_DIR / f'{split}.jsonl'
           for split in migration.SPLITS},
        ledger['merged_name']: DATA_DIR / ledger['merged_name'],
    }


def _restore(path, ledger):
    """현 파일에 원장을 역적용해 마이그레이션 전 바이트를 만든다."""
    rows = migration.read_rows(path)
    ids = {row['id'] for row in rows}
    migration.reverse_apply_to_rows(
        rows,
        [e for e in ledger['move'] if e['id'] in ids],
        [e for e in ledger['remove'] if e['id'] in ids],
    )
    return migration.dump_rows(rows).encode('utf-8')


# ── 데이터 없이 도는 검사 ────────────────────────────────────────────────

def test_the_migration_never_imports_the_replay_path():
    """수락 기준 1 — 스크립트가 주입 모듈을 import 하지 않는다.

    주입 산출물의 NER 엔티티를 `extract_spans`+`merge_entities` 로 다시
    도출하면 `test_injection_replays_exactly` 가 항진명제가 된다 — 그 테스트가
    바로 그 경로로 기대치를 만들어 산출물과 대조하기 때문이다.

    **문자열 검색이 아니라 AST 로 본다.** 스크립트 docstring 이 그 이름들을
    "안 쓴다" 고 적고 있어, 문면을 훑으면 설명하는 문장 자체가 걸린다.
    """
    tree = ast.parse(SCRIPT_PATH.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = {
        name for name in imported
        if name == REPLAY_MODULE_PREFIX
        or name.startswith(REPLAY_MODULE_PREFIX + '.')
    }
    assert offenders == set(), offenders


def test_the_ledger_rows_are_well_formed():
    """수락 기준 1·2 — 모든 행이 `ORG` 에서 출발하고 판정과 맞는다."""
    ledger = _ledger()
    for entry in ledger['move']:
        assert entry['verdict'] == 'LOC', entry
        assert entry['before'] == migration.LABEL_BEFORE, entry
        assert entry['after'] == 'LOC', entry
    for entry in ledger['remove']:
        assert entry['verdict'] == 'DROP', entry
        assert entry['before'] == migration.LABEL_BEFORE, entry
        assert 'after' not in entry, entry
    for entry in ledger['move'] + ledger['remove']:
        # 역적용이 되꽂을 자리 — 없으면 전 지문을 못 만든다.
        for field in ('entity_index', 'pii_start_char', 'pii_end_char'):
            assert isinstance(entry[field], int), entry


def test_the_applied_and_skipped_counts_are_pinned():
    """수락 기준 2·4 — 손댄 수와 **안 손댄 수**를 함께 박는다.

    안 손댄 쪽을 안 박으면 모호 판정을 조용히 없애고 전부 옮겨도 아무 검사가
    안 걸린다 — 후 지문은 그 결과를 그대로 봉인하기 때문이다.
    """
    ledger = _ledger()
    assert len(ledger['move']) == GOLDEN_MOVED
    assert len(ledger['remove']) == GOLDEN_REMOVED
    assert len(ledger['not_applied']['ambiguous']) == GOLDEN_AMBIGUOUS
    assert len(ledger['not_applied']['absent_from_pii']) == GOLDEN_ABSENT
    assert ledger['verdict_counts'] == GOLDEN_TABLE_VERDICTS
    assert sum(GOLDEN_TABLE_VERDICTS.values()) == (
        GOLDEN_MOVED + GOLDEN_REMOVED + GOLDEN_AMBIGUOUS + GOLDEN_ABSENT
    )


def test_every_touched_span_is_listed_once():
    """한 span 이 두 칸에 동시에 오르지 않는다.

    겹치면 역적용이 같은 자리를 두 번 손대 전 지문이 안 나오고, 개수 등식도
    양쪽에서 세어 부풀어 오른다.
    """
    keys = [_span_key(entry) for entry in _entries(_ledger())]
    assert len(keys) == len(set(keys))


# ── 원천이 있어야 도는 검사 ──────────────────────────────────────────────

raw_only = pytest.mark.skipif(
    not (RAW_DIR / 'label.json').exists(),
    reason=f'OntoNotes5 source not present at {RAW_DIR}',
)


@raw_only
def test_the_ledger_population_comes_from_the_verdict_table():
    """수락 기준 2 — 원장이 다루는 자리가 판정 표에서 나온다.

    모집단을 손으로 고르게 두면 불리한 자리를 조용히 빼는 길이 열린다. 표를
    고치면 이 검사가 먼저 붉어져 원장도 함께 고치게 된다.
    """
    targets, _ = migration.derive_targets(RAW_DIR)
    derived = {
        (t.split, t.id, t.start_char, t.end_char, t.verdict) for t in targets
    }
    listed = {_span_key(entry) for entry in _entries(_ledger())}
    assert listed == derived, listed ^ derived


@raw_only
def test_the_conversion_output_is_reproducible_from_raw():
    """수락 기준 3 — `{split}.jsonl` 이 현행 코드의 산물 그대로다.

    원장이 이 파일들을 손대지 않는 대신 지문만 남기는 근거가 이것이다 —
    원천 + 코드에서 결정적으로 다시 만들어지므로 원장 없이도 반증된다.
    낡아 있으면 주입 쪽만 옮겨도 재생 대조가 어긋난다.
    """
    ledger = _ledger()
    id2label = load_id2label(RAW_DIR / 'label.json')
    for split in SPLIT_FILES:
        records, problems, _, _ = convert_one_split(RAW_DIR, split, id2label)
        assert problems == [], problems[:5]
        digest = hashlib.sha256(
            migration.dump_rows(records).encode('utf-8')
        ).hexdigest()
        assert digest == ledger['conversion_sha256'][split], split
        assert digest == hashlib.sha256(
            (DATA_DIR / f'{split}.jsonl').read_bytes()
        ).hexdigest(), split


# ── 주입 산출물이 있어야 도는 검사 ──────────────────────────────────────

corpus_only = pytest.mark.skipif(
    not all((PII_DIR / f'{s}.jsonl').exists() for s in migration.SPLITS)
    or not (DATA_DIR / 'origin.jsonl').exists(),
    reason=f'EN PII corpus not present at {PII_DIR}',
)


@corpus_only
def test_the_corpus_matches_the_after_fingerprints():
    """수락 기준 3 — 마이그레이션 뒤 조용한 편집을 거부한다."""
    ledger = _ledger()
    for name, path in _corpus_files(ledger).items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == ledger['after_sha256'][name], (
            f'{name} 가 원장이 기록한 판본과 다르다. 코퍼스를 바꿨다면 원장을'
            f' 다시 떠서 그 변경을 커밋에 남겨라.'
            f' ledger={ledger["after_sha256"][name][:12]} actual={digest[:12]}'
        )


@corpus_only
def test_reverse_applying_the_ledger_reproduces_the_before_fingerprints():
    """수락 기준 1·3 — 이 파일의 핵심 검사.

    되돌린 결과가 **바이트 단위로** 마이그레이션 전 파일이어야 한다. 바이트를
    요구하는 것이 요점이다 — 라벨만 맞추면 offset 을 다시 계산해 넣어도
    지나가는데, 그 경로가 바로 수락 기준 1 이 막는 재생 경로다.
    """
    ledger = _ledger()
    for name, path in _corpus_files(ledger).items():
        digest = hashlib.sha256(_restore(path, ledger)).hexdigest()
        assert digest == ledger['before_sha256'][name], (
            f'{name}: 역적용이 마이그레이션 전 파일을 재현하지 못했다 —'
            f' 원장에 없는 변경이 섞였거나 있는 변경이 빠졌다.'
            f' ledger={ledger["before_sha256"][name][:12]} actual={digest[:12]}'
        )


@corpus_only
def test_the_listed_spans_are_actually_in_the_corpus():
    """원장이 가리키는 자리에 정말 그 엔티티가 있다.

    지문 검사는 "전체가 그 판본이다" 만 말하고 어느 자리가 왜 그런지는 안
    말한다. 여기서 행 안 자리까지 내려가 대조한다.
    """
    ledger = _ledger()
    for split in migration.SPLITS:
        rows = {r['id']: r
                for r in migration.read_rows(PII_DIR / f'{split}.jsonl')}
        for entry in ledger['move']:
            if entry['split'] != split:
                continue
            entity = rows[entry['id']]['entities'][entry['entity_index']]
            assert entity['label'] == entry['after'], entry
            assert entity['text'] == entry['text'], entry
            assert entity['start_char'] == entry['pii_start_char'], entry
        for entry in ledger['not_applied']['absent_from_pii']:
            if entry['split'] != split:
                continue
            row = rows.get(entry['id'])
            hits = [] if row is None else [
                e for e in row['entities']
                if e['label'] == migration.LABEL_BEFORE
                and e['text'] == entry['text']
            ]
            assert hits == [], entry
