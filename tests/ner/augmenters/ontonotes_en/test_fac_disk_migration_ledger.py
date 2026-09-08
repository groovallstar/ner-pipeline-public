"""EN `FAC` 디스크 마이그레이션 원장 — 손댄 span 을 지문에 묶는다.

`data/**` 는 gitignore 이고 주입 텍스트는 LLM 산출이라 다시 만들 수 없다.
그래서 "어떤 span 이 왜 손대졌는가" 가 저장소에 남는 자리는 이 원장뿐이고,
원장이 스스로 어긋나지 않는지를 여기서 본다.

**원천·주입 산출물을 읽던 검사는 내렸다.** `data/ontonotes_en/` 은 다른 언어
폴더와 같은 구성으로 정리돼 `raw/`·`pii/`·변환 직후 split 이 디스크에서
내려갔다. 그 파일을 읽던 검사를 skip 으로 남겨 두면 통과 수만 세어지고 무엇이
실제로 반증되는지가 흐려지므로 함께 지웠다 — 되살리려면 원천을 받아
`ner.augmenters.ontonotes_en` 을 다시 돌리는 것이 전제라, 그때 검사도 함께
되살리는 편이 맞다.

남은 것은 둘이다. 원장 자체의 형태·개수 등식은 데이터 없이 돌고, 병합 산출물
(`origin.jsonl`)의 후 지문은 gitignore 된 코퍼스를 나중에 조용히 편집하는 것을
막는다 — 원장은 그대로인데 실물만 달라지면 그 어긋남이 여기서 붉어진다.
"""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from ner.scripts import migrate_en_fac_disk_corpus as migration

_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _ROOT / 'data' / 'ontonotes_en'
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


# ── 데이터 없이 도는 검사 ────────────────────────────────────────────────

def test_the_migration_never_imports_the_replay_path():
    """수락 기준 1 — 스크립트가 주입 모듈을 import 하지 않는다.

    주입 산출물의 NER 엔티티를 `extract_spans`+`merge_entities` 로 다시
    도출하면 마이그레이션이 주입을 재생한 셈이 돼, 산출물과의 어떤 대조도
    같은 코드가 만든 값끼리 맞춰 보는 항진명제가 된다. 판정 표를 읽어
    옮기는 경로와 주입 경로는 끝까지 갈라져 있어야 한다.

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


# ── 병합 산출물이 있어야 도는 검사 ──────────────────────────────────────

@pytest.mark.skipif(
    not (DATA_DIR / 'origin.jsonl').exists(),
    reason=f'EN merged corpus not present at {DATA_DIR}',
)
def test_the_merged_corpus_matches_the_after_fingerprint():
    """수락 기준 3 — 마이그레이션 뒤 조용한 편집을 거부한다.

    학습이 읽는 파일이 이것 하나라(`classifier` 의 `--data`) 여기만 봉인해도
    gold 가 바뀌면 붉어진다. 주입 split 의 지문은 그 파일들이 디스크에서
    내려가 대조할 상대가 없어졌다.
    """
    ledger = _ledger()
    name = ledger['merged_name']
    digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
    assert digest == ledger['after_sha256'][name], (
        f'{name} 가 원장이 기록한 판본과 다르다. 코퍼스를 바꿨다면 원장을'
        f' 다시 떠서 그 변경을 커밋에 남겨라.'
        f' ledger={ledger["after_sha256"][name][:12]} actual={digest[:12]}'
    )
