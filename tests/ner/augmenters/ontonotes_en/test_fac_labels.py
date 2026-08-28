"""EN `FAC` 판정 표를 검사한다 — 표가 정본이고 어휘는 검사자다.

원본 `FAC` 한 타입에 개별 구조물(`the Golden Gate Bridge`)과 경로
(`East Third Ring Road`)가 함께 들어 있어, canonical §3 인프라 규칙대로
가르려면 span 별 판별이 필요하다. 판별은 규칙이 아니라 **표면 전량을
열거한 표**로 한다 — 규칙을 쓰면 규칙이 못 가른 몫이 조용히 기본값으로
흐르고, 실측으로 어휘 규칙은 634 표면 중 391 만 가른다.

**어휘는 판정자가 아니라 검사자다.** 이 파일이 어휘 목록을 갖는 것은
표를 되짚기 위해서이고, 표가 gold 를 정한다. 어휘를 고쳐도 gold 는 안
움직인다 — 어긋나면 이 파일이 붉어질 뿐이다.

코퍼스 없이도 도는 검사와 코퍼스가 있어야 도는 검사를 가른다. `data/` 는
gitignore 라 클론 직후에는 원본이 없고, 그때도 표가 반증되어야 한다.
"""
import json
import re
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.convert import decode_bio, load_id2label

_ROOT = Path(__file__).resolve().parents[4]
TABLE_PATH = (
    _ROOT / 'src' / 'ner' / 'augmenters' / 'ontonotes_en'
    / 'data' / 'fac_labels.json'
)
RAW_DIR = _ROOT / 'data' / 'ontonotes_en' / 'raw'
SPLIT_FILES = {
    'train': ['train00.json', 'train01.json', 'train02.json', 'train03.json'],
    'valid': ['valid.json'],
    'test': ['test.json'],
}

# 원본 태그에서 직접 센 값 — 매핑·표와 독립이라 표가 바뀌어도 안 움직인다.
GOLDEN_SOURCE_FAC = {'train': 860, 'valid': 115, 'test': 135}

VERDICTS = frozenset({'ORG', 'LOC', 'DROP'})
REASONS = frozenset({
    'route:lead', 'route:head', 'route:number', 'route:world',
    'structure:head', 'structure:world',
    'noise:source', 'nonentity:other',
})

# ── 어휘 목록 (검사자) ────────────────────────────────────────────────
ROUTE_HEADS = frozenset({
    'road', 'roads', 'street', 'streets', 'avenue', 'avenues', 'boulevard',
    'boulevards', 'lane', 'lanes', 'drive', 'parkway', 'parkways', 'highway',
    'highways', 'expressway', 'expressways', 'freeway', 'freeways',
    'turnpike', 'motorway', 'causeway', 'beltway', 'route', 'routes',
    'railway', 'railways', 'railroad', 'railroads', 'line', 'lines', 'trail',
    'trails', 'path', 'paths', 'canal', 'canals', 'waterway', 'waterways',
    'corridor', 'corridors',
})
ROUTE_LEADS = frozenset({
    'rue', 'rues', 'route', 'highway', 'interstate', 'autoroute',
})
STRUCT_HEADS = frozenset({
    'bridge', 'bridges', 'airport', 'airports', 'station', 'stations',
    'park', 'hotel', 'building', 'buildings', 'tower', 'towers', 'plant',
    'hospital', 'museum', 'stadium', 'stadiums', 'center', 'centre', 'hall',
    'temple', 'church', 'cathedral', 'harbor', 'harbour', 'port', 'tunnel',
    'dam', 'plaza', 'square', 'mosque', 'palace', 'house', 'houses',
    'terminal', 'base', 'prison', 'camp', 'school', 'university', 'college',
    'library', 'theater', 'theatre', 'arena', 'mall', 'complex', 'field',
    'pier', 'wharf', 'embassy', 'tomb', 'gate', 'wall', 'monument',
    'memorial', 'zoo', 'farm', 'ranch', 'mine', 'refinery', 'reactor',
    'factory', 'arch', 'shrine', 'villa', 'castle', 'fort', 'gallery',
    'observatory', 'cemetery', 'reservoir', 'garden', 'gardens',
    'guesthouse', 'restaurant', 'bowl', 'dome', 'resort', 'landing',
})
ARTICLES = frozenset({'the', 'a', 'an', 'la', 'le', 'les', 'this', 'that'})
NUMBERED_ROUTE = re.compile(
    r'^(?:the\s+)?(?:U\.S\.|US|I|SR|Route|Highway|Interstate)?\s*\d+\s*$',
    re.IGNORECASE,
)

# 표면 형태가 같아 어떤 표면 술어도 못 가르는 자리 — 잔여여야 한다.
# `the Beijing - Kowloon`(철도 노선)과 `Vermont - Slauson`(쇼핑센터)이
# 같은 모양이므로, 한쪽에 술어 있는 코드가 붙었다면 그 술어는 다른 쪽을
# 틀리게 판정한다.
RESIDUAL_FLOOR = (
    'Datong - Hongdong', 'Lugou Bridge - Handan', 'Tianjin - Dezhou',
    'Wanhua - Panchiao', 'Vermont - Slauson', 'the Anhui - Jiangxi',
    'the Beijing - Kowloon', 'the Wuhan - Jiujiang', 'the Yingtan - Xiamen',
    'the Zhejiang - Jiangxi', 'Chengyu', 'Chuanqing', 'Xiangyu',
    'Metrorail', 'Underground', 'MRT',
)

# 이슈가 소유하는 표본 — 구현자가 고르지 않는다.
PROBE = {
    'U.S. 460': 'LOC', '288': 'LOC', '101': 'LOC',
    'the Beijing - Kowloon': 'LOC',
    'Vermont - Slauson': 'ORG', 'Wall Street': 'LOC',
}

GOLDEN_LEXICAL_EXCEPTIONS = 2
GOLDEN_CONFLICTS = 32


def words(surface: str) -> list[str]:
    """표면에서 낱말만 뽑는다 — 따옴표·숫자·구두점은 낱말이 아니다."""
    return re.findall(r"[A-Za-z][A-Za-z\-']*", surface)


def lexical(surface: str) -> str | None:
    """어휘·형태로 갈리는 판정. 못 가르면 `None`.

    이것이 gold 를 정하지 않는다 — 표가 정한다. 여기서 나온 값은 표를
    되짚는 데만 쓰이고, 어긋나면 표의 예외 목록에 사유가 있어야 한다.
    """
    lowered = [w.lower() for w in words(surface)]
    if not lowered:
        return None
    lead = (
        lowered[1] if (lowered[0] in ARTICLES and len(lowered) > 1)
        else lowered[0]
    )
    if lead in ROUTE_LEADS:
        return 'LOC'
    if lowered[-1] in ROUTE_HEADS:
        return 'LOC'
    if lowered[-1] in STRUCT_HEADS:
        return 'ORG'
    return None


@pytest.fixture(scope='module')
def table():
    return json.loads(TABLE_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def by_surface(table):
    return {e['surface']: e for e in table['entries']}


# ── 코퍼스 없이 도는 검사 ─────────────────────────────────────────────

def test_table_declares_its_criterion_and_population(table):
    """표가 자기 판정 기준과 모집단을 문장으로 갖는다."""
    for key in ('canonical_section', 'criterion', 'population', 'decision'):
        assert table[key].strip(), f'{key} is empty'


def test_every_entry_is_well_formed(table):
    """모든 항목이 판정·사유·합의·등장수를 갖고 값역 안에 있다."""
    for e in table['entries']:
        assert e['verdict'] in VERDICTS, e
        assert e['reason'] in REASONS, e
        assert e['agreement'] in ('high', 'conflict'), e
        assert e['decided_by'] in ('models', 'human'), e
        assert set(e['occurrences']) == {'train', 'valid', 'test'}, e


def test_surfaces_are_unique(table):
    """한 표면에 답은 하나다 — 표가 이름당 한 줄만 갖는다."""
    surfaces = [e['surface'] for e in table['entries']]
    assert len(surfaces) == len(set(surfaces))


def test_inventory_sums_to_the_pinned_source_counts(table):
    """등장 수 합이 원본 태그에서 직접 센 값과 맞는다.

    `data/` 가 없는 환경에서 표를 반증하는 유일한 자리다. 표에 이름을
    더하거나 빼면 이 합이 어긋난다.
    """
    for split, want in GOLDEN_SOURCE_FAC.items():
        got = sum(e['occurrences'][split] for e in table['entries'])
        assert got == want, f'{split}: {got} != {want}'


def test_source_count_equals_the_three_verdicts(table):
    """`원본 FAC 수 = ORG + LOC + DROP` 이 split 별로 성립한다.

    `LOC`·`ORG` 개별 수는 표가 정하는 값이라 골든으로 박지 않는다 —
    박으면 표를 고칠 때마다 골든도 고치게 되어 골든이 표의 사본이 된다.
    이 등식은 표를 어떻게 고쳐도 성립해야 하고, 깨지면 span 이 새거나
    겹친 것이다.
    """
    for split, want in GOLDEN_SOURCE_FAC.items():
        parts = {
            v: sum(
                e['occurrences'][split]
                for e in table['entries'] if e['verdict'] == v
            )
            for v in VERDICTS
        }
        assert sum(parts.values()) == want, f'{split}: {parts} != {want}'


def test_probe_surfaces_hold(by_surface):
    """이슈가 소유한 표본 여섯이 표에 그대로 있다."""
    for surface, want in PROBE.items():
        assert by_surface[surface]['verdict'] == want, surface


def test_lexical_reasons_satisfy_their_own_predicate(table):
    """술어가 있는 사유코드는 그 술어를 실제로 만족한다.

    사유코드를 아무 데나 못 붙이게 하는 자리다. 이것이 없으면 잔여로
    가야 할 표면이 술어 있는 코드로 세탁되고, 잔여 수가 낮게 나와
    "다 기계로 뒷받침됐다" 는 틀린 신호를 준다.
    """
    for e in table['entries']:
        surface, reason = e['surface'], e['reason']
        lowered = [w.lower() for w in words(surface)]
        if reason == 'route:lead':
            lead = (
                lowered[1] if (lowered[0] in ARTICLES and len(lowered) > 1)
                else lowered[0]
            )
            assert lead in ROUTE_LEADS, surface
        elif reason == 'route:head':
            assert lowered[-1] in ROUTE_HEADS, surface
        elif reason == 'structure:head':
            assert lowered[-1] in STRUCT_HEADS, surface
        elif reason == 'route:number':
            assert NUMBERED_ROUTE.match(surface.strip()), surface


def test_residual_codes_are_not_lexically_decidable(table, by_surface):
    """잔여 코드는 어휘가 못 가른 자리이거나 명시 예외다."""
    exceptions = {x['surface'] for x in table['lexical_exceptions']}
    for e in table['entries']:
        if not e['reason'].endswith(':world'):
            continue
        if e['surface'] in exceptions:
            continue
        assert lexical(e['surface']) is None, (
            f"{e['surface']!r} is lexically decidable but coded as residual"
        )


def test_residual_floor_is_held(by_surface):
    """표면 형태가 같아 못 가르는 자리가 전부 잔여로 남아 있다."""
    for surface in RESIDUAL_FLOOR:
        reason = by_surface[surface]['reason']
        assert reason.endswith(':world'), f'{surface}: {reason}'


def test_lexical_agreement_outside_the_exception_list(table):
    """어휘가 가르는 표면은 표의 판정과 일치하고, 예외만 어긋난다.

    어휘를 판정자가 아니라 **검사자**로 쓰는 자리다. 예외 수를 못 박아
    조용히 늘리지 못하게 한다.
    """
    exceptions = {
        x['surface']: x for x in table['lexical_exceptions']
    }
    assert len(exceptions) == GOLDEN_LEXICAL_EXCEPTIONS
    for x in exceptions.values():
        assert x['why'].strip(), x
        assert lexical(x['surface']) == x['lexical'], x
        assert x['lexical'] != x['verdict'], x

    for e in table['entries']:
        if e['verdict'] == 'DROP' or e['surface'] in exceptions:
            continue
        want = lexical(e['surface'])
        if want is not None:
            assert want == e['verdict'], (
                f"{e['surface']!r}: table={e['verdict']} lexical={want}"
            )


def test_conflicts_were_settled_by_a_human(table):
    """두 모델이 갈린 항목은 전부 사람이 판정했고 그 수가 박혀 있다."""
    conflicts = [
        e for e in table['entries'] if e['agreement'] == 'conflict'
    ]
    assert len(conflicts) == GOLDEN_CONFLICTS
    for e in conflicts:
        assert e['decided_by'] == 'human', e['surface']


def test_dropped_entries_carry_a_split_reason(table):
    """`DROP` 은 원본 오류와 다른 타입을 갈라 적는다.

    이 파이프라인의 위험은 "있는 것을 버리다 잃는" 쪽이라(§4.3),
    무엇을 왜 버렸는지가 표에 남아야 한다.
    """
    for e in table['entries']:
        if e['verdict'] == 'DROP':
            assert e['reason'] in ('noise:source', 'nonentity:other'), e
            if e['reason'] == 'nonentity:other':
                assert e.get('note', '').strip(), e['surface']
        else:
            assert e['reason'] not in ('noise:source', 'nonentity:other'), e


def test_separate_entity_surfaces_are_listed_with_their_verdict(
    table, by_surface,
):
    """같은 이름의 별개 실체가 실재하는 자리가 열거돼 있다.

    §3 환유 항은 실체가 하나면 실체 판정을 유지하고, 별개 실체가
    실재하면 §2.1 로 보낸다. 표는 이름당 답이 하나라 그런 자리는 사람이
    판정해 여기 남긴다.
    """
    listed = table['separate_entities']
    assert listed, 'separate_entities must not be silently empty'
    for x in listed:
        assert x['why'].strip(), x
        assert by_surface[x['surface']]['verdict'] == x['verdict'], x


# ── 코퍼스가 있어야 도는 검사 ─────────────────────────────────────────

@pytest.fixture(scope='module')
def source_surfaces():
    if not RAW_DIR.exists():
        pytest.skip(f'source corpus not present at {RAW_DIR}')
    id2label = load_id2label(RAW_DIR / 'label.json')
    found: dict[str, dict[str, int]] = {}
    for split, files in SPLIT_FILES.items():
        for name in files:
            with (RAW_DIR / name).open(encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens = row['tokens']
                    for start, end, src in decode_bio(row['tags'], id2label):
                        if src != 'FAC':
                            continue
                        surface = ' '.join(tokens[start:end + 1])
                        bucket = found.setdefault(
                            surface, {'train': 0, 'valid': 0, 'test': 0},
                        )
                        bucket[split] += 1
    return found


def test_table_covers_the_corpus_both_ways(table, source_surfaces):
    """표와 원본이 양쪽 방향으로 맞는다.

    한쪽만 보면 죽은 줄을 놓친다 — 표에만 있는 이름은 원본 스냅샷이
    바뀌었거나 표면 생성 규칙이 갈렸다는 신호다.
    """
    in_table = {e['surface'] for e in table['entries']}
    in_source = set(source_surfaces)
    assert not (in_source - in_table), sorted(in_source - in_table)[:20]
    assert not (in_table - in_source), sorted(in_table - in_source)[:20]


def test_inventory_matches_the_corpus(table, source_surfaces):
    """표에 적힌 등장 수가 원본에서 다시 센 값과 같다."""
    for e in table['entries']:
        assert e['occurrences'] == source_surfaces[e['surface']], e['surface']
