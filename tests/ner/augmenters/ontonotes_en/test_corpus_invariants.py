"""전량 불변식 — split·group-key·canonical 완료 조건과 `FAC` 분할의 앵커.

원천(`--raw-dir`)이 없으면 통째로 건너뛴다. `data/` 는 gitignore 되므로
클론 직후에는 데이터가 없고, 그때 실패시키면 무관한 변경까지 붉어진다.

**골든 넘버는 이 파일에 둔다.** 변환 모듈에서 가져오면 매핑을 고칠 때 기대치도
함께 움직여 검사가 자기참조가 된다. 매핑을 의도적으로 바꾸는 사람은 이 숫자를
손으로 고쳐야 하고, 그 마찰이 목적이다 — 자를 바꾸면 사람이 멈춰 선다.

**`LOC`·`ORG` 개별 수는 골든이 아니다.** 그 둘은 `FAC` 판정 표가 정하는 값이라
표를 고치면 함께 움직인다 — 골든에 박으면 표를 손볼 때마다 골든도 고치게 되어
골든이 표의 사본이 되고, 사본은 원본을 검사하지 못한다. 대신 **표에서 도출한
기대 수**와 실제 변환 결과를 대조한다. 표를 고치면 기대값도 따라 움직이므로
골든이 아니지만, 표면 생성·BIO 디코드·표 조회 경로가 틀어지면 어긋난다.

고정하는 것은 표를 어떻게 고쳐도 안 변하는 것뿐이다 — 원본 태그 수
(`GOLDEN_SOURCE_SPANS`)와 이 분할이 안 건드리는 canonical 타입 넷
(`GOLDEN_UNTOUCHED_SPANS`)이다.
"""
import json
from pathlib import Path

import pytest

from ner.augmenters.ontonotes_en.convert import decode_bio, load_id2label
from ner.augmenters.ontonotes_en.mapping import (
    FAC_VERDICTS,
    assert_fac_coverage,
)
from ner.augmenters.ontonotes_en.__main__ import SPLIT_FILES, convert_one_split

_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = _ROOT / 'data' / 'ontonotes_en' / 'raw'
TABLE_PATH = (
    _ROOT / 'src' / 'ner' / 'augmenters' / 'ontonotes_en'
    / 'data' / 'fac_labels.json'
)

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / 'label.json').exists(),
    reason=f'OntoNotes5 source not present at {RAW_DIR}',
)

# 이 분할이 건드리지 않는 canonical 타입 — 표를 어떻게 고쳐도 안 변한다.
# `LOC`·`ORG` 는 여기 없다(위 docstring).
GOLDEN_UNTOUCHED_SPANS = {
    'PER':  {'train': 15429, 'valid': 2020, 'test': 1988},
    'DAT':  {'train': 10922, 'valid': 1507, 'test': 1602},
    'PROD': {'train': 1580,  'valid': 214,  'test': 242},
    'EVT':  {'train': 748,   'valid': 143,  'test': 63},
}
EXPECTED_LABELS = frozenset({'PER', 'LOC', 'ORG', 'PROD', 'EVT', 'DAT'})

# 타입 통째로 버리는 원본 타입 — **리터럴로 적는다.** 매핑표에서 읽어 오면
# 매핑축이 등식 양변에 들어가, `DATE` 를 드롭으로 뒤집어도 기대치가 함께
# 줄어 통과한다.
SOURCE_TYPES_DROPPED_WHOLE = frozenset({
    'LAW', 'NORP', 'LANGUAGE', 'TIME', 'QUANTITY', 'MONEY',
    'PERCENT', 'ORDINAL', 'CARDINAL',
})

GOLDEN_SENTENCES = {'train': 59924, 'valid': 8528, 'test': 8262}

# 갈림 후보 두 그물의 규모. 원본 태그에서 다시 뽑아 대조한다.
#
# ① 원본이 `FAC` 와 **다른 타입으로도** 태그한 표면. 다른 타입은 매핑표가
#    따로 라벨을 내 판정 표가 못 건드리므로 최종 데이터에 섞임이 남는다 —
#    규칙의 불일치가 아니라 원본 태그의 불일치라 규모만 박는다.
# ② `FAC` 로 2 회 이상 등장한 표면. 여기서 볼 것은 같은 이름이 별개 실체
#    둘을 가리키는가이고, 해당하는 자리는 표가 답을 하나만 담으므로 사람이
#    개별로 정해 표의 `separate_entities` 에 남는다.
#
# 교집합은 ① 우선이라 ②' 는 ① 을 뺀 나머지다.
GOLDEN_MIXED_TYPE_SURFACES = 56
GOLDEN_MIXED_TYPE_FAC_SPANS = 239
GOLDEN_REPEATED_SURFACES = 162
GOLDEN_REPEATED_FAC_SPANS = 638
GOLDEN_REPEATED_ONLY_SURFACES = 133
GOLDEN_REPEATED_ONLY_FAC_SPANS = 426

# 같은 이름의 별개 실체가 실재해 사람이 개별로 정한 자리의 수. 비어 있어도
# "비어 있음" 을 세어 대조한다 — 조용히 줄여 사람 검수를 건너뛰지 못하게 한다.
#
# **그물이 후보를 뽑을 뿐 답을 가두지는 않는다.** 셋 중 하나
# (`` the West Wing ''` — 백악관 서관과 같은 이름의 TV 프로그램)는 `FAC` 로
# 한 번만 등장하고 다른 타입 태그도 없어 두 그물 어디에도 안 걸린다. 그것이
# 판정을 그물 안에서만 돌리지 않고 **634 표면 전량**에 돌린 이유다 — 그물만
# 봤으면 이 자리는 안 보였다. 두 수를 함께 박아 그 비율이 조용히 뒤집히지
# 않게 한다.
GOLDEN_SEPARATE_ENTITIES = 3
GOLDEN_SEPARATE_ENTITIES_IN_NETS = 2

# 원본 타입별 span 수 — 매핑표를 거치지 않고 태그에서 직접 센 값이다.
# canonical 쪽 골든과 **함께** 고정하는 이유는 실패를 단계별로 가르기
# 위해서다: 원본 수만 움직이면 BIO 디코드가, canonical 수만 움직이면
# 매핑이 원인이다. 한쪽만 두면 어느 단계가 깨졌는지 모른다.
GOLDEN_SOURCE_SPANS = {
    'CARDINAL': {'train': 7367, 'valid': 938, 'test': 935},
    'DATE': {'train': 10922, 'valid': 1507, 'test': 1602},
    'EVENT': {'train': 748, 'valid': 143, 'test': 63},
    'FAC': {'train': 860, 'valid': 115, 'test': 135},
    'GPE': {'train': 15405, 'valid': 2268, 'test': 2240},
    'LANGUAGE': {'train': 304, 'valid': 33, 'test': 22},
    'LAW': {'train': 282, 'valid': 40, 'test': 40},
    'LOC': {'train': 1514, 'valid': 204, 'test': 179},
    'MONEY': {'train': 2434, 'valid': 274, 'test': 314},
    'NORP': {'train': 6870, 'valid': 847, 'test': 841},
    'ORDINAL': {'train': 1640, 'valid': 232, 'test': 195},
    'ORG': {'train': 12820, 'valid': 1740, 'test': 1795},
    'PERCENT': {'train': 1763, 'valid': 177, 'test': 349},
    'PERSON': {'train': 15429, 'valid': 2020, 'test': 1988},
    'PRODUCT': {'train': 606, 'valid': 72, 'test': 76},
    'QUANTITY': {'train': 657, 'valid': 100, 'test': 105},
    'TIME': {'train': 1233, 'valid': 214, 'test': 212},
    'WORK_OF_ART': {'train': 974, 'valid': 142, 'test': 166},
}
GOLDEN_TOTAL_SOURCE_SPANS = 104151

# 공식 split 이 물려준 중복 문장. 제거하지 않는 대신 규모를 못 박는다 —
# 조용히 커지면 test 점수가 그만큼 부푼다.
GOLDEN_UNIQUE_TEXTS = {'train': 55154, 'valid': 7996, 'test': 7782}
GOLDEN_ROWS_ALSO_IN_TRAIN = {'valid': 659, 'test': 625}
GOLDEN_SPANS_ALSO_IN_TRAIN = {'valid': 79, 'test': 67}


@pytest.fixture(scope='module')
def converted():
    """원천 전량을 split 별로 변환하고 검사 결과를 함께 모은다."""
    id2label = load_id2label(RAW_DIR / 'label.json')
    by_split: dict[str, list[dict]] = {}
    problems: list[str] = []
    spans: dict[str, dict[str, int]] = {}
    source_spans: dict[str, dict[str, int]] = {}
    mismatched_slices = 0

    fac_surfaces: dict[str, int] = {}

    for split in SPLIT_FILES:
        records, split_problems, split_source, split_surfaces = (
            convert_one_split(RAW_DIR, split, id2label)
        )
        source_spans[split] = dict(split_source)
        for surface, count in split_surfaces.items():
            fac_surfaces[surface] = fac_surfaces.get(surface, 0) + count
        problems += split_problems
        by_split[split] = records
        for record in records:
            for entity in record['entities']:
                bucket = spans.setdefault(entity['label'], {})
                bucket[split] = bucket.get(split, 0) + 1
                sliced = record['text'][
                    entity['start_char']:entity['end_char']
                ]
                if sliced != entity['text']:
                    mismatched_slices += 1

    return {
        'by_split': by_split,
        'records': [r for rs in by_split.values() for r in rs],
        'problems': problems,
        'sentences': {s: len(r) for s, r in by_split.items()},
        'spans': spans,
        'source_spans': source_spans,
        'fac_surfaces': fac_surfaces,
        'mismatched_slices': mismatched_slices,
    }


@pytest.fixture(scope='module')
def source_tags():
    """원본 태그에서 표면별 타입을 다시 뽑는다 — 갈림 후보 두 그물의 재료.

    변환 산출물이 아니라 **원본 태그**를 읽는 것이 요점이다. 산출물에서 뽑으면
    판정 표가 이미 답을 준 뒤라 "표가 못 건드리는 자리" 가 안 보인다.
    """
    id2label = load_id2label(RAW_DIR / 'label.json')
    fac: dict[str, int] = {}
    other_types: dict[str, set] = {}
    for split, filenames in SPLIT_FILES.items():
        for filename in filenames:
            with (RAW_DIR / filename).open(encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens = row['tokens']
                    for start, end, src in decode_bio(row['tags'], id2label):
                        surface = ' '.join(tokens[start:end + 1])
                        if src == 'FAC':
                            fac[surface] = fac.get(surface, 0) + 1
                        else:
                            other_types.setdefault(surface, set()).add(src)
    return {'fac': fac, 'other_types': other_types}


@pytest.fixture(scope='module')
def verdict_table():
    return json.loads(TABLE_PATH.read_text(encoding='utf-8'))


def _table_totals(table) -> dict[str, dict[str, int]]:
    """판정별·split별 등장 수를 표에서 합산한다."""
    totals = {
        verdict: {split: 0 for split in GOLDEN_SENTENCES}
        for verdict in ('ORG', 'LOC', 'DROP')
    }
    for entry in table['entries']:
        for split in GOLDEN_SENTENCES:
            totals[entry['verdict']][split] += entry['occurrences'][split]
    return totals


def test_no_entity_token_mismatch(converted):
    """수락 기준 1 — 엔티티 표면이 원본 토큰과 (공백을 뺀 채) 같다."""
    problems = converted['problems']
    assert problems == [], f'{len(problems)} mismatches, first: {problems[:5]}'


def test_source_span_counts_match_golden(converted):
    """원본 타입별 span 수를 고정한다 — 매핑과 독립인 앵커.

    canonical 골든만 두면 BIO 디코드가 바뀌었는지 매핑이 바뀌었는지 갈리지
    않는다. 이 층은 매핑표를 거치지 않은 수라 디코드 단계만 본다.
    """
    actual = {
        source_type: {
            split: converted['source_spans'][split].get(source_type, 0)
            for split in GOLDEN_SENTENCES
        }
        for source_type in GOLDEN_SOURCE_SPANS
    }
    assert actual == GOLDEN_SOURCE_SPANS
    total = sum(
        n for counts in converted['source_spans'].values()
        for n in counts.values()
    )
    assert total == GOLDEN_TOTAL_SOURCE_SPANS


def test_slice_equality_is_tautological_not_a_guard(converted):
    """`text[start:end] == entity.text` 는 변환 산출물에서 항진명제다.

    `convert_record` 가 `entity['text']` 를 그 슬라이스로 대입하므로 늘 참이며,
    한때 이것을 수락 기준으로 세었던 것은 **철회**했다. 값이 0 인 것을 확인은
    하되 이 파일에서 보증하는 것으로 세지 않는다 — offset 을 실제로 검증하는
    것은 `test_no_entity_token_mismatch` 다(원본 토큰이 비교 상대).
    """
    assert converted['mismatched_slices'] == 0


def test_untouched_span_counts_match_golden(converted):
    """이 분할이 안 건드리는 타입 넷은 한 span 도 안 움직인다.

    판정 표를 어떻게 고쳐도 `PER`·`DAT`·`PROD`·`EVT` 는 그대로여야 한다 —
    움직였다면 표가 아니라 디코드나 매핑이 바뀐 것이다.
    """
    actual = {
        label: counts for label, counts in converted['spans'].items()
        if label in GOLDEN_UNTOUCHED_SPANS
    }
    assert actual == GOLDEN_UNTOUCHED_SPANS


def test_loc_and_org_match_the_counts_derived_from_the_table(
    converted, verdict_table,
):
    """`LOC`·`ORG` 가 **표에서 도출한 기대 수**와 같다.

    골든이 아니라 도출값이다 — 표를 고치면 기대값도 따라 움직인다. 그래도
    검사가 되는 것은 기대값이 표를 지나 오고 실제값은 표면 생성·BIO 디코드·
    표 조회를 지나 오기 때문이다. 그 경로가 틀어지면 둘이 어긋난다.

    표를 전량 `ORG` 로 무력화해도 여기서는 안 걸린다(양변이 함께 움직인다).
    그 자리를 막는 것은 `test_fac_labels.py` 의 표본 판정과 잔여 하한이다.
    """
    totals = _table_totals(verdict_table)
    expected = {
        'LOC': {
            split: (GOLDEN_SOURCE_SPANS['GPE'][split]
                    + GOLDEN_SOURCE_SPANS['LOC'][split]
                    + totals['LOC'][split])
            for split in GOLDEN_SENTENCES
        },
        'ORG': {
            split: GOLDEN_SOURCE_SPANS['ORG'][split] + totals['ORG'][split]
            for split in GOLDEN_SENTENCES
        },
    }
    actual = {
        label: converted['spans'][label] for label in ('LOC', 'ORG')
    }
    assert actual == expected


def test_source_fac_count_equals_the_three_verdicts(converted, verdict_table):
    """`원본 FAC 수 = ORG + LOC + 버림` 이 split 별로 성립한다.

    표를 어떻게 고쳐도 성립해야 하는 등식이다. 깨지면 span 이 새거나 겹친
    것이다 — 판정이 옳은지가 아니라 **하나도 안 잃었는지**를 본다.
    """
    totals = _table_totals(verdict_table)
    for split, want in GOLDEN_SOURCE_SPANS['FAC'].items():
        got = sum(totals[verdict][split] for verdict in totals)
        assert got == want, f'{split}: {got} != {want}'


def test_no_unexpected_labels_appear(converted):
    """드롭해야 할 타입이 canonical 라벨로 새어 들어오지 않는다."""
    assert set(converted['spans']) == EXPECTED_LABELS


def test_total_span_count(converted, verdict_table):
    """전량 합은 골든이 아니라 **원본 수에서 버린 몫을 뺀** 값이다.

    `DROP` 판정이 생기면 합이 줄므로 고정값을 못 쓴다. 대신 원본 골든에서
    버린 몫을 빼 기대치를 만든다 — 버린 몫은 타입 통째로 버리는 원본 타입
    (리터럴 목록)과 표가 버린 `FAC` 표면 둘로 갈린다.
    """
    dropped_whole = sum(
        n for source_type in SOURCE_TYPES_DROPPED_WHOLE
        for n in GOLDEN_SOURCE_SPANS[source_type].values()
    )
    dropped_by_table = sum(_table_totals(verdict_table)['DROP'].values())
    total = sum(
        n for split_counts in converted['spans'].values()
        for n in split_counts.values()
    )
    assert total == (
        GOLDEN_TOTAL_SOURCE_SPANS - dropped_whole - dropped_by_table
    )


def test_the_verdict_table_covers_the_corpus_both_ways(converted):
    """피복 검사를 전량 코퍼스에 실제로 태운다.

    변환 CLI 가 매 실행 거는 것과 같은 검사다. 여기서 한 번 더 태우는 것은
    CLI 를 안 돌리고 테스트만 도는 경로가 있기 때문이다.
    """
    assert_fac_coverage(converted['fac_surfaces'], list(GOLDEN_SENTENCES))
    assert len(converted['fac_surfaces']) == len(FAC_VERDICTS)


def test_mixed_type_candidate_net_is_pinned(source_tags):
    """그물 ① — 원본이 `FAC` 와 다른 타입으로도 태그한 표면의 규모.

    다른 타입은 매핑표가 따로 라벨을 내므로 판정 표가 못 건드린다. 최종
    데이터에 섞임이 남는 것은 규칙의 불일치가 아니라 **원본 태그의 불일치**라
    없애지 않고 규모만 박는다. 새 표면이 생기면 이 수가 움직인다.
    """
    net = {
        surface for surface in source_tags['fac']
        if surface in source_tags['other_types']
    }
    assert len(net) == GOLDEN_MIXED_TYPE_SURFACES
    spans = sum(source_tags['fac'][surface] for surface in net)
    assert spans == GOLDEN_MIXED_TYPE_FAC_SPANS


def test_repeated_surface_candidate_net_is_pinned(source_tags):
    """그물 ② — `FAC` 로 2 회 이상 등장한 표면의 규모.

    같은 이름이 별개 실체 둘을 가리키는지 사람이 보는 자리다. 교집합은
    ① 우선이므로 ① 을 뺀 나머지도 함께 박는다 — 둘 중 하나만 박으면 표면이
    그물 사이를 옮겨 다녀도 합이 맞아 안 걸린다.
    """
    repeated = {
        surface for surface, count in source_tags['fac'].items() if count >= 2
    }
    assert len(repeated) == GOLDEN_REPEATED_SURFACES
    assert sum(
        source_tags['fac'][s] for s in repeated
    ) == GOLDEN_REPEATED_FAC_SPANS

    only = repeated - set(source_tags['other_types'])
    assert len(only) == GOLDEN_REPEATED_ONLY_SURFACES
    assert sum(
        source_tags['fac'][s] for s in only
    ) == GOLDEN_REPEATED_ONLY_FAC_SPANS


def test_separate_entity_surfaces_are_listed_and_real(
    source_tags, verdict_table,
):
    """사람이 "별개 실체 둘" 로 정한 자리가 열거돼 있고 코퍼스에 실재한다.

    수를 박아 조용히 줄이지 못하게 한다 — 비어 있어도 "비어 있음" 을 세어
    대조하는 자리다. 그물 안에서 몇 개가 나왔는지도 함께 박는다: 그물은
    후보를 뽑을 뿐 답을 가두지 않으므로(`GOLDEN_SEPARATE_ENTITIES_IN_NETS`
    주석), 그 비율이 뒤집히면 판정을 전량이 아니라 그물 안에서만 돌렸다는
    신호다.
    """
    listed = verdict_table['separate_entities']
    assert len(listed) == GOLDEN_SEPARATE_ENTITIES
    candidates = {
        surface for surface in source_tags['fac']
        if surface in source_tags['other_types']
        or source_tags['fac'][surface] >= 2
    }
    in_nets = 0
    for entry in listed:
        # 지어낸 이름이 아니라 원본이 `FAC` 로 태그한 표면이어야 한다.
        assert entry['surface'] in source_tags['fac'], entry['surface']
        in_nets += entry['surface'] in candidates
    assert in_nets == GOLDEN_SEPARATE_ENTITIES_IN_NETS


def test_sentence_counts_match_original_split(converted):
    """원본 split 보존 — 저장소 관례인 80/10/10 재분할을 하지 않는다."""
    assert converted['sentences'] == GOLDEN_SENTENCES


def test_every_record_declares_its_split(converted):
    for record in converted['records']:
        assert record['split'] in GOLDEN_SENTENCES


def test_output_passes_canonical_label_validation(converted, tmp_path):
    """canonical 10 종 밖 라벨이 없다 — load_jsonl 이 위반 시 ValueError."""
    from ner.classifier.data_utils import load_jsonl

    path = tmp_path / 'origin.jsonl'
    with path.open('w', encoding='utf-8') as fh:
        for record in converted['records']:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    assert len(load_jsonl(str(path))) == sum(GOLDEN_SENTENCES.values())


def test_group_key_orig_is_accepted_per_split(converted):
    """split 파일 안에서 `orig` 보다 강한 후보가 없다.

    split 을 필드로 섞어 한 파일에 두면 `split`(3 값)이 후보로 잡혀 올바른
    형제 키를 밀어낸다. 파일을 가르면 파일 안에서 상수(그룹 1)라 빠진다.
    """
    from ner.classifier.data_utils import validate_group_key

    for records in converted['by_split'].values():
        validate_group_key(records, 'orig')


PII_DIR = RAW_DIR.parent / 'pii'


PII_SPLITS = ('train', 'valid', 'test')


def _read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')
            if line.strip()]


@pytest.mark.skipif(
    not all((PII_DIR / f'{s}.jsonl').exists() for s in PII_SPLITS),
    reason=f'PII-injected output incomplete at {PII_DIR} '
           f'(needs all of {PII_SPLITS})',
)
def test_injected_output_keeps_the_group_key():
    """주입 산출물에 `orig` 가 살아 있고 그룹 보호가 실제로 작동한다.

    `augmenters.pii` 의 Record 가 이 필드를 버리므로 복원 단계
    (`restore_groups`)를 지나야 한다. 안 지나면 후속 학습에서 `--group-key`
    를 줄 수단이 없고, 같은 원문에서 나온 행이 train 과 test 로 갈린다.

    **가드가 세 파일 전부를 요구하는 이유**는 주입이 split 을 하나씩 뱉기
    때문이다. 한 파일만 보고 켜면 나머지가 없어 `FileNotFoundError` 로
    붉어지고, 반대로 있는 것만 골라 검사하면 주입이 도는 중인지 끝났는지가
    구별되지 않는다. 세 파일이 다 나온 뒤에만 돌고, 그때 `orig` 가 없으면
    `restore_groups` 를 안 지났다는 뜻이라 **skip 이 아니라 실패**다 —
    복원 단계 누락을 잡는 자동 검사가 이것뿐이다.
    """
    from ner.classifier.data_utils import load_jsonl, validate_group_key

    for split in PII_SPLITS:
        path = PII_DIR / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        assert rows, f'{split}: empty'
        assert all('orig' in r and 'split' in r for r in rows)
        assert {r['split'] for r in rows} == {split}
        validate_group_key(rows, 'orig')
        # 주입이 문장을 다시 써 text 는 갈라지지만 orig 는 원문으로 묶는다 —
        # 묶이지 않으면 그룹 보호가 no-op 이다.
        assert len({r['orig'] for r in rows}) < len({r['text'] for r in rows})
        load_jsonl(str(path))


def test_unique_text_counts_are_pinned(converted):
    """중복 문장 규모를 고정한다 — 늘면 test 점수가 그만큼 부푼다."""
    actual = {
        split: len({r['text'] for r in records})
        for split, records in converted['by_split'].items()
    }
    assert actual == GOLDEN_UNIQUE_TEXTS


def test_cross_split_exposure_is_pinned(converted):
    """train 에도 있는 valid/test 행과 그 span 수를 고정한다.

    공식 split 이 물려준 성질이라 제거하지 않는다. 근거는 "외부 공개 수치와
    비교" 가 **아니다** — 18→6 매핑으로 라벨 공간이 이미 달라져
    published OntoNotes NER F1 과 나란히 못 놓는다. 남은 근거는 공식 split 이
    재현·인용에 유리하고 노출이 작다는 것(현재 test span 의 0.81%)이며,
    커지면 유지 결정을 다시 봐야 한다. 그래서 규모를 못 박는다.
    """
    train_texts = {r['text'] for r in converted['by_split']['train']}
    rows = {}
    spans = {}
    for split in ('valid', 'test'):
        shared = [
            r for r in converted['by_split'][split]
            if r['text'] in train_texts
        ]
        rows[split] = len(shared)
        spans[split] = sum(len(r['entities']) for r in shared)
    assert rows == GOLDEN_ROWS_ALSO_IN_TRAIN
    assert spans == GOLDEN_SPANS_ALSO_IN_TRAIN


@pytest.mark.skipif(
    not all((PII_DIR / f'{s}.jsonl').exists() for s in PII_SPLITS),
    reason=f'PII-injected output incomplete at {PII_DIR} '
           f'(needs all of {PII_SPLITS})',
)
def test_injection_replays_exactly():
    """주입 단계를 재생해 산출물과 대조한다 — 이 이슈의 외부 앵커.

    주입에서 비결정적인 부품은 LLM 하나이고 그 산출 문장은 `text` 로 남아
    있다. 나머지 두 입력은 복구된다 — 원천 gold 는 변환 산출물에, 합성 PII
    값은 `random.Random(42)` 의 결정론에. 그래서 주입 단계 전체를 다시 돌려
    최종 파일과 맞춰볼 수 있다.

    **이것이 "검증이 NER 을 안 건드렸다" 의 유일한 유효 증거다.** 검증기
    자신의 카운터로는 못 보인다 — `dropped_count` 는 `drop_span` 정책에서
    정의상 0 이고, `exempt_count` 도 `_apply_policy` 가 exempt 를 통째로
    `keep_keys` 에 넣으므로 "exempt 수 = 최종 NER 수" 가 눈먼 라벨러로도
    성립한다. 재생은 검증·`restore_groups` 코드를 한 줄도 쓰지 않는다.

    **민감도는 같은 대조가 실증한다** — 재생에만 있고 산출물에 없는 것이
    검증이 지운 PII 다. 대조가 삭제를 못 보는 것이었다면 그 값이 0 이 나왔을
    것이므로 양성 대조 노릇을 한다. 다만 아래 `assert` 는 "0 보다 크다" 만
    보므로, 이 파일이 보증하는 것은 **대조가 삭제를 본다** 까지다.

    **이 검사가 못 보는 것을 분명히 해 둔다.**

    - 비교가 집합이라 **같은 엔티티가 한 행에 중복**되는 것은 안 갈린다.
    - 산출 행이 **통째로 사라진** 경우는 `continue` 로 지나간다 — 행 수는
      §현 상태 가 문서로 못 박고 여기서는 안 본다.
    - 검증이 PII 를 **과다 삭제**해도 통과한다(문턱이 0 이라 642 든 842 든
      지난다). 삭제 규모는 `verify.json` 이 기록한다.
    - **산출 파일 자체가 온전한지는 안 본다** — 없는 행, 늘어난 행, 중복
      `id` 는 원본을 기준으로 도는 이 루프가 그냥 지나간다(실증됐다). 그쪽은
      행 수를 못 박은 §현 상태 와 `test_injected_output_keeps_the_group_key`
      의 그룹 키 검사가 본다.

    전부 이 앵커의 목적(주입 단계가 재생되나 · 검증이 NER 을 건드렸나) 밖이라
    문턱을 올리지 않았다. 목적을 넓히려면 검사를 나누는 편이 낫다.
    """
    import logging

    from ner.augmenters.pii.llm_injector import (
        LLMInjector, extract_spans, merge_entities,
    )
    from ner.augmenters.pii.schema import Entity

    logging.disable(logging.WARNING)  # 못 찾은 gold 를 행마다 경고한다
    try:
        pii_labels = ['EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD']
        ner_mismatch = missing = verifier_removed = 0
        for split in PII_SPLITS:
            source = _read_jsonl(RAW_DIR.parent / f'{split}.jsonl')
            out = {r['id']: r for r in
                   _read_jsonl(PII_DIR / f'{split}.jsonl')}
            injector = LLMInjector(
                client=None, lang='en', seed=42, pii_labels=pii_labels,
            )
            for row in source:
                # rng 는 산출에 남지 않은 행에서도 소비된다 — 순서를 지켜야
                # 뒤 행의 PII 값이 어긋나지 않는다.
                labels = injector._pick_labels(injector._sample_n())
                values = (injector._generate_pii_values(labels)
                          if labels else {})
                record = out.get(row['id'])
                if record is None:
                    continue
                if not labels:
                    # 주입할 것이 없으면 `_inject_async` 는 원본 엔티티를
                    # 그대로 복사한다 — offset 재계산이 없다.
                    replayed = {
                        (e['label'], e['start_char'], e['end_char'], e['text'])
                        for e in row['entities']
                    }
                else:
                    spans = extract_spans(
                        record['text'], pii_values=values,
                        original_entities=[
                            Entity(label=e['label'],
                                   start_char=e['start_char'],
                                   end_char=e['end_char'], text=e['text'])
                            for e in row['entities']
                        ],
                    )
                    replayed = {
                        (e.label, e.start_char, e.end_char, e.text)
                        for e in merge_entities(spans, injector._merge_rules)
                    }
                final = {
                    (e['label'], e['start_char'], e['end_char'], e['text'])
                    for e in record['entities']
                }
                missing += len(final - replayed)
                only_replayed = replayed - final
                verifier_removed += len(
                    [x for x in only_replayed if x[0] in pii_labels]
                )
                ner_replayed = {x for x in replayed if x[0] not in pii_labels}
                ner_final = {x for x in final if x[0] not in pii_labels}
                if ner_replayed != ner_final:
                    ner_mismatch += 1
    finally:
        logging.disable(logging.NOTSET)

    assert missing == 0, f'{missing} span(s) in output that replay cannot make'
    assert ner_mismatch == 0, f'{ner_mismatch} row(s) differ on NER spans'
    # 양성 대조 — 대조가 삭제를 실제로 검출한다.
    assert verifier_removed > 0, 'replay detected no verifier deletion at all'
