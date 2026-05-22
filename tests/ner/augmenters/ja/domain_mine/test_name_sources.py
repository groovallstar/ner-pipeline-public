"""name_sources 순수 파서·정규화 단위 테스트 (네트워크 호출 없음)."""

from ner.augmenters.ja.domain_mine import name_sources as ns

_LAW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DataRoot>
  <Result><Code>0</Code><Message/></Result>
  <ApplData>
    <Category>2</Category>
    <LawNameListInfo>
      <LawId>321CONSTITUTION</LawId>
      <LawName>日本国憲法</LawName>
      <LawNo>昭和二十一年憲法</LawNo>
      <PromulgationDate>19461103</PromulgationDate>
    </LawNameListInfo>
    <LawNameListInfo>
      <LawId>106DF0000000065</LawId>
      <LawName>絞罪器械図式</LawName>
      <LawNo>明治六年太政官布告第六十五号</LawNo>
      <PromulgationDate>18730220</PromulgationDate>
    </LawNameListInfo>
  </ApplData>
</DataRoot>"""

_SPARQL_JSON = {
    'results': {
        'bindings': [
            {'itemLabel': {'type': 'literal', 'value': '吾輩は猫である'}},
            {'itemLabel': {'type': 'literal', 'value': '坊っちゃん'}},
            # 라벨 없는 항목은 Q-ID 폴백 → 제외돼야 함
            {'itemLabel': {'type': 'literal', 'value': 'Q494'}},
        ]
    }
}


def test_parse_law_list_extracts_names():
    names = ns.parse_law_list(_LAW_XML)
    assert names == ['日本国憲法', '絞罪器械図式']


def test_parse_law_list_empty_on_no_info():
    xml = '<DataRoot><ApplData/></DataRoot>'
    assert ns.parse_law_list(xml) == []


def test_parse_sparql_names_drops_qid_fallback():
    names = ns.parse_sparql_names(_SPARQL_JSON)
    assert names == ['吾輩は猫である', '坊っちゃん']


def test_parse_sparql_names_empty_bindings():
    assert ns.parse_sparql_names({'results': {'bindings': []}}) == []


def test_normalize_names_dedup_preserves_order():
    raw = ['Suica', 'PASMO', 'Suica', 'ICOCA']
    assert ns.normalize_names(raw) == ['Suica', 'PASMO', 'ICOCA']


def test_normalize_names_length_filters():
    raw = ['A', '  ', 'AB', 'x' * 41, '有効な名前']
    assert ns.normalize_names(raw) == ['AB', '有効な名前']


def test_collect_transit_card_uses_constant_no_network():
    names = ns.collect_domain_names('transit_card')
    assert 'Suica' in names and 'ICOCA' in names
    # 정규화로 중복 제거됨
    assert len(names) == len(set(names))


def test_collect_unknown_domain_raises():
    try:
        ns.collect_domain_names('food')
        assert False, 'expected ValueError'
    except ValueError as exc:
        assert 'unknown domain' in str(exc)
