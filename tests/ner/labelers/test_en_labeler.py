"""영어 라벨러 — 프롬프트·설정 계약. 네트워크를 쓰지 않는다.

라벨러의 실제 추출 품질은 vLLM 이 있어야 재므로 여기서는 다루지 않는다.
이 파일이 고정하는 것은 **라벨 공간과 경계 선언이 매핑표와 어긋나지 않는가**
이며, 어긋나면 주입 검증이 gold 와 다른 자로 재게 된다.
"""
from ner.augmenters.ontonotes_en.mapping import PRODUCED_LABELS
from ner.labelers.en import DEFAULT_ENTITY_TYPES, SINGLE_PROMPT_TEMPLATE
from ner.labelers.en.vllm_ner_labeler import VllmNERLabeler

CANONICAL_10 = {
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
    'DAT', 'EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD',
}


def test_entity_types_are_the_canonical_ten():
    assert set(DEFAULT_ENTITY_TYPES) == CANONICAL_10
    assert len(DEFAULT_ENTITY_TYPES) == 10


def test_labeler_covers_every_label_the_converter_produces():
    """변환기가 만드는 라벨을 라벨러가 모르면 검증에서 전부 missed 가 된다."""
    assert PRODUCED_LABELS <= set(DEFAULT_ENTITY_TYPES)


def test_prompt_takes_the_expected_placeholders():
    rendered = SINGLE_PROMPT_TEMPLATE.format(
        entity_types='/'.join(DEFAULT_ENTITY_TYPES),
        sentence='Boston grew.',
    )
    assert 'Boston grew.' in rendered
    assert '{sentence}' not in rendered


def test_prompt_declares_facilities_as_org():
    """EN 은 JA·VI 관례를 따른다 — 매핑표의 `FAC` → `ORG` 와 같은 선언이다."""
    lowered = SINGLE_PROMPT_TEMPLATE.lower()
    assert 'man-made facilit' in lowered
    for facility in ('airport', 'hospital', 'stadium', 'museum'):
        assert facility in lowered


def test_prompt_excludes_the_dropped_source_types():
    """NORP·LANGUAGE·LAW·수량류는 매핑표가 드롭하므로 라벨러도 뽑으면 안 된다."""
    lowered = SINGLE_PROMPT_TEMPLATE.lower()
    assert 'do not label' in lowered
    for excluded in ('nationality', 'language names', 'laws', 'percentages'):
        assert excluded in lowered


def test_labeler_declares_english_for_the_sentence_splitter():
    labeler = VllmNERLabeler(base_url='http://localhost:1/v1', model='x')
    assert labeler.lang == 'en'
    assert labeler.entity_types == DEFAULT_ENTITY_TYPES


def _examples():
    """프롬프트의 `Input:`/`Output:` 예시 쌍을 뽑는다 (마지막 자리표시자 제외)."""
    import json
    import re

    pairs = re.findall(
        r'^Input: (.+)\nOutput: (\[.*\])$', SINGLE_PROMPT_TEMPLATE, re.M,
    )
    return [
        (sentence, json.loads(payload.replace('{{', '{').replace('}}', '}')))
        for sentence, payload in pairs
    ]


def test_no_example_slips_past_the_parser():
    """예시가 하나도 검사에서 조용히 빠지지 않는다.

    아래 검사들은 정규식으로 예시를 뽑는데 그 정규식은 `Output:` 이 한 줄일
    때만 맞는다. 여러 줄로 쓴 예시가 생기면 파싱되지 않고, 그 예시에 위반이
    있어도 검사가 초록으로 통과한다 — 검사 셋 전체가 fail-open 이 된다.
    그래서 파싱된 수와 실제 예시 수를 대조한다.
    """
    import re

    # 마지막 `Input: {sentence}` 는 예시가 아니라 자리표시자다.
    declared = len(re.findall(r'^Input: ', SINGLE_PROMPT_TEMPLATE, re.M)) - 1
    assert len(_examples()) == declared, (
        f'{declared - len(_examples())} example(s) not parsed — '
        f'the checks below would skip them silently'
    )


def test_every_example_spans_appear_in_its_own_sentence():
    """예시 정답의 span 이 그 예시 문장 안에 글자 그대로 있다.

    프롬프트 첫 규칙이 "입력에 나온 대로 글자 하나까지 복사하라" 인데 예시가
    그걸 어기면 모델에게 규칙과 반례를 함께 주는 셈이다.
    """
    examples = _examples()
    assert len(examples) >= 6
    for sentence, spans in examples:
        for span in spans:
            assert span['text'] in sentence, (
                f'{span["text"]!r} not in {sentence!r}'
            )


def test_every_example_label_is_canonical():
    for _, spans in _examples():
        for span in spans:
            assert span['type'] in CANONICAL_10


def test_examples_do_not_teach_that_legislatures_are_non_entities():
    """`legislatures` 를 ORG 로 선언해 놓고 예시에서 버리면 안 된다.

    실제로 그랬다 — `Congress passed the Clean Air Act...` 의 정답이 `[]`
    였다. 법률을 빼라고 가르치려다 같은 문장의 입법부까지 버려, 검증 LLM 이
    `Congress` 를 놓치고 그 gold 가 지워졌다.
    """
    for sentence, spans in _examples():
        if 'Congress' not in sentence:
            continue
        assert any(s['text'] == 'Congress' and s['type'] == 'ORG'
                   for s in spans), (
            f'example teaches Congress is not an entity: {sentence!r}'
        )
