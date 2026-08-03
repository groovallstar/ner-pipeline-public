"""한국어 NER 프롬프트의 규칙 정합성 테스트.

**few-shot 예시도 규칙의 일부다**(`docs/specs/coding-conventions.md`). 규칙 문장을
고치고 예시를 안 고치면 프롬프트가 같은 모양에 두 답을 가르치는데, 그건 어느 한쪽을
고르는 것보다 무조건 나쁘고 **어떤 기계 검사도 안 잡는다** — 게이트는 diff 의 글자만
읽고, 라벨러 출력은 LLM 을 불러야 보인다. 그래서 예시에서 기계로 확인할 수 있는 것을
여기서 고정한다.
"""

import json
import re

from ner.labelers.ko.ko_evt_axis1_audit import AXIS1_HEADS
from ner.labelers.ko.ner_prompts import SINGLE_PROMPT_TEMPLATE, SYSTEM_PROMPT

_PAIR = re.compile(r"입력: (?P<text>.+)\n출력: (?P<spans>\[.*\])")
_QUOTED = re.compile(r'"([^"]+)"')


def _few_shots():
    """`{{` 이스케이프를 되돌려 (문장, [span]) 쌍으로 읽는다."""
    body = SINGLE_PROMPT_TEMPLATE.replace("{{", "{").replace("}}", "}")
    out = []
    for match in _PAIR.finditer(body):
        text = match.group("text")
        if text.strip() == "{sentence}":
            continue
        out.append((text, json.loads(match.group("spans"))))
    return out


def test_few_shot_examples_are_present_and_parse():
    shots = _few_shots()
    assert len(shots) >= 8
    assert all(spans for _, spans in shots)


def _squeeze(text: str) -> str:
    """공백을 지운 좌표계.

    프롬프트는 복합 행정지명·날짜를 **일부러 붙여서** 낸다(`서울 삼성동` →
    `서울삼성동`, 절대 분리 금지 규칙). 그래서 원문 그대로의 부분문자열 검사는
    의도된 출력을 오탐한다. 공백만 지우면 그 규칙은 통과시키면서 "입력에 없는
    문자열" 은 그대로 걸린다.
    """
    return re.sub(r"\s+", "", text)


def test_every_few_shot_span_occurs_in_its_own_input():
    """예시 출력이 입력에 없는 문자열이면 모델에게 없는 자리를 가르치는 것이다."""
    for text, spans in _few_shots():
        squeezed = _squeeze(text)
        for span in spans:
            assert _squeeze(span["text"]) in squeezed, (span["text"], text)


def test_no_two_same_type_spans_are_separated_only_by_whitespace():
    """붙어 있는 같은 타입 span 둘은 쪼갤지 합칠지가 안 정해졌다는 신호다.

    실제로 이 검사가 잡은 것 — `천안함 침몰` + `추모식` 을 두 EVT 로 가르면서 세 줄
    위에서는 같은 모양(`소치올림픽 폐막식`)을 한 span 으로 가르쳤다. canonical 은
    고유명을 낀 것을 축3 최대-span 으로 정해 두었으므로 답은 하나여야 하고, 한
    프롬프트가 같은 모양에 두 답을 가르치는 것은 어느 한쪽을 고르는 것보다 나쁘다.
    """
    for text, spans in _few_shots():
        squeezed = _squeeze(text)
        placed = []
        for span in spans:
            body = _squeeze(span["text"])
            start = squeezed.index(body)
            placed.append((start, start + len(body), span["type"]))
        placed.sort()
        for (_, end, left), (start, _, right) in zip(placed, placed[1:]):
            assert not (left == right and start == end), (
                f"{left} spans touch in: {text}")


def test_axis1_head_list_matches_the_module():
    """프롬프트가 canonical 보다 좁은 head 를 가르치면 라벨링이 gold 와 어긋난다."""
    line = next(ln for ln in SINGLE_PROMPT_TEMPLATE.splitlines()
                if "고유명이 조사 없이 바로 앞에 오면" in ln)
    quoted = _QUOTED.findall(line)
    heads = [q for q in quoted if q in AXIS1_HEADS]
    assert set(heads) == set(AXIS1_HEADS)


def test_both_backends_teach_the_axis1_rule():
    """vLLM 과 OpenAI 는 템플릿이 두 벌이라 한쪽만 고치면 백엔드가 갈린다."""
    for template in (SINGLE_PROMPT_TEMPLATE, SYSTEM_PROMPT):
        assert "고유명" in template and "사건 head" in template


def test_particle_bearing_modifier_is_taught_as_out_of_span():
    """`파리에서 테러` 는 복합명사가 아니라 문장 성분이다 — 예시가 그걸 보여야 한다."""
    shots = dict(_few_shots())
    text = next(t for t in shots if "파리에서 테러" in t)
    types = {s["text"]: s["type"] for s in shots[text]}
    assert types["파리"] == "LOC"
    assert types["테러"] == "EVT"
    assert "파리에서 테러" not in types
