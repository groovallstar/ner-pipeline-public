"""프롬프트가 컨텍스트 예산을 넘지 않는지 고정한다.

**왜 이 검사가 필요한가.** vLLM 은 `프롬프트 + max_tokens(출력 예약) > max-model-len`
이면 요청을 받자마자 400 으로 거절한다. `max_tokens` 는 실제로 쓴 양이 아니라 미리
잡아두는 자리라, 출력이 짧아도 예약분 전체가 한도 계산에 들어간다. 한국어 프롬프트가
규칙 추가로 계속 자라 이 선을 넘었고, **넘은 뒤에도 두 달 가까이 아무 신호가 없었다** —
벤치마크 러너가 예외를 세기만 하고 리포트는 정상 생성해서, 로그를 안 보면 "요청이
서버에 닿지도 않았다" 가 "모델 성능이 나쁘다" 로 읽혔기 때문이다.

**임계값은 벽에 붙이지 않는다.** 실제 벽은 `max-model-len − 출력 예약` 이지만, 거기에
맞춰 잡으면 넘는 순간에야 걸려 고칠 여유가 없다. 그래서 출력 예약 한 벌만큼(`RESERVED`)
완충을 더 뺀 값을 예산으로 쓴다 — 걸린 시점에도 실제 요청은 아직 200 이므로, 프롬프트를
줄이든 서버 설정을 올리든 판단할 시간이 남는다. 입력 문장 자체도 이 완충 안에서 소화된다
(여기서 재는 것은 문장을 뺀 고정 지시문이다).

**문자 수가 아니라 토큰으로 센다.** 400 을 만드는 것은 토큰이고, 한국어는 문자당 토큰
비율이 텍스트에 따라 흔들려 문자 수 근사는 벽 근처에서 빗나간다. 대신 토크나이저가
로컬 HF 캐시에 있어야 하므로, 캐시가 없으면 건너뛴다 —
`tests/ner/CLAUDE.md` 의 "로컬 자원 부재 → skip" 관례를 따른다.
"""

import inspect
import os
import re
from pathlib import Path

import pytest

from ner.labelers.base_vllm_labeler import BaseVllmLabeler
from ner.labelers.ja.ner_prompts import (
    DEFAULT_ENTITY_TYPES as JA_TYPES,
    SINGLE_PROMPT_TEMPLATE as JA_TEMPLATE,
)
from ner.labelers.ko.ner_prompts import (
    DEFAULT_ENTITY_TYPES as KO_TYPES,
    SINGLE_PROMPT_TEMPLATE as KO_TEMPLATE,
)
from ner.labelers.vi.ner_prompts import (
    DEFAULT_ENTITY_TYPES as VI_TYPES,
    SINGLE_PROMPT_TEMPLATE as VI_TEMPLATE,
)

_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE = _ROOT / "docker" / "vllm" / "docker-compose.yml"

# 측정 기준 토크나이저. 서버마다 토크나이저가 달라 값이 조금씩 다르지만, 예산은 벽에서
# 한 벌 물러나 있어 그 차이를 흡수한다.
TOKENIZER_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-8bit"

# 라벨러가 실제로 예약하는 출력 자리. 기본값을 올리면 여유가 그만큼 줄어야 하므로
# 상수로 베끼지 않고 서명에서 읽는다.
RESERVED = inspect.signature(BaseVllmLabeler.__init__).parameters["max_tokens"].default


def _max_model_len() -> int:
    """서버가 실제로 뜨는 컨텍스트 길이를 compose 기본값에서 읽는다.

    숫자를 테스트에 베껴 두면 서버 설정이 바뀌어도 검사가 옛 전제로 계속 통과한다.
    """
    text = _COMPOSE.read_text(encoding="utf-8")
    match = re.search(r"--max-model-len \$\{VLLM_MAX_MODEL_LEN:-(\d+)\}", text)
    assert match, f"compose 에서 max-model-len 기본값을 못 읽었다: {_COMPOSE}"
    return int(match.group(1))


MAX_MODEL_LEN = _max_model_len()
WALL = MAX_MODEL_LEN - RESERVED           # 이 선을 넘으면 서버가 400 으로 거절한다
BUDGET = WALL - RESERVED                  # 벽에서 예약 한 벌만큼 물러난 자리

_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_CACHE = _HF_HOME / "hub" / ("models--" + TOKENIZER_MODEL.replace("/", "--"))
_HAS_TOKENIZER = any(_CACHE.glob("snapshots/*/tokenizer.json"))

CASES = [
    ("ko", KO_TEMPLATE, KO_TYPES),
    ("ja", JA_TEMPLATE, JA_TYPES),
    ("vi", VI_TEMPLATE, VI_TYPES),
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, local_files_only=True)


def test_every_labeler_reserves_the_same_output_room():
    """예산은 베이스의 예약값으로 계산한다 — 서브클래스가 따로 들면 그 언어만 벗어난다.

    ko·ja·vi 는 각자 `max_tokens` 기본값을 다시 적으므로, 한 곳만 올려도 베이스에서
    읽은 예산은 그대로다. 그러면 이 파일의 검사가 그 언어에 대해서는 틀린 벽을 본다.
    """
    from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler as JA
    from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler as KO
    from ner.labelers.vi.vllm_ner_labeler import VllmNERLabeler as VI

    for cls in (KO, JA, VI):
        reserved = inspect.signature(cls.__init__).parameters["max_tokens"].default
        assert reserved == RESERVED, f"{cls.__module__} 의 예약값 {reserved} ≠ 베이스 {RESERVED}"


def test_the_budget_leaves_room_between_itself_and_the_wall():
    """예산이 벽과 같아지면 이 검사는 400 을 예고하지 못하고 사후 확인만 한다."""
    assert RESERVED > 0
    assert BUDGET > 0, f"출력 예약 {RESERVED} 가 컨텍스트 {MAX_MODEL_LEN} 에 비해 크다"
    assert WALL - BUDGET == RESERVED


@pytest.mark.skipif(not _HAS_TOKENIZER, reason=f"{TOKENIZER_MODEL} tokenizer not in local HF cache")
@pytest.mark.parametrize("lang,template,types", CASES, ids=[c[0] for c in CASES])
def test_fixed_instruction_fits_the_context_budget(tokenizer, lang, template, types):
    """고정 지시문이 예산 안이어야 한다. 넘으면 규칙을 줄이거나 서버 설정을 올린다."""
    rendered = template.format(entity_types=", ".join(types), sentence="")
    count = len(tokenizer.encode(rendered))
    assert count <= BUDGET, (
        f"{lang} 지시문 {count} 토큰 > 예산 {BUDGET} "
        f"(벽 {WALL} = max-model-len {MAX_MODEL_LEN} − 출력 예약 {RESERVED})"
    )
    assert count + RESERVED < MAX_MODEL_LEN
