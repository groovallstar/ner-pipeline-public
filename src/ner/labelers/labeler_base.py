"""JSON 응답 파싱을 위한 LLM 라벨러 공통 유틸리티."""
import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> List[dict]:
    """LLM 출력에서 JSON span 리스트를 추출한다. 실패 시 []를 반환한다.

    처리 대상: 단독 배열, 래핑된 딕셔너리, think 태그, 마크다운 펜스.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "text" in data and "type" in data:
                return [data]
            for val in data.values():
                if isinstance(val, list):
                    return val
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []
