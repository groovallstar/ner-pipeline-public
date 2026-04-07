"""Shared LLM labeler utilities for JSON response parsing."""
import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> List[dict]:
    """Extract JSON span list from LLM output. Returns [] on failure.

    Handles: bare arrays, wrapped dicts, think tags, markdown fences.
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
