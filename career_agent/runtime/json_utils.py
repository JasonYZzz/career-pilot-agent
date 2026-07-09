from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取单个 JSON 对象。

    参数: text 模型原始输出，可能包含 ```json 代码块或前后解释文字。
    返回: 解析后的 dict。
    """
    candidates = _candidate_json_strings(text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        raise ValueError("model JSON must be an object")
    raise ValueError("no JSON object found in model output")


def _candidate_json_strings(text: str) -> list[str]:
    """按可靠性顺序返回可能的 JSON 字符串候选。"""
    stripped = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates = [item.strip() for item in fenced]
    candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates
