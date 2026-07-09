import pytest

from career_agent.runtime.json_utils import extract_json_object


def test_extract_json_object_accepts_plain_json() -> None:
    payload = extract_json_object('{"decision":"final_answer","reason":"ok"}')
    assert payload == {"decision": "final_answer", "reason": "ok"}


def test_extract_json_object_accepts_fenced_json() -> None:
    text = '```json\n{"decision":"call_tool","tool_name":"read_file"}\n```'
    payload = extract_json_object(text)
    assert payload["decision"] == "call_tool"
    assert payload["tool_name"] == "read_file"


def test_extract_json_object_accepts_explanatory_prefix() -> None:
    text = '下面是决策：\n{"decision":"load_skill","skill_name":"role_matching"}\n请执行。'
    payload = extract_json_object(text)
    assert payload["decision"] == "load_skill"
    assert payload["skill_name"] == "role_matching"


def test_extract_json_object_rejects_array() -> None:
    with pytest.raises(ValueError, match="object"):
        extract_json_object('[{"decision":"final_answer"}]')


def test_extract_json_object_rejects_missing_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        extract_json_object("我需要继续分析。")
