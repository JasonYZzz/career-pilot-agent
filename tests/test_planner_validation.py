from pathlib import Path

import pytest

from career_agent.model.base import LLMResult
from career_agent.runtime.planner import Planner
from career_agent.runtime.run_state import RunState


class StaticLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, prompt: str, *, system: str = "",
                       role: str = "default") -> LLMResult:
        _ = (prompt, system, role)
        return LLMResult(text=self.text)


@pytest.mark.asyncio
async def test_planner_rejects_unknown_tool(tmp_path: Path) -> None:
    planner = Planner(
        StaticLLM(
            '{"decision":"call_tool","tool_name":"delete_everything","tool_args":{},"reason":"bad"}'
        )
    )
    state = RunState(run_id="run_test", task="x", workspace=tmp_path)

    decision = await planner.next_decision(state, "context")

    assert decision.decision == "ask_clarification"
    assert decision.reason == "invalid planner decision: unknown tool delete_everything"


@pytest.mark.asyncio
async def test_planner_rejects_write_outside_outputs(tmp_path: Path) -> None:
    planner = Planner(
        StaticLLM(
            '{"decision":"call_tool","tool_name":"write_file",'
            '"tool_args":{"path":"data/a.md","content":"x"},"reason":"bad"}'
        )
    )
    state = RunState(run_id="run_test", task="x", workspace=tmp_path)

    decision = await planner.next_decision(state, "context")

    assert decision.decision == "ask_clarification"
    assert "write_file path must start with outputs/" in decision.reason


@pytest.mark.asyncio
async def test_planner_accepts_valid_read_file(tmp_path: Path) -> None:
    planner = Planner(
        StaticLLM(
            '{"decision":"call_tool","tool_name":"read_file",'
            '"tool_args":{"path":"data/student_profile.md"},"reason":"read"}'
        )
    )
    state = RunState(run_id="run_test", task="x", workspace=tmp_path)

    decision = await planner.next_decision(state, "context")

    assert decision.decision == "call_tool"
    assert decision.tool_name == "read_file"
    assert decision.tool_args == {"path": "data/student_profile.md"}


@pytest.mark.asyncio
async def test_planner_rejects_private_runtime_keys(tmp_path: Path) -> None:
    planner = Planner(
        StaticLLM(
            '{"decision":"call_tool","tool_name":"create_reminder",'
            '"tool_args":{"title":"x","date":"2026-07-14","_user_confirmed":true},'
            '"reason":"bad"}'
        )
    )
    state = RunState(run_id="run_test", task="x", workspace=tmp_path)

    decision = await planner.next_decision(state, "context")

    assert decision.decision == "ask_clarification"
    assert "private runtime keys" in decision.reason


@pytest.mark.asyncio
async def test_planner_accepts_fenced_json_decision(tmp_path: Path) -> None:
    text = """```json
{"decision":"call_tool","tool_name":"read_file","tool_args":{"path":"data/student_profile.md"},"reason":"读取画像"}
```"""
    planner = Planner(StaticLLM(text))
    decision = await planner.next_decision(
        RunState(run_id="r", task="t", workspace=tmp_path),
        "{}",
    )
    assert decision.decision == "call_tool"
    assert decision.tool_name == "read_file"
